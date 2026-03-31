"""Standalone local semantic scoring runtime for chonkify wheels."""

from __future__ import annotations

import gc
import importlib
import inspect
import json
import math
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, Mapping, Optional, Sequence

PoolingMode = Literal["mean", "cls"]
CanonicalEmbeddingRole = Literal["query", "passage"]
SemanticPairModelKind = Literal["bi", "cross"]

_EMBEDDING_MODEL_SPEC_FILENAME: Final[str] = "embedding_model_spec.json"
_PAIR_MODEL_SPEC_FILENAME: Final[str] = "semantic_pair_model_spec.json"
_QUERY_PREFIX_BGE_EN: Final[str] = "Represent this sentence for searching relevant passages: "
_QUERY_PREFIX_BGE_ZH: Final[str] = "为这个句子生成表示以用于检索相关文章："


@dataclass(frozen=True, slots=True)
class EmbeddingModelSpec:
    """Exact embedding metadata needed for source-faithful local scoring."""

    default_pooling: Optional[PoolingMode] = None
    query_prefix: str = ""
    passage_prefix: str = ""
    revision: Optional[str] = None


@dataclass(frozen=True, slots=True)
class SemanticPairModelSpec:
    """Exact runtime metadata needed to score semantic text pairs."""

    kind: SemanticPairModelKind


@dataclass(frozen=True, slots=True)
class LocalSemanticRuntimeConfig:
    """Environment-resolved runtime settings for local semantic scoring."""

    semantic_pair_device: str = "cpu"
    semantic_pair_local_files_only: bool = False

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
    ) -> "LocalSemanticRuntimeConfig":
        values = env if env is not None else os.environ
        return cls(
            semantic_pair_device=_env_str(
                values,
                "CHONKIFY_SEMANTIC_PAIR_DEVICE",
                "SLM_SEMANTIC_PAIR_DEVICE",
                default="cpu",
            ),
            semantic_pair_local_files_only=_env_bool(
                values,
                "CHONKIFY_SEMANTIC_PAIR_LOCAL_FILES_ONLY",
                "SLM_SEMANTIC_PAIR_LOCAL_FILES_ONLY",
                default=False,
            ),
        )


@dataclass(frozen=True, slots=True)
class LocalSemanticProfile:
    """Minimal local rerank profile registry for standalone chonkify wheels."""

    profile_id: str
    model: str
    model_kind: SemanticPairModelKind
    provider: Literal["local_cross_encoder"] = "local_cross_encoder"


_EMPTY_MODEL_SPEC: Final[EmbeddingModelSpec] = EmbeddingModelSpec()
_EXACT_EMBEDDING_MODEL_SPECS: Final[Mapping[str, EmbeddingModelSpec]] = {
    "intfloat/e5-small-v2": EmbeddingModelSpec(
        default_pooling="mean",
        query_prefix="query: ",
        passage_prefix="passage: ",
    ),
    "intfloat/e5-base-v2": EmbeddingModelSpec(
        default_pooling="mean",
        query_prefix="query: ",
        passage_prefix="passage: ",
    ),
    "intfloat/multilingual-e5-small": EmbeddingModelSpec(
        default_pooling="mean",
        query_prefix="query: ",
        passage_prefix="passage: ",
        revision="d829207ab28e6a5fb3aafb5d4c44111b8146db32",
    ),
    "BAAI/bge-small-en-v1.5": EmbeddingModelSpec(
        default_pooling="cls",
        query_prefix=_QUERY_PREFIX_BGE_EN,
        revision="b49342cba6a5914c1760cd4aae1d75a6f2e8fc4c",
    ),
    "BAAI/bge-base-en-v1.5": EmbeddingModelSpec(
        default_pooling="cls",
        query_prefix=_QUERY_PREFIX_BGE_EN,
    ),
    "BAAI/bge-large-en-v1.5": EmbeddingModelSpec(
        default_pooling="cls",
        query_prefix=_QUERY_PREFIX_BGE_EN,
    ),
    "BAAI/bge-small-zh-v1.5": EmbeddingModelSpec(
        default_pooling="cls",
        query_prefix=_QUERY_PREFIX_BGE_ZH,
    ),
    "BAAI/bge-base-zh-v1.5": EmbeddingModelSpec(
        default_pooling="cls",
        query_prefix=_QUERY_PREFIX_BGE_ZH,
    ),
    "BAAI/bge-large-zh-v1.5": EmbeddingModelSpec(
        default_pooling="cls",
        query_prefix=_QUERY_PREFIX_BGE_ZH,
    ),
}
_ROLE_ALIASES: Final[Mapping[str, CanonicalEmbeddingRole]] = {
    "query": "query",
    "passage": "passage",
    "document": "passage",
    "doc": "passage",
}
_EXACT_PAIR_MODEL_SPECS: Final[Mapping[str, SemanticPairModelSpec]] = {
    "baai/bge-reranker-v2-m3": SemanticPairModelSpec(kind="cross"),
    "cross-encoder/ms-marco-minilm-l-6-v2": SemanticPairModelSpec(kind="cross"),
}
_DEFAULT_LOCAL_SEMANTIC_PROFILES: Final[Mapping[str, LocalSemanticProfile]] = {
    "cross_encoder_minilm_l6_local": LocalSemanticProfile(
        profile_id="cross_encoder_minilm_l6_local",
        model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        model_kind="cross",
    ),
    "bge_reranker_v2_m3_local": LocalSemanticProfile(
        profile_id="bge_reranker_v2_m3_local",
        model="BAAI/bge-reranker-v2-m3",
        model_kind="cross",
    ),
}

_MODEL_LOCK = threading.RLock()
_MODEL_CACHE: dict[str, Any] = {}
_MODEL_KIND_CACHE: dict[str, SemanticPairModelKind] = {}
_MODEL_DISABLED = False
_MODEL_FAILURE_NAME = ""
_MODEL_LAST_ERROR = ""
_MODEL_LOAD_FAILS = 0
_MODEL_DISABLED_UNTIL = 0.0


def _env_str(values: Mapping[str, str], *keys: str, default: str) -> str:
    for key in keys:
        value = str(values.get(key, "") or "").strip()
        if value:
            return value
    return str(default)


def _env_bool(values: Mapping[str, str], *keys: str, default: bool) -> bool:
    for key in keys:
        raw = str(values.get(key, "") or "").strip().lower()
        if not raw:
            continue
        if raw in {"1", "true", "yes", "on"}:
            return True
        if raw in {"0", "false", "no", "off"}:
            return False
        raise ValueError(f"invalid_bool_env {key}={raw!r}")
    return bool(default)


def resolve_local_semantic_profile(profile_id: str) -> LocalSemanticProfile:
    """Resolve a standalone local semantic profile by id."""

    try:
        return _DEFAULT_LOCAL_SEMANTIC_PROFILES[str(profile_id)]
    except KeyError as exc:
        raise ValueError(f"unknown_local_semantic_profile {profile_id!r}") from exc


def normalize_embedding_role(role: str) -> CanonicalEmbeddingRole:
    """Normalize embedding role aliases to the canonical contract."""

    role_norm = str(role or "").strip().lower()
    mapped = _ROLE_ALIASES.get(role_norm)
    if mapped is None:
        raise RuntimeError(f"invalid_embedding_role:{role_norm or 'empty'}")
    return mapped


def apply_embedding_prefix_once(text: str, prefix: str) -> str:
    """Prefix text idempotently when a model spec requires prompts."""

    if not prefix:
        return text
    if str(text or "").lower().startswith(prefix.lower()):
        return str(text)
    return f"{prefix}{text}"


def _load_sidecar_embedding_model_spec(model_name: str) -> EmbeddingModelSpec | None:
    model_path = Path(str(model_name or "")).expanduser()
    if not model_path.is_dir():
        return None
    spec_path = model_path / _EMBEDDING_MODEL_SPEC_FILENAME
    if not spec_path.is_file():
        return None
    try:
        payload = json.loads(spec_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"embedding_model_spec_invalid_json:{spec_path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"embedding_model_spec_not_object:{spec_path}")

    pooling_raw = payload.get("default_pooling")
    pooling: Optional[PoolingMode]
    if pooling_raw is None or str(pooling_raw).strip() == "":
        pooling = None
    else:
        pooling_s = str(pooling_raw).strip().lower()
        if pooling_s not in {"mean", "cls"}:
            raise RuntimeError(f"embedding_model_spec_invalid_pooling:{spec_path}")
        pooling = pooling_s  # type: ignore[assignment]

    revision_raw = str(payload.get("revision") or "").strip()
    return EmbeddingModelSpec(
        default_pooling=pooling,
        query_prefix=str(payload.get("query_prefix") or ""),
        passage_prefix=str(payload.get("passage_prefix") or ""),
        revision=revision_raw or None,
    )


def resolve_embedding_model_spec(model_name: str) -> EmbeddingModelSpec:
    """Resolve exact embedding metadata without model-name heuristics."""

    sidecar = _load_sidecar_embedding_model_spec(model_name)
    if sidecar is not None:
        return sidecar
    return _EXACT_EMBEDDING_MODEL_SPECS.get(str(model_name or "").strip(), _EMPTY_MODEL_SPEC)


def _normalize_model_kind(model_kind: str | None) -> SemanticPairModelKind | None:
    raw = str(model_kind or "").strip().lower()
    if not raw:
        return None
    if raw not in {"bi", "cross"}:
        raise RuntimeError(f"semantic_pair_model_kind_invalid:{raw}")
    return raw  # type: ignore[return-value]


def _load_sidecar_pair_model_spec(model_name: str) -> SemanticPairModelSpec | None:
    model_path = Path(str(model_name or "")).expanduser()
    if not model_path.is_dir():
        return None
    spec_path = model_path / _PAIR_MODEL_SPEC_FILENAME
    if not spec_path.is_file():
        return None
    try:
        payload = json.loads(spec_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"semantic_pair_model_spec_invalid_json:{spec_path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"semantic_pair_model_spec_not_object:{spec_path}")

    kind = _normalize_model_kind(str(payload.get("kind") or "")) or None
    if kind is None:
        raise RuntimeError(f"semantic_pair_model_spec_missing_kind:{spec_path}")
    return SemanticPairModelSpec(kind=kind)


def _infer_local_model_kind_from_files(model_name: str) -> SemanticPairModelKind | None:
    model_path = Path(str(model_name or "")).expanduser()
    if not model_path.is_dir():
        return None
    if (model_path / "modules.json").is_file():
        return "bi"

    config_path = model_path / "config.json"
    if not config_path.is_file():
        return None
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, Mapping):
        return None
    architectures = payload.get("architectures")
    if not isinstance(architectures, list):
        return None
    names = [str(item or "") for item in architectures if str(item or "").strip()]
    if any(name.endswith("ForSequenceClassification") for name in names):
        return "cross"
    return None


def _embedding_spec_is_defined(model_name: str) -> bool:
    spec = resolve_embedding_model_spec(str(model_name or ""))
    return bool(
        spec.default_pooling is not None
        or spec.query_prefix
        or spec.passage_prefix
        or spec.revision
    )


def resolve_semantic_pair_model_kind(
    model_name: str,
    *,
    model_kind: str | None = None,
    model: Any | None = None,
) -> SemanticPairModelKind:
    """Resolve bi-encoder versus cross-encoder behavior deterministically."""

    explicit_kind = _normalize_model_kind(model_kind)
    if explicit_kind is not None:
        return explicit_kind

    if model is not None:
        has_predict = callable(getattr(model, "predict", None))
        has_encode = callable(getattr(model, "encode", None))
        if has_predict and not has_encode:
            return "cross"
        if has_encode and not has_predict:
            return "bi"

    model_name_s = str(model_name or "").strip()
    if not model_name_s:
        return "bi"

    sidecar_spec = _load_sidecar_pair_model_spec(model_name_s)
    if sidecar_spec is not None:
        return sidecar_spec.kind

    exact_spec = _EXACT_PAIR_MODEL_SPECS.get(model_name_s.lower())
    if exact_spec is not None:
        return exact_spec.kind

    local_kind = _infer_local_model_kind_from_files(model_name_s)
    if local_kind is not None:
        return local_kind

    if _embedding_spec_is_defined(model_name_s):
        return "bi"

    raise RuntimeError(f"semantic_pair_model_kind_unknown:{model_name_s}")


def apply_semantic_role_prefix(text: str, *, role: str, model_name: str) -> str:
    """Apply exact query/passage prefixes only when a model spec requires them."""

    raw = str(text or "")
    try:
        normalized_role = normalize_embedding_role(str(role or ""))
    except RuntimeError:
        return raw
    spec = resolve_embedding_model_spec(str(model_name or ""))
    prefix = spec.query_prefix if normalized_role == "query" else spec.passage_prefix
    return apply_embedding_prefix_once(raw, prefix)


def _filter_kwargs_for_callable(target: Any, kwargs: Mapping[str, Any]) -> dict[str, Any]:
    try:
        sig = inspect.signature(target)
    except Exception:
        return dict(kwargs)
    params = dict(sig.parameters)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()):
        return dict(kwargs)
    return {str(key): value for key, value in dict(kwargs).items() if str(key) in params}


def _is_backend_available(torch_module: Any, backend_name: str) -> bool:
    backend = getattr(torch_module, backend_name, None)
    is_available = getattr(backend, "is_available", None)
    if callable(is_available):
        return bool(is_available())
    return backend is not None


def _iter_cache_clearers(torch_module: Any) -> list[tuple[str, Any]]:
    clearers: list[tuple[str, Any]] = []
    for backend_name in ("cuda", "mps", "xpu"):
        backend = getattr(torch_module, backend_name, None)
        clearer = getattr(backend, "empty_cache", None)
        if callable(clearer):
            clearers.append((backend_name, clearer))
    return clearers


def ensure_accelerate_clear_device_cache_compat() -> dict[str, Any]:
    """Backfill accelerate cache clearing when older installs omit the symbol."""

    try:
        memory_mod = importlib.import_module("accelerate.utils.memory")
    except Exception as exc:
        return {
            "ok": False,
            "patched": False,
            "reason": f"memory_import_failed:{type(exc).__name__}",
        }

    if hasattr(memory_mod, "clear_device_cache"):
        return {"ok": True, "patched": False, "reason": "already_present"}

    def _clear_device_cache(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        diag: dict[str, Any] = {
            "gc_collected": False,
            "torch_loaded": False,
            "backend_results": {},
        }
        try:
            gc.collect()
            diag["gc_collected"] = True
        except Exception as exc:
            diag["gc_error"] = type(exc).__name__
        try:
            import torch  # noqa: PLC0415
        except ImportError:
            diag["torch_error"] = "ImportError"
            return diag
        except Exception as exc:
            diag["torch_error"] = type(exc).__name__
            return diag

        diag["torch_loaded"] = True
        for backend_name, clear_fn in _iter_cache_clearers(torch):
            backend_diag: dict[str, Any] = {"available": False, "cleared": False}
            try:
                backend_diag["available"] = _is_backend_available(torch, backend_name)
            except Exception as exc:
                backend_diag["availability_error"] = type(exc).__name__
            if backend_diag["available"]:
                try:
                    clear_fn()
                    backend_diag["cleared"] = True
                except Exception as exc:
                    backend_diag["clear_error"] = type(exc).__name__
            diag["backend_results"][backend_name] = backend_diag
        return diag

    setattr(memory_mod, "clear_device_cache", _clear_device_cache)
    return {"ok": True, "patched": True, "reason": "installed_missing_symbol"}


def load_local_sentence_transformer_model(
    model_name: str,
    *,
    device: str = "cpu",
    local_files_only: bool = True,
    trust_remote_code: bool = False,
    model_kind: str | None = None,
) -> Any | None:
    """Load a local bi/cross encoder lazily with deterministic caching."""

    global _MODEL_DISABLED
    global _MODEL_FAILURE_NAME
    global _MODEL_LAST_ERROR
    global _MODEL_LOAD_FAILS
    global _MODEL_DISABLED_UNTIL

    model_name_s = str(model_name or "").strip()
    if not model_name_s or _MODEL_DISABLED:
        return None

    cached = _MODEL_CACHE.get(model_name_s)
    if cached is not None:
        return cached

    now = time.monotonic()
    if str(_MODEL_FAILURE_NAME) == model_name_s and float(now) < float(_MODEL_DISABLED_UNTIL):
        return None

    with _MODEL_LOCK:
        cached = _MODEL_CACHE.get(model_name_s)
        if cached is not None:
            return cached

        now = time.monotonic()
        if str(_MODEL_FAILURE_NAME) == model_name_s and float(now) < float(_MODEL_DISABLED_UNTIL):
            return None

        try:
            resolved_kind = resolve_semantic_pair_model_kind(
                model_name_s,
                model_kind=model_kind,
            )
        except Exception as exc:
            _MODEL_LAST_ERROR = str(exc)
            _MODEL_FAILURE_NAME = model_name_s
            _MODEL_LOAD_FAILS = int(min(8, int(_MODEL_LOAD_FAILS) + 1))
            _MODEL_DISABLED_UNTIL = float(now) + float(min(300.0, 2.0 ** float(_MODEL_LOAD_FAILS)))
            return None

        try:
            ensure_accelerate_clear_device_cache_compat()
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415
        except ImportError:
            _MODEL_DISABLED = True
            _MODEL_LAST_ERROR = "import_error:SentenceTransformer"
            return None
        except Exception as exc:
            _MODEL_LAST_ERROR = f"import_error:{type(exc).__name__}"
            _MODEL_FAILURE_NAME = model_name_s
            _MODEL_LOAD_FAILS = int(min(8, int(_MODEL_LOAD_FAILS) + 1))
            _MODEL_DISABLED_UNTIL = float(now) + float(min(300.0, 2.0 ** float(_MODEL_LOAD_FAILS)))
            return None

        try:
            kwargs = {
                "device": str(device or "cpu"),
                "local_files_only": bool(local_files_only),
                "trust_remote_code": bool(trust_remote_code),
            }
            if resolved_kind == "cross":
                from sentence_transformers import CrossEncoder  # noqa: PLC0415

                init_kwargs = _filter_kwargs_for_callable(CrossEncoder, kwargs)
                model_obj = CrossEncoder(model_name_s, **init_kwargs)
            else:
                init_kwargs = _filter_kwargs_for_callable(SentenceTransformer, kwargs)
                model_obj = SentenceTransformer(model_name_s, **init_kwargs)
            _MODEL_CACHE[model_name_s] = model_obj
            _MODEL_KIND_CACHE[model_name_s] = resolved_kind
            _MODEL_FAILURE_NAME = ""
            _MODEL_LAST_ERROR = ""
            _MODEL_LOAD_FAILS = 0
            _MODEL_DISABLED_UNTIL = 0.0
            return model_obj
        except Exception as exc:
            _MODEL_LAST_ERROR = f"load_error:{type(exc).__name__}"
            _MODEL_FAILURE_NAME = model_name_s
            _MODEL_LOAD_FAILS = int(min(8, int(_MODEL_LOAD_FAILS) + 1))
            _MODEL_DISABLED_UNTIL = float(time.monotonic()) + float(
                min(300.0, 2.0 ** float(_MODEL_LOAD_FAILS))
            )
            return None


def semantic_model_status_snapshot(model_name: str) -> dict[str, Any]:
    """Return the current local semantic model health snapshot."""

    model_name_s = str(model_name or "").strip()
    try:
        resolved_kind = resolve_semantic_pair_model_kind(model_name_s)
        kind = str(_MODEL_KIND_CACHE.get(model_name_s, resolved_kind))
        kind_error = ""
    except Exception as exc:
        kind = str(_MODEL_KIND_CACHE.get(model_name_s, "unknown"))
        kind_error = str(exc)
    return {
        "configured_model": model_name_s,
        "loaded_model": str(model_name_s if model_name_s in _MODEL_CACHE else ""),
        "kind": kind,
        "kind_resolution_error": kind_error,
        "disabled": bool(_MODEL_DISABLED),
        "failure_name": str(_MODEL_FAILURE_NAME or ""),
        "last_error": str(_MODEL_LAST_ERROR or ""),
        "load_fail_count": int(_MODEL_LOAD_FAILS),
    }


def normalize_relevance_scores(scores: Sequence[float]) -> list[float]:
    """Normalize raw relevance scores into stable [0, 1] probabilities."""

    values = [float(value) for value in list(scores or [])]
    if not values:
        return []
    if all(0.0 <= float(value) <= 1.0 for value in values):
        return list(values)
    normalized: list[float] = []
    for value in values:
        clipped = max(-30.0, min(30.0, float(value)))
        normalized.append(float(1.0 / (1.0 + math.exp(-clipped))))
    return normalized


def semantic_similarity_matrix(
    *,
    queries: Sequence[str],
    docs: Sequence[str],
    model_name: str,
    model: Any | None = None,
    model_kind: str | None = None,
    query_role: str = "query",
    doc_role: str = "passage",
    batch_size: int = 16,
) -> list[list[float]] | None:
    """Score query-document pairs with a local cross-encoder or bi-encoder."""

    query_list = [str(text or "") for text in list(queries or []) if str(text or "").strip()]
    doc_list = [str(text or "") for text in list(docs or []) if str(text or "").strip()]
    if not query_list or not doc_list:
        return None

    resolved_model = model or load_local_sentence_transformer_model(
        str(model_name or ""),
        model_kind=model_kind,
    )
    if resolved_model is None:
        return None

    try:
        resolved_kind = resolve_semantic_pair_model_kind(
            str(model_name or ""),
            model_kind=model_kind,
            model=resolved_model,
        )
    except Exception:
        return None

    if resolved_kind == "cross" and hasattr(resolved_model, "predict"):
        pairs = [
            (
                apply_semantic_role_prefix(query, role=str(query_role), model_name=str(model_name)),
                apply_semantic_role_prefix(doc, role=str(doc_role), model_name=str(model_name)),
            )
            for query in query_list
            for doc in doc_list
        ]
        if not pairs:
            return None
        try:
            try:
                raw_scores = resolved_model.predict(
                    pairs,
                    batch_size=min(int(batch_size), max(1, len(pairs))),
                    show_progress_bar=False,
                )
            except TypeError:
                raw_scores = resolved_model.predict(pairs)
        except Exception:
            return None
        scores = normalize_relevance_scores([float(value) for value in list(raw_scores)])
        n_docs = int(len(doc_list))
        if n_docs <= 0 or len(scores) < int(len(query_list) * n_docs):
            return None
        return [
            [float(value) for value in scores[index * n_docs : (index + 1) * n_docs]]
            for index in range(len(query_list))
        ]

    prepared_queries = [
        apply_semantic_role_prefix(query, role=str(query_role), model_name=str(model_name))
        for query in query_list
    ]
    prepared_docs = [
        apply_semantic_role_prefix(doc, role=str(doc_role), model_name=str(model_name))
        for doc in doc_list
    ]
    try:
        emb_q = resolved_model.encode(
            prepared_queries,
            batch_size=min(int(batch_size), max(1, len(prepared_queries))),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        emb_d = resolved_model.encode(
            prepared_docs,
            batch_size=min(int(batch_size), max(1, len(prepared_docs))),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        sims = emb_q @ emb_d.T
        return [[float(value) for value in list(row)] for row in sims]
    except Exception:
        return None
