"""Embedding backend adapters for the protected chonkify winner core."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

from chonkify.config import azure_embedding_config_from_env
from chonkify.engine import AzureOpenAIEmbeddingProvider
from chonkify.types import EmbeddingProvider

EMBEDDING_DIMENSIONS = 768
DEFAULT_LOCAL_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"


def _resolve_value(
    explicit: str | None,
    env: Mapping[str, str],
    *keys: str,
) -> str:
    if explicit is not None and explicit.strip():
        return explicit.strip()
    for key in keys:
        value = str(env.get(key, "")).strip()
        if value:
            return value
    return ""


def _coerce_vector(raw_vector: Any, *, provider_name: str, index: int) -> list[float]:
    if hasattr(raw_vector, "tolist"):
        raw_vector = raw_vector.tolist()
    if not isinstance(raw_vector, Sequence) or isinstance(raw_vector, (str, bytes)):
        raise TypeError(
            f"{provider_name} returned a non-sequence embedding at index {index}: "
            f"{type(raw_vector).__name__}"
        )
    return [float(value) for value in raw_vector]


def _validate_vectors(
    raw_vectors: Sequence[Any] | Any,
    *,
    provider_name: str,
    expected_count: int,
    expected_dimensions: int = EMBEDDING_DIMENSIONS,
) -> list[list[float]]:
    if hasattr(raw_vectors, "tolist"):
        raw_vectors = raw_vectors.tolist()
    if not isinstance(raw_vectors, Sequence) or isinstance(raw_vectors, (str, bytes)):
        raise TypeError(f"{provider_name} returned a non-sequence embedding matrix.")

    vectors: list[list[float]] = []
    for index, raw_vector in enumerate(raw_vectors):
        vector = _coerce_vector(raw_vector, provider_name=provider_name, index=index)
        if len(vector) != expected_dimensions:
            raise ValueError(
                f"{provider_name} embedding dimension mismatch at index {index}: "
                f"expected {expected_dimensions}, got {len(vector)}"
            )
        vectors.append(vector)

    if len(vectors) != expected_count:
        raise ValueError(
            f"{provider_name} returned the wrong number of vectors: "
            f"expected {expected_count}, got {len(vectors)}"
        )
    return vectors


@dataclass(frozen=True, slots=True)
class OpenAIEmbeddingConfig:
    """Runtime configuration for OpenAI-compatible embedding backends."""

    api_key: str
    model: str
    provider_kind: Literal["openai", "openai-compatible"] = "openai"
    base_url: str | None = None
    dimensions: int = EMBEDDING_DIMENSIONS
    max_batch_size: int = 32
    timeout_seconds: float = 30.0
    send_dimensions_parameter: bool = True


@dataclass(frozen=True, slots=True)
class LocalEmbeddingConfig:
    """Runtime configuration for local SentenceTransformer embeddings."""

    model_name: str = DEFAULT_LOCAL_MODEL_NAME
    device: str = "cpu"
    batch_size: int = 32
    cache_folder: str | None = None
    dimensions: int = EMBEDDING_DIMENSIONS


def build_azure_embedding_provider_from_env(
    *,
    endpoint: str | None = None,
    api_key: str | None = None,
    api_version: str | None = None,
    deployment: str | None = None,
    model_label: str | None = None,
    max_batch_size: int = 32,
    timeout_seconds: float = 30.0,
    env: Mapping[str, str] | None = None,
) -> AzureOpenAIEmbeddingProvider:
    """Build the Azure embedding provider from explicit overrides and environment."""

    config = azure_embedding_config_from_env(
        endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
        deployment=deployment,
        model_label=model_label,
        max_batch_size=max_batch_size,
        timeout_seconds=timeout_seconds,
        env=env,
    )
    return AzureOpenAIEmbeddingProvider(config)


def openai_embedding_config_from_env(
    *,
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    provider_kind: Literal["openai", "openai-compatible"] = "openai",
    max_batch_size: int = 32,
    timeout_seconds: float = 30.0,
    send_dimensions_parameter: bool = True,
    env: Mapping[str, str] | None = None,
) -> OpenAIEmbeddingConfig:
    """Resolve OpenAI/OpenAI-compatible embedding config from env plus overrides."""

    values = env if env is not None else os.environ
    resolved_api_key = _resolve_value(api_key, values, "OPENAI_API_KEY", "CHONKIFY_OPENAI_API_KEY")
    resolved_model = _resolve_value(
        model,
        values,
        "CHONKIFY_OPENAI_EMBEDDING_MODEL",
        "OPENAI_EMBEDDING_MODEL",
    )
    resolved_base_url = _resolve_value(
        base_url,
        values,
        "CHONKIFY_OPENAI_BASE_URL",
        "OPENAI_BASE_URL",
    )
    if not resolved_model:
        resolved_model = "text-embedding-3-large"

    missing: list[str] = []
    if not resolved_api_key:
        missing.append("OPENAI_API_KEY")
    if provider_kind == "openai-compatible" and not resolved_base_url:
        missing.append("CHONKIFY_OPENAI_BASE_URL")
    if missing:
        raise ValueError(f"Missing OpenAI embedding configuration: {', '.join(missing)}")
    if max_batch_size <= 0:
        raise ValueError("max_batch_size must be positive.")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive.")

    return OpenAIEmbeddingConfig(
        api_key=resolved_api_key,
        model=resolved_model,
        provider_kind=provider_kind,
        base_url=resolved_base_url or None,
        dimensions=EMBEDDING_DIMENSIONS,
        max_batch_size=int(max_batch_size),
        timeout_seconds=float(timeout_seconds),
        send_dimensions_parameter=bool(send_dimensions_parameter),
    )


def local_embedding_config(
    *,
    model_name: str = DEFAULT_LOCAL_MODEL_NAME,
    device: str = "cpu",
    batch_size: int = 32,
    cache_folder: str | None = None,
) -> LocalEmbeddingConfig:
    """Build validated local embedding config for SentenceTransformer runtimes."""

    resolved_model_name = str(model_name).strip()
    resolved_device = str(device).strip()
    if not resolved_model_name:
        raise ValueError("model_name must be non-empty.")
    if not resolved_device:
        raise ValueError("device must be non-empty.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    resolved_cache_folder = str(cache_folder).strip() if cache_folder is not None else None
    return LocalEmbeddingConfig(
        model_name=resolved_model_name,
        device=resolved_device,
        batch_size=int(batch_size),
        cache_folder=resolved_cache_folder or None,
        dimensions=EMBEDDING_DIMENSIONS,
    )


class OpenAIEmbeddingProvider:
    """Embedding provider using the official OpenAI client surface."""

    def __init__(self, config: OpenAIEmbeddingConfig) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("The openai package is required for OpenAI embeddings.") from exc

        client_kwargs: dict[str, Any] = {
            "api_key": config.api_key,
            "timeout": float(config.timeout_seconds),
        }
        if config.base_url:
            client_kwargs["base_url"] = config.base_url
        self._config = config
        self._client = OpenAI(**client_kwargs)
        self.name = f"{config.provider_kind}:{config.model}"

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []

        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._config.max_batch_size):
            batch = list(texts[start : start + self._config.max_batch_size])
            request_kwargs: dict[str, Any] = {
                "model": self._config.model,
                "input": batch,
            }
            if self._config.send_dimensions_parameter:
                request_kwargs["dimensions"] = self._config.dimensions
            response = self._client.embeddings.create(**request_kwargs)
            ordered = sorted(response.data, key=lambda item: int(item.index))
            batch_vectors = _validate_vectors(
                [item.embedding for item in ordered],
                provider_name=self.name,
                expected_count=len(batch),
                expected_dimensions=self._config.dimensions,
            )
            vectors.extend(batch_vectors)

        return _validate_vectors(
            vectors,
            provider_name=self.name,
            expected_count=len(texts),
            expected_dimensions=self._config.dimensions,
        )


class LocalSentenceTransformerEmbeddingProvider:
    """Embedding provider backed by a local SentenceTransformer runtime."""

    def __init__(self, config: LocalEmbeddingConfig) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "The sentence-transformers package is required for local embeddings. "
                "Install sentence-transformers separately or use the chonkify[local] extra."
            ) from exc

        self._config = config
        self._model = SentenceTransformer(
            config.model_name,
            device=config.device,
            cache_folder=config.cache_folder,
            trust_remote_code=False,
        )
        declared_dimension = getattr(self._model, "get_sentence_embedding_dimension", lambda: None)()
        if declared_dimension is not None and int(declared_dimension) != config.dimensions:
            raise ValueError(
                f"Local model {config.model_name} exposes {declared_dimension} dimensions; "
                f"chonkify requires {config.dimensions}."
            )
        self.name = f"local-sentence-transformer:{config.model_name}@{config.device}"

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []

        vectors = self._model.encode(
            list(texts),
            batch_size=self._config.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
            device=self._config.device,
        )
        return _validate_vectors(
            vectors,
            provider_name=self.name,
            expected_count=len(texts),
            expected_dimensions=self._config.dimensions,
        )


def build_openai_embedding_provider_from_env(
    *,
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    max_batch_size: int = 32,
    timeout_seconds: float = 30.0,
    send_dimensions_parameter: bool = True,
    env: Mapping[str, str] | None = None,
) -> OpenAIEmbeddingProvider:
    """Build the official OpenAI embedding provider."""

    config = openai_embedding_config_from_env(
        api_key=api_key,
        model=model,
        base_url=base_url,
        provider_kind="openai",
        max_batch_size=max_batch_size,
        timeout_seconds=timeout_seconds,
        send_dimensions_parameter=send_dimensions_parameter,
        env=env,
    )
    return OpenAIEmbeddingProvider(config)


def build_openai_compatible_embedding_provider_from_env(
    *,
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    max_batch_size: int = 32,
    timeout_seconds: float = 30.0,
    send_dimensions_parameter: bool = True,
    env: Mapping[str, str] | None = None,
) -> OpenAIEmbeddingProvider:
    """Build an OpenAI-compatible remote embedding provider."""

    config = openai_embedding_config_from_env(
        api_key=api_key,
        model=model,
        base_url=base_url,
        provider_kind="openai-compatible",
        max_batch_size=max_batch_size,
        timeout_seconds=timeout_seconds,
        send_dimensions_parameter=send_dimensions_parameter,
        env=env,
    )
    return OpenAIEmbeddingProvider(config)


def build_local_embedding_provider(
    *,
    model_name: str = DEFAULT_LOCAL_MODEL_NAME,
    device: str = "cpu",
    batch_size: int = 32,
    cache_folder: str | None = None,
) -> EmbeddingProvider:
    """Build the local SentenceTransformer embedding provider."""

    config = local_embedding_config(
        model_name=model_name,
        device=device,
        batch_size=batch_size,
        cache_folder=cache_folder,
    )
    return LocalSentenceTransformerEmbeddingProvider(config)
