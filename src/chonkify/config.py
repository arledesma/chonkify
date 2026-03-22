"""Configuration helpers for the chonkify Azure embedding runtime."""

from __future__ import annotations

import os
from typing import Mapping

EMBEDDING_DIMENSIONS = 768


def _resolve_value(
    explicit: str | None,
    env: Mapping[str, str],
    *keys: str,
) -> str:
    """Return the first non-empty value from explicit, then env keys in order."""
    if explicit is not None and explicit.strip():
        return explicit.strip()
    for key in keys:
        value = str(env.get(key, "")).strip()
        if value:
            return value
    return ""


class AzureEmbeddingConfig:
    """Runtime configuration for the Azure OpenAI embedding backend."""

    __slots__ = (
        "endpoint",
        "api_key",
        "api_version",
        "deployment",
        "model_label",
        "dimensions",
        "max_batch_size",
        "timeout_seconds",
    )

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        api_version: str,
        deployment: str,
        model_label: str = "",
        dimensions: int = EMBEDDING_DIMENSIONS,
        max_batch_size: int = 32,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.endpoint = endpoint
        self.api_key = api_key
        self.api_version = api_version
        self.deployment = deployment
        self.model_label = model_label
        self.dimensions = dimensions
        self.max_batch_size = max_batch_size
        self.timeout_seconds = timeout_seconds


def azure_embedding_config_from_env(
    *,
    endpoint: str | None = None,
    api_key: str | None = None,
    api_version: str | None = None,
    deployment: str | None = None,
    model_label: str | None = None,
    max_batch_size: int = 32,
    timeout_seconds: float = 30.0,
    env: Mapping[str, str] | None = None,
) -> AzureEmbeddingConfig:
    """Resolve Azure embedding config from explicit overrides plus environment."""

    values = env if env is not None else os.environ

    resolved_endpoint = _resolve_value(
        endpoint, values, "AZURE_OPENAI_ENDPOINT"
    )
    resolved_api_key = _resolve_value(
        api_key, values, "AZURE_OPENAI_API_KEY"
    )
    resolved_api_version = _resolve_value(
        api_version, values, "AZURE_OPENAI_API_VERSION"
    )
    resolved_deployment = _resolve_value(
        deployment, values, "CHONKIFY_AZURE_EMBEDDING_DEPLOYMENT"
    )
    resolved_model_label = _resolve_value(
        model_label, values, "CHONKIFY_AZURE_EMBEDDING_MODEL"
    )

    missing: list[str] = []
    if not resolved_endpoint:
        missing.append("AZURE_OPENAI_ENDPOINT")
    if not resolved_api_key:
        missing.append("AZURE_OPENAI_API_KEY")
    if not resolved_api_version:
        missing.append("AZURE_OPENAI_API_VERSION")
    if not resolved_deployment:
        missing.append("CHONKIFY_AZURE_EMBEDDING_DEPLOYMENT")
    if missing:
        raise ValueError(
            f"Missing Azure embedding configuration: {', '.join(missing)}"
        )
    if max_batch_size <= 0:
        raise ValueError("max_batch_size must be positive.")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive.")

    return AzureEmbeddingConfig(
        endpoint=resolved_endpoint,
        api_key=resolved_api_key,
        api_version=resolved_api_version,
        deployment=resolved_deployment,
        model_label=resolved_model_label or resolved_deployment,
        dimensions=EMBEDDING_DIMENSIONS,
        max_batch_size=int(max_batch_size),
        timeout_seconds=float(timeout_seconds),
    )
