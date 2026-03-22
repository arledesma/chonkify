"""Public package surface for chonkify."""

from chonkify.backends import (
    DEFAULT_LOCAL_MODEL_NAME,
    EMBEDDING_DIMENSIONS,
    LocalEmbeddingConfig,
    LocalSentenceTransformerEmbeddingProvider,
    OpenAIEmbeddingConfig,
    OpenAIEmbeddingProvider,
    build_azure_embedding_provider_from_env,
    build_local_embedding_provider,
    build_openai_compatible_embedding_provider_from_env,
    build_openai_embedding_provider_from_env,
    local_embedding_config,
    openai_embedding_config_from_env,
)
from chonkify.config import AzureEmbeddingConfig, azure_embedding_config_from_env
from chonkify.engine import AzureOpenAIEmbeddingProvider, compress_documents
from chonkify.types import CompressionRequest, CompressionResult, Document

__all__ = [
    "AzureEmbeddingConfig",
    "AzureOpenAIEmbeddingProvider",
    "CompressionRequest",
    "CompressionResult",
    "DEFAULT_LOCAL_MODEL_NAME",
    "Document",
    "EMBEDDING_DIMENSIONS",
    "LocalEmbeddingConfig",
    "LocalSentenceTransformerEmbeddingProvider",
    "OpenAIEmbeddingConfig",
    "OpenAIEmbeddingProvider",
    "azure_embedding_config_from_env",
    "build_azure_embedding_provider_from_env",
    "build_local_embedding_provider",
    "build_openai_compatible_embedding_provider_from_env",
    "build_openai_embedding_provider_from_env",
    "compress_documents",
    "local_embedding_config",
    "openai_embedding_config_from_env",
]

__version__ = "0.2.2"
