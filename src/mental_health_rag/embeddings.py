"""Configuration and construction of Chinese text embeddings."""

from __future__ import annotations

import re
from dataclasses import dataclass

import torch
from langchain_huggingface import HuggingFaceEmbeddings

DEFAULT_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
DEFAULT_MODEL_REVISION = "7999e1d3359715c523056ef9478215996d62a620"
DEFAULT_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："
EXPECTED_EMBEDDING_DIMENSION = 512


@dataclass(frozen=True, slots=True)
class EmbeddingConfig:
    """Define a reproducible BGE embedding runtime configuration."""

    model_name: str = DEFAULT_MODEL_NAME
    revision: str = DEFAULT_MODEL_REVISION
    device: str = "auto"
    batch_size: int = 32
    normalize_embeddings: bool = True
    query_instruction: str = DEFAULT_QUERY_INSTRUCTION
    allow_download: bool = False

    def __post_init__(self) -> None:
        if not self.model_name.strip():
            raise ValueError("model_name must be a non-empty string")
        if not re.fullmatch(r"[0-9a-f]{40}", self.revision):
            raise ValueError("revision must be a 40-character lowercase commit SHA")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")
        if not self.query_instruction.strip():
            raise ValueError("query_instruction must be a non-empty string")
        if self.device != "auto" and self.device != "cpu" and not self.device.startswith(
            "cuda"
        ):
            raise ValueError("device must be auto, cpu, cuda, or cuda:<index>")


def resolve_device(requested_device: str) -> str:
    """Resolve an explicit CPU/CUDA device without silently changing requests."""
    if requested_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested_device == "cpu":
        return "cpu"
    if requested_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return "cuda"
    if requested_device.startswith("cuda:"):
        try:
            device_index = int(requested_device.removeprefix("cuda:"))
        except ValueError as error:
            raise ValueError("CUDA device must use the form cuda:<index>") from error
        if device_index < 0:
            raise ValueError("CUDA device index must be at least 0")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        if device_index >= torch.cuda.device_count():
            raise ValueError(
                f"CUDA device index {device_index} is unavailable; "
                f"detected {torch.cuda.device_count()} device(s)"
            )
        return requested_device
    raise ValueError("device must be auto, cpu, cuda, or cuda:<index>")


def create_embeddings(config: EmbeddingConfig | None = None) -> HuggingFaceEmbeddings:
    """Create LangChain embeddings with separate query and document settings.

    Downloads are disabled by default. The BGE retrieval instruction is added
    only to queries through Sentence Transformers' ``prompt`` argument.
    """
    resolved_config = config or EmbeddingConfig()
    device = resolve_device(resolved_config.device)
    common_encode_kwargs = {
        "batch_size": resolved_config.batch_size,
        "normalize_embeddings": resolved_config.normalize_embeddings,
    }
    return HuggingFaceEmbeddings(
        model=resolved_config.model_name,
        model_kwargs={
            "device": device,
            "revision": resolved_config.revision,
            "local_files_only": not resolved_config.allow_download,
        },
        encode_kwargs=dict(common_encode_kwargs),
        query_encode_kwargs={
            **common_encode_kwargs,
            "prompt": resolved_config.query_instruction,
        },
        show_progress=False,
    )
