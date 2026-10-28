import numpy as np
import pytest

from mental_health_rag.embeddings import (
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_REVISION,
    DEFAULT_QUERY_INSTRUCTION,
    EmbeddingConfig,
    create_embeddings,
    resolve_device,
)


class FakeSentenceTransformer:
    init_args: tuple[object, ...] = ()
    init_kwargs: dict[str, object] = {}
    encode_calls: list[tuple[list[str], dict[str, object]]] = []

    def __init__(self, *args: object, **kwargs: object) -> None:
        type(self).init_args = args
        type(self).init_kwargs = kwargs
        type(self).encode_calls = []

    def encode(self, texts: list[str], **kwargs: object) -> np.ndarray:
        type(self).encode_calls.append((texts, kwargs))
        return np.ones((len(texts), 3), dtype=np.float32)


def test_default_config_is_pinned_and_offline() -> None:
    config = EmbeddingConfig()

    assert config.model_name == DEFAULT_MODEL_NAME
    assert config.revision == DEFAULT_MODEL_REVISION
    assert config.query_instruction == DEFAULT_QUERY_INSTRUCTION
    assert config.batch_size == 32
    assert config.normalize_embeddings is True
    assert config.allow_download is False


def test_auto_device_prefers_cuda_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)

    assert resolve_device("auto") == "cuda"


def test_auto_device_falls_back_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)

    assert resolve_device("auto") == "cpu"


def test_explicit_unavailable_cuda_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)

    with pytest.raises(RuntimeError, match="not available"):
        resolve_device("cuda")


def test_out_of_range_cuda_index_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.device_count", lambda: 1)

    with pytest.raises(ValueError, match="detected 1 device"):
        resolve_device("cuda:1")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"model_name": " "}, "model_name"),
        ({"revision": "main"}, "revision"),
        ({"batch_size": 0}, "batch_size"),
        ({"query_instruction": " "}, "query_instruction"),
        ({"device": "gpu"}, "device"),
    ],
)
def test_invalid_config_is_rejected(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        EmbeddingConfig(**kwargs)  # type: ignore[arg-type]


def test_create_embeddings_passes_reproducible_model_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "sentence_transformers.SentenceTransformer", FakeSentenceTransformer
    )
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)

    create_embeddings()

    assert FakeSentenceTransformer.init_args == (DEFAULT_MODEL_NAME,)
    assert FakeSentenceTransformer.init_kwargs == {
        "cache_folder": None,
        "device": "cuda",
        "revision": DEFAULT_MODEL_REVISION,
        "local_files_only": True,
    }


def test_query_instruction_is_not_added_to_documents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "sentence_transformers.SentenceTransformer", FakeSentenceTransformer
    )
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    embeddings = create_embeddings(EmbeddingConfig(batch_size=8))

    document_vectors = embeddings.embed_documents(["第一段\n正文", "第二段正文"])
    query_vector = embeddings.embed_query("如何缓解压力？")

    assert len(document_vectors) == 2
    assert len(query_vector) == 3
    document_texts, document_kwargs = FakeSentenceTransformer.encode_calls[0]
    query_texts, query_kwargs = FakeSentenceTransformer.encode_calls[1]
    assert document_texts == ["第一段 正文", "第二段正文"]
    assert document_kwargs["batch_size"] == 8
    assert document_kwargs["normalize_embeddings"] is True
    assert "prompt" not in document_kwargs
    assert query_texts == ["如何缓解压力？"]
    assert query_kwargs["prompt"] == DEFAULT_QUERY_INSTRUCTION
    assert query_kwargs["normalize_embeddings"] is True


def test_allow_download_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sentence_transformers.SentenceTransformer", FakeSentenceTransformer
    )
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)

    create_embeddings(EmbeddingConfig(allow_download=True))

    assert FakeSentenceTransformer.init_kwargs["local_files_only"] is False
