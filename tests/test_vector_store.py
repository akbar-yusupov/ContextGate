import pytest
from qdrant_client import QdrantClient, models

from contextgate.adapters.fastembed.embeddings import DeterministicEmbeddingProvider
from contextgate.adapters.qdrant.vector_index import VectorStore
from contextgate.config import PolicyConfig, Settings
from contextgate.domain.documents import Chunk


def test_hybrid_late_interaction_pipeline() -> None:
    settings = Settings(
        embedding_backend="deterministic",
        dense_dimension=64,
        late_dimension=32,
    )
    embedder = DeterministicEmbeddingProvider(64, 32)
    store = VectorStore(
        settings=settings,
        embedder=embedder,
        client=QdrantClient(":memory:"),
    )
    chunks = [
        Chunk(
            chunk_id="orders:0",
            document_id="orders",
            text="Cancel an order before it is handed to the courier.",
            source="orders.md",
            language="en",
        ),
        Chunk(
            chunk_id="payments:0",
            document_id="payments",
            text="Cards and bank transfers are accepted.",
            source="payments.md",
            language="en",
        ),
        Chunk(
            chunk_id="orders-ru:0",
            document_id="orders-ru",
            text="Р—Р°РєР°Р· РјРѕР¶РЅРѕ РѕС‚РјРµРЅРёС‚СЊ РґРѕ РїРµСЂРµРґР°С‡Рё РєСѓСЂСЊРµСЂСѓ.",
            source="orders-ru.md",
            language="ru",
        ),
    ]
    store.upsert_chunks("test", chunks)
    policy = PolicyConfig(
        dense_limit=20,
        sparse_limit=20,
        prefetch_limit=20,
        output_limit=10,
        use_late_interaction=True,
        use_cross_encoder=False,
        abstention_threshold=0,
    )

    hits = store.policy_search(
        "test",
        "How can I cancel my order?",
        policy,
        limit=10,
        language="en",
    )

    assert hits
    assert hits[0].chunk_id == "orders:0"

    russian_hits = store.policy_search(
        "test",
        "РљРѕРіРґР° РјРѕР¶РЅРѕ РѕС‚РјРµРЅРёС‚СЊ Р·Р°РєР°Р·?",
        policy,
        limit=10,
        language="ru",
    )
    assert russian_hits
    assert russian_hits[0].chunk_id == "orders-ru:0"


def test_existing_collection_schema_is_validated() -> None:
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name="incompatible",
        vectors_config={"dense": models.VectorParams(size=8, distance=models.Distance.COSINE)},
    )
    settings = Settings(
        embedding_backend="deterministic",
        dense_dimension=64,
        late_dimension=32,
    )
    store = VectorStore(
        settings=settings,
        embedder=DeterministicEmbeddingProvider(64, 32),
        client=client,
    )

    with pytest.raises(ValueError, match="Incompatible Qdrant collection"):
        store.ensure_collection("incompatible")
