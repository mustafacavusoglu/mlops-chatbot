import uuid

from qdrant_client import AsyncQdrantClient, models

from app.config import settings
from app.llm import embed_query

client = AsyncQdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)


async def setup() -> None:
    """Koleksiyon yoksa oluştur; repo ve path için keyword index ekle."""
    if not await client.collection_exists(settings.qdrant_collection):
        await client.create_collection(
            settings.qdrant_collection,
            vectors_config=models.VectorParams(size=settings.embedding_dim, distance=models.Distance.COSINE),
        )
    for field in ("repo", "path"):  # zaten varsa Qdrant sorunsuz geçer
        await client.create_payload_index(
            settings.qdrant_collection, field, field_schema=models.PayloadSchemaType.KEYWORD
        )


def point_id(repo: str, path: str, chunk: int) -> str:
    # Sabit id: aynı chunk tekrar yazılırsa kopya üretmez, üzerine yazar
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{repo}:{path}:{chunk}"))


async def search(query: str, k: int = settings.top_k, repo: str | None = None) -> list[dict]:
    """Payload: {repo, path, url, text, chunk, file_hash} + score."""
    flt = None
    if repo:
        flt = models.Filter(must=[models.FieldCondition(key="repo", match=models.MatchValue(value=repo))])
    response = await client.query_points(
        settings.qdrant_collection, query=await embed_query(query), limit=k, query_filter=flt
    )
    return [{**h.payload, "score": h.score} for h in response.points]
