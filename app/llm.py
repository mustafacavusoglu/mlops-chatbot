import asyncio

from langchain_openai import ChatOpenAI
from openai import AsyncOpenAI

from app.config import settings

chat = ChatOpenAI(
    model=settings.llm_model,
    base_url=settings.llm_base_url,
    api_key=settings.llm_api_key,
    temperature=settings.llm_temperature,
    timeout=settings.llm_timeout,
    max_retries=2,
    streaming=True,
)

_embed_client = AsyncOpenAI(
    base_url=settings.embedding_base_url, api_key=settings.embedding_api_key, max_retries=3, timeout=60
)
_embed_limit = asyncio.Semaphore(settings.embedding_concurrency)


async def _embed_batch(texts: list[str]) -> list[list[float]]:
    async with _embed_limit:
        response = await _embed_client.embeddings.create(model=settings.embedding_model, input=texts)
    return [d.embedding for d in response.data]


async def embed_documents(texts: list[str]) -> list[list[float]]:
    """Batch'lere böler ve embedding_concurrency kadar paralel gönderir."""
    size = settings.embedding_batch_size
    batches = await asyncio.gather(*(_embed_batch(texts[i : i + size]) for i in range(0, len(texts), size)))
    return [vector for batch in batches for vector in batch]


async def embed_query(text: str) -> list[float]:
    return (await _embed_batch([settings.embedding_query_prefix + text]))[0]
