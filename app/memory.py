"""mem0: kullanıcıya özel, sohbetler arası uzun vadeli hafıza (ör. "X projesinde çalışıyor", "Python 3.11 kullanıyor").

Hafıza yardımcıdır, zorunlu değil: kurulamazsa ya da hata verirse cevap hafızasız üretilir.
"""

import asyncio
import logging
import os

os.environ.setdefault("MEM0_TELEMETRY", "False")  # mem0 varsayılan olarak kullanım verisi gönderir

from mem0 import AsyncMemory  # noqa: E402
from qdrant_client import QdrantClient  # noqa: E402

from app.config import settings  # noqa: E402

log = logging.getLogger(__name__)
_memory: AsyncMemory | None = None


def _config() -> dict:
    llm = {"openai_base_url": settings.llm_base_url, "api_key": settings.llm_api_key}
    embed = {"openai_base_url": settings.embedding_base_url, "api_key": settings.embedding_api_key}
    return {
        "llm": {"provider": "openai", "config": {"model": settings.memory_llm_model, "temperature": 0, **llm}},
        "embedder": {
            "provider": "openai",
            "config": {"model": settings.embedding_model, "embedding_dims": settings.embedding_dim, **embed},
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "client": QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key),
                "collection_name": settings.memory_collection,
                "embedding_model_dims": settings.embedding_dim,
            },
        },
        "history_db_path": os.path.join(os.environ.get("MEM0_DIR", "/tmp/.mem0"), "history.db"),
    }


async def setup() -> None:
    global _memory
    if not settings.memory_enabled:
        return
    try:
        # from_config senkron ve Qdrant'a bağlanıyor: event loop'u bloklamasın
        _memory = await asyncio.to_thread(AsyncMemory.from_config, _config())
    except Exception:
        log.exception("mem0 kurulamadı, hafıza kapalı")


async def recall(query: str, user_id: str) -> str:
    if _memory is None:
        return ""
    try:
        hits = (await _memory.search(query, filters={"user_id": user_id}, top_k=5))["results"]
    except Exception:
        log.exception("mem0 arama başarısız")
        return ""
    return "\n".join(f"- {h['memory']}" for h in hits)


async def remember(question: str, answer: str, user_id: str) -> None:
    if _memory is None:
        return
    # mem0 konuşmadan kalıcı gerçekleri LLM ile çıkarır, eskileriyle birleştirir
    messages = [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
    await _memory.add(messages, user_id=user_id)
