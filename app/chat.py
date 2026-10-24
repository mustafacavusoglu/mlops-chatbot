"""ChatKit sunucusu.

Cevap, HTTP isteğinden bağımsız bir asyncio görevinde üretilir. Kullanıcı başka sohbete geçince tarayıcı
stream'i keser; görev çalışmaya devam eder ve cevabı store'a kendisi yazar. Kullanıcı geri döndüğünde cevap
yerindedir. Farklı sohbetler birbirini beklemez; aynı sohbette yeni soru önceki cevabın bitmesini bekler.
"""

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from datetime import datetime

from chatkit.server import ChatKitServer
from chatkit.types import (
    Annotation,
    AssistantMessageContent,
    AssistantMessageContentPartAdded,
    AssistantMessageContentPartTextDelta,
    AssistantMessageItem,
    ProgressUpdateEvent,
    StreamOptions,
    ThreadItemAddedEvent,
    ThreadItemDoneEvent,
    ThreadItemUpdatedEvent,
    ThreadMetadata,
    ThreadStreamEvent,
    URLSource,
    UserMessageItem,
)
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from app import agent, memory
from app.config import settings
from app.store import Context, store

log = logging.getLogger(__name__)

# ponytail: süreç içi; "cevap üretiliyor" göstergesi pod başına. Route sticky cookie ile kullanıcı aynı poda gider.
running: dict[str, asyncio.Task] = {}
_background: set[asyncio.Task] = set()  # referansı tutulmayan task çöp toplayıcıya gidebilir


def _user_text(item: UserMessageItem) -> str:
    return " ".join(part.text for part in item.content).strip()


def _sources(text: str, hits: list[dict]) -> list[Annotation]:
    """Cevapta adı/linki geçen dosyalar kaynak kartı olur."""
    seen, annotations = set(), []
    for h in hits:
        if h["url"] in seen or (h["path"] not in text and h["url"] not in text):
            continue
        seen.add(h["url"])
        title = f"{h['repo']}/{h['path']}"
        annotations.append(Annotation(source=URLSource(title=title, url=h["url"], group=h["repo"])))
    return annotations


def _done(task: asyncio.Task) -> None:
    _background.discard(task)
    if not task.cancelled() and task.exception():
        log.error("arka plan görevi başarısız", exc_info=task.exception())


def _spawn(coro) -> None:
    task = asyncio.create_task(coro)
    _background.add(task)
    task.add_done_callback(_done)


class MLOpsChatServer(ChatKitServer[Context]):
    def get_stream_options(self, thread: ThreadMetadata, context: Context) -> StreamOptions:
        # Durdur butonu yok: bağlantı kopması = sohbet değiştirme, cevap arkada tamamlanır
        return StreamOptions(allow_cancel=False)

    async def handle_stream_cancelled(self, thread, pending_items, context) -> None:
        # Varsayılan davranış yarım cevabı kaydedip "kullanıcı iptal etti" notu ekler; biz görevi sürdürüyoruz
        return None

    async def respond(
        self, thread: ThreadMetadata, input_user_message: UserMessageItem | None, context: Context
    ) -> AsyncIterator[ThreadStreamEvent]:
        if input_user_message is None:
            return
        if previous := running.get(thread.id):
            await asyncio.wait([previous])  # aynı sohbette sırayla; wait iptal edilse de görevi iptal etmez
        if not thread.title:
            thread.title = _user_text(input_user_message)[:60] or "Yeni sohbet"
            await self.store.save_thread(thread, context)

        queue: asyncio.Queue = asyncio.Queue()
        task = asyncio.create_task(self._generate(thread, input_user_message, context, queue))
        running[thread.id] = task
        task.add_done_callback(lambda t: running.pop(thread.id, None) if running.get(thread.id) is t else None)
        while (event := await queue.get()) is not None:
            yield event

    async def add_feedback(self, thread_id, item_ids, feedback, context) -> None:
        await self.store.save_feedback(thread_id, item_ids, feedback, context)

    async def _trace(self, item, message, text, searches, annotations, started, first_token, failed, context) -> None:
        """RAG'i iyileştirmek için cevabın izi: hangi sorgularla ne geldi, skorlar, neye atıf yapıldı, ne kadar sürdü."""
        await self.store.save_trace({
            "item_id": item.id, "thread_id": item.thread_id, "user_id": context["user_id"],
            "created_at": item.created_at, "question": _user_text(message), "answer": text,
            "searches": [
                {**s, "hits": [{k: h.get(k) for k in ("repo", "path", "url", "score", "chunk", "text")} for h in s["hits"]]}
                for s in searches
            ],
            "cited": [a.source.url for a in annotations],
            "model": settings.llm_model,
            "latency_ms": int((time.monotonic() - started) * 1000),
            "first_token_ms": int((first_token - started) * 1000) if first_token else None,
            "error": failed,
        })

    async def _history(self, thread: ThreadMetadata, context: Context) -> list[BaseMessage]:
        page = await self.store.load_thread_items(thread.id, None, settings.history_limit, "desc", context)
        messages: list[BaseMessage] = []
        for item in reversed(page.data):
            if isinstance(item, UserMessageItem):
                messages.append(HumanMessage(_user_text(item)))
            elif isinstance(item, AssistantMessageItem) and item.content:
                messages.append(AIMessage("".join(c.text for c in item.content)))
        return messages

    async def _generate(
        self, thread: ThreadMetadata, message: UserMessageItem, context: Context, queue: asyncio.Queue
    ) -> None:
        try:
            await self._answer(thread, message, context, queue)
        finally:
            queue.put_nowait(None)  # sohbet silinip görev iptal edilse de bağlı istemcinin stream'i kapanır

    async def _answer(
        self, thread: ThreadMetadata, message: UserMessageItem, context: Context, queue: asyncio.Queue
    ) -> None:
        item = AssistantMessageItem(
            id=self.store.generate_item_id("message", thread, context),
            thread_id=thread.id,
            created_at=datetime.now(),
            content=[],
        )
        queue.put_nowait(ThreadItemAddedEvent(item=item))
        queue.put_nowait(ThreadItemUpdatedEvent(
            item_id=item.id, update=AssistantMessageContentPartAdded(content_index=0, content=AssistantMessageContent(text=""))
        ))
        queue.put_nowait(ProgressUpdateEvent(icon="search", text="Dokümanlar taranıyor…"))
        parts: list[str] = []
        searches: list[dict] = []
        started, first_token, failed = time.monotonic(), None, False
        try:
            history = await self._history(thread, context)
            async for kind, data in agent.stream(history, context["user_id"]):
                if kind == "token":
                    first_token = first_token or time.monotonic()
                    parts.append(data)
                    queue.put_nowait(ThreadItemUpdatedEvent(
                        item_id=item.id, update=AssistantMessageContentPartTextDelta(content_index=0, delta=data)
                    ))
                elif kind == "progress":
                    queue.put_nowait(ProgressUpdateEvent(icon="search", text=data))
                elif kind == "search":
                    searches.append(data)
        except Exception:
            log.exception("cevap üretilemedi: thread=%s", thread.id)
            failed = True
            parts.append("\n\n⚠ Cevap üretilirken bir hata oluştu, lütfen tekrar dener misin?")
        text = "".join(parts)
        annotations = _sources(text, [h for s in searches for h in s["hits"]])
        item.content = [AssistantMessageContent(text=text, annotations=annotations)]
        try:
            await self.store.add_thread_item(thread.id, item, context)  # istemci gitmiş olsa da kaydolur
        except Exception:
            log.exception("cevap kaydedilemedi: thread=%s", thread.id)
        queue.put_nowait(ThreadItemDoneEvent(item=item))
        _spawn(self._trace(item, message, text, searches, annotations, started, first_token, failed, context))
        # mem0 kaydı birkaç LLM çağrısı sürer; cevabı bekletmez
        # ponytail: pod kapanırsa bitmemiş hafıza kaydı kaybolur; kalıcı olmalıysa kuyruk (arq/celery)
        _spawn(memory.remember(_user_text(message), text, context["user_id"]))


server = MLOpsChatServer(store)
