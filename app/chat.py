"""ChatKit sunucusu.

Cevap, HTTP isteğinden bağımsız bir asyncio görevinde üretilir. Kullanıcı başka sohbete geçince tarayıcı
stream'i keser; görev çalışmaya devam eder ve cevabı store'a kendisi yazar. Kullanıcı geri döndüğünde cevap
yerindedir. Farklı sohbetler birbirini beklemez; aynı sohbette yeni soru önceki cevabın bitmesini bekler.
"""

import asyncio
import logging
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
        hits: list[dict] = []
        try:
            history = await self._history(thread, context)
            async for kind, data in agent.stream(history, context["user_id"]):
                if kind == "token":
                    parts.append(data)
                    queue.put_nowait(ThreadItemUpdatedEvent(
                        item_id=item.id, update=AssistantMessageContentPartTextDelta(content_index=0, delta=data)
                    ))
                elif kind == "progress":
                    queue.put_nowait(ProgressUpdateEvent(icon="search", text=data))
                elif kind == "hits":
                    hits.extend(data)
        except Exception:
            log.exception("cevap üretilemedi: thread=%s", thread.id)
            parts.append("\n\n⚠ Cevap üretilirken bir hata oluştu, lütfen tekrar dener misin?")
        text = "".join(parts)
        item.content = [AssistantMessageContent(text=text, annotations=_sources(text, hits))]
        try:
            await self.store.add_thread_item(thread.id, item, context)  # istemci gitmiş olsa da kaydolur
        except Exception:
            log.exception("cevap kaydedilemedi: thread=%s", thread.id)
        queue.put_nowait(ThreadItemDoneEvent(item=item))
        queue.put_nowait(None)
        # mem0 kaydı birkaç LLM çağrısı sürer; cevabı bekletmez
        # ponytail: pod kapanırsa bitmemiş hafıza kaydı kaybolur; kalıcı olmalıysa kuyruk (arq/celery)
        _spawn(memory.remember(_user_text(message), text, context["user_id"]))


server = MLOpsChatServer(store)
