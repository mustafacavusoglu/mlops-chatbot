"""ChatKit Store: sohbetler ve mesajlar. SQLite (tek pod) veya Postgres (çok pod), aynı kod.

Her kayıt pydantic JSON olarak `data` kolonunda; sorgu için gereken alanlar ayrı kolon.
Tüm sorgular `user_id` ile süzülür: kullanıcı yalnızca kendi sohbetlerini görür.
"""

from pathlib import Path

from chatkit.store import NotFoundError, Store
from chatkit.types import Attachment, Page, ThreadItem, ThreadMetadata
from pydantic import TypeAdapter
from sqlalchemy import Column, DateTime, Index, MetaData, String, Table, Text, and_, delete, insert, or_, select, update
from sqlalchemy.ext.asyncio import create_async_engine

from app.config import settings

Context = dict  # {"user_id": str}

_meta = MetaData()
threads = Table(
    "threads", _meta,
    Column("id", String(64), primary_key=True),
    Column("user_id", String(255), nullable=False),
    Column("created_at", DateTime, nullable=False),
    Column("data", Text, nullable=False),
    Index("ix_threads_user_created", "user_id", "created_at"),
)
items = Table(
    "items", _meta,
    Column("id", String(64), primary_key=True),
    Column("thread_id", String(64), nullable=False),
    Column("user_id", String(255), nullable=False),
    Column("created_at", DateTime, nullable=False),
    Column("data", Text, nullable=False),
    Index("ix_items_thread_created", "thread_id", "created_at"),
)

_item_adapter = TypeAdapter(ThreadItem)


def _page(table, user_col, user_id, after_row, order, limit):
    """(created_at, id) ile sabit sıralı imleç sayfalama."""
    desc = order == "desc"
    query = select(table.c.data, table.c.id).where(user_col == user_id)
    if after_row is not None:
        created, row_id = after_row
        cmp = (table.c.created_at < created) if desc else (table.c.created_at > created)
        tie = (table.c.id < row_id) if desc else (table.c.id > row_id)
        query = query.where(or_(cmp, and_(table.c.created_at == created, tie)))
    keys = (table.c.created_at.desc(), table.c.id.desc()) if desc else (table.c.created_at, table.c.id)
    return query.order_by(*keys).limit(limit + 1)


class SqlStore(Store[Context]):
    def __init__(self, url: str = settings.database_url):
        if url.startswith("sqlite"):
            Path(url.split("///", 1)[1]).parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_async_engine(url, pool_pre_ping=True)

    async def setup(self) -> None:
        async with self.engine.begin() as conn:
            await conn.run_sync(_meta.create_all)

    async def _upsert(self, table, row: dict) -> None:
        async with self.engine.begin() as conn:
            result = await conn.execute(update(table).where(table.c.id == row["id"]).values(data=row["data"]))
            if result.rowcount == 0:
                await conn.execute(insert(table).values(**row))

    # --- threads
    async def load_thread(self, thread_id: str, context: Context) -> ThreadMetadata:
        async with self.engine.connect() as conn:
            data = await conn.scalar(
                select(threads.c.data).where(threads.c.id == thread_id, threads.c.user_id == context["user_id"])
            )
        if data is None:
            raise NotFoundError(f"Thread {thread_id} bulunamadı")
        return ThreadMetadata.model_validate_json(data)

    async def save_thread(self, thread: ThreadMetadata, context: Context) -> None:
        data = ThreadMetadata.model_validate(thread.model_dump()).model_dump_json()  # Thread -> items'sız metadata
        row = {"id": thread.id, "user_id": context["user_id"], "created_at": thread.created_at, "data": data}
        await self._upsert(threads, row)

    async def load_threads(self, limit: int, after: str | None, order: str, context: Context) -> Page[ThreadMetadata]:
        async with self.engine.connect() as conn:
            after_row = None
            if after:
                after_row = (await conn.execute(
                    select(threads.c.created_at, threads.c.id).where(threads.c.id == after)
                )).first()
            rows = (await conn.execute(_page(threads, threads.c.user_id, context["user_id"], after_row, order, limit))).all()
        data = [ThreadMetadata.model_validate_json(r.data) for r in rows[:limit]]
        return Page(data=data, has_more=len(rows) > limit, after=data[-1].id if data else None)

    async def delete_thread(self, thread_id: str, context: Context) -> None:
        async with self.engine.begin() as conn:
            result = await conn.execute(
                delete(threads).where(threads.c.id == thread_id, threads.c.user_id == context["user_id"])
            )
            if result.rowcount:
                await conn.execute(delete(items).where(items.c.thread_id == thread_id))

    # --- items
    async def load_thread_items(
        self, thread_id: str, after: str | None, limit: int, order: str, context: Context
    ) -> Page[ThreadItem]:
        await self.load_thread(thread_id, context)  # sahiplik kontrolü
        async with self.engine.connect() as conn:
            after_row = None
            if after:
                after_row = (await conn.execute(select(items.c.created_at, items.c.id).where(items.c.id == after))).first()
            rows = (await conn.execute(_page(items, items.c.thread_id, thread_id, after_row, order, limit))).all()
        data = [_item_adapter.validate_json(r.data) for r in rows[:limit]]
        return Page(data=data, has_more=len(rows) > limit, after=data[-1].id if data else None)

    async def add_thread_item(self, thread_id: str, item: ThreadItem, context: Context) -> None:
        # Upsert: cevap hem arka plan görevi hem ChatKit tarafından kaydedilebilir (bkz. chat.py)
        await self.save_item(thread_id, item, context)

    async def save_item(self, thread_id: str, item: ThreadItem, context: Context) -> None:
        row = {
            "id": item.id, "thread_id": thread_id, "user_id": context["user_id"],
            "created_at": item.created_at, "data": item.model_dump_json(),
        }
        await self._upsert(items, row)

    async def load_item(self, thread_id: str, item_id: str, context: Context) -> ThreadItem:
        async with self.engine.connect() as conn:
            data = await conn.scalar(
                select(items.c.data).where(
                    items.c.id == item_id, items.c.thread_id == thread_id, items.c.user_id == context["user_id"]
                )
            )
        if data is None:
            raise NotFoundError(f"Item {item_id} bulunamadı")
        return _item_adapter.validate_json(data)

    async def delete_thread_item(self, thread_id: str, item_id: str, context: Context) -> None:
        async with self.engine.begin() as conn:
            await conn.execute(
                delete(items).where(
                    items.c.id == item_id, items.c.thread_id == thread_id, items.c.user_id == context["user_id"]
                )
            )

    # --- ek dosya: arayüzde kapalı
    async def save_attachment(self, attachment: Attachment, context: Context) -> None:
        raise NotImplementedError("Ek dosya desteklenmiyor")

    async def load_attachment(self, attachment_id: str, context: Context) -> Attachment:
        raise NotFoundError(attachment_id)

    async def delete_attachment(self, attachment_id: str, context: Context) -> None:
        return None


store = SqlStore()
