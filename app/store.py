"""ChatKit Store: sohbetler ve mesajlar. SQLite (tek pod) veya Postgres (çok pod), aynı kod.

Her kayıt pydantic JSON olarak `data` kolonunda; sorgu için gereken alanlar ayrı kolon.
Tüm sorgular `user_id` ile süzülür: kullanıcı yalnızca kendi sohbetlerini görür.

Analiz için iki tablo daha (sohbet silinse de kalır):
  traces   her cevabın izi: soru, cevap, arama sorguları + gelen parçalar/skorlar, atıflar, süre, model
  feedback 👍/👎 (kullanıcı başına cevap başına bir oy; tekrar oy öncekini ezer)
Ayrıca index_runs: indeksleme çalıştırmaları (CronJob + manuel), admin panelindeki geçmiş için.
"""

from collections.abc import AsyncIterator
from datetime import datetime, timedelta
from pathlib import Path

import msgspec

from chatkit.store import NotFoundError, Store
from chatkit.types import Attachment, Page, ThreadItem, ThreadMetadata
from pydantic import TypeAdapter
from sqlalchemy import (
    Boolean, Column, DateTime, Float, Index, Integer, MetaData, String, Table, Text, and_, delete, func, insert, or_,
    select, update,
)
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

traces = Table(
    "traces", _meta,
    Column("item_id", String(64), primary_key=True),
    Column("thread_id", String(64), nullable=False),
    Column("user_id", String(255), nullable=False),
    Column("created_at", DateTime, nullable=False, index=True),
    Column("question", Text, nullable=False),
    Column("answer", Text, nullable=False),
    Column("searches", Text, nullable=False),  # JSON: [{query, repo, hits: [{repo, path, url, score, chunk, text}]}]
    Column("cited", Text, nullable=False),  # JSON: cevabın altında kaynak olarak gösterilen url'ler
    Column("top_score", Float),  # en iyi retrieval skoru: düşükse doküman eksik olabilir
    Column("n_search", Integer, nullable=False, default=0),
    Column("model", String(255), nullable=False),
    Column("latency_ms", Integer, nullable=False),
    Column("first_token_ms", Integer),
    Column("error", Boolean, nullable=False, default=False),
)
feedback = Table(
    "feedback", _meta,
    Column("item_id", String(64), primary_key=True),
    Column("thread_id", String(64), nullable=False),
    Column("user_id", String(255), nullable=False),
    Column("kind", String(16), nullable=False),  # positive | negative
    Column("created_at", DateTime, nullable=False, index=True),
)

index_runs = Table(
    "index_runs", _meta,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("trigger", String(255), nullable=False),  # schedule | manual:<kullanıcı>
    Column("started_at", DateTime, nullable=False, index=True),
    Column("finished_at", DateTime),
    Column("results", Text),  # JSON: {repo: {files_indexed, files_skipped, files_deleted, chunks, error}}
)
STALE_AFTER_S = 3 * 3600  # bitmeden bu kadar geçtiyse pod ölmüş sayılır

_item_adapter = TypeAdapter(ThreadItem)
_JSON = ("searches", "cited", "results")


def _repo_path(url: str) -> dict:
    """Atıf url'sinden repo/yol (GitHub /blob/<branch>/ veya Azure ?path=) çıkarır; gösterim için."""
    if "?path=/" in url:
        base, _, rest = url.partition("?path=/")
        return {"repo": base.rstrip("/").rsplit("/", 1)[-1], "path": rest.split("&", 1)[0]}
    head, _, rest = url.partition("/blob/")
    return {"repo": head.rstrip("/").rsplit("/", 1)[-1], "path": rest.split("/", 1)[1] if "/" in rest else rest}


def _trace_row(row) -> dict:
    out = dict(row._mapping)
    for key in _JSON:
        if out.get(key) is not None:
            out[key] = msgspec.json.decode(out[key])
    for key in ("created_at", "feedback_at"):
        if isinstance(out.get(key), datetime):
            out[key] = out[key].isoformat()
    return out


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


def engine_url(url: str) -> str:
    """OpenShift'teki Postgres secret'larının `uri` değeri (postgresql://...) doğrudan verilebilsin."""
    scheme, sep, rest = url.partition("://")
    return f"postgresql+asyncpg://{rest}" if scheme in ("postgres", "postgresql") else url


class SqlStore(Store[Context]):
    def __init__(self, url: str = settings.database_url):
        url = engine_url(url)
        if url.startswith("sqlite"):
            Path(url.split("///", 1)[1]).parent.mkdir(parents=True, exist_ok=True)
            self.engine = create_async_engine(url)
        else:
            self.engine = create_async_engine(
                url, pool_pre_ping=True, pool_size=settings.db_pool_size, max_overflow=settings.db_pool_size
            )

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

    # --- analiz: trace + feedback
    async def save_trace(self, row: dict) -> None:
        row = {**row, **{k: msgspec.json.encode(row[k]).decode() for k in ("searches", "cited")}}
        async with self.engine.begin() as conn:
            await conn.execute(insert(traces).values(**row))

    async def save_feedback(self, thread_id: str, item_ids: list[str], kind: str, context: Context) -> None:
        for item_id in item_ids:
            await self.load_item(thread_id, item_id, context)  # sahiplik: başkasının cevabına oy yok -> 404
            row = {"item_id": item_id, "thread_id": thread_id, "user_id": context["user_id"],
                   "kind": kind, "created_at": datetime.now()}
            async with self.engine.begin() as conn:
                result = await conn.execute(
                    update(feedback).where(feedback.c.item_id == item_id)
                    .values(kind=kind, created_at=row["created_at"])
                )
                if result.rowcount == 0:
                    await conn.execute(insert(feedback).values(**row))

    async def thread_feedback(self, thread_id: str, context: Context) -> dict[str, str]:
        async with self.engine.connect() as conn:
            rows = await conn.execute(
                select(feedback.c.item_id, feedback.c.kind)
                .where(feedback.c.thread_id == thread_id, feedback.c.user_id == context["user_id"])
            )
            return {r.item_id: r.kind for r in rows}

    async def feedback_rows(self, kind: str | None, limit: int) -> list[dict]:
        query = (
            select(feedback, traces.c.question, traces.c.answer, traces.c.searches, traces.c.cited,
                   traces.c.model, traces.c.latency_ms)
            .select_from(feedback.outerjoin(traces, traces.c.item_id == feedback.c.item_id))
            .order_by(feedback.c.created_at.desc())
            .limit(limit)
        )
        if kind:
            query = query.where(feedback.c.kind == kind)
        async with self.engine.connect() as conn:
            return [_trace_row(r) for r in await conn.execute(query)]

    async def traces_page(
        self, q: str, user: str, fb: str, errors: bool, days: int, limit: int, offset: int
    ) -> dict:
        query = select(
            traces.c.item_id, traces.c.thread_id, traces.c.user_id, traces.c.created_at, traces.c.question,
            func.substr(traces.c.answer, 1, 200).label("answer_preview"), traces.c.error, traces.c.latency_ms,
            traces.c.first_token_ms, traces.c.model, traces.c.n_search, traces.c.top_score, traces.c.cited,
            feedback.c.kind.label("feedback"),
        ).select_from(traces.outerjoin(feedback, feedback.c.item_id == traces.c.item_id))
        if q:
            query = query.where(or_(traces.c.question.ilike(f"%{q}%"), traces.c.answer.ilike(f"%{q}%")))
        if user:
            query = query.where(traces.c.user_id == user)
        if fb == "none":
            query = query.where(feedback.c.kind.is_(None))
        elif fb in ("positive", "negative"):
            query = query.where(feedback.c.kind == fb)
        if errors:
            query = query.where(traces.c.error.is_(True))
        if days:
            query = query.where(traces.c.created_at >= datetime.now() - timedelta(days=days))
        async with self.engine.connect() as conn:
            total = await conn.scalar(select(func.count()).select_from(query.subquery()))
            rows = await conn.execute(query.order_by(traces.c.created_at.desc()).limit(limit).offset(offset))
            out = []
            for r in rows:
                row = _trace_row(r)
                row["n_cited"] = len(row.pop("cited"))
                out.append(row)
        return {"total": total, "rows": out}

    async def trace(self, item_id: str) -> dict:
        query = (
            select(traces, feedback.c.kind.label("feedback"), feedback.c.created_at.label("feedback_at"))
            .select_from(traces.outerjoin(feedback, feedback.c.item_id == traces.c.item_id))
            .where(traces.c.item_id == item_id)
        )
        async with self.engine.connect() as conn:
            row = (await conn.execute(query)).first()
        if row is None:
            raise NotFoundError(item_id)
        out = _trace_row(row)
        out["n_cited"] = len(out["cited"])
        return out

    async def overview(self, days: int) -> dict:
        """Genel bakış: KPI'lar, günlük seri, en çok atıf alan dokümanlar, en zayıf aramalar."""
        # ponytail: pencere Python'da toplanıyor; günde on binlerce soruya çıkarsa SQL'de group by'a taşı
        since = datetime.now() - timedelta(days=days)
        async with self.engine.connect() as conn:
            rows = (await conn.execute(
                select(traces.c.item_id, traces.c.created_at, traces.c.user_id, traces.c.thread_id,
                       traces.c.latency_ms, traces.c.first_token_ms, traces.c.error, traces.c.cited,
                       traces.c.top_score, traces.c.question)
                .where(traces.c.created_at >= since)
            )).all()
            votes = (await conn.execute(
                select(feedback.c.kind, feedback.c.created_at).where(feedback.c.created_at >= since)
            )).all()
        start = since.date()
        daily = {start + timedelta(days=i): {"questions": 0, "users": set(), "errors": 0, "positive": 0, "negative": 0}
                 for i in range((datetime.now().date() - start).days + 1)}
        cited: dict[str, int] = {}
        for r in rows:
            day = daily.setdefault(r.created_at.date(), {"questions": 0, "users": set(), "errors": 0,
                                                         "positive": 0, "negative": 0})
            day["questions"] += 1
            day["users"].add(r.user_id)
            day["errors"] += bool(r.error)
            for url in msgspec.json.decode(r.cited):
                cited[url] = cited.get(url, 0) + 1
        for v in votes:
            if (day := daily.get(v.created_at.date())) is not None:
                day[v.kind] += 1
        latencies = sorted(r.latency_ms for r in rows if not r.error)
        first_tokens = [r.first_token_ms for r in rows if r.first_token_ms]
        totals = {
            "questions": len(rows),
            "users": len({r.user_id for r in rows}),
            "threads": len({r.thread_id for r in rows}),
            "positive": sum(v.kind == "positive" for v in votes),
            "negative": sum(v.kind == "negative" for v in votes),
            "errors": sum(bool(r.error) for r in rows),
            "avg_latency_ms": int(sum(latencies) / len(latencies)) if latencies else None,
            "p95_latency_ms": latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))] if latencies else None,
            "avg_first_token_ms": int(sum(first_tokens) / len(first_tokens)) if first_tokens else None,
        }
        top_docs = []
        for url, count in sorted(cited.items(), key=lambda kv: -kv[1])[:10]:
            repo_path = _repo_path(url)
            top_docs.append({**repo_path, "url": url, "count": count})
        weakest = sorted((r for r in rows if r.top_score is not None), key=lambda r: r.top_score)[:10]
        return {
            "days": days,
            "totals": totals,
            "daily": [
                {"date": d.isoformat(), **{**v, "users": len(v["users"])}} for d, v in sorted(daily.items())
            ],
            "top_docs": top_docs,
            "weakest": [
                {"item_id": r.item_id, "question": r.question, "top_score": r.top_score,
                 "created_at": r.created_at.isoformat()}
                for r in weakest
            ],
        }

    # --- indeksleme geçmişi (CronJob pod'u da buraya yazar)
    async def start_index_run(self, trigger: str) -> int:
        async with self.engine.begin() as conn:
            result = await conn.execute(insert(index_runs).values(trigger=trigger, started_at=datetime.now()))
            return result.inserted_primary_key[0]

    async def finish_index_run(self, run_id: int, results: dict) -> None:
        async with self.engine.begin() as conn:
            await conn.execute(
                update(index_runs).where(index_runs.c.id == run_id)
                .values(finished_at=datetime.now(), results=msgspec.json.encode(results).decode())
            )

    async def index_run_rows(self, limit: int) -> list[dict]:
        async with self.engine.connect() as conn:
            rows = await conn.execute(select(index_runs).order_by(index_runs.c.started_at.desc()).limit(limit))
            out = []
            for r in rows:
                results = msgspec.json.decode(r.results) if r.results else {}
                if r.finished_at:
                    status = "error" if any(v.get("error") for v in results.values()) else "ok"
                    duration = (r.finished_at - r.started_at).total_seconds()
                else:
                    stale = (datetime.now() - r.started_at).total_seconds() > STALE_AFTER_S
                    status, duration = ("stale" if stale else "running"), None
                keys = ("files_indexed", "files_skipped", "files_deleted", "chunks")
                out.append({
                    "id": r.id, "trigger": r.trigger, "started_at": r.started_at.isoformat(),
                    "finished_at": r.finished_at.isoformat() if r.finished_at else None,
                    "status": status, "duration_s": duration,
                    "totals": {k: sum(v.get(k) or 0 for v in results.values()) for k in keys},
                    "results": results,
                })
        return out

    async def export_traces(self) -> AsyncIterator[dict]:
        """Tüm cevap izleri + varsa oy; notebook'ta analiz için (JSONL)."""
        query = (
            select(traces, feedback.c.kind.label("feedback"), feedback.c.created_at.label("feedback_at"))
            .select_from(traces.outerjoin(feedback, feedback.c.item_id == traces.c.item_id))
            .order_by(traces.c.created_at)
        )
        async with self.engine.connect() as conn:
            async for row in await conn.stream(query):
                yield _trace_row(row)

    # --- ek dosya: arayüzde kapalı
    async def save_attachment(self, attachment: Attachment, context: Context) -> None:
        raise NotImplementedError("Ek dosya desteklenmiyor")

    async def load_attachment(self, attachment_id: str, context: Context) -> Attachment:
        raise NotFoundError(attachment_id)

    async def delete_attachment(self, attachment_id: str, context: Context) -> None:
        return None


store = SqlStore()
