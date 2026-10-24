import asyncio
import logging
from datetime import datetime
from contextlib import asynccontextmanager
from pathlib import Path

import msgspec
from chatkit.server import StreamingResult
from chatkit.store import NotFoundError
from fastapi import Depends, FastAPI, Request, Response
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

from app import auth, indexer, logs, memory, vectorstore
from app.auth import User, current_user, require_admin, require_user
from app.chat import running, server
from app.config import load_sources, settings
from app.store import store

logs.setup(settings.log_level, settings.log_format)
log = logging.getLogger(__name__)
STATIC = Path(__file__).with_name("static")

if settings.auth_mode == "openshift" and settings.session_secret == "change-me":
    raise RuntimeError("SESSION_SECRET ayarlanmalı (Secret'tan rastgele bir değer)")


def to_json(data, status: int = 200) -> Response:
    return Response(msgspec.json.encode(data), status_code=status, media_type="application/json")


@asynccontextmanager
async def lifespan(_: FastAPI):
    await store.setup()
    if settings.auth_mode == "openshift" and settings.database_url.startswith("sqlite"):
        log.warning("DATABASE_URL sqlite: pod yeniden başlayınca sohbetler kaybolur, Postgres kullan")
    # Qdrant/mem0 geçici olarak erişilemezse uygulama yine açılsın; arama hatası cevapta görünür
    for name, setup in (("qdrant", vectorstore.setup), ("mem0", memory.setup)):
        try:
            await setup()
        except Exception:
            log.exception("%s kurulumu başarısız", name)
    yield


app = FastAPI(title=settings.app_title, lifespan=lifespan, docs_url=None, redoc_url=None)
app.add_middleware(
    SessionMiddleware,
    secret_key=settings.session_secret,
    max_age=settings.session_max_age,
    same_site="lax",
    https_only=settings.auth_mode == "openshift",
)
app.include_router(auth.router)
app.mount("/static", StaticFiles(directory=STATIC), name="static")  # vendor JS (marked, DOMPurify): CDN yok


@app.exception_handler(NotFoundError)
async def _not_found(_: Request, exc: NotFoundError) -> Response:
    # Store sahiplik kontrolü: başkasının ya da silinmiş sohbet -> 404 (500 değil)
    return to_json({"detail": "Sohbet bulunamadı"}, 404)


@app.get("/healthz", include_in_schema=False)
def healthz() -> Response:
    return to_json({"ok": True})


@app.get("/", include_in_schema=False)
def ui(request: Request) -> Response:
    if current_user(request) is None:
        return RedirectResponse("/auth/login")
    return FileResponse(STATIC / "index.html")


@app.get("/admin", include_in_schema=False)
def admin_page(request: Request) -> Response:
    user = current_user(request)
    if user is None:
        return RedirectResponse("/auth/login")
    if not user.is_admin:
        return Response("Bu sayfa için admin yetkisi gerekiyor.", status_code=403, media_type="text/plain; charset=utf-8")
    return FileResponse(STATIC / "admin.html")


@app.get("/api/me")
def me(user: User = Depends(require_user)) -> Response:
    return to_json({
        "user": user.name,
        "is_admin": user.is_admin,
        "title": settings.app_title,
        "chatkit_js_url": settings.chatkit_js_url,
        "chatkit_domain_key": settings.chatkit_domain_key,
        "ui_mode": settings.ui_mode,
    })


@app.post("/chatkit")
async def chatkit(request: Request, user: User = Depends(require_user)) -> Response:
    result = await server.process(await request.body(), {"user_id": user.name})
    if isinstance(result, StreamingResult):
        return StreamingResponse(result, media_type="text/event-stream", headers={"X-Accel-Buffering": "no"})
    return Response(result.json, media_type="application/json")


@app.get("/api/threads")
async def threads(user: User = Depends(require_user)) -> Response:
    page = await store.load_threads(100, None, "desc", {"user_id": user.name})
    return to_json([
        {"id": t.id, "title": t.title, "created_at": t.created_at.isoformat(), "running": t.id in running}
        for t in page.data
    ])


@app.get("/api/threads/{thread_id}/feedback")
async def thread_feedback(thread_id: str, user: User = Depends(require_user)) -> Response:
    return to_json(await store.thread_feedback(thread_id, {"user_id": user.name}))


@app.delete("/api/threads/{thread_id}")
async def delete_thread(thread_id: str, user: User = Depends(require_user)) -> Response:
    context = {"user_id": user.name}
    await store.load_thread(thread_id, context)  # başkasının sohbeti / yok -> 404
    if task := running.get(thread_id):
        task.cancel()  # silinen sohbete cevap yazılmasın
        await asyncio.wait([task], timeout=5)
    await store.delete_thread(thread_id, context)
    return Response(status_code=204)


# --- admin: indeksleme
_index_task: asyncio.Task | None = None


@app.post("/api/admin/reindex")
async def reindex(source: str | None = None, user: User = Depends(require_admin)) -> Response:
    global _index_task
    if indexer.is_running():
        return to_json({"started": False}, 202)
    log.info("manuel indeksleme: %s, kaynak=%s", user.name, source or "hepsi")
    _index_task = asyncio.create_task(indexer.run(source, trigger=f"manual:{user.name}"))
    return to_json({"started": True}, 202)


@app.get("/api/admin/index-status")
def index_status(_: User = Depends(require_admin)) -> Response:
    return to_json(indexer.status)


@app.get("/api/admin/overview")
async def admin_overview(days: int = 30, _: User = Depends(require_admin)) -> Response:
    return to_json(await store.overview(min(max(days, 1), 365)))


@app.get("/api/admin/traces")
async def admin_traces(
    q: str = "", user: str = "", feedback: str = "all", errors: int = 0, days: int = 30,
    limit: int = 50, offset: int = 0, _: User = Depends(require_admin),
) -> Response:
    page = await store.traces_page(q.strip(), user.strip(), feedback, bool(errors), days, min(limit, 200), max(offset, 0))
    return to_json(page)


@app.get("/api/admin/traces/{item_id}")
async def admin_trace(item_id: str, _: User = Depends(require_admin)) -> Response:
    return to_json(await store.trace(item_id))


@app.get("/api/admin/index-runs")
async def admin_index_runs(limit: int = 30, _: User = Depends(require_admin)) -> Response:
    return to_json(await store.index_run_rows(min(limit, 200)))


@app.get("/api/admin/feedback")
async def admin_feedback(kind: str = "negative", limit: int = 50, _: User = Depends(require_admin)) -> Response:
    kind = None if kind == "all" else kind
    return to_json(await store.feedback_rows(kind, min(limit, 500)))


@app.get("/api/admin/export.jsonl")
async def export(_: User = Depends(require_admin)) -> StreamingResponse:
    async def lines():
        async for row in store.export_traces():
            yield msgspec.json.encode(row) + b"\n"
    name = f"mlops-chatbot-traces-{datetime.now():%Y%m%d-%H%M}.jsonl"
    return StreamingResponse(
        lines(), media_type="application/x-ndjson", headers={"Content-Disposition": f'attachment; filename="{name}"'}
    )


@app.get("/api/admin/sources")
def sources(_: User = Depends(require_admin)) -> Response:
    return to_json([
        {"name": s.name, "provider": indexer.provider(s), "url": indexer.clean_url(s.url), "branch": s.branch}
        for s in load_sources()
    ])

