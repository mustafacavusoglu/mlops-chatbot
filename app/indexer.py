"""Git repolarını klonlayıp Qdrant'a indeksler (artımlı: değişmeyen dosya yeniden embed edilmez).

CLI (OpenShift CronJob): python -m app.indexer [--source NAME]
"""

import argparse
import asyncio
import base64
import hashlib
import json
import logging
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit, urlunsplit

from qdrant_client import models

from app import logs, vectorstore
from app.config import Source, load_sources, settings
from app.llm import embed_documents

log = logging.getLogger(__name__)

CLONE_TIMEOUT = 300
UPSERT_BATCH = 128

status: dict = {"running": False, "started_at": None, "finished_at": None, "trigger": None, "results": {}}
_lock = asyncio.Lock()


def is_running() -> bool:
    return _lock.locked()


# ---------- saf fonksiyonlar (testlenebilir) ----------


def matches(path: str, patterns: list[str]) -> bool:
    """`**/*.md` kökteki README.md'yi de yakalar (full_match: ** sıfır klasör olabilir)."""
    p = PurePosixPath(path)
    return any(p.full_match(pat) for pat in patterns)


def wanted(path: str, include: list[str], exclude: list[str]) -> bool:
    if ".git" in PurePosixPath(path).parts:
        return False
    return matches(path, include) and not matches(path, exclude)


def notebook_text(raw: str) -> str:
    """Markdown hücreleri düz metin, kod hücreleri ```python bloğu; çıktılar atılır."""
    parts = []
    for cell in json.loads(raw).get("cells", []):
        src = cell.get("source", "")
        src = "".join(src) if isinstance(src, list) else src
        if not src.strip():
            continue
        if cell.get("cell_type") == "code":
            parts.append(f"```python\n{src.rstrip()}\n```")
        elif cell.get("cell_type") == "markdown":
            parts.append(src.rstrip())
    return "\n\n".join(parts)


def clean_url(url: str) -> str:
    """URL'den user:pass@ kısmını at."""
    s = urlsplit(url)
    return urlunsplit(s._replace(netloc=s.netloc.rpartition("@")[2]))


def provider(src: Source) -> str:
    return src.provider or ("azure" if "/_git/" in src.url else "github")


def web_url(src: Source, path: str) -> str:
    base = clean_url(src.url)
    if provider(src) == "azure":
        return f"{base}?path=/{path}&version=GB{src.branch}"
    return f"{base.removesuffix('.git')}/blob/{src.branch}/{path}"


def _cut(text: str, start: int, end: int) -> tuple[int, bool]:
    """[start, end) içinde en iyi bölme noktası: başlık > boş satır > satır. (nokta, başlıkta_mı)"""
    lo = start + (end - start) // 2  # çok küçük parçalar çıkmasın
    for sep, shift in (("\n#", 1), ("\n\n", 2), ("\n", 1)):
        i = text.rfind(sep, lo, end)
        if i > start:
            return i + shift, sep == "\n#"
    return end, False


def chunk(text: str, header: str, size: int = settings.chunk_size, overlap: int = settings.chunk_overlap) -> list[str]:
    """~size karakterlik parçalar; her parçanın başına `# repo/path` başlığı eklenir."""
    text = text.strip()
    out, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        at_heading = False
        if end < len(text):
            end, at_heading = _cut(text, start, end)
        body = text[start:end].strip()
        if body:
            out.append(f"{header}\n\n{body}")
        if end >= len(text):
            break
        # başlıkta bölündüyse örtüşme gerekmez; yoksa overlap kadar geri git (ama ilerle)
        start = end if at_heading else max(end - overlap, start + 1)
    return out


# ---------- git ----------


async def _clone(src: Source, dest: Path) -> None:
    cmd = ["git"]
    token = os.environ.get(src.token_env) if src.token_env else None
    if token:
        b64 = base64.b64encode(f"x-access-token:{token}".encode()).decode()
        cmd += ["-c", f"http.extraHeader=Authorization: Basic {b64}"]
    cmd += ["clone", "--depth", "1", "--branch", src.branch, "--single-branch", clean_url(src.url), str(dest)]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        _, err = await asyncio.wait_for(proc.communicate(), CLONE_TIMEOUT)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise RuntimeError(f"git clone {CLONE_TIMEOUT}s içinde bitmedi") from None
    if proc.returncode:
        msg = err.decode(errors="replace").strip()
        if token:
            msg = msg.replace(token, "***").replace(b64, "***")
        raise RuntimeError(f"git clone başarısız: {msg}")


# ---------- dosya tarama ----------


def _scan(root: Path, src: Source) -> dict[str, tuple[str, str]]:
    """path -> (file_hash, metin). Büyük, ikili ve boş dosyalar atlanır."""
    files = {}
    for dirpath, dirs, names in os.walk(root):
        dirs[:] = [d for d in dirs if d != ".git"]
        for name in names:
            f = Path(dirpath) / name
            path = f.relative_to(root).as_posix()
            if f.is_symlink() or not wanted(path, src.include, src.exclude):
                continue
            if f.stat().st_size > src.max_file_kb * 1024:
                continue
            data = f.read_bytes()
            try:
                text = data.decode("utf-8")
                if path.endswith(".ipynb"):
                    text = notebook_text(text)
            except ValueError:  # UnicodeDecodeError ve JSONDecodeError ValueError'dır
                continue
            if text.strip():
                files[path] = (hashlib.sha256(data).hexdigest(), text)
    return files


async def _existing(repo: str) -> dict[str, str]:
    """Qdrant'taki path -> file_hash (vektörsüz scroll)."""
    flt = models.Filter(must=[models.FieldCondition(key="repo", match=models.MatchValue(value=repo))])
    hashes, offset = {}, None
    while True:
        points, offset = await vectorstore.client.scroll(
            settings.qdrant_collection,
            scroll_filter=flt,
            limit=1000,
            offset=offset,
            with_payload=["path", "file_hash"],
            with_vectors=False,
        )
        for p in points:
            hashes[p.payload["path"]] = p.payload["file_hash"]
        if offset is None:
            return hashes


async def _delete(repo: str, paths: list[str]) -> None:
    for i in range(0, len(paths), 200):
        flt = models.Filter(
            must=[
                models.FieldCondition(key="repo", match=models.MatchValue(value=repo)),
                models.FieldCondition(key="path", match=models.MatchAny(any=paths[i : i + 200])),
            ]
        )
        await vectorstore.client.delete(settings.qdrant_collection, points_selector=models.FilterSelector(filter=flt))


async def index_source(src: Source) -> dict:
    result = {"files_indexed": 0, "files_skipped": 0, "files_deleted": 0, "chunks": 0, "error": None}
    dest = Path(settings.index_workdir) / src.name
    shutil.rmtree(dest, ignore_errors=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        await _clone(src, dest)
        files = await asyncio.to_thread(_scan, dest, src)
        old = await _existing(src.name)

        changed = [p for p, (h, _) in files.items() if old.get(p) != h]
        gone = [p for p in old if p not in files]
        result["files_skipped"] = len(files) - len(changed)

        # önce embed et: hata olursa eski noktalar yerinde kalır
        meta = []  # (path, hash, chunk_no, text)
        for path in changed:
            file_hash, text = files[path]
            for n, c in enumerate(chunk(text, f"# {src.name}/{path}")):
                meta.append((path, file_hash, n, c))
        vectors = await embed_documents([m[3] for m in meta])

        await _delete(src.name, changed + gone)
        points = [
            models.PointStruct(
                id=vectorstore.point_id(src.name, path, n),
                vector=vec,
                payload={
                    "repo": src.name, "path": path, "url": web_url(src, path),
                    "text": text, "chunk": n, "file_hash": file_hash,
                },
            )
            for (path, file_hash, n, text), vec in zip(meta, vectors)
        ]
        for i in range(0, len(points), UPSERT_BATCH):
            await vectorstore.client.upsert(settings.qdrant_collection, points=points[i : i + UPSERT_BATCH])

        result.update(files_indexed=len({m[0] for m in meta}), files_deleted=len(gone), chunks=len(points))
    except Exception as e:
        log.exception("indeksleme hatası: %s", src.name)
        result["error"] = f"{type(e).__name__}: {e}"
    finally:
        shutil.rmtree(dest, ignore_errors=True)
    return result


async def run(only: str | None = None, trigger: str = "manual") -> dict:
    if _lock.locked():
        return status
    async with _lock:
        status.update(
            running=True, started_at=datetime.now(timezone.utc).isoformat(), finished_at=None,
            trigger=trigger, results={},
        )
        run_id = await _record_start(trigger)
        try:
            await vectorstore.setup()
            sources = [s for s in load_sources() if only in (None, s.name)]
            if only and not sources:
                status["results"][only] = {"error": "kaynak bulunamadı", "files_indexed": 0,
                                           "files_skipped": 0, "files_deleted": 0, "chunks": 0}
            limit = asyncio.Semaphore(settings.index_concurrency)

            async def one(s: Source) -> None:
                async with limit:
                    status["results"][s.name] = await index_source(s)

            await asyncio.gather(*(one(s) for s in sources))
        except Exception as e:  # setup/kaynak okuma hatası
            log.exception("indeksleme başlatılamadı")
            status["results"]["_run"] = {"error": f"{type(e).__name__}: {e}", "files_indexed": 0,
                                         "files_skipped": 0, "files_deleted": 0, "chunks": 0}
        finally:
            status.update(running=False, finished_at=datetime.now(timezone.utc).isoformat())
            await _record_finish(run_id)
    return status


# Geçmiş Postgres'te: CronJob pod'unun çalıştırmaları da admin panelinde görünür.
# DB'ye yazılamazsa indeksleme yine yapılır, sadece geçmişe düşmez.
async def _record_start(trigger: str) -> int | None:
    from app.store import store  # indeksleme CLI'ı DB'siz de çalışabilsin diye geç import

    try:
        await store.setup()
        return await store.start_index_run(trigger)
    except Exception:
        log.exception("indeksleme geçmişi yazılamadı")
        return None


async def _record_finish(run_id: int | None) -> None:
    from app.store import store

    for repo, result in status["results"].items():
        log.info("repo indekslendi", extra={"event": "index", "repo": repo, "trigger": status["trigger"], **result})
    if run_id is None:
        return
    try:
        await store.finish_index_run(run_id, status["results"])
    except Exception:
        log.exception("indeksleme geçmişi yazılamadı")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=None)
    args = ap.parse_args()
    logs.setup(settings.log_level, settings.log_format)
    result = asyncio.run(run(args.source, trigger="schedule"))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    sys.exit(1 if any(r.get("error") for r in result["results"].values()) else 0)


if __name__ == "__main__":
    main()
