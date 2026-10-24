"""Log biçimi: LOG_FORMAT=json (OpenShift Logging / Loki'de alan bazlı süzmek için) veya text (lokal).

JSON satırı: {"ts", "level", "logger", "msg", ...extra alanlar, "exc"}.
Olay alanları `log.info("...", extra={"thread_id": ..., "user": ...})` ile eklenir.
"""

import logging
from datetime import datetime, timezone

import msgspec

_STANDARD = set(vars(logging.makeLogRecord({}))) | {"message", "asctime", "taskName"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        out = {
            "ts": datetime.fromtimestamp(record.created, timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        out.update({k: v for k, v in vars(record).items() if k not in _STANDARD})
        if record.exc_info:
            out["exc"] = self.formatException(record.exc_info)
        return msgspec.json.encode(out, enc_hook=str).decode()


def setup(level: str, fmt: str) -> None:
    """Kök logger'ı kurar; uvicorn'un kendi logger'ları da aynı biçime akar."""
    handler = logging.StreamHandler()
    handler.setFormatter(
        JsonFormatter() if fmt == "json" else logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.propagate = True


if __name__ == "__main__":
    import io

    stream = io.StringIO()
    h = logging.StreamHandler(stream)
    h.setFormatter(JsonFormatter())
    log = logging.getLogger("test")
    log.addHandler(h)
    log.propagate = False
    log.warning("merhaba %s", "dünya", extra={"thread_id": "thr_1", "latency_ms": 12})
    try:
        1 / 0
    except ZeroDivisionError:
        log.exception("hata")
    first, second = (msgspec.json.decode(line) for line in stream.getvalue().splitlines() if line.startswith("{"))
    assert first["msg"] == "merhaba dünya" and first["thread_id"] == "thr_1" and first["level"] == "WARNING", first
    assert "ZeroDivisionError" in second["exc"], second
    print("ok")
