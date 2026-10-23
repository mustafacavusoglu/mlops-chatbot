"""OpenShift OAuth girişi: "OpenShift'e girebilen herkes girer", admin yetkisi OpenShift'ten gelir.

Akış: ServiceAccount OAuth client (client_id = system:serviceaccount:<ns>:<sa>, secret = SA token).
SA üzerinde `serviceaccounts.openshift.io/oauth-redirecturi.*` (veya `oauth-redirectreference.*`)
annotation'ı redirect_uri ile eşleşmeli. SA yalnızca şu scope'ları isteyebilir: user:info,
user:check-access, role:<rol>:<kendi-namespace'i> (Red Hat docs, "Using a service account as an OAuth client").
user:check-access = boş user/groups ile SelfSubjectAccessReview yapabilme (docs: "Scoping tokens").
Doğrulanamayan: user:check-access ile SAR'ın canlı kümede gerçekten izin verdiği (yalnızca dokümandan okundu).
OpenShift access token'ı cookie'ye KONMAZ; yalnızca ad/grup/admin bilgisi oturumda tutulur.
"""

import logging
import os
import secrets
import ssl
import time
from pathlib import Path
from urllib.parse import urlencode

import httpx
import msgspec
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from app.config import settings

log = logging.getLogger(__name__)
router = APIRouter()

SCOPE = "user:info user:check-access"
NS_FILE = Path("/var/run/secrets/kubernetes.io/serviceaccount/namespace")


class User(msgspec.Struct):
    name: str
    groups: list[str] = []
    is_admin: bool = False


# --- HTTP istemcisi -------------------------------------------------------------------------

_client: httpx.AsyncClient | None = None
_endpoints: dict[str, str] = {}  # .well-known sonucu (bir kez)


def _verify() -> ssl.SSLContext | bool:
    if not settings.tls_verify:
        return False
    # Sistem CA'ları + SA ca.crt: token endpoint'i (public route) ingress CA ile imzalı olabilir
    ctx = ssl.create_default_context()
    for f in (os.environ.get("SSL_CERT_FILE"), os.environ.get("REQUESTS_CA_BUNDLE"), settings.ca_file):
        if f and Path(f).is_file():
            ctx.load_verify_locations(cafile=f)
    return ctx


def client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(verify=_verify(), timeout=10.0)
    return _client


async def _endpoint(key: str, configured: str | None) -> str:
    if configured:
        return configured
    if not _endpoints:
        r = await client().get(f"{settings.openshift_api_url}/.well-known/oauth-authorization-server")
        r.raise_for_status()
        _endpoints.update(r.json())
    return _endpoints[key]


# --- Admin kararı ---------------------------------------------------------------------------


async def is_admin(name: str, groups: list[str], token: str) -> bool:
    """Kullanıcı listesi, grup veya kullanıcının kendi token'ıyla SelfSubjectAccessReview. Hata -> False."""
    if name in settings.admin_user_set or set(groups) & settings.admin_group_set:
        return True
    ns = settings.admin_namespace or (NS_FILE.read_text().strip() if NS_FILE.exists() else "")
    if not ns:
        return False
    body = {
        "apiVersion": "authorization.k8s.io/v1",
        "kind": "SelfSubjectAccessReview",
        "spec": {"resourceAttributes": {
            "namespace": ns, "verb": settings.admin_sar_verb,
            "group": settings.admin_sar_group, "resource": settings.admin_sar_resource,
        }},
    }
    try:
        r = await client().post(
            f"{settings.openshift_api_url}/apis/authorization.k8s.io/v1/selfsubjectaccessreviews",
            json=body, headers={"Authorization": f"Bearer {token}"},
        )
        r.raise_for_status()
        return r.json().get("status", {}).get("allowed") is True
    except Exception:
        log.warning("SAR başarısız, admin değil sayılıyor", exc_info=True)
        return False


# --- Oturum ve bağımlılıklar ----------------------------------------------------------------


def current_user(request: Request) -> User | None:
    if settings.auth_mode == "dev":
        return User(settings.dev_user, [], settings.dev_admin)
    data = request.session.get("user")
    if not data or time.time() - request.session.get("login_at", 0) > settings.session_max_age:
        return None
    return msgspec.convert(data, User)


def require_user(request: Request) -> User:
    user = current_user(request)
    if user is None:
        raise HTTPException(401, "Giriş yapmanız gerekiyor.")
    return user


def require_admin(request: Request) -> User:
    user = require_user(request)
    if not user.is_admin:
        raise HTTPException(403, "Bu işlem için admin yetkisi gerekiyor.")
    return user


# --- Rotalar --------------------------------------------------------------------------------


@router.get("/auth/login")
async def login(request: Request):
    if settings.auth_mode == "dev":
        return RedirectResponse("/")
    try:
        authorize = await _endpoint("authorization_endpoint", settings.oauth_authorize_url)
    except Exception:
        log.exception("OAuth uç noktası keşfedilemedi")
        raise HTTPException(502, "OpenShift OAuth sunucusuna ulaşılamadı.")
    state = secrets.token_urlsafe(32)
    request.session["oauth_state"] = state
    q = urlencode({
        "response_type": "code", "client_id": settings.oauth_client_id,
        "redirect_uri": settings.oauth_redirect_url, "scope": SCOPE, "state": state,
    })
    return RedirectResponse(f"{authorize}?{q}")


@router.get("/auth/callback")
async def callback(request: Request, code: str = "", state: str = "", error: str = "", error_description: str = ""):
    expected = request.session.pop("oauth_state", "")
    if error:
        raise HTTPException(400, f"OpenShift girişi reddedildi: {error_description or error}")
    if not expected or not state or not secrets.compare_digest(state, expected):
        raise HTTPException(400, "Geçersiz oturum durumu (state). Lütfen girişi baştan başlatın.")
    if not code:
        raise HTTPException(400, "Yetkilendirme kodu eksik.")
    try:
        token_url = await _endpoint("token_endpoint", settings.oauth_token_url)
        r = await client().post(token_url, data={
            "grant_type": "authorization_code", "code": code, "redirect_uri": settings.oauth_redirect_url,
            "client_id": settings.oauth_client_id, "client_secret": settings.oauth_client_secret,
        })
        if r.status_code != 200:
            log.warning("token endpoint %s: %s", r.status_code, r.text[:200])
            raise HTTPException(502, "OpenShift token alınamadı. Lütfen tekrar deneyin.")
        token = r.json()["access_token"]
        r = await client().get(
            f"{settings.openshift_api_url}/apis/user.openshift.io/v1/users/~",
            headers={"Authorization": f"Bearer {token}"},
        )
        r.raise_for_status()
        info = r.json()
        name, groups = info["metadata"]["name"], info.get("groups") or []
        admin = await is_admin(name, groups, token)
    except HTTPException:
        raise
    except Exception:
        log.exception("OAuth callback hatası")
        raise HTTPException(502, "OpenShift ile iletişim kurulamadı.")
    request.session["user"] = msgspec.to_builtins(User(name, groups, admin))
    request.session["login_at"] = int(time.time())
    return RedirectResponse("/")


@router.get("/auth/logout")
async def logout(request: Request):
    request.session.clear()
    if settings.auth_mode == "dev":
        return RedirectResponse("/")
    # Doğrudan "/" yerine sayfa: OpenShift oturumu açıksa kullanıcı anında geri girmesin
    return HTMLResponse(
        '<!doctype html><meta charset="utf-8"><title>Çıkış yapıldı</title>'
        '<p>Çıkış yapıldı. <a href="/auth/login">Tekrar giriş yap</a></p>'
    )
