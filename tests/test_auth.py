"""python -m tests.test_auth  (pytest yok, düz assert)"""

import asyncio
import json
from types import SimpleNamespace

import httpx

from app import auth
from app.auth import User, current_user, is_admin
from app.config import settings

settings.admin_namespace = "ns"
settings.admin_users, settings.admin_groups = "root", "mlops-admins"


def mock(handler):
    auth._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))


def sar(allowed):
    def h(req):
        body = json.loads(req.content)["spec"]["resourceAttributes"]
        assert body["namespace"] == "ns" and req.headers["authorization"] == "Bearer t"
        return httpx.Response(201, json={"status": {"allowed": allowed}})
    return h


def run(name, groups=()):
    return asyncio.run(is_admin(name, list(groups), "t"))


def boom(req):
    raise AssertionError("SAR çağrılmamalıydı")


mock(boom)
assert run("root") and run("x", ["mlops-admins"])  # liste/grup: SAR yok
mock(sar(True)); assert run("x")
mock(sar(False)); assert not run("x")
mock(lambda r: httpx.Response(500)); assert not run("x")  # hata -> fail closed
mock(lambda r: (_ for _ in ()).throw(httpx.ConnectError("x"))); assert not run("x")

settings.auth_mode = "dev"
assert current_user(SimpleNamespace()) == User(settings.dev_user, [], settings.dev_admin)
settings.auth_mode = "openshift"
req = SimpleNamespace(session={})
assert current_user(req) is None
req.session = {"user": {"name": "a", "groups": ["g"], "is_admin": True}, "login_at": 1}
assert current_user(req) is None  # süresi dolmuş
import time
req.session["login_at"] = time.time()
assert current_user(req) == User("a", ["g"], True)
print("ok")
