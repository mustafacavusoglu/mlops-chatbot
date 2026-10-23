"""Tüm ayarlar ortam değişkenlerinden gelir (OpenShift: ConfigMap + Secret -> envFrom).

Değişken adı = alan adının büyük harfi, örn. `llm_model` -> `LLM_MODEL`.
Repo listesi ayrı bir YAML dosyasıdır (ConfigMap volume): `SOURCES_FILE`.
"""

import os
from pathlib import Path

import msgspec
import msgspec.yaml

DEFAULT_INCLUDE = [
    "**/*.md", "**/*.adoc", "**/*.rst", "**/*.ipynb", "**/*.py", "**/*.yaml", "**/*.yml", "**/*.toml", "**/*.cfg",
    "**/*.txt", "**/*.sh", "**/Dockerfile", "**/*.proto", "**/*.json",
]


class Source(msgspec.Struct, forbid_unknown_fields=True):
    """İndekslenecek bir git reposu: Azure DevOps Server (on-prem), Azure DevOps Services veya GitHub."""

    name: str  # kısa ad; aramada filtre ve atıf için
    url: str  # https clone adresi
    provider: str | None = None  # azure | github; boşsa URL'den anlaşılır ("/_git/" -> azure)
    branch: str = "main"
    token_env: str | None = None  # PAT'i tutan ortam değişkeninin ADI (değer Secret'ta)
    include: list[str] = msgspec.field(default_factory=lambda: list(DEFAULT_INCLUDE))
    exclude: list[str] = msgspec.field(default_factory=list)
    max_file_kb: int = 512  # daha büyük dosyalar atlanır (lock dosyaları, veri)


class SourcesFile(msgspec.Struct):
    sources: list[Source] = []


class Settings(msgspec.Struct):
    app_title: str = "MLOps Asistanı"
    log_level: str = "INFO"

    # LLM: OpenAI uyumlu herhangi bir uç (vLLM, LiteLLM, Azure OpenAI v1, OpenAI)
    llm_base_url: str | None = None
    llm_api_key: str = "not-set"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.0
    llm_timeout: float = 120.0
    memory_llm_model: str | None = None  # mem0'ın gerçek çıkarımı için; boşsa llm_model

    # Embedding: boş bırakılanlar LLM ayarlarından gelir
    embedding_base_url: str | None = None
    embedding_api_key: str | None = None
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536
    embedding_query_prefix: str = ""  # bazı modeller (Qwen3, e5) sorguda talimat ister
    embedding_batch_size: int = 64
    embedding_concurrency: int = 4

    # Qdrant (adres + anahtar Secret'tan)
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "mlops_docs"
    memory_collection: str = "mlops_memories"
    memory_enabled: bool = True

    # Retrieval / agent
    top_k: int = 6
    chunk_size: int = 1500
    chunk_overlap: int = 200
    history_limit: int = 20  # LLM'e giden son mesaj sayısı
    recursion_limit: int = 12  # agent <-> tools döngü sınırı

    # Sohbet deposu: OpenShift'te Postgres (postgresql://user:pass@host:5432/db), lokalde sqlite
    database_url: str = "sqlite+aiosqlite:///./data/chat.db"
    db_pool_size: int = 10

    # İndeksleme
    sources_file: str = "./sources.yaml"
    index_workdir: str = "/tmp/mlops-chatbot-repos"
    index_concurrency: int = 4  # aynı anda klonlanan/işlenen repo sayısı

    # Kimlik doğrulama: openshift | dev (dev: lokal geliştirme, giriş yok)
    auth_mode: str = "openshift"
    dev_user: str = "developer"
    dev_admin: bool = True
    session_secret: str = "change-me"
    session_max_age: int = 8 * 3600
    openshift_api_url: str = "https://kubernetes.default.svc"
    oauth_client_id: str = ""  # system:serviceaccount:<ns>:<sa>
    oauth_client_secret: str = ""  # service account token
    oauth_redirect_url: str = ""  # https://<route>/auth/callback
    oauth_authorize_url: str | None = None  # boşsa .well-known'dan keşfedilir
    oauth_token_url: str | None = None
    ca_file: str | None = "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"
    tls_verify: bool = True
    # Admin: listedeki grup/kullanıcılar VEYA namespace'te bu yetkiye sahip olanlar (SubjectAccessReview)
    admin_groups: str = ""  # virgülle ayrılmış
    admin_users: str = ""
    admin_namespace: str | None = None  # boşsa pod'un namespace'i
    admin_sar_verb: str = "create"
    admin_sar_resource: str = "rolebindings"
    admin_sar_group: str = "rbac.authorization.k8s.io"

    # Arayüz
    chatkit_js_url: str = "https://cdn.platform.openai.com/deployments/chatkit/chatkit.js"
    chatkit_domain_key: str = "local-dev"

    def __post_init__(self):
        self.embedding_base_url = self.embedding_base_url or self.llm_base_url
        self.embedding_api_key = self.embedding_api_key or self.llm_api_key
        self.memory_llm_model = self.memory_llm_model or self.llm_model

    @property
    def admin_group_set(self) -> set[str]:
        return {g.strip() for g in self.admin_groups.split(",") if g.strip()}

    @property
    def admin_user_set(self) -> set[str]:
        return {u.strip() for u in self.admin_users.split(",") if u.strip()}


def load() -> Settings:
    fields = set(Settings.__struct_fields__)
    env = {k.lower(): v for k, v in os.environ.items() if k.lower() in fields}
    return msgspec.convert(env, Settings, strict=False)  # strict=False: "true" -> bool, "5" -> int


def load_sources() -> list[Source]:
    """Her indekslemede yeniden okunur: ConfigMap değişince pod yeniden başlatmaya gerek yok."""
    path = Path(settings.sources_file)
    if not path.exists():
        return []
    return msgspec.yaml.decode(path.read_bytes(), type=SourcesFile).sources


settings = load()
