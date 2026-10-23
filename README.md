# mlops-chatbot

Veri bilimcilerin **Red Hat OpenShift AI** üzerindeki MLOps platformu, workbench/notebook'ları ve projeleri
hakkında soru sorduğu agentic RAG asistanı. Şirket içindeki **Azure DevOps Server (on-prem Azure Repos)**
repolarını indeksler, cevabı bu repolardaki doküman ve koda dayandırır, kullandığı dosyaya link verir.
GitHub ve Azure DevOps Services repoları da desteklenir.

> *"Projemde gRPC desteğini nasıl açarım?"*, *"Notebook'ta custom package nasıl kurarım?"*,
> *"Notebook'a API key'i nasıl eklerim?"*, *"Özel LLM modeline erişimi nasıl alırım?"*

**Teknolojiler:** Python 3.13 · FastAPI · msgspec · LangGraph · Qdrant · mem0 · OpenAI ChatKit · OpenShift

```
Tarayıcı (ChatKit + kenar çubuğu)
   │  OpenShift OAuth ile giriş (oturum çerezi)
   ▼
FastAPI ──/chatkit──> ChatKitServer ──arka plan görevi──> LangGraph agent
   │                       │                               ├─ prepare: mem0 hafıza ∥ Qdrant arama (paralel)
   │                       │                               ├─ agent (LLM, tool calling)
   │                       │                               └─ tools: search_docs (repo filtresiyle)
   │                       └─ Store: SQLite / Postgres (sohbetler, mesajlar)
   └─/api/admin/reindex──> indexer ──git clone──> Azure DevOps Server (on-prem) ──embedding──> Qdrant
                              ▲
              OpenShift CronJob (belirli aralıklarla)
```

## Özellikler

- **ChatGPT benzeri arayüz:** solda sohbet listesi, yeni sohbet, silme, açık/koyu tema (varsayılan açık).
  Mobilde ☰ ile açılır.
- **Bağımsız, asenkron sohbetler:** A sohbetinde soru sorup cevap beklemeden B sohbetine geçip soru
  sorabilirsin.
  - Cevap HTTP isteğinden bağımsız bir asyncio görevinde üretilir. Sohbet değişince bağlantı kopar ama
    sunucu cevabı bitirip kaydeder.
  - Kenar çubuğunda bekleyen sohbette dönen işaret, arkada tamamlananda mavi nokta görünür.
  - Aynı sohbette ikinci soru, ilk cevap bitince işlenir.
- **OpenShift girişi:** OpenShift'e girebilen herkes girer. Uygulamanın ServiceAccount'u OAuth client
  olarak kullanılır, cluster-admin gerekmez. OpenShift token'ı çerezde tutulmaz.
- **Admin yetkisi OpenShift'ten gelir:** `ADMIN_GROUPS` / `ADMIN_USERS` listesinde olan, veya namespace'te
  `ADMIN_SAR_*` yetkisine sahip olan admin sayılır. Varsayılan yetki `create rolebindings`, yani
  projenin admin'leri. Yetki SelfSubjectAccessReview ile sorulur.
- **İndeksleme:**
  - Belirli aralıklarla OpenShift CronJob'ı çalışır.
  - Manuel indeksleme için admin'ler arayüzdeki **⚙ İndeksleme** panelinden tüm repoları ya da tek repoyu
    başlatır ve durumu canlı izler.
  - İndeksleme artımlıdır: dosya hash'i değişmediyse yeniden embedding yapılmaz, silinen dosyaların
    parçaları Qdrant'tan da silinir.
  - Repolar paralel klonlanır, embedding'ler batch halinde paralel gönderilir.
- **Her şey parametrik:** repo listesi, modeller, Qdrant, admin kuralları ConfigMap'ten; anahtarlar
  Secret'tan gelir. ConfigMap'teki `sources.yaml` her indekslemede yeniden okunur, pod'u yeniden
  başlatmaya gerek yok.

### Agentic AI best practice'leri

| Uygulama | Nerede |
|---|---|
| **Agentic RAG:** ilk arama LLM'i beklemeden yapılır. Yetmezse agent `search_docs`'u farklı sorgu veya repo filtresiyle tekrar çağırır. | `app/agent.py` |
| **Sınırlı döngü:** `RECURSION_LIMIT` ile agent ↔ tool döngüsü sınırlıdır. | `app/agent.py` |
| **Paralellik:** mem0 hafızası ve doküman araması `asyncio.gather` ile aynı anda çalışır. | `prepare` node |
| **Uzun vadeli hafıza (mem0):** kullanıcıya özeldir ve sohbetler arası taşınır. Cevabı bekletmez, arka planda yazılır. Hata verirse cevap hafızasız üretilir. | `app/memory.py` |
| **Kısa vadeli hafıza:** sohbet geçmişinin tek kaynağı ChatKit store'dur. LLM'e son `HISTORY_LIMIT` mesaj gider. | `app/chat.py` |
| **Groundedness:** cevap yalnızca bulunan parçalara dayanır. Bulunamazsa "bilmiyorum" der, komut uydurmaz. | system prompt |
| **Atıf:** kullanılan dosyalar markdown link olarak yazılır ve cevabın altında kaynak kartı olarak görünür. | `_sources()` |
| **Prompt injection:** doküman içindeki talimatlar veri sayılır ve uygulanmaz. | system prompt |
| **Secret hijyeni:** gerçek API key yazılmaz veya istenmez. Git token'ı URL'ye ve loglara girmez. | prompt, `indexer.py` |
| **Dayanıklılık:** LLM ve embedding için retry ve timeout vardır. Bir repo hata verirse diğerleri devam eder. Qdrant veya mem0 kapalıysa uygulama yine açılır. | genel |
| **Gözlemlenebilirlik:** yapılandırılmış log. LangSmith için `LANGSMITH_TRACING=true` ve `LANGSMITH_API_KEY` vermek yeterli (LangGraph otomatik izler). | env |

## Proje yapısı

```
app/
  main.py         FastAPI: /chatkit, /api/threads, /api/admin/*, /healthz
  chat.py         ChatKitServer: arka plan görevleriyle bağımsız sohbetler
  agent.py        LangGraph grafiği ve search_docs aracı
  store.py        ChatKit Store (SQLAlchemy async: sqlite / postgres)
  indexer.py      git clone → parçala → embedding → Qdrant (CLI: python -m app.indexer)
  vectorstore.py  Qdrant koleksiyonu ve arama
  memory.py       mem0 (Qdrant üzerinde)
  auth.py         OpenShift OAuth + admin kararı
  llm.py          OpenAI uyumlu LLM ve embedding istemcileri
  config.py       Tüm ayarlar (msgspec, ortam değişkenlerinden)
  static/index.html   Arayüz (ChatKit web bileşeni + kenar çubuğu, build yok)
openshift/        BuildConfig, Deployment, Route, ConfigMap, Secret örneği, CronJob...
tests/            python -m tests.test_indexer / python -m tests.test_auth
```

## Lokal kurulum

Gerekenler: Python 3.13, Docker (Qdrant için), OpenAI uyumlu bir LLM ve embedding ucu (OpenAI, vLLM,
LiteLLM, Azure OpenAI...). Kurulum `uv` değil **pip** iledir.

```bash
git clone https://github.com/mustafacavusoglu/mlops-chatbot.git
cd mlops-chatbot
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

```bash
docker compose up -d                       # Qdrant: http://localhost:6333/dashboard
cp .env.example .env                       # LLM_*, EMBEDDING_* değerlerini doldur
cp sources.example.yaml sources.yaml       # indekslenecek repolar
```

`sources.example.yaml` içindeki `odh-docs` kaynağı OpenShift AI'ın upstream dokümantasyonudur (Open Data
Hub, public, token istemez). Şirket ağına bağlı değilken denemek için `sources.yaml`'da yalnızca onu
bırakabilirsin.

```bash
set -a; source .env; set +a
python -m app.indexer                      # repoları indeksle (tek repo: --source odh-docs)
uvicorn app.main:app --reload --port 8000  # http://localhost:8000
```

Lokalde `AUTH_MODE=dev` olduğu için giriş istenmez. Kullanıcı `DEV_USER`, admin yetkisi `DEV_ADMIN`'den
gelir. Testler:

```bash
python -m tests.test_indexer
```

```bash
python -m tests.test_auth
```

## OpenShift'e kurulum

### 1. Proje ve Secret

```bash
oc new-project mlops-chatbot
```

```bash
oc create secret generic mlops-chatbot-secrets \
  --from-literal=QDRANT_URL=http://qdrant.qdrant.svc:6333 \
  --from-literal=QDRANT_API_KEY=... \
  --from-literal=LLM_API_KEY=... \
  --from-literal=EMBEDDING_API_KEY=... \
  --from-literal=SESSION_SECRET="$(openssl rand -hex 32)" \
  --from-literal=AZURE_DEVOPS_PAT=...
```

Anahtarların tam listesi `openshift/secret.example.yaml` dosyasında. Gerçek değerleri repoya koyma.

### 2. ConfigMap'i düzenle

`openshift/configmap.yaml` içinde şunları değiştir:

| Anahtar | Değer |
|---|---|
| `OAUTH_CLIENT_ID` | `system:serviceaccount:<namespace>:mlops-chatbot` |
| `OAUTH_REDIRECT_URL` | `https://<route-host>/auth/callback` |
| `LLM_MODEL`, `EMBEDDING_MODEL`, `EMBEDDING_DIM` | kullanılan modeller; `EMBEDDING_DIM` modelin vektör boyutuyla aynı olmalı |
| `LLM_BASE_URL`, `EMBEDDING_BASE_URL` | şirket içi LLM ucu (ör. vLLM/LiteLLM), gerekirse ekle |
| `ADMIN_GROUPS` | admin sayılacak OpenShift grupları (virgülle) |
| `sources.yaml` | indekslenecek Azure DevOps Server repoları |

Deployment ve CronJob'daki `image-registry.openshift-image-registry.svc:5000/<namespace>/...` alanında da
`<namespace>`'i değiştir.

### 3. Uygula ve build al

```bash
oc apply -k openshift/
```

```bash
oc start-build mlops-chatbot --follow
```

BuildConfig, Dockerfile'ı kullanır ve kaynağı şirket içi Azure DevOps Server'daki bu repodan çeker. Bu
repoyu oraya mirror'la ve `openshift/buildconfig.yaml` içindeki `uri`'yi kendi adresinle değiştir. Git
kimliği için PAT'li bir basic-auth secret gerekir; sunucu şirket CA'sıyla imzalıysa `ca.crt` de eklenir:

```bash
oc create secret generic azure-repos-pat --type=kubernetes.io/basic-auth \
  --from-literal=username=pat --from-literal=password=<PAT> \
  --from-file=ca.crt=sirket-ca.pem
```

İmaj ImageStream'e yazılır. Deployment ve CronJob yeni imajı otomatik alır (`image.openshift.io/triggers`).
Git'ten build mümkün değilse lokal klasörden binary build al: BuildConfig'te `source: {type: Binary}`
yapıp `oc start-build mlops-chatbot --from-dir=. --follow` çalıştır.

Şirket içi PyPI mirror'ı (Nexus/Artifactory) ve base imaj için BuildConfig'e `buildArgs` ekle:
`PIP_INDEX_URL`, `PIP_TRUSTED_HOST`, `BASE_IMAGE`.

### 4. İlk indeksleme

```bash
oc create job --from=cronjob/mlops-chatbot-indexer ilk-indeksleme
```

Sonraki indekslemeler CronJob'ın `schedule` alanına göre (varsayılan 6 saatte bir) çalışır. Admin'ler
arayüzden de istediği zaman başlatabilir.

### OpenShift girişi nasıl çalışıyor?

1. Kullanıcı `/auth/login`'e gelir ve OpenShift OAuth sunucusuna yönlenir. Uygulama `user:info` ve
   `user:check-access` scope'larını ister.
2. OpenShift'e giriş yapıp izin verdikten sonra `/auth/callback`'e döner. Kod, token ile değiştirilir.
3. `users/~` çağrısından kullanıcı adı ve gruplar alınır. Admin kararı grup/kullanıcı listesiyle ya da
   SelfSubjectAccessReview ile verilir.
4. Oturum çerezine yalnızca ad, gruplar ve admin bilgisi yazılır. Çerez `SESSION_SECRET` ile imzalanır ve
   süresi `SESSION_MAX_AGE` (varsayılan 8 saat) kadardır.

OAuth client olarak `mlops-chatbot` ServiceAccount'u kullanılır:
- Redirect adresi SA üzerindeki `serviceaccounts.openshift.io/oauth-redirectreference.primary`
  annotation'ından Route'a bağlanır.
- Client secret `mlops-chatbot-oauth-token` secret'ındaki SA token'ıdır.

Bunlar `openshift/serviceaccount.yaml` dosyasında hazır.

OAuth sunucusu ve git sunucuları şirket CA'sı ile imzalıysa `mlops-chatbot-trusted-ca` ConfigMap'i
kümenin güvenilir CA paketini enjekte eder ve `SSL_CERT_FILE` ile kullanılır.

## Ayarlar

Tüm ayarlar `app/config.py` içinde. Ortam değişkeni adı, alan adının büyük harfidir.

| Değişken | Varsayılan | Açıklama |
|---|---|---|
| `LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL` | – / – / `gpt-4o-mini` | OpenAI uyumlu sohbet modeli |
| `MEMORY_LLM_MODEL` | `LLM_MODEL` | mem0'ın bilgi çıkarımı için (daha ucuz bir model olabilir) |
| `EMBEDDING_BASE_URL` / `EMBEDDING_API_KEY` | LLM değerleri | embedding ucu |
| `EMBEDDING_MODEL` / `EMBEDDING_DIM` | `text-embedding-3-small` / `1536` | boyut modelle aynı olmalı |
| `EMBEDDING_QUERY_PREFIX` | boş | Qwen3/e5 gibi modellerde sorgu talimatı |
| `QDRANT_URL` / `QDRANT_API_KEY` | `http://localhost:6333` / – | Secret'tan |
| `QDRANT_COLLECTION` / `MEMORY_COLLECTION` | `mlops_docs` / `mlops_memories` | |
| `MEMORY_ENABLED` | `true` | mem0'ı kapatmak için `false` |
| `TOP_K`, `CHUNK_SIZE`, `CHUNK_OVERLAP` | `6`, `1500`, `200` | retrieval / parçalama |
| `HISTORY_LIMIT`, `RECURSION_LIMIT` | `20`, `12` | LLM'e giden geçmiş, agent döngü sınırı |
| `DATABASE_URL` | sqlite | çok pod için `postgresql+asyncpg://user:pass@host/db` |
| `SOURCES_FILE` | `./sources.yaml` | repo listesi |
| `INDEX_CONCURRENCY`, `EMBEDDING_CONCURRENCY`, `EMBEDDING_BATCH_SIZE` | `4`, `4`, `64` | paralellik |
| `AUTH_MODE` | `openshift` | lokalde `dev` |
| `ADMIN_GROUPS`, `ADMIN_USERS` | boş | virgülle ayrılmış |
| `ADMIN_SAR_VERB` / `_RESOURCE` / `_GROUP` | `create` / `rolebindings` / `rbac.authorization.k8s.io` | bu yetkiye sahip olan admin |
| `SESSION_SECRET` | – | **zorunlu** (openshift modunda) |
| `CHATKIT_JS_URL`, `CHATKIT_DOMAIN_KEY` | OpenAI CDN, `local-dev` | aşağıya bak |

### Repo listesi (`sources.yaml`)

```yaml
sources:
  - name: mlops-platform              # aramada filtre ve atıf adı
    url: https://azuredevops.sirket.local/DefaultCollection/MLOps/_git/mlops-platform
    branch: main
    token_env: AZURE_DEVOPS_PAT       # PAT'i tutan ORTAM DEĞİŞKENİNİN adı (değer Secret'ta)
    include: ["docs/**/*.md", "**/*.ipynb", "**/*.py"]
    exclude: ["tests/**"]
  - name: workbench-images
    url: https://azuredevops.sirket.local/DefaultCollection/MLOps/_git/workbench-images
    branch: develop
    token_env: AZURE_DEVOPS_PAT
```

- `provider` yazılmazsa URL'den anlaşılır: `/_git/` geçiyorsa Azure DevOps, değilse GitHub. Kaynak
  kartlarındaki linkler buna göre üretilir.
- `include` boş bırakılırsa md, adoc, rst, ipynb, py, yaml, toml, Dockerfile, proto gibi varsayılan desenler
  kullanılır.
- Notebook'ların markdown ve kod hücreleri indekslenir, çıktılar atılır.
- Token git'e `http.extraHeader` ile verilir, URL'ye yazılmaz.

### On-prem Azure DevOps Server

- **Adres biçimi:** `https://<sunucu>/<collection>/<project>/_git/<repo>`. Eski TFS kurulumlarında
  `https://<sunucu>/tfs/<collection>/<project>/_git/<repo>` olur. Adresi repo sayfasındaki **Clone**
  butonundan kopyala.
- **PAT:** kullanıcı ayarlarındaki **Personal access tokens** ekranından **Code → Read** yetkisiyle
  oluştur ve Secret'a `AZURE_DEVOPS_PAT` olarak koy.
  - Indexer token'ı `git -c http.extraHeader="Authorization: Basic ..."` ile verir; URL'ye ve loglara
    yazmaz.
  - Sunucuda PAT ile basic auth açık olmalıdır. Azure DevOps Server 2019 ve sonrasında HTTPS üzerinden
    varsayılan olarak açıktır.
  - Birden fazla collection ya da farklı yetki gerekiyorsa her repo için ayrı `token_env` kullan, örneğin
    `token_env: PAT_DATA_TEAM`, ve değeri Secret'a ekle.
- **Şirket CA'sı:** git, Python'un okuduğu `SSL_CERT_FILE` değişkenine bakmaz. Bu yüzden Deployment ve
  CronJob'da `GIT_SSL_CAINFO` da kümenin güvenilir CA paketine (`mlops-chatbot-trusted-ca`) bakar. Şirket
  CA'sı kümenin `trustedCA` ayarında yoksa paketi kendi `ca.crt`'inle değiştir.
- **Proxy:** küme çıkışı proxy üzerindeyse Azure DevOps sunucusunu, Qdrant'ı ve LLM ucunu `NO_PROXY`'ye ekle.

## Önemli: ChatKit'in internet ihtiyacı

ChatKit arayüzü OpenAI tarafından barındırılan bir web bileşenidir. Self-host edilemiyor. Tarayıcının şu
adreslere erişmesi gerekir:

- `cdn.platform.openai.com`: `chatkit.js` ve bileşenin iframe'i.
- `api.openai.com`: domain key doğrulaması.

Sohbet verisi OpenAI'a gitmez. Mesajlar yalnızca bu uygulamanın `/chatkit` ucuna ve senin LLM'ine gider.
Kapalı ağda yapılacaklar:

1. Kullanıcı tarayıcılarının proxy/firewall üzerinden bu iki adrese erişimini aç.
2. [OpenAI Platform → Settings → Security → Domain allowlist](https://platform.openai.com/settings/organization/security/domain-allowlist)
   ekranında Route adresini ekle ve verilen anahtarı `CHATKIT_DOMAIN_KEY` olarak gir. `localhost`'ta
   doğrulama atlanır, `local-dev` yeterlidir.
3. Bileşen yüklenemezse arayüz bunu açıklayan bir uyarı gösterir.

## Ölçek ve sınırlar

- **Tek pod + SQLite** varsayılandır (PVC, `strategy: Recreate`). Birden çok pod için
  `DATABASE_URL`'i Postgres yap, PVC'yi kaldır ve `replicas`'ı artır.
- "Cevap üretiliyor" göstergesi ve aynı sohbette sıraya koyma pod içidir. Route'un varsayılan sticky
  cookie'si kullanıcıyı aynı poda götürür.
- Pod tam cevap üretirken kapanırsa o cevap ve bekleyen mem0 kaydı kaybolur. Kalıcı olması gerekirse
  bir kuyruk (arq/celery) eklenebilir.
- `WORKERS=1` önerilir. Ölçeklemeyi pod sayısıyla yap: uygulama tamamen async olduğu için tek süreç
  çok sayıda eşzamanlı sohbeti kaldırır.
- Admin yetkisi girişte hesaplanır. OpenShift'te yetkisi alınan kullanıcı, oturum süresi dolana kadar
  admin kalır.
