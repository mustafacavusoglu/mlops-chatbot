# mlops-chatbot

Veri bilimcilerin **Red Hat OpenShift AI** üzerindeki MLOps platformu, workbench/notebook'ları ve projeleri
hakkında soru sorduğu agentic RAG asistanı. Şirket içindeki **Azure DevOps Server (on-prem Azure Repos)**
repolarını indeksler, cevabı bu repolardaki doküman ve koda dayandırır, kullandığı dosyaya link verir.
GitHub ve Azure DevOps Services repoları da desteklenir.

> *"Projemde gRPC desteğini nasıl açarım?"*, *"Notebook'ta custom package nasıl kurarım?"*,
> *"Notebook'a API key'i nasıl eklerim?"*, *"Özel LLM modeline erişimi nasıl alırım?"*

**Teknolojiler:** Python 3.13 · FastAPI · msgspec · LangGraph · Qdrant · mem0 · OpenAI ChatKit (protokol) · OpenShift

```
Tarayıcı (kendi sohbet arayüzümüz veya ChatKit bileşeni + kenar çubuğu)
   │  OpenShift OAuth ile giriş (oturum çerezi)
   ▼
FastAPI ──/chatkit──> ChatKitServer ──arka plan görevi──> LangGraph agent
   │                       │                               ├─ prepare: mem0 hafıza ∥ Qdrant arama (paralel)
   │                       │                               ├─ agent (LLM, tool calling)
   │                       │                               └─ tools: search_docs (repo filtresiyle)
   │                       └─ Store: PostgreSQL (sohbetler, mesajlar)
   └─/api/admin/reindex──> indexer ──git clone──> Azure DevOps Server (on-prem) ──embedding──> Qdrant
                              ▲
              OpenShift CronJob (belirli aralıklarla)
```

## Özellikler

- **ChatGPT benzeri arayüz:** solda sohbet listesi, yeni sohbet, açık/koyu tema (varsayılan açık).
- **Geri bildirim:** cevapların altında 👍/👎; admin'ler oyları ve her cevabın arama izini panelden görür,
  JSONL olarak indirip notebook'ta analiz eder (bkz. [RAG kalitesini inceleme](#rag-kalitesini-inceleme)).
- **Yönetim paneli (`/admin`):** kullanım KPI'ları ve grafik, tüm konuşmalar ve arama izleri, geri bildirimler,
  indeksleme ve çalıştırma geçmişi (bkz. [Yönetim paneli](#yönetim-paneli-admin)). Loglar JSON.
- **Sohbet silme:** listede her sohbetin yanında × butonu var, silmeden önce onay ister.
  - Cevabı hâlâ üretilen bir sohbet silinirse arka plandaki görev durdurulur ve silinen sohbete cevap
    yazılmaz.
  - Kullanıcı yalnızca kendi sohbetlerini görebilir ve silebilir. Başkasının ya da olmayan bir sohbet 404
    döner.
  - Sohbetten mem0'a öğrenilmiş genel bilgiler (ör. "X projesinde çalışıyor") kullanıcı hafızasında
    kalır. Cevap izleri ve oylar da analiz için kalır.
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
  - Manuel indeksleme için admin'ler arayüzdeki **⚙ Yönetim** panelinden tüm repoları ya da tek repoyu
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
  store.py        ChatKit Store (SQLAlchemy async: PostgreSQL, lokalde sqlite)
  indexer.py      git clone → parçala → embedding → Qdrant (CLI: python -m app.indexer)
  vectorstore.py  Qdrant koleksiyonu ve arama
  memory.py       mem0 (Qdrant üzerinde)
  auth.py         OpenShift OAuth + admin kararı
  llm.py          OpenAI uyumlu LLM ve embedding istemcileri
  config.py       Tüm ayarlar (msgspec, ortam değişkenlerinden)
  logs.py         Log biçimi: JSON (Loki) veya text
  static/index.html   Arayüz: kenar çubuğu + native sohbet ekranı / ChatKit bileşeni (build yok)
  static/admin.html   Yönetim paneli (/admin): genel bakış, konuşmalar, geri bildirimler, indeksleme
  static/vendor/      marked + DOMPurify (imajın içinde, CDN yok)
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

### 0. Önceden hazır olması gerekenler

| Gereken | Not |
|---|---|
| `oc` ile kümeye giriş | Namespace içinde admin yetkin olmalı |
| PostgreSQL | Boş bir DB ve kullanıcı yeterli; tabloları uygulama açılırken kendisi oluşturur (bkz. [Sohbetler nerede tutuluyor?](#sohbetler-nerede-tutuluyor)) |
| Qdrant | Kümede çalışıyor olmalı; adresi ve varsa API key'i lazım. Yoksa resmi Helm chart'ıyla (`qdrant/qdrant`) kurulabilir |
| LLM ve embedding ucu | OpenAI uyumlu olmalı, ör. OpenShift AI'da vLLM ile serve edilen model. Adres, model adı ve embedding boyutu lazım |
| Azure DevOps PAT | **Code → Read** yetkisi yeterli; hem repoları indekslemek hem build için kullanılır |
| İnternet | **Gerekmez.** Varsayılan arayüz (`UI_MODE=native`) tamamen kümede çalışır. Yalnızca `UI_MODE=chatkit` seçilirse tarayıcıların OpenAI CDN'ine erişmesi gerekir (bkz. [Arayüz modları](#arayüz-modları-native-ve-chatkit)) |

### 1. Kodu Azure DevOps Server'a koy

Build runner'ları şirket içi git'ten çektiği için repoyu oraya mirror'la:

```bash
git clone https://github.com/mustafacavusoglu/mlops-chatbot.git && cd mlops-chatbot
```

```bash
git remote add azure https://azuredevops.sirket.local/DefaultCollection/MLOps/_git/mlops-chatbot
```

```bash
git push azure main
```

Git'ten build mümkün değilse lokal klasörden binary build alınabilir: BuildConfig'te `source: {type: Binary}`
yapıp 6. adımda `oc start-build mlops-chatbot --from-dir=. --follow` çalıştır.

### 2. Proje aç ve yer tutucuları doldur

```bash
oc new-project mlops-chatbot
```

```bash
perl -pi -e 's/<namespace>/mlops-chatbot/g' openshift/*.yaml
```

Ardından dosyalarda şunları kendi değerlerinle değiştir:

- **`openshift/buildconfig.yaml` → `uri`:** 1. adımda push ettiğin repo adresi.
- **`openshift/configmap.yaml`:**
  - `LLM_BASE_URL` ve `EMBEDDING_BASE_URL`: yorum satırını aç.
  - `LLM_MODEL`, `EMBEDDING_MODEL`, `EMBEDDING_DIM`: `EMBEDDING_DIM`, modelin vektör boyutuyla aynı olmalı.
  - `ADMIN_GROUPS`: admin sayılacak OpenShift grupları (virgülle).
  - `sources.yaml`: indekslenecek Azure Repos adresleri (bkz. [Repo listesi](#repo-listesi-sourcesyaml)).
- **`OAUTH_REDIRECT_URL`:** şimdilik bırak, 5. adımda dolduracağız.

Şirket içi PyPI mirror'ı (Nexus/Artifactory) veya farklı bir base imaj kullanılacaksa BuildConfig'e
`buildArgs` ekle: `PIP_INDEX_URL`, `PIP_TRUSTED_HOST`, `BASE_IMAGE`.

### 3. Secret'ları oluştur

Uygulamanın kullandığı anahtarlar:

```bash
oc create secret generic mlops-chatbot-secrets \
  --from-literal=QDRANT_URL=http://qdrant.qdrant.svc:6333 \
  --from-literal=QDRANT_API_KEY=<qdrant-key> \
  --from-literal=LLM_API_KEY=<llm-key> \
  --from-literal=EMBEDDING_API_KEY=<embedding-key> \
  --from-literal=SESSION_SECRET="$(openssl rand -hex 32)" \
  --from-literal=DATABASE_URL='postgresql://chatbot:<sifre>@postgres.db.svc:5432/mlops_chatbot' \
  --from-literal=AZURE_DEVOPS_PAT=<pat>
```

Build'in git'ten kod çekebilmesi için PAT ve şirket CA'sı:

```bash
oc create secret generic azure-repos-pat --type=kubernetes.io/basic-auth \
  --from-literal=username=pat --from-literal=password=<pat> \
  --from-file=ca.crt=sirket-ca.pem
```

- Postgres şifresinde `@` veya `:` varsa URL-encode et (`@` → `%40`, `:` → `%3A`).
- LLM'inde key yoksa `LLM_API_KEY` için herhangi bir değer verebilirsin.
- Anahtarların tam listesi `openshift/secret.example.yaml` dosyasında. Gerçek değerleri repoya koyma.

### 4. Kaynakları uygula

```bash
oc apply -k openshift/
```

Bu komut şunları oluşturur: ImageStream, BuildConfig, ServiceAccount (OAuth client olarak kullanılır) ve
onun token'ı, ConfigMap, güvenilir CA ConfigMap'i, Deployment, Service, Route ve indeksleme CronJob'ı.

### 5. Route adresini OAuth ayarına yaz

Route oluşmadan adresi belli olmuyor, bu yüzden bu adım `apply`'dan sonra:

```bash
oc get route mlops-chatbot -o jsonpath='{.spec.host}'
```

Çıkan adresi ConfigMap'e yaz:

```bash
oc patch configmap mlops-chatbot-config --type merge -p '{"data":{"OAUTH_REDIRECT_URL":"https://<cikan-host>/auth/callback"}}'
```

`UI_MODE=chatkit` kullanıyorsan aynı adresi OpenAI Platform'daki domain allowlist ekranına da ekle. İleride `oc apply -k` tekrar
çalıştırılacaksa `configmap.yaml` içindeki `OAUTH_REDIRECT_URL`'i de bu adresle güncelle; yoksa patch ezilir.

### 6. Build al

```bash
oc start-build mlops-chatbot --follow
```

Build bitince Deployment ve CronJob yeni imajı otomatik alır (`image.openshift.io/triggers`). Pod daha
önce ayağa kalktıysa, değişen ConfigMap'i okuması için yeniden başlat:

```bash
oc rollout restart deployment/mlops-chatbot
```

### 7. İlk indeksleme

```bash
oc create job --from=cronjob/mlops-chatbot-indexer ilk-indeksleme
```

```bash
oc logs -f job/ilk-indeksleme
```

Bundan sonra CronJob her 6 saatte bir çalışır; aralık `openshift/cronjob-indexer.yaml` içindeki
`schedule` alanında. Admin'ler arayüzdeki **⚙ Yönetim** panelinden istediği zaman da başlatabilir.

### 8. Kontrol

```bash
oc get pods
```

```bash
oc logs deployment/mlops-chatbot
```

Route adresini tarayıcıda aç. OpenShift giriş ekranı gelmeli, giriş yapınca sohbet ekranı açılmalı.
Admin'sen sol altta "Admin" etiketi ve İndeksleme butonu görünür.

### Sık karşılaşılabilecek sorunlar

| Belirti | Sebep / çözüm |
|---|---|
| Girişte `redirect_uri` hatası | `OAUTH_REDIRECT_URL` Route adresiyle birebir aynı değil (https ve `/auth/callback` dahil) |
| Pod açılmıyor: `SESSION_SECRET ayarlanmalı` | Secret'ta `SESSION_SECRET` eksik |
| (`UI_MODE=chatkit`) Sohbet ekranı boş ve uyarı çıkıyor | Tarayıcı OpenAI CDN'ine erişemiyor ya da domain key yanlış; `UI_MODE=native` kullan |
| İndekslemede `git clone başarısız` | PAT yetkisi yetersiz ya da sertifika hatası: şirket CA'sı kümenin trustedCA ayarında değil (bkz. [On-prem Azure DevOps Server](#on-prem-azure-devops-server)) |
| Admin görünmüyor | Kullanıcı `ADMIN_GROUPS` içindeki grupta değil ya da namespace'te admin değil. Yetki değişince çıkış yapıp tekrar gir |
| Postgres'e bağlanamıyor | `DATABASE_URL` yanlış veya şifre URL-encode edilmemiş; NetworkPolicy DB namespace'ine erişimi engelliyor olabilir |

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
| `DATABASE_URL` | sqlite (lokal) | OpenShift'te **Secret'tan** `postgresql://user:pass@host:5432/db` |
| `DB_POOL_SIZE` | `10` | pod başına Postgres bağlantı havuzu (+ aynı kadar taşma) |
| `SOURCES_FILE` | `./sources.yaml` | repo listesi |
| `INDEX_CONCURRENCY`, `EMBEDDING_CONCURRENCY`, `EMBEDDING_BATCH_SIZE` | `4`, `4`, `64` | paralellik |
| `AUTH_MODE` | `openshift` | lokalde `dev` |
| `ADMIN_GROUPS`, `ADMIN_USERS` | boş | virgülle ayrılmış |
| `ADMIN_SAR_VERB` / `_RESOURCE` / `_GROUP` | `create` / `rolebindings` / `rbac.authorization.k8s.io` | bu yetkiye sahip olan admin |
| `SESSION_SECRET` | – | **zorunlu** (openshift modunda) |
| `LOG_LEVEL` / `LOG_FORMAT` | `INFO` / `text` | OpenShift'te `json` (ConfigMap'te ayarlı) |
| `UI_MODE` | `native` | `native` (internet gerekmez) veya `chatkit` |
| `CHATKIT_JS_URL`, `CHATKIT_DOMAIN_KEY` | OpenAI CDN, `local-dev` | yalnızca `UI_MODE=chatkit` için |

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

## Arayüz modları: native ve ChatKit

Backend [OpenAI ChatKit](https://github.com/openai/chatkit-python) protokolünü konuşur (`/chatkit`:
JSON + SSE). Bu protokolü iki farklı arayüz kullanabilir; seçim ConfigMap'teki `UI_MODE` ile yapılır.

| | `UI_MODE=native` (varsayılan) | `UI_MODE=chatkit` |
|---|---|---|
| Arayüz | Kendi sohbet ekranımız (`app/static/index.html`) | OpenAI'ın ChatKit web bileşeni |
| İnternet | **Gerekmez.** Markdown için `marked`, güvenlik için `DOMPurify` imajın içinde (`app/static/vendor/`) | Tarayıcı `cdn.platform.openai.com` ve `api.openai.com`'a erişmeli |
| Ek ayar | Yok | OpenAI Platform'dan domain key (`CHATKIT_DOMAIN_KEY`) |
| Sohbet değiştirme | Her sohbetin akışı ayrı tutulur; cevap gelirken başka sohbete geçip dönünce cevap kaldığı yerden akmaya devam eder | Sohbet değişince bileşen yeniden kurulur, cevap arkada biter ve dönünce görünür |

Her iki modda da kenar çubuğu, sohbet silme, admin indeksleme paneli ve açık/koyu tema aynıdır. Sohbet
verisi hiçbir modda OpenAI'a gitmez; mesajlar yalnızca bu uygulamaya ve senin LLM'ine gider.

`UI_MODE=chatkit` kullanılacaksa:

1. Kullanıcı tarayıcılarının proxy/firewall üzerinden yukarıdaki iki adrese erişimini aç.
2. [OpenAI Platform → Settings → Security → Domain allowlist](https://platform.openai.com/settings/organization/security/domain-allowlist)
   ekranında Route adresini ekle ve verilen anahtarı `CHATKIT_DOMAIN_KEY` olarak gir. `localhost`'ta
   doğrulama atlanır, `local-dev` yeterlidir.
3. Bileşen yüklenemezse arayüz bunu açıklayan bir uyarı gösterir; `UI_MODE=native`'e dönmek yeterlidir.

Vendor dosyalarının sürümleri: `marked` 18.0.13 (MIT), `DOMPurify` 3.4.15 (Apache-2.0 / MPL-2.0).

## Sohbetler nerede tutuluyor?

Her şey **PostgreSQL**'de, beş tabloda tutulur:

| Tablo | İçerik | Sohbet silinince |
|---|---|---|
| `threads` | sohbet: `id`, `user_id`, `created_at`, `data` (başlık vb. JSON) | silinir |
| `items` | mesaj: `id`, `thread_id`, `user_id`, `created_at`, `data` (soru/cevap ve kaynak kartları, JSON) | silinir |
| `traces` | her cevabın izi: soru, cevap, agent'ın yaptığı **her arama** (sorgu, repo filtresi, gelen parçalar, metinleri ve skorları), atıf yapılan dosyalar, model, toplam süre ve ilk token süresi, hata | **kalır** |
| `feedback` | 👍/👎: hangi cevap, kim, ne zaman. Kullanıcı başına cevap başına bir oy; yeniden oy öncekini ezer | **kalır** |
| `index_runs` | indeksleme çalıştırmaları (CronJob ve manuel): kim/ne başlattı, ne zaman, repo bazında sonuç ve hatalar | – |

- Tablolar ve indeksler uygulama açılırken yoksa oluşturulur. Ayrıca migration gerekmez; DB ve kullanıcının
  hazır olması yeterli:

  ```sql
  CREATE USER chatbot WITH PASSWORD '...';
  CREATE DATABASE mlops_chatbot OWNER chatbot;
  ```

- Bağlantı Secret'taki `DATABASE_URL` ile verilir. `postgresql://` ile başlayan adres kabul edilir;
  async sürücü (asyncpg) otomatik seçilir. Şifrede özel karakter varsa URL-encode et (`@` → `%40`,
  `:` → `%3A`).
- Postgres operator'ü (Crunchy PGO vb.) kullanıcı secret'ı üretiyorsa, Deployment'ta yorum satırındaki
  `secretKeyRef` ile o secret'ın `uri` anahtarı doğrudan kullanılabilir.
- Vektörler (doküman parçaları, mem0 hafızası) Postgres'te değil, Qdrant'tadır.
- Lokal geliştirmede `DATABASE_URL` verilmezse sqlite (`./data/chat.db`) kullanılır. OpenShift modunda
  sqlite görülürse uygulama uyarı loglar, çünkü pod yeniden başlayınca sohbetler kaybolur.

## Yönetim paneli (`/admin`)

Admin'ler sohbet ekranındaki **⚙ Yönetim** butonuyla ya da doğrudan `/admin` adresiyle girer. Admin
olmayanlar 403 alır; admin kararı [OpenShift girişi](#openshift-girişi-nasıl-çalışıyor) bölümündeki gibidir.

| Sekme | İçerik |
|---|---|
| **Genel bakış** | 7 / 30 / 90 günlük KPI'lar: soru, aktif kullanıcı, sohbet, memnuniyet (👍 oranı), ortalama ve p95 cevap süresi, ilk token süresi, hata oranı. Günlük soru grafiği (hatalı günler işaretli), en çok atıf alan dokümanlar ve **en zayıf aramalar**: en iyi retrieval skoru en düşük sorular, yani dokümanın büyük ihtimalle eksik olduğu konular |
| **Konuşmalar** | Tüm cevaplar: metin, kullanıcı, oy, "sadece hatalılar" ve dönem süzgeçleri; sayfalı tablo. Satıra tıklayınca detay paneli: soru, markdown cevap, model, süre, oy ve **her arama** (sorgu, gelen dosyalar, skor çubukları, "✓ atıf", parçanın metni) |
| **Geri bildirimler** | 👎 / 👍 / tümü listesi, detay paneli ve **JSONL indir** |
| **İndeksleme** | Kaynak repolar; tek tek ya da hepsini indeksleme, canlı ilerleme ve **çalıştırma geçmişi**. Geçmiş Postgres'te tutulduğu için CronJob'ın zamanlanmış çalıştırmaları da "Zamanlanmış" olarak görünür. Satır açılınca repo bazında sonuç ve hatalar |

### Loglar

Panel uygulamanın anlamlı olaylarını gösterir. Teknik pod logları ise OpenShift'te kalır: konsoldaki pod
**Logs** sekmesi, ya da kümede OpenShift Logging (Loki) varsa **Observe → Logs**.

OpenShift'te loglar JSON'dur (`LOG_FORMAT=json`); her satır `ts`, `level`, `logger`, `msg` ve olaya özel
alanlar taşır. Uvicorn'un erişim logları da aynı biçimdedir. İki olay tipi özellikle işe yarar:

```json
{"ts":"…","level":"INFO","logger":"app.chat","msg":"cevap tamamlandı","event":"answer","item_id":"msg_…","thread_id":"thr_…","user":"ayse","latency_ms":10026,"first_token_ms":4241,"n_search":3,"top_score":0.75,"n_cited":5,"error":false}
{"ts":"…","level":"INFO","logger":"app.indexer","msg":"repo indekslendi","event":"index","repo":"mlops-platform","trigger":"schedule","files_indexed":12,"files_skipped":630,"files_deleted":1,"chunks":48,"error":null}
```

Soru ve cevap metinleri loga yazılmaz, sadece `traces` tablosunda durur. Loki'de örnek sorgular:

```
{kubernetes_namespace_name="mlops-chatbot"} | json | level="ERROR"
{kubernetes_namespace_name="mlops-chatbot"} | json | event="answer" | latency_ms > 20000
{kubernetes_namespace_name="mlops-chatbot"} | json | event="index" | error != ""
```

Lokalde `LOG_FORMAT` verilmezse okunabilir düz metin kullanılır.

## RAG kalitesini inceleme

Kullanıcılar cevapların altındaki 👍/👎 ile oy verir; ChatKit modunda ChatKit'in kendi butonları aynı isteği
gönderir. Her cevabın izi (`traces`) ayrıca otomatik kaydedilir, yani oy verilmeyen cevaplar da incelenebilir.

- **Admin paneli:** `/admin` sayfasında **Konuşmalar** tüm cevapları, **Geri bildirimler** 👎/👍 alanları
  listeler. Bir satıra tıklayınca soru, cevap, agent'ın yaptığı aramalar, gelen dosyalar ve skorları görünür;
  atıf yapılan dosyalar "✓ atıf" ile işaretlidir (bkz. [Yönetim paneli](#yönetim-paneli-admin)).
- **Toplu analiz:** aynı sekmedeki **JSONL indir** (`/api/admin/export.jsonl`) tüm izleri oylarıyla birlikte
  indirir. Dosya bir OpenShift AI workbench'inde doğrudan açılabilir:

  ```python
  import pandas as pd

  df = pd.read_json("mlops-chatbot-traces.jsonl", lines=True)
  df["top_score"] = df.searches.apply(lambda s: max((h["score"] for x in s for h in x["hits"]), default=0))
  df["n_search"] = df.searches.str.len()

  df.groupby("feedback")[["top_score", "n_search", "latency_ms"]].mean()   # 👎 cevaplarda skor düşük mü?
  df[(df.feedback == "negative") & (df.top_score < 0.6)][["question"]]      # dokümanı eksik konular
  ```

Bakılacak sinyaller:

| Sinyal | Ne anlama gelir | Ne yapılır |
|---|---|---|
| 👎 ve en yüksek skor düşük | Bu konuda doküman yok ya da bulunamıyor | Repoya doküman ekle, `include` desenlerini genişlet |
| 👎 ama skor yüksek ve atıf var | Doğru doküman geldi, cevap yine kötü | Prompt'u veya modeli gözden geçir; doküman güncel mi bak |
| İlk aramanın (ham soru) skoru, agent'ın yeniden yazdığı sorgudan çok düşük | Ham soru embedding'e uygun değil | Sorgu yeniden yazma adımı ekle, `EMBEDDING_QUERY_PREFIX`'i dene |
| Aynı dosya çok sık geliyor ama hiç atıf yok | Parça gürültü üretiyor | `exclude` ile çıkar veya `CHUNK_SIZE`'ı küçült |
| `n_search` yüksek | Agent ilk aramada bulamıyor, tekrar tekrar arıyor | `TOP_K`'yı artır, chunking'i gözden geçir |

SQL ile doğrudan da bakılabilir:

```sql
SELECT f.kind, t.question, t.cited, t.latency_ms
FROM feedback f JOIN traces t USING (item_id)
WHERE f.kind = 'negative'
ORDER BY f.created_at DESC;
```

**Gizlilik:** `traces` ve `feedback`, kullanıcı sohbeti silse de analiz için kalır ve kullanıcı adı içerir.
Şirketin KVKK / veri saklama politikası belli bir süre sonra silmeyi gerektiriyorsa bunu Postgres'te
zamanlanmış bir sorguyla yap, örneğin `DELETE FROM traces WHERE created_at < now() - interval '180 days'`.

## Ölçek ve sınırlar

- Sohbetler Postgres'te olduğu için pod'lar durumsuzdur, `replicas` artırılabilir. PVC kullanılmaz.
- "Cevap üretiliyor" göstergesi ve aynı sohbette sıraya koyma pod içidir. Route'un varsayılan sticky
  cookie'si kullanıcıyı aynı poda götürür.
- Pod tam cevap üretirken kapanırsa o cevap ve bekleyen mem0 kaydı kaybolur. Kalıcı olması gerekirse
  bir kuyruk (arq/celery) eklenebilir.
- `WORKERS=1` önerilir. Ölçeklemeyi pod sayısıyla yap: uygulama tamamen async olduğu için tek süreç
  çok sayıda eşzamanlı sohbeti kaldırır.
- Admin yetkisi girişte hesaplanır. OpenShift'te yetkisi alınan kullanıcı, oturum süresi dolana kadar
  admin kalır.
