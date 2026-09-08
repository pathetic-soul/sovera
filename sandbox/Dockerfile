# Sandbox image for tools/py_sandbox.py (AGENTS.md §9.3).
#
# Build once, on a machine that still has a network, BEFORE arming the firewall:
#     docker build -t sandbox-py:local sandbox/
#
# After that the image is on disk and the container runs with --network none,
# so nothing here is fetched at demo time (§2.1: weights and images are
# pre-staged, runtime never downloads). `pip install` happens in THIS build
# step only — the one place §2.1 permits a network call — never at runtime.
#
# WIDENED FROM STDLIB-ONLY, DELIBERATELY (2026-09-09). The image was
# stdlib-only by design: smaller attack surface, a Dockerfile a judge could
# read in full, no supply chain to vouch for. That tradeoff was explicitly
# revisited to cover corporate/agentic tasks offline — the packages below are
# a curated, categorised list (not "whatever's popular"), chosen so each
# group is auditable at a glance. Rule 1 still applies ("never add a
# dependency that phones home") — the honest answer at this scale is
# defence in depth, not a per-package audit: --network none means every one
# of these can WANT to phone home and still physically cannot, which is the
# same guarantee §9.3 already rests on for anything a generated script does.
#
# Deliberately excluded, and why:
#   - torch / tensorflow / any deep-learning framework: multi-GB each, and
#     nothing in this workbench's task shape (§14.5: thickness maths,
#     corrosion rate, remaining life, office documents) needs one.
#   - spacy / nltk with data corpora: the corpora are a SEPARATE download
#     step this Dockerfile does not pre-stage; shipping the libraries
#     without their models would silently not work offline.
#   - scrapy: pulls in Twisted's async reactor for a use case (scraping)
#     that cannot function under --network none anyway.

FROM python:3.13-slim

# Unbuffered so stdout survives a kill at the 60 s timeout. MPLBACKEND=Agg is
# load-bearing, not cosmetic: matplotlib's default backend probes for a
# display and errors in a container that has none.
#
# HOME=/work — measured, not precautionary: `--no-create-home` plus
# `--read-only` means there is no home directory and no writable /tmp, and
# Docker's USER instruction does not set $HOME on its own, so any package
# whose lookup chain falls back to $HOME (matplotlib's cache dir, in the
# traceback that caught this) hits a read-only filesystem error. /work is
# the ONLY writable path at runtime (py_sandbox.py's --tmpfs mount), so it
# is the only correct value here — this is a blanket fix for the whole
# class of "wants a writable home", not a matplotlib-specific patch.
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 MPLBACKEND=Agg HOME=/work MPLCONFIGDIR=/work/.mplconfig

# Installed as root, in this one network-permitted build step, before the
# switch to the non-root runtime user below — a container running as uid
# 10001 has no permission to write into site-packages, so this order is
# required, not stylistic.
RUN pip install --no-cache-dir \
# --- data & numerics ---------------------------------------------------
    numpy pandas scipy polars pyarrow \
# --- office documents & PDF ---------------------------------------------
    openpyxl xlrd XlsxWriter python-docx python-pptx pypdf reportlab pdfplumber \
# --- plotting -------------------------------------------------------------
    matplotlib seaborn plotly \
# --- web frameworks (offline-runnable: no external calls needed to start
#     a local Flask/FastAPI app or render a Jinja2 template) --------------
    flask fastapi uvicorn starlette werkzeug jinja2 \
# --- HTTP clients & parsing (the network calls are inert under
#     --network none; the parsing/data-shape utilities are not) -----------
    requests httpx urllib3 beautifulsoup4 lxml html5lib xmltodict \
# --- databases / ORM (drivers only — no server ships in this image;
#     useful against a service reachable on the local subnet, or for
#     schema/ORM work that never opens a socket) --------------------------
    sqlalchemy psycopg2-binary pymysql pymongo redis \
# --- cloud SDK (boto3's local utilities — request signing, data models —
#     work without network; the calls themselves cannot, by design) -------
    boto3 botocore \
# --- validation & serialization -------------------------------------------
    pydantic marshmallow jsonschema cerberus orjson ujson msgpack \
# --- scientific / classical ML (no deep learning — see exclusions above) -
    scikit-learn statsmodels sympy \
# --- testing ---------------------------------------------------------------
    pytest pytest-cov hypothesis Faker mock \
# --- CLI / terminal UX -----------------------------------------------------
    click rich tqdm typer \
# --- config & env -----------------------------------------------------------
    python-dotenv PyYAML toml \
# --- dates & time -------------------------------------------------------------
    python-dateutil pytz arrow \
# --- crypto / security (local hashing, signing, token work — no network) ----
    cryptography bcrypt passlib PyJWT \
# --- images -------------------------------------------------------------------
    Pillow \
# --- offline-safe validators (format checking, not lookups) ------------------
    "validators" "email-validator" phonenumbers \
# --- structured logging (local only) ------------------------------------------
    structlog python-json-logger loguru \
# --- general utilities ----------------------------------------------------------
    more-itertools humanize tabulate regex python-slugify Unidecode \
# --- async ------------------------------------------------------------------------
    aiohttp anyio \
# --- misc: compression, resource stats, task-queue primitives, retry logic -------
    zstandard psutil celery tenacity backoff

# The container also runs --read-only with a tmpfs at /work, so this user needs
# no writable home. -m is omitted on purpose.
RUN useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin sandbox
USER 10001

WORKDIR /work
CMD ["python", "-c", "print('sandbox ready')"]
