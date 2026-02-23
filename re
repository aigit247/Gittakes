import os
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from services.metadata_service import MetadataService
from services.rag_search_service import RagSearchService

load_dotenv()

AWS_REGION = os.getenv("AWS_REGION", "us-east-1").strip()

FAISS_INDEX_DIR = (os.getenv("FAISS_INDEX_DIR") or "").strip()
CLIENT_GROUP_XLSX_PATH = (os.getenv("CLIENT_GROUP_XLSX_PATH") or "").strip()

XLSX_CLIENT_COL = (os.getenv("XLSX_CLIENT_COL") or "clientName").strip()
XLSX_GROUP_COL = (os.getenv("XLSX_GROUP_COL") or "groupid").strip()

if not FAISS_INDEX_DIR.lower().startswith("s3://"):
    raise RuntimeError(f"FAISS_INDEX_DIR must be s3://bucket/prefix, got: {FAISS_INDEX_DIR!r}")
if not CLIENT_GROUP_XLSX_PATH:
    raise RuntimeError("CLIENT_GROUP_XLSX_PATH is required (s3://... or local path)")

svc = MetadataService(
    client_group_xlsx_path=CLIENT_GROUP_XLSX_PATH,
    faiss_index_dir=FAISS_INDEX_DIR,
    xlsx_client_col=XLSX_CLIENT_COL,
    xlsx_group_col=XLSX_GROUP_COL,
    bedrock_region=AWS_REGION,
)

rag = RagSearchService(
    metadata_svc=svc,
    faiss_index_dir=FAISS_INDEX_DIR,
    bedrock_region=AWS_REGION,
    embedding_model_id=os.getenv("BEDROCK_EMBEDDING_MODEL"),
    default_top_k=int(os.getenv("RAG_TOP_K", "8")),
    default_category=os.getenv("RAG_DEFAULT_CATEGORY", "finance"),
    faiss_cache_dir=os.getenv("FAISS_INDEX_DIR_LOCAL", "/tmp/faiss_indices"),
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    try:
        svc.ensure_loaded()
        print("[INFO] XLSX mapping loaded")
    except Exception as e:
        # Don't crash the app if you prefer, but usually better to crash early:
        print(f"[WARN] startup preload failed: {e}")
        # raise

    yield

    # shutdown (optional cleanup)
    # boto3 clients support .close() in botocore; safe to call if present
    for c in [getattr(svc, "_s3", None), getattr(rag, "_s3", None), getattr(rag, "_bedrock_runtime", None)]:
        try:
            if c and hasattr(c, "close"):
                c.close()
        except Exception:
            pass


app = FastAPI(
    title="Structured Metadata + RAG Retrieval API",
    version="4.0.0",
    lifespan=lifespan,
)


# -------------------- Request Models --------------------

class MetadataLookupRequest(BaseModel):
    client: str
    # NOTE: kept as "country" for backward compatibility; acts as "scope":
    #   "US"/"IN"/etc OR "Ultimate Parent" OR "Corporate Family"
    country: Optional[str] = None
    category: str = "finance"
    include_manifest: bool = True
    max_docs_per_manifest: int = 200


class RagSearchRequest(BaseModel):
    query: str

    # preferred: pass group_id returned by metadata lookup
    group_id: Optional[str] = None

    # fallback: resolve group_id via metadata service if group_id not provided
    client: Optional[str] = None
    country: Optional[str] = None  # scope input
    category: str = "finance"

    top_k: Optional[int] = None
    include_manifest: bool = False
    max_docs_per_manifest: int = 100000


# -------------------- Endpoints --------------------

@app.post("/api/metadata/lookup", response_model=Dict[str, Any], tags=["metadata"])
def metadata_lookup(payload: MetadataLookupRequest):
    client = (payload.client or "").strip()
    if not client:
        raise HTTPException(status_code=400, detail="client is required")

    try:
        res = svc.lookup_structured(
            client=client,
            country=payload.country,
            category=payload.category,
            include_manifest=payload.include_manifest,
            max_docs_per_manifest=payload.max_docs_per_manifest,
        )
        if res.get("error"):
            raise HTTPException(status_code=404, detail=res["error"])
        return res
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"metadata lookup failed: {e}")


@app.post("/api/rag/search", response_model=Dict[str, Any], tags=["rag"])
def rag_search(payload: RagSearchRequest):
    query = (payload.query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="query is required")

    cat = (payload.category or "finance").strip().lower()
    if cat not in {"finance", "legal"}:
        cat = "finance"

    gid = (payload.group_id or "").strip()
    client = (payload.client or "").strip()

    # Resolve group_id if not provided
    if not gid:
        if not client:
            raise HTTPException(status_code=400, detail="Provide either group_id or client (to resolve group_id)")
        m = svc.lookup_structured(
            client=client,
            country=payload.country,
            category=cat,
            include_manifest=False,
        )
        if m.get("error"):
            raise HTTPException(status_code=404, detail=m["error"])
        gid = (m.get("group_id") or "").strip()
        if not gid:
            raise HTTPException(status_code=404, detail="group_id not found for provided client/scope")

    try:
        return rag.retrieve_chunks_structured(
            query=query,
            group_id=gid,
            category=cat,
            top_k=payload.top_k,
            include_manifest=payload.include_manifest,
            client_for_manifest=client if payload.include_manifest else None,
            scope_for_manifest=payload.country if payload.include_manifest else None,
            max_docs_per_manifest=payload.max_docs_per_manifest,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"rag retrieval failed: {e}")


@app.get("/health", tags=["ops"])
def health():
    return {"status": "ok"}
