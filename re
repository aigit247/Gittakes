# indexer.py
from __future__ import annotations

import argparse
import io
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

import boto3
import fitz  # PyMuPDF
import pandas as pd
from boto3 import Session
from langchain_aws import BedrockEmbeddings
from langchain_community.document_loaders import AmazonTextractPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

try:
    from docx import Document as DocxDocument
    _HAS_PYTHON_DOCX = True
except Exception:
    _HAS_PYTHON_DOCX = False


# ======================================================================================
# CONFIG
# ======================================================================================

LOCAL_DOC_ROOT = Path(os.getenv("LOCAL_DOC_ROOT", "ce")).resolve()

CATEGORY_ROOTS: Dict[str, Path] = {
    "legal": LOCAL_DOC_ROOT / "legal",
    "finance": LOCAL_DOC_ROOT / "finance",
}

# Local output where indices are created
FAISS_INDEX_DIR = Path(os.getenv("FAISS_INDEX_DIR_LOCAL", "/opt/ml/faiss_indices")).resolve()

# XLSX mapping (required)
# Columns:
#   clientName: "Client - US" OR "Client - Ultimate Parent" OR "Client - Corporate Family"
#   groupid:    folder name to store the index for that client+scope row
CLIENT_GROUP_XLSX_PATH = (os.getenv("CLIENT_GROUP_XLSX_PATH") or "").strip()
XLSX_CLIENT_COL = (os.getenv("XLSX_CLIENT_COL") or "clientName").strip()
XLSX_GROUP_COL = (os.getenv("XLSX_GROUP_COL") or "groupid").strip()

# Textract reads the S3 copy of the same contract data
S3_DOC_BUCKET = (os.getenv("S3_DOC_BUCKET") or "").strip()
S3_DOC_PREFIX = (os.getenv("S3_DOC_PREFIX") or "user/d/ce").strip()

REGION = os.getenv("AWS_REGION", "us-east-1")
EMBED_MODEL = os.getenv("BEDROCK_EMBED_MODEL", "amazon.titan-embed-text-v2:0").strip()

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

FUZZY_THRESHOLD = int(os.getenv("FUZZY_THRESHOLD", "80"))


# ======================================================================================
# AUTH: separate sessions (S3/Textract) vs (Bedrock)
# ======================================================================================

# TEMP TESTING ONLY (DO NOT COMMIT)
S3_AWS_ACCESS_KEY_ID = ""
S3_AWS_SECRET_ACCESS_KEY = ""
S3_AWS_SESSION_TOKEN = ""

BEDROCK_AWS_ACCESS_KEY_ID = ""
BEDROCK_AWS_SECRET_ACCESS_KEY = ""
BEDROCK_AWS_SESSION_TOKEN = ""


def _pick_static_creds_for(profile_env: str) -> Tuple[str, str, str]:
    if profile_env == "AWS_PROFILE_S3":
        ak = os.getenv("AWS_ACCESS_KEY_ID_S3") or S3_AWS_ACCESS_KEY_ID
        sk = os.getenv("AWS_SECRET_ACCESS_KEY_S3") or S3_AWS_SECRET_ACCESS_KEY
        st = os.getenv("AWS_SESSION_TOKEN_S3") or S3_AWS_SESSION_TOKEN
        return ak.strip(), sk.strip(), st.strip()
    if profile_env == "AWS_PROFILE_BEDROCK":
        ak = os.getenv("AWS_ACCESS_KEY_ID_BEDROCK") or BEDROCK_AWS_ACCESS_KEY_ID
        sk = os.getenv("AWS_SECRET_ACCESS_KEY_BEDROCK") or BEDROCK_AWS_SECRET_ACCESS_KEY
        st = os.getenv("AWS_SESSION_TOKEN_BEDROCK") or BEDROCK_AWS_SESSION_TOKEN
        return ak.strip(), sk.strip(), st.strip()
    return "", "", ""


def _make_session(region: str, profile_env: str, role_arn_env: str, session_name: str) -> boto3.Session:
    """
    Priority:
      1) Static creds (env/hardcoded)
      2) Profile
      3) Default chain
      4) Assume role (optional)
    """
    ak, sk, st = _pick_static_creds_for(profile_env)
    if ak and sk:
        return boto3.Session(
            aws_access_key_id=ak,
            aws_secret_access_key=sk,
            aws_session_token=(st or None),
            region_name=region,
        )

    profile = (os.getenv(profile_env) or "").strip()
    base = boto3.Session(profile_name=profile, region_name=region) if profile else boto3.Session(region_name=region)

    role_arn = (os.getenv(role_arn_env) or "").strip()
    if not role_arn:
        return base

    sts = base.client("sts", region_name=region)
    resp = sts.assume_role(RoleArn=role_arn, RoleSessionName=session_name)
    c = resp["Credentials"]
    return boto3.Session(
        aws_access_key_id=c["AccessKeyId"],
        aws_secret_access_key=c["SecretAccessKey"],
        aws_session_token=c["SessionToken"],
        region_name=region,
    )


S3_SESSION: Session = _make_session(REGION, "AWS_PROFILE_S3", "AWS_ROLE_ARN_S3", "indexer-s3")
BEDROCK_SESSION: Session = _make_session(REGION, "AWS_PROFILE_BEDROCK", "AWS_ROLE_ARN_BEDROCK", "indexer-bedrock")

S3_CLIENT = S3_SESSION.client("s3", region_name=REGION)
TEXTRACT_CLIENT = S3_SESSION.client("textract", region_name=REGION)
BEDROCK_RUNTIME = BEDROCK_SESSION.client("bedrock-runtime", region_name=REGION)


# ======================================================================================
# TEXT HELPERS
# ======================================================================================

_LEGAL_SUFFIX_TOKENS = {"inc", "ltd", "llc", "plc", "as", "sa", "ag", "gmbh", "bv", "nv", "s"}


def _clean_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def canonicalize_name(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return _clean_spaces(s)


def canonicalize_compact(s: str) -> str:
    return canonicalize_name(s).replace(" ", "")


def acronym(name: str) -> str:
    toks = canonicalize_name(name).split()
    while toks and toks[-1] in _LEGAL_SUFFIX_TOKENS:
        toks.pop()
    return "".join(t[0] for t in toks).upper() if toks else ""


def normalize_country(country: Optional[str]) -> Optional[str]:
    if not country:
        return None
    c = str(country).strip().upper()

    if c in {"UNITED STATES", "U.S.", "U.S", "US", "USA"}:
        return "US"
    if c in {"UNITED KINGDOM", "GREAT BRITAIN", "GB", "UK"}:
        return "UK"
    if c in {"INDIA", "IN", "IND"}:
        return "IN"
    return c or None


def clean_text(t: str) -> str:
    return re.sub(r"[ \t]+\n", "\n", t or "").strip()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Tuple[int, int, str]]:
    text = text or ""
    n = len(text)
    if n <= chunk_size:
        return [(0, n, text)]
    chunks: List[Tuple[int, int, str]] = []
    start = 0
    while start < n:
        end = min(n, start + chunk_size)
        chunks.append((start, end, text[start:end]))
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks


# ======================================================================================
# S3 URI HELPERS
# ======================================================================================

def _is_s3_uri(p: str) -> bool:
    return isinstance(p, str) and p.lower().startswith("s3://")


def _parse_s3_uri(uri: str) -> Tuple[str, str]:
    u = urlparse((uri or "").strip())
    bucket = (u.netloc or "").strip()
    key = (u.path or "").lstrip("/")
    if not bucket or not key:
        raise ValueError(f"Invalid S3 URI (need s3://bucket/key): {uri}")
    return bucket, key


def _read_bytes(path: str) -> bytes:
    if _is_s3_uri(path):
        b, k = _parse_s3_uri(path)
        obj = S3_CLIENT.get_object(Bucket=b, Key=k)  # Key (capital K)
        return obj["Body"].read()
    return Path(path).read_bytes()


# ======================================================================================
# XLSX MAPPING PARSER
# ======================================================================================

_SCOPE_UP = canonicalize_name("Ultimate Parent")
_SCOPE_CF = canonicalize_name("Corporate Family")


def _split_clientname_pattern(s: str) -> Tuple[str, str]:
    """
    Split "Client - Suffix" using the last separator occurrence.
    Supports: " - ", " – ", " — " (space-dash-space variants).
    If no separator, returns (s, "").
    """
    s = _clean_spaces(s)
    for sep in [" - ", " – ", " — "]:
        if sep in s:
            base, suf = s.rsplit(sep, 1)
            return base.strip(), suf.strip()
    return s.strip(), ""


def _scope_kind_and_value(scope_raw: str) -> Tuple[str, Optional[str]]:
    """
    kind: "country" | "ultimate_parent" | "corporate_family"
    value: country_code for "country", else None
    """
    sr = _clean_spaces(scope_raw)
    sc = canonicalize_name(sr)
    if sc == _SCOPE_UP:
        return "ultimate_parent", None
    if sc == _SCOPE_CF:
        return "corporate_family", None
    return "country", normalize_country(sr)


@dataclass(frozen=True)
class MappingRow:
    client_display: str
    client_canon: str
    client_compact: str
    client_acronym: str
    scope_kind: str
    scope_value: Optional[str]
    group_id: str


class ClientGroupMapper:
    """
    Loads XLSX with:
      - clientName : "Client - US" / "Client - Ultimate Parent" / "Client - Corporate Family"
      - groupid    : folder name where index should be written (CAPID only for country rows per your rule)
    """

    def __init__(self, xlsx_path: str, client_col: str, group_col: str):
        self.xlsx_path = (xlsx_path or "").strip()
        self.client_col = (client_col or "").strip()
        self.group_col = (group_col or "").strip()

        self.rows: List[MappingRow] = []
        self._by_key: Dict[Tuple[str, str, Optional[str]], str] = {}
        self._display_by_canon: Dict[str, str] = {}
        self._clients_canon: List[str] = []

    def load(self) -> None:
        if self.rows:
            return
        if not self.xlsx_path:
            raise ValueError("CLIENT_GROUP_XLSX_PATH is required (s3://... or local)")

        data = _read_bytes(self.xlsx_path)
        df = pd.read_excel(io.BytesIO(data), dtype=str, keep_default_na=False)
        df.columns = [str(c).strip() for c in df.columns]

        if self.client_col not in df.columns:
            raise ValueError(f"XLSX missing column '{self.client_col}'. Found: {list(df.columns)[:50]}")
        if self.group_col not in df.columns:
            raise ValueError(f"XLSX missing column '{self.group_col}'. Found: {list(df.columns)[:50]}")

        out: List[MappingRow] = []
        by_key: Dict[Tuple[str, str, Optional[str]], str] = {}

        for _, r in df.iterrows():
            raw_client = str(r.get(self.client_col) or "").strip()
            raw_gid = str(r.get(self.group_col) or "").strip()
            if not raw_client or not raw_gid:
                continue

            base, scope = _split_clientname_pattern(raw_client)
            kind, val = _scope_kind_and_value(scope)

            canon = canonicalize_name(base)
            comp = canonicalize_compact(base)
            acr = acronym(base)

            key = (canon, kind, val)
            if key not in by_key:
                by_key[key] = raw_gid.strip()

            out.append(
                MappingRow(
                    client_display=base.strip(),
                    client_canon=canon,
                    client_compact=comp,
                    client_acronym=acr,
                    scope_kind=kind,
                    scope_value=val,
                    group_id=raw_gid.strip(),
                )
            )

            if canon and canon not in self._display_by_canon:
                self._display_by_canon[canon] = base.strip()

        self.rows = out
        self._by_key = by_key
        self._clients_canon = sorted({r.client_canon for r in out if r.client_canon})

        print(f"[MAP] Loaded XLSX mapping rows={len(out)} from {self.xlsx_path}")

    def _resolve_client_canon(self, client_input: str, allow_fuzzy: bool, debug: bool) -> Tuple[Optional[str], str]:
        """
        Client-name matching:
          1) exact canonical
          2) exact compact (spaces removed)
          3) acronym unique
          4) substring (longest)
          5) fuzzy (optional)
        """
        ci = (client_input or "").strip()
        if not ci:
            return None, "[MAP] empty client"

        c = canonicalize_name(ci)
        cc = canonicalize_compact(ci)
        acr = ci.strip().upper()

        # 1) exact canonical
        if c in self._display_by_canon:
            return c, "[MAP] client exact canonical"

        # 2) exact compact
        for r in self.rows:
            if r.client_compact and r.client_compact == cc:
                return r.client_canon, "[MAP] client exact compact"

        # 3) acronym unique
        acr_hits = {r.client_canon for r in self.rows if r.client_acronym == acr and r.client_canon}
        if len(acr_hits) == 1:
            return next(iter(acr_hits)), "[MAP] client acronym unique"
        if len(acr_hits) > 1:
            return None, f"[MAP] client acronym ambiguous ({len(acr_hits)})"

        # 4) substring (longest)
        hits = [canon for canon in self._clients_canon if canon and (canon in c or c in canon)]
        if hits:
            hits.sort(key=len, reverse=True)
            return hits[0], f"[MAP] client substring hit='{hits[0]}'"

        # 5) fuzzy optional
        if allow_fuzzy:
            try:
                from rapidfuzz import process, fuzz  # type: ignore

                best = process.extractOne(c, self._clients_canon, scorer=fuzz.token_set_ratio)
                if best:
                    best_key, score, _ = best
                    if debug:
                        print(f"[MAP] fuzzy client best='{best_key}' score={score}")
                    if float(score) >= float(FUZZY_THRESHOLD):
                        return best_key, f"[MAP] client fuzzy hit='{best_key}' score={score}"
            except Exception as e:
                return None, f"[MAP] fuzzy client failed: {e}"

        return None, "[MAP] client no match"

    def resolve_group_id(
        self,
        client_input: str,
        scope_input: str,
        *,
        allow_fuzzy: bool = False,
        debug: bool = False,
    ) -> Tuple[Optional[str], Optional[str], str, str, Optional[str]]:
        """
        Returns:
          group_id, client_resolved_display, reason, scope_kind, scope_value
        """
        self.load()

        client_canon, why_client = self._resolve_client_canon(client_input, allow_fuzzy=allow_fuzzy, debug=debug)
        if not client_canon:
            return None, None, why_client, "unknown", None

        client_disp = self._display_by_canon.get(client_canon)

        kind, val = _scope_kind_and_value(scope_input)

        gid = self._by_key.get((client_canon, kind, val))
        if gid:
            return gid, client_disp, f"{why_client}; scope exact", kind, val

        # If scope not found but scope_input empty: try UP then CF then single-country fallback
        if not _clean_spaces(scope_input):
            up = self._by_key.get((client_canon, "ultimate_parent", None))
            if up:
                return up, client_disp, f"{why_client}; scope empty -> Ultimate Parent fallback", "ultimate_parent", None

            cf = self._by_key.get((client_canon, "corporate_family", None))
            if cf:
                return cf, client_disp, f"{why_client}; scope empty -> Corporate Family fallback", "corporate_family", None

            countries = [k for k in self._by_key.keys() if k[0] == client_canon and k[1] == "country"]
            countries = sorted(countries, key=lambda x: (x[2] or ""))
            if len(countries) == 1:
                only = countries[0]
                return self._by_key[only], client_disp, f"{why_client}; scope empty -> single country fallback", "country", only[2]

        return None, client_disp, f"{why_client}; no mapping for scope ({kind},{val})", kind, val


# ======================================================================================
# CLI
# ======================================================================================

def _parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--category", default=os.getenv("TARGET_CATEGORY", "").strip().lower())  # legal/finance optional
    p.add_argument("--client-folder", default=os.getenv("TARGET_CLIENT_FOLDER", "").strip())
    p.add_argument("--scope", default=os.getenv("TARGET_SCOPE", "").strip())  # country/UP/CF
    p.add_argument("--groupid", default=os.getenv("TARGET_GROUPID", "").strip())  # filter by mapped groupid

    p.add_argument("--check", action="store_true")  # mapping check only
    p.add_argument("--debug-mapping", action="store_true")
    p.add_argument("--allow-fuzzy", action="store_true")

    return p.parse_args()


# ======================================================================================
# Local -> S3 mapping for Textract
# ======================================================================================

def local_path_to_s3_key(path: Path) -> Optional[str]:
    """
    Map local path under LOCAL_DOC_ROOT to s3://S3_DOC_BUCKET/S3_DOC_PREFIX/<relative>
    """
    try:
        rel = path.resolve().relative_to(LOCAL_DOC_ROOT.resolve())
    except ValueError:
        print(f"[WARN] Path {path} is not under LOCAL_DOC_ROOT {LOCAL_DOC_ROOT}; cannot map to S3 key.")
        return None

    rel_key = rel.as_posix()
    if S3_DOC_PREFIX:
        return f"{S3_DOC_PREFIX.rstrip('/')}/{rel_key}"
    return rel_key


# ======================================================================================
# Optional file-wise metadata (manifest only)
# ======================================================================================

def load_metadata_for_file(doc_path: Path) -> Optional[dict]:
    candidates = [
        doc_path.with_name(doc_path.name + ".metadata.json"),
        doc_path.with_name(doc_path.name + ".metadata"),
    ]
    for mp in candidates:
        if not mp.exists():
            continue
        try:
            text = mp.read_text(encoding="utf-8", errors="ignore").strip()
            data = json.loads(text)
        except Exception as e:
            print(f"[WARN] Failed to read metadata for {doc_path} from {mp}: {e}")
            continue
        return data.get("metadataAtrributes") or data.get("metadataAttributes") or data
    return None


# ======================================================================================
# Embeddings
# ======================================================================================

def embedder():
    return BedrockEmbeddings(model_id=EMBED_MODEL, client=BEDROCK_RUNTIME)


# ======================================================================================
# Scanned PDF detection
# ======================================================================================

def is_scanned_pdf(pdf_path: Path, sample_pages: int = 3, char_threshold: int = 50) -> bool:
    try:
        with fitz.open(str(pdf_path)) as doc:
            n_pages = doc.page_count
            if n_pages == 0:
                return True
            to_sample = min(n_pages, sample_pages)
            total_chars = 0
            for i in range(to_sample):
                page = doc.load_page(i)
                txt = page.get_text("text") or ""
                total_chars += len(txt.strip())
            return total_chars < char_threshold * to_sample
    except Exception as e:
        print(f"[WARN] Failed scanned detection {pdf_path}: {e}")
        return True


# ======================================================================================
# Chunking
# ======================================================================================

def _yield_textract_chunks_with_loader(
    pdf_path: Path,
    category: str,
    group_id: str,
    client_folder: str,
    scope_input: str,
) -> Iterable[Document]:
    if not S3_DOC_BUCKET:
        print("[WARN] S3_DOC_BUCKET not set; cannot run Textract loader for scanned PDF.")
        return

    s3_key = local_path_to_s3_key(pdf_path)
    if not s3_key:
        print(f"[WARN] Could not map {pdf_path} to S3 key; skipping Textract loader.")
        return

    s3_uri = f"s3://{S3_DOC_BUCKET}/{s3_key}"
    print(f"[TEXTRACT] AmazonTextractPDFLoader: {s3_uri}")

    try:
        loader = AmazonTextractPDFLoader(file_path=s3_uri, client=TEXTRACT_CLIENT)
        tex_docs = loader.load()
    except Exception as e:
        print(f"[WARN] AmazonTextractPDFLoader failed for {pdf_path}: {e}")
        return

    if not tex_docs:
        return

    concatenated = ""
    page_ranges: List[Tuple[int, int, int]] = []
    cursor = 0

    for d in tex_docs:
        txt = clean_text(d.page_content or "")
        if not txt:
            continue
        page_num = 1
        if isinstance(d.metadata, dict):
            p = d.metadata.get("page") or d.metadata.get("Page")
            try:
                page_num = int(p)
            except Exception:
                page_num = 1

        start = cursor
        concatenated += txt + "\n"
        cursor = len(concatenated)
        page_ranges.append((start, cursor, page_num))

    if not concatenated.strip():
        return

    policy_id = pdf_path.name

    for c_start, c_end, chunk in chunk_text(concatenated):
        pages = sorted(
            {
                page_num
                for p_start, p_end, page_num in page_ranges
                if not (p_end <= c_start or p_start >= c_end)
            }
        )
        if not pages:
            continue

        meta: Dict[str, Any] = {
            "category": category,
            "group_id": group_id,
            "client_folder": client_folder,
            "scope": scope_input,
            "policy_id": policy_id,
            "page_start": pages[0],
            "page_end": pages[-1],
            "source_file": str(pdf_path.resolve()),
            "extracted_by": "amazon_textract_loader",
        }
        yield Document(page_content=chunk, metadata=meta)


def make_chunks_for_pdf(
    pdf_path: Path,
    category: str,
    group_id: str,
    client_folder: str,
    scope_input: str,
) -> Iterable[Document]:
    policy_id = pdf_path.name

    if is_scanned_pdf(pdf_path):
        yield from _yield_textract_chunks_with_loader(pdf_path, category, group_id, client_folder, scope_input)
        return

    try:
        with fitz.open(str(pdf_path)) as doc:
            concatenated = ""
            page_ranges: List[Tuple[int, int, int]] = []
            cursor = 0
            for i in range(doc.page_count):
                page = doc.load_page(i)
                txt = clean_text(page.get_text("text") or "")
                if not txt:
                    continue
                start = cursor
                concatenated += txt + "\n"
                cursor = len(concatenated)
                page_ranges.append((start, cursor, i + 1))
    except Exception as e:
        print(f"[WARN] Failed to read PDF {pdf_path}: {e}")
        return

    if not concatenated.strip():
        return

    for c_start, c_end, chunk in chunk_text(concatenated):
        pages = sorted(
            {
                page_num
                for p_start, p_end, page_num in page_ranges
                if not (p_end <= c_start or p_start >= c_end)
            }
        )
        if not pages:
            continue

        meta: Dict[str, Any] = {
            "category": category,
            "group_id": group_id,
            "client_folder": client_folder,
            "scope": scope_input,
            "policy_id": policy_id,
            "page_start": pages[0],
            "page_end": pages[-1],
            "source_file": str(pdf_path.resolve()),
            "extracted_by": "pymupdf_text",
        }
        yield Document(page_content=chunk, metadata=meta)


def make_chunks_for_docx(
    doc_path: Path,
    category: str,
    group_id: str,
    client_folder: str,
    scope_input: str,
) -> Iterable[Document]:
    if not _HAS_PYTHON_DOCX:
        print(f"[WARN] python-docx not available; skipping {doc_path}")
        return
    if doc_path.suffix.lower() != ".docx":
        return

    policy_id = doc_path.name
    try:
        doc = DocxDocument(str(doc_path))
    except Exception as e:
        print(f"[WARN] Failed to open DOCX {doc_path}: {e}")
        return

    texts: List[str] = []
    for para in doc.paragraphs:
        t = (para.text or "").strip()
        if t:
            texts.append(t)

    full_text = "\n".join(texts).strip()
    if not full_text:
        return

    for logical_page, (_, _, chunk) in enumerate(chunk_text(full_text), start=1):
        meta: Dict[str, Any] = {
            "category": category,
            "group_id": group_id,
            "client_folder": client_folder,
            "scope": scope_input,
            "policy_id": policy_id,
            "page": logical_page,
            "source_file": str(doc_path.resolve()),
            "extracted_by": "python-docx",
        }
        yield Document(page_content=chunk, metadata=meta)


def make_chunks_for_file(
    path: Path,
    category: str,
    group_id: str,
    client_folder: str,
    scope_input: str,
) -> Iterable[Document]:
    suf = path.suffix.lower()
    if suf == ".pdf":
        yield from make_chunks_for_pdf(path, category, group_id, client_folder, scope_input)
    elif suf == ".docx":
        yield from make_chunks_for_docx(path, category, group_id, client_folder, scope_input)
    elif suf == ".doc":
        print(f"[INFO] Skipping .doc: {path}")
    else:
        print(f"[INFO] Skipping unsupported file type: {path}")


# ======================================================================================
# Manifest helpers
# ======================================================================================

@dataclass
class DocInfo:
    policy_id: str
    source_file: str
    pages: int
    modified_time: int
    metadata: Optional[dict] = None


def build_manifest_entry(path: Path) -> DocInfo:
    pages = 0
    if path.suffix.lower() == ".pdf":
        try:
            with fitz.open(str(path)) as doc:
                pages = doc.page_count
        except Exception:
            pages = 0

    stat = path.stat()
    meta = load_metadata_for_file(path)

    return DocInfo(
        policy_id=path.name,
        source_file=str(path.resolve()),
        pages=pages,
        modified_time=int(stat.st_mtime),
        metadata=meta,
    )


# ======================================================================================
# Scan contract data
# ======================================================================================

def scan_category_client_files() -> Dict[str, Dict[str, List[Path]]]:
    """
    Produces keys:
      - "ClientFolder/Scope" when scope subfolders exist
      - "ClientFolder" when there are no scope subfolders
    Works for both legal and finance trees.
    """
    out: Dict[str, Dict[str, List[Path]]] = {}

    for category, root in CATEGORY_ROOTS.items():
        if not root.exists():
            print(f"[WARN] Category root does not exist: {root}")
            continue

        cat_map: Dict[str, List[Path]] = {}

        for client_dir in root.iterdir():
            if not client_dir.is_dir():
                continue

            scope_dirs = [d for d in client_dir.iterdir() if d.is_dir()]
            if scope_dirs:
                for scope_dir in scope_dirs:
                    scope = scope_dir.name.strip()
                    if not scope:
                        continue
                    key = f"{client_dir.name.strip()}/{scope}"
                    pdfs = list(scope_dir.rglob("*.pdf"))
                    docs = list(scope_dir.rglob("*.doc")) + list(scope_dir.rglob("*.docx"))
                    files = sorted(pdfs + docs, key=lambda p: p.name.lower())
                    if files:
                        cat_map.setdefault(key, []).extend(files)
            else:
                key = client_dir.name.strip()
                pdfs = list(client_dir.rglob("*.pdf"))
                docs = list(client_dir.rglob("*.doc")) + list(client_dir.rglob("*.docx"))
                files = sorted(pdfs + docs, key=lambda p: p.name.lower())
                if files:
                    cat_map.setdefault(key, []).extend(files)

        if cat_map:
            out[category] = cat_map

    return out


# ======================================================================================
# Indexing
# ======================================================================================

def index_group(
    category: str,
    client_key: str,
    files: List[Path],
    mapper: ClientGroupMapper,
    *,
    allow_fuzzy: bool,
    debug_mapping: bool,
) -> None:
    parts = (client_key or "").split("/", 1)
    client_folder = parts[0].strip()
    scope_input = parts[1].strip() if len(parts) > 1 else ""  # may be ""

    gid, client_resolved, why, scope_kind, scope_value = mapper.resolve_group_id(
        client_folder, scope_input, allow_fuzzy=allow_fuzzy, debug=debug_mapping
    )

    if debug_mapping:
        print(
            f"[MAP] key={client_key!r} -> group_id={gid!r} "
            f"client_resolved={client_resolved!r} scope_kind={scope_kind} scope_value={scope_value} ({why})"
        )

    if not gid:
        print(f"[SKIP] No mapping for '{client_key}'")
        return

    out_dir = FAISS_INDEX_DIR / category / gid
    out_dir.mkdir(parents=True, exist_ok=True)

    docs: List[Document] = []
    manifest_rows: List[DocInfo] = []

    tag = f"{category}/{gid}"
    for path in files:
        print(f"[{tag}] Processing {path}")
        manifest_rows.append(build_manifest_entry(path))

        for doc in make_chunks_for_file(path, category, gid, client_folder, scope_input):
            md = doc.metadata or {}
            md["client_resolved"] = client_resolved
            md["scope_kind"] = scope_kind
            md["scope_value"] = scope_value
            doc.metadata = md
            docs.append(doc)

    if not docs:
        print(f"[{tag}] No extractable text; skipping index.")
        return

    print(f"[{tag}] Building FAISS with {len(docs)} chunks…")
    vs = FAISS.from_documents(docs, embedder())
    vs.save_local(str(out_dir))

    manifest: Dict[str, Any] = {
        "category": category,
        "group_id": gid,
        "client_folder": client_folder,
        "client_resolved": client_resolved,
        "scope": scope_input,
        "scope_kind": scope_kind,
        "scope_value": scope_value,
        "docs": [
            {
                "policy_id": m.policy_id,
                "source_file": m.source_file,
                "pages": m.pages,
                "modified_time": m.modified_time,
                "metadata": m.metadata,
            }
            for m in manifest_rows
        ],
        "generated_time": int(time.time()),
    }

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[{tag}] Done. Index at {out_dir}")


# ======================================================================================
# Main
# ======================================================================================

def main():
    args = _parse_args()

    if not CLIENT_GROUP_XLSX_PATH:
        raise RuntimeError("CLIENT_GROUP_XLSX_PATH is required (s3://... or local path)")

    FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)

    by_cat_client = scan_category_client_files()
    if not by_cat_client:
        print("No files found under category roots.")
        return

    # Optional category filter
    if args.category in {"legal", "finance"}:
        by_cat_client = {args.category: by_cat_client.get(args.category, {})}

    mapper = ClientGroupMapper(CLIENT_GROUP_XLSX_PATH, XLSX_CLIENT_COL, XLSX_GROUP_COL)
    mapper.load()

    # Check-only mode: resolve one client/scope
    if args.check:
        ck = args.client_folder.strip()
        sc = args.scope.strip()
        if not ck:
            print("[CHECK] Provide --client-folder and optionally --scope")
            return
        gid, client_resolved, why, scope_kind, scope_value = mapper.resolve_group_id(
            ck, sc, allow_fuzzy=args.allow_fuzzy, debug=args.debug_mapping
        )
        print(f"[CHECK] client={ck!r} scope={sc!r} -> group_id={gid!r} client_resolved={client_resolved!r}")
        print(f"[CHECK] scope_kind={scope_kind} scope_value={scope_value} ({why})")
        return

    # Selection filters: client-folder (+scope) and/or groupid
    if args.client_folder:
        cf = args.client_folder.strip()
        sc = args.scope.strip()
        new: Dict[str, Dict[str, List[Path]]] = {}
        for category, cmap in by_cat_client.items():
            sub: Dict[str, List[Path]] = {}
            for key, files in cmap.items():
                if sc:
                    if key == f"{cf}/{sc}":
                        sub[key] = files
                else:
                    if key == cf or key.startswith(cf + "/"):
                        sub[key] = files
            if sub:
                new[category] = sub
        by_cat_client = new

    if args.groupid:
        target = args.groupid.strip()
        new2: Dict[str, Dict[str, List[Path]]] = {}
        for category, cmap in by_cat_client.items():
            sub2: Dict[str, List[Path]] = {}
            for key, files in cmap.items():
                parts = key.split("/", 1)
                client_folder = parts[0].strip()
                scope_input = parts[1].strip() if len(parts) > 1 else ""
                gid, _, _, _, _ = mapper.resolve_group_id(client_folder, scope_input, allow_fuzzy=args.allow_fuzzy, debug=False)
                if gid == target:
                    sub2[key] = files
            if sub2:
                new2[category] = sub2
        by_cat_client = new2

    if not by_cat_client or all(not v for v in by_cat_client.values()):
        print("[WARN] Nothing matched selection.")
        return

    for category, clients in by_cat_client.items():
        for client_key, files in clients.items():
            print(f"\n==== Indexing {category}/{client_key} ({len(files)} files) ====")
            index_group(
                category,
                client_key,
                files,
                mapper,
                allow_fuzzy=args.allow_fuzzy,
                debug_mapping=args.debug_mapping,
            )


if __name__ == "__main__":

####


# services/metadata_service.py
from __future__ import annotations

import io
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import boto3
import pandas as pd


# ======================================================================================
# TEXT HELPERS (shared with indexer logic)
# ======================================================================================

_LEGAL_SUFFIX_TOKENS = {"inc", "ltd", "llc", "plc", "as", "sa", "ag", "gmbh", "bv", "nv", "s"}


def _clean_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def canonicalize_name(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return _clean_spaces(s)


def canonicalize_compact(s: str) -> str:
    return canonicalize_name(s).replace(" ", "")


def acronym(name: str) -> str:
    toks = canonicalize_name(name).split()
    while toks and toks[-1] in _LEGAL_SUFFIX_TOKENS:
        toks.pop()
    return "".join(t[0] for t in toks).upper() if toks else ""


def normalize_country(country: Optional[str]) -> Optional[str]:
    if not country:
        return None
    c = str(country).strip().upper()

    if c in {"UNITED STATES", "U.S.", "U.S", "US", "USA"}:
        return "US"
    if c in {"UNITED KINGDOM", "GREAT BRITAIN", "GB", "UK"}:
        return "UK"
    if c in {"INDIA", "IN", "IND"}:
        return "IN"
    return c or None


# ======================================================================================
# S3 + PATH HELPERS
# ======================================================================================

def _is_s3_uri(s: str) -> bool:
    return isinstance(s, str) and s.lower().startswith("s3://")


def _parse_s3_uri(uri: str) -> Tuple[str, str]:
    u = urlparse((uri or "").strip())
    bucket = (u.netloc or "").strip()
    key = (u.path or "").lstrip("/")
    if not bucket or not key:
        raise ValueError(f"Invalid S3 URI (need s3://bucket/key): {uri}")
    return bucket, key


def _join_key(*parts: str) -> str:
    out = []
    for p in parts:
        p = str(p or "").strip("/")
        if p:
            out.append(p)
    return "/".join(out)


def _s3_uri(bucket: str, key: str) -> str:
    return f"s3://{bucket}/{key.lstrip('/')}"


# ======================================================================================
# SESSION HELPERS (separate S3 session like your earlier code)
# ======================================================================================

# TEMP TESTING ONLY (DO NOT COMMIT)
S3_AWS_ACCESS_KEY_ID = ""
S3_AWS_SECRET_ACCESS_KEY = ""
S3_AWS_SESSION_TOKEN = ""


def _pick_static_creds_for(profile_env: str) -> Tuple[str, str, str]:
    if profile_env == "AWS_PROFILE_S3":
        ak = os.getenv("AWS_ACCESS_KEY_ID_S3") or S3_AWS_ACCESS_KEY_ID
        sk = os.getenv("AWS_SECRET_ACCESS_KEY_S3") or S3_AWS_SECRET_ACCESS_KEY
        st = os.getenv("AWS_SESSION_TOKEN_S3") or S3_AWS_SESSION_TOKEN
        return ak.strip(), sk.strip(), st.strip()
    return "", "", ""


def _make_session(region: str, profile_env: str, role_arn_env: str, session_name: str) -> boto3.Session:
    """
    Priority:
      1) static creds (env/hardcoded)
      2) profile
      3) default chain
      4) assume role (optional)
    """
    ak, sk, st = _pick_static_creds_for(profile_env)
    if ak and sk:
        return boto3.Session(
            aws_access_key_id=ak,
            aws_secret_access_key=sk,
            aws_session_token=(st or None),
            region_name=region,
        )

    profile = (os.getenv(profile_env) or "").strip()
    base = boto3.Session(profile_name=profile, region_name=region) if profile else boto3.Session(region_name=region)

    role_arn = (os.getenv(role_arn_env) or "").strip()
    if not role_arn:
        return base

    sts = base.client("sts", region_name=region)
    resp = sts.assume_role(RoleArn=role_arn, RoleSessionName=session_name)
    c = resp["Credentials"]
    return boto3.Session(
        aws_access_key_id=c["AccessKeyId"],
        aws_secret_access_key=c["SecretAccessKey"],
        aws_session_token=c["SessionToken"],
        region_name=region,
    )


# ======================================================================================
# XLSX MAPPING LOGIC
# ======================================================================================

_SCOPE_UP = canonicalize_name("Ultimate Parent")
_SCOPE_CF = canonicalize_name("Corporate Family")


def _split_clientname_pattern(s: str) -> Tuple[str, str]:
    """
    Split "Client - Suffix" using last separator.
    Supports " - ", " – ", " — ".
    If no separator, returns (s, "").
    """
    s = _clean_spaces(s)
    for sep in [" - ", " – ", " — "]:
        if sep in s:
            base, suf = s.rsplit(sep, 1)
            return base.strip(), suf.strip()
    return s.strip(), ""


def _scope_kind_and_value(scope_raw: str) -> Tuple[str, Optional[str]]:
    """
    kind: "country" | "ultimate_parent" | "corporate_family"
    value: country_code for country, else None
    """
    sr = _clean_spaces(scope_raw)
    sc = canonicalize_name(sr)
    if sc == _SCOPE_UP:
        return "ultimate_parent", None
    if sc == _SCOPE_CF:
        return "corporate_family", None
    return "country", normalize_country(sr)


@dataclass(frozen=True)
class MappingRow:
    client_display: str
    client_canon: str
    client_compact: str
    client_acronym: str
    scope_kind: str
    scope_value: Optional[str]
    group_id: str


class ClientGroupMapper:
    """
    Loads XLSX with:
      - clientName : "Client - US" OR "Client - Ultimate Parent" OR "Client - Corporate Family"
      - groupid    : index folder key

    Resolution:
      - Resolve client by 4+ ways (canon, compact, acronym, substring, fuzzy optional)
      - Resolve group_id by exact (client, scope_kind, scope_value)
      - If scope empty -> UP fallback -> CF fallback -> single-country fallback
    """

    def __init__(self, xlsx_path: str, client_col: str = "clientName", group_col: str = "groupid"):
        self.xlsx_path = (xlsx_path or "").strip()
        self.client_col = (client_col or "").strip()
        self.group_col = (group_col or "").strip()

        self.rows: List[MappingRow] = []
        self._by_key: Dict[Tuple[str, str, Optional[str]], str] = {}
        self._display_by_canon: Dict[str, str] = {}
        self._clients_canon: List[str] = []
        self._clients_compact: Dict[str, str] = {}  # compact -> canon
        self._acr_to_canons: Dict[str, List[str]] = {}

    def load(self, s3_client) -> None:
        if self.rows:
            return
        if not self.xlsx_path:
            raise ValueError("CLIENT_GROUP_XLSX is required")

        if _is_s3_uri(self.xlsx_path):
            b, k = _parse_s3_uri(self.xlsx_path)
            obj = s3_client.get_object(Bucket=b, Key=k)
            data = obj["Body"].read()
        else:
            data = open(self.xlsx_path, "rb").read()

        df = pd.read_excel(io.BytesIO(data), dtype=str, keep_default_na=False)
        df.columns = [str(c).strip() for c in df.columns]
        if self.client_col not in df.columns:
            raise ValueError(f"XLSX missing '{self.client_col}'. Found: {list(df.columns)[:50]}")
        if self.group_col not in df.columns:
            raise ValueError(f"XLSX missing '{self.group_col}'. Found: {list(df.columns)[:50]}")

        out: List[MappingRow] = []
        by_key: Dict[Tuple[str, str, Optional[str]], str] = {}
        display_by_canon: Dict[str, str] = {}
        clients_canon: List[str] = []
        clients_compact: Dict[str, str] = {}
        acr_to_canons: Dict[str, List[str]] = {}

        for _, r in df.iterrows():
            raw_client = str(r.get(self.client_col) or "").strip()
            raw_gid = str(r.get(self.group_col) or "").strip()
            if not raw_client or not raw_gid:
                continue

            base, scope = _split_clientname_pattern(raw_client)
            kind, val = _scope_kind_and_value(scope)

            canon = canonicalize_name(base)
            comp = canonicalize_compact(base)
            acr = acronym(base)

            key = (canon, kind, val)
            if canon and key not in by_key:
                by_key[key] = raw_gid.strip()

            out.append(
                MappingRow(
                    client_display=base.strip(),
                    client_canon=canon,
                    client_compact=comp,
                    client_acronym=acr,
                    scope_kind=kind,
                    scope_value=val,
                    group_id=raw_gid.strip(),
                )
            )

            if canon and canon not in display_by_canon:
                display_by_canon[canon] = base.strip()
                clients_canon.append(canon)
            if comp and canon and comp not in clients_compact:
                clients_compact[comp] = canon
            if acr and canon:
                acr_to_canons.setdefault(acr, [])
                if canon not in acr_to_canons[acr]:
                    acr_to_canons[acr].append(canon)

        self.rows = out
        self._by_key = by_key
        self._display_by_canon = display_by_canon
        self._clients_canon = sorted(set(clients_canon))
        self._clients_compact = clients_compact
        self._acr_to_canons = acr_to_canons

    def _resolve_client_canon(self, client_input: str, allow_fuzzy: bool = False) -> Optional[str]:
        ci = (client_input or "").strip()
        if not ci:
            return None

        c = canonicalize_name(ci)
        cc = canonicalize_compact(ci)
        acr = ci.strip().upper()

        # 1) exact canonical
        if c in self._display_by_canon:
            return c

        # 2) exact compact
        if cc in self._clients_compact:
            return self._clients_compact[cc]

        # 3) acronym unique
        hits = self._acr_to_canons.get(acr) or []
        if len(hits) == 1:
            return hits[0]

        # 4) substring (longest)
        subs = [canon for canon in self._clients_canon if canon and (canon in c or c in canon)]
        if subs:
            subs.sort(key=len, reverse=True)
            return subs[0]

        # 5) fuzzy optional
        if allow_fuzzy:
            try:
                from rapidfuzz import process, fuzz  # type: ignore
                best = process.extractOne(c, self._clients_canon, scorer=fuzz.token_set_ratio)
                if best:
                    best_key, score, _ = best
                    if float(score) >= 80:
                        return best_key
            except Exception:
                pass

        return None

    def resolve_group_id(
        self,
        client_input: str,
        scope_input: Optional[str],
        *,
        allow_fuzzy: bool = False,
    ) -> Tuple[Optional[str], Optional[str], str, str, Optional[str]]:
        """
        Returns: group_id, client_resolved_display, reason, scope_kind, scope_value
        """
        client_canon = self._resolve_client_canon(client_input, allow_fuzzy=allow_fuzzy)
        if not client_canon:
            return None, None, "client no match", "unknown", None

        client_disp = self._display_by_canon.get(client_canon)

        kind, val = _scope_kind_and_value(scope_input or "")
        gid = self._by_key.get((client_canon, kind, val))
        if gid:
            return gid, client_disp, "exact match", kind, val

        if not _clean_spaces(scope_input or ""):
            # empty scope fallback order
            up = self._by_key.get((client_canon, "ultimate_parent", None))
            if up:
                return up, client_disp, "scope empty -> Ultimate Parent fallback", "ultimate_parent", None
            cf = self._by_key.get((client_canon, "corporate_family", None))
            if cf:
                return cf, client_disp, "scope empty -> Corporate Family fallback", "corporate_family", None

            countries = [k for k in self._by_key.keys() if k[0] == client_canon and k[1] == "country"]
            countries = sorted(countries, key=lambda x: (x[2] or ""))
            if len(countries) == 1:
                only = countries[0]
                return self._by_key[only], client_disp, "scope empty -> single country fallback", "country", only[2]

        return None, client_disp, f"no mapping for scope ({kind},{val})", kind, val


# ======================================================================================
# MANIFEST TYPES
# ======================================================================================

@dataclass
class ManifestDoc:
    policy_id: str
    source_file: Optional[str]
    metadata: Dict[str, Any]


# ======================================================================================
# METADATA SERVICE (updated)
# ======================================================================================

class MetadataService:
    """
    Updated to use XLSX mapping:
      - Input: client + "country" (which can be country code OR 'Ultimate Parent' OR 'Corporate Family')
      - Resolve group_id using mapping XLSX (clientName + groupid)
      - For country scope: capid == group_id (per your rule)
      - For UP/CF scope: capid=None, group_id returned
      - Manifest is fetched from: s3://<bucket>/<prefix>/<category>/<group_id>/manifest.json
    """

    def __init__(
        self,
        client_group_xlsx_path: str,
        faiss_index_dir: str,
        xlsx_client_col: str = "clientName",
        xlsx_group_col: str = "groupid",
        bedrock_region: Optional[str] = None,
        manifest_file_name: Optional[str] = None,
        **_ignored_kwargs,
    ):
        if not client_group_xlsx_path:
            raise ValueError("CLIENT_GROUP_XLSX_PATH is required")
        if not (faiss_index_dir or "").lower().startswith("s3://"):
            raise ValueError("FAISS_INDEX_DIR must be s3://bucket/prefix")

        self.client_group_xlsx_path = client_group_xlsx_path.strip()
        self.faiss_index_dir = faiss_index_dir.strip().rstrip("/")
        self.bedrock_region = bedrock_region or os.getenv("AWS_REGION", "us-east-1")

        self.manifest_file_name = (manifest_file_name or os.getenv("MANIFEST_FILE_NAME") or "manifest.json").strip()

        self._s3_session = _make_session(
            region=self.bedrock_region,
            profile_env="AWS_PROFILE_S3",
            role_arn_env="AWS_ROLE_ARN_S3",
            session_name="metadata-s3",
        )
        self._s3 = self._s3_session.client("s3", region_name=self.bedrock_region)

        self.mapper = ClientGroupMapper(
            xlsx_path=self.client_group_xlsx_path,
            client_col=xlsx_client_col,
            group_col=xlsx_group_col,
        )
        self._loaded = False

    def ensure_loaded(self) -> None:
        if not self._loaded:
            self.mapper.load(self._s3)
            self._loaded = True

    def _get_bytes(self, bucket: str, key: str) -> bytes:
        obj = self._s3.get_object(Bucket=bucket, Key=key)
        return obj["Body"].read()

    def _read_manifest_docs_from_s3(self, bucket: str, manifest_key: str) -> List[ManifestDoc]:
        try:
            raw = self._get_bytes(bucket, manifest_key)
            text = raw.decode("utf-8-sig", errors="replace")
            data = json.loads(text)
        except Exception:
            return []

        docs_out: List[ManifestDoc] = []
        for d in data.get("docs", []) or []:
            policy_id = str(d.get("policy_id") or "").strip()
            source_file = d.get("source_file")

            md = d.get("metadata") or {}
            # handle your spellings
            if isinstance(md, dict) and "metadataAttributes" in md and isinstance(md["metadataAttributes"], dict):
                md = md["metadataAttributes"]
            if isinstance(md, dict) and "metadataAtrributes" in md and isinstance(md["metadataAtrributes"], dict):
                md = md["metadataAtrributes"]

            if isinstance(md, dict):
                docs_out.append(ManifestDoc(policy_id=policy_id, source_file=source_file, metadata=md))

        return docs_out

    def _manifest_paths(self, category: str, group_id: str) -> Tuple[str, str, str]:
        b, base_prefix = _parse_s3_uri(self.faiss_index_dir)
        base_prefix = base_prefix.rstrip("/")
        folder_key = _join_key(base_prefix, category, group_id)
        manifest_key = _join_key(folder_key, self.manifest_file_name)
        return b, manifest_key, folder_key

    def lookup_structured(
        self,
        client: str,
        country: Optional[str] = None,
        category: Optional[str] = "finance",
        include_manifest: bool = True,
        max_docs_per_manifest: int = 200,
    ) -> Dict[str, Any]:
        self.ensure_loaded()

        client_in = (client or "").strip()
        if not client_in:
            return {"error": "client is required"}

        cat = (category or "finance").strip().lower()
        if cat not in {"finance", "legal"}:
            cat = "finance"

        # IMPORTANT: we interpret `country` as "scope input":
        #  - "US"/"IN"/... => country scope
        #  - "Ultimate Parent" => UP scope
        #  - "Corporate Family" => CF scope
        scope_input = (country or "").strip()

        gid, client_resolved, why, scope_kind, scope_value = self.mapper.resolve_group_id(
            client_in,
            scope_input,
            allow_fuzzy=True,
        )

        if not gid:
            return {
                "client_input": client_in,
                "client_resolved": client_resolved,
                "category": cat,
                "scope_input": scope_input,
                "scope_kind": scope_kind,
                "scope_value": scope_value,
                "group_id": None,
                "capid": None,
                "manifest": {"manifests_found": 0, "manifests": [], "metadata_keys": [], "docs_truncated": False},
                "error": f"Could not resolve group_id from XLSX ({why}).",
            }

        # Per your rule:
        # - If scope_kind is country AND scope_value exists -> group_id is CAPID
        # - Otherwise group_id is group id (not capid)
        capid = gid if scope_kind == "country" and scope_value else None

        manifest_payload: Dict[str, Any] = {
            "manifests_found": 0,
            "manifests": [],
            "metadata_keys": [],
            "docs_truncated": False,
        }

        if include_manifest:
            bucket, manifest_key, folder_key = self._manifest_paths(cat, gid)

            docs = self._read_manifest_docs_from_s3(bucket, manifest_key)

            truncated = False
            if len(docs) > max_docs_per_manifest:
                docs = docs[:max_docs_per_manifest]
                truncated = True
                manifest_payload["docs_truncated"] = True

            all_keys = set()
            docs_out = []
            for d in docs:
                for k in (d.metadata or {}).keys():
                    all_keys.add(str(k))
                docs_out.append({"policy_id": d.policy_id, "source_file": d.source_file, "metadata": d.metadata})

            manifest_payload["manifests"].append(
                {
                    "folder": _s3_uri(bucket, folder_key),
                    "manifest_path": _s3_uri(bucket, manifest_key),
                    "exists": bool(docs_out),
                    "docs_count_returned": len(docs_out),
                    "docs_truncated": truncated,
                    "docs": docs_out,
                }
            )
            manifest_payload["manifests_found"] = len(manifest_payload["manifests"])
            manifest_payload["metadata_keys"] = sorted(all_keys)

        return {
            "client_input": client_in,
            "client_resolved": client_resolved,
            "category": cat,
            "scope_input": scope_input,
            "scope_kind": scope_kind,
            "scope_value": scope_value,
            "group_id": gid,
            "capid": capid,
            "manifest": manifest_payload,
            "error": None,
        }

  #####


import os
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from services.metadata_service import MetadataService
from services.rag_search_service import RagSearchService

load_dotenv()

app = FastAPI(title="Structured Metadata + RAG Retrieval API", version="4.0.0")

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


# -------------------- Request Models --------------------

class MetadataLookupRequest(BaseModel):
    client: str
    # NOTE: we keep field name "country" for compatibility; it can be:
    #   "US" / "IN" / etc  OR  "Ultimate Parent"  OR  "Corporate Family"
    country: Optional[str] = None
    category: str = "finance"
    include_manifest: bool = True
    max_docs_per_manifest: int = 200


class RagSearchRequest(BaseModel):
    query: str

    # NEW: if orchestrator already called metadata, pass group_id directly
    group_id: Optional[str] = None

    # Backward compatible: if group_id is not provided, we will resolve it using metadata service
    client: Optional[str] = None
    country: Optional[str] = None  # same meaning as metadata endpoint scope
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
            country=payload.country,  # scope input
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

    # If group_id not provided, resolve it using metadata service
    if not gid:
        if not client:
            raise HTTPException(status_code=400, detail="Provide either group_id or client (to resolve group_id)")

        try:
            m = svc.lookup_structured(
                client=client,
                country=payload.country,  # scope input: US/Ultimate Parent/Corporate Family
                category=cat,
                include_manifest=False,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"group_id resolve failed: {e}")

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
            # if you enable include_manifest, we need these inputs to call metadata again
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


@app.on_event("startup")
def startup():
    try:
        svc.ensure_loaded()
        print("[INFO] XLSX mapping loaded")
    except Exception as e:
        print(f"[WARN] startup preload failed: {e}")


@app.get("/health", tags=["ops"])
def health():
    return {"status": "ok"}

###


from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import boto3
from dotenv import load_dotenv
from langchain_aws import BedrockEmbeddings

from services.metadata_service import MetadataService

try:
    from langchain_community.vectorstores import FAISS  # type: ignore
except Exception:
    from langchain.vectorstores import FAISS  # type: ignore


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    u = urlparse(uri)
    bucket = (u.netloc or "").strip()
    key = (u.path or "").lstrip("/")
    if not bucket:
        raise ValueError(f"Invalid S3 URI (missing bucket): {uri}")
    return bucket, key


def _join_key(*parts: str) -> str:
    out = []
    for p in parts:
        p = str(p or "").strip("/")
        if p:
            out.append(p)
    return "/".join(out)


# ---------- separate sessions (S3 vs Bedrock) ----------
S3_AWS_ACCESS_KEY_ID = ""
S3_AWS_SECRET_ACCESS_KEY = ""
S3_AWS_SESSION_TOKEN = ""

BEDROCK_AWS_ACCESS_KEY_ID = ""
BEDROCK_AWS_SECRET_ACCESS_KEY = ""
BEDROCK_AWS_SESSION_TOKEN = ""


def _pick_static_creds_for(profile_env: str) -> tuple[str, str, str]:
    if profile_env == "AWS_PROFILE_S3":
        ak = os.getenv("AWS_ACCESS_KEY_ID_S3") or S3_AWS_ACCESS_KEY_ID
        sk = os.getenv("AWS_SECRET_ACCESS_KEY_S3") or S3_AWS_SECRET_ACCESS_KEY
        st = os.getenv("AWS_SESSION_TOKEN_S3") or S3_AWS_SESSION_TOKEN
        return ak.strip(), sk.strip(), st.strip()
    if profile_env == "AWS_PROFILE_BEDROCK":
        ak = os.getenv("AWS_ACCESS_KEY_ID_BEDROCK") or BEDROCK_AWS_ACCESS_KEY_ID
        sk = os.getenv("AWS_SECRET_ACCESS_KEY_BEDROCK") or BEDROCK_AWS_SECRET_ACCESS_KEY
        st = os.getenv("AWS_SESSION_TOKEN_BEDROCK") or BEDROCK_AWS_SESSION_TOKEN
        return ak.strip(), sk.strip(), st.strip()
    return "", "", ""


def _make_session(region: str, profile_env: str, role_arn_env: str, session_name: str) -> boto3.Session:
    ak, sk, st = _pick_static_creds_for(profile_env)
    if ak and sk:
        return boto3.Session(
            aws_access_key_id=ak,
            aws_secret_access_key=sk,
            aws_session_token=(st or None),
            region_name=region,
        )

    profile = (os.getenv(profile_env) or "").strip()
    base = boto3.Session(profile_name=profile, region_name=region) if profile else boto3.Session(region_name=region)

    role_arn = (os.getenv(role_arn_env) or "").strip()
    if not role_arn:
        return base

    sts = base.client("sts", region_name=region)
    resp = sts.assume_role(RoleArn=role_arn, RoleSessionName=session_name)
    c = resp["Credentials"]
    return boto3.Session(
        aws_access_key_id=c["AccessKeyId"],
        aws_secret_access_key=c["SecretAccessKey"],
        aws_session_token=c["SessionToken"],
        region_name=region,
    )


@dataclass
class RagChunk:
    rank: int
    score: float
    text: str
    metadata: Dict[str, Any]


class RagSearchService:
    """
    GroupID-based FAISS layout:
      s3://<bucket>/<base_prefix>/<category>/<group_id>/(index.faiss + index.pkl + manifest.json)

    NOTE: group_id is the folder name from XLSX mapping (groupid column).
    """

    def __init__(
        self,
        metadata_svc: MetadataService,
        faiss_index_dir: str,  # s3://bucket/prefix
        bedrock_region: str = "us-east-1",
        embedding_model_id: Optional[str] = None,
        default_top_k: int = 8,
        default_category: str = "finance",
        mmr_fetch_k: Optional[int] = None,
        mmr_lambda_mult: float = 0.5,
        faiss_cache_dir: Optional[str] = None,
        **_ignored_kwargs,
    ):
        load_dotenv()

        faiss_index_dir = (faiss_index_dir or "").strip()
        if not faiss_index_dir.lower().startswith("s3://"):
            raise ValueError(f"FAISS_INDEX_DIR must be an S3 prefix like s3://bucket/path, got: {faiss_index_dir!r}")

        self.meta = metadata_svc
        self.bedrock_region = bedrock_region

        self.bucket, self.base_prefix = _parse_s3_uri(faiss_index_dir.rstrip("/"))
        self.base_prefix = self.base_prefix.rstrip("/")

        self.embedding_model_id = embedding_model_id or os.getenv(
            "BEDROCK_EMBEDDING_MODEL",
            "amazon.titan-embed-text-v1",
        )

        self.default_top_k = int(default_top_k)
        self.default_category = (default_category or "finance").strip().lower()
        if self.default_category not in {"finance", "legal"}:
            self.default_category = "finance"

        self.mmr_fetch_k = int(mmr_fetch_k) if mmr_fetch_k is not None else max(self.default_top_k * 5, 25)
        self.mmr_lambda_mult = float(mmr_lambda_mult)

        self.faiss_cache_dir = (faiss_cache_dir or os.getenv("FAISS_INDEX_DIR_LOCAL") or "/tmp/faiss_indices").strip()
        if not self.faiss_cache_dir:
            self.faiss_cache_dir = "/tmp/faiss_indices"

        # S3 client session (separate)
        self._s3_session = _make_session(
            region=self.bedrock_region,
            profile_env="AWS_PROFILE_S3",
            role_arn_env="AWS_ROLE_ARN_S3",
            session_name="rag-s3",
        )
        self._s3 = self._s3_session.client("s3", region_name=self.bedrock_region)

        # Bedrock runtime session (separate)
        self._br_session = _make_session(
            region=self.bedrock_region,
            profile_env="AWS_PROFILE_BEDROCK",
            role_arn_env="AWS_ROLE_ARN_BEDROCK",
            session_name="rag-bedrock",
        )
        self._bedrock_runtime = self._br_session.client("bedrock-runtime", region_name=self.bedrock_region)

        self._embeddings = None

    # ---------------- S3 helpers ----------------
    def _list_keys(self, prefix: str) -> List[str]:
        prefix = prefix.rstrip("/") + "/"
        paginator = self._s3.get_paginator("list_objects_v2")
        out: List[str] = []
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []) or []:
                k = obj.get("Key") or ""
                if k and not k.endswith("/"):
                    out.append(k)
        return out

    def _download(self, key: str, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        self._s3.download_file(self.bucket, key, str(dest))

    # ---------------- Embeddings ----------------
    def _emb(self):
        if self._embeddings is not None:
            return self._embeddings
        self._embeddings = BedrockEmbeddings(model_id=self.embedding_model_id, client=self._bedrock_runtime)
        return self._embeddings

    # ---------------- Find faiss files ----------------
    def _pick_faiss_files_in_folder(self, folder_prefix: str) -> tuple[str, str]:
        folder_prefix = folder_prefix.rstrip("/") + "/"
        keys = self._list_keys(folder_prefix)

        faiss_candidates = [k for k in keys if k.lower().endswith(".faiss")]
        pkl_candidates = [k for k in keys if k.lower().endswith(".pkl")]

        if not faiss_candidates or not pkl_candidates:
            raise FileNotFoundError(
                f"Could not find both .faiss and .pkl under s3://{self.bucket}/{folder_prefix}"
            )

        return faiss_candidates[0], pkl_candidates[0]

    # ---------------- FAISS ----------------
    def _load_faiss_store(self, folder: Path):
        try:
            return FAISS.load_local(str(folder), self._emb(), allow_dangerous_deserialization=True)
        except TypeError:
            return FAISS.load_local(str(folder), self._emb())

    def _json_sanitize(self, obj: Any) -> Any:
        if obj is None:
            return None
        if isinstance(obj, (str, int, float, bool)):
            return obj
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, dict):
            return {str(k): self._json_sanitize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple, set)):
            return [self._json_sanitize(v) for v in obj]
        return str(obj)

    # ---------------- Public ----------------
    def retrieve_chunks_structured(
        self,
        query: str,
        group_id: str,
        category: Optional[str] = "finance",
        top_k: Optional[int] = None,
        include_manifest: bool = False,
        client_for_manifest: Optional[str] = None,
        scope_for_manifest: Optional[str] = None,
        max_docs_per_manifest: int = 100000,
    ) -> Dict[str, Any]:
        q = (query or "").strip()
        if not q:
            raise ValueError("query is required")

        gid = (group_id or "").strip()
        if not gid:
            raise ValueError("group_id is required")

        cat = (category or self.default_category).strip().lower()
        if cat not in {"finance", "legal"}:
            cat = self.default_category

        k = int(top_k) if top_k else self.default_top_k

        folder_prefix = _join_key(self.base_prefix, cat, gid)
        faiss_key, pkl_key = self._pick_faiss_files_in_folder(folder_prefix)

        local_folder = Path(self.faiss_cache_dir) / self.base_prefix / cat / gid
        self._download(faiss_key, local_folder / Path(faiss_key).name)
        self._download(pkl_key, local_folder / Path(pkl_key).name)

        store = self._load_faiss_store(local_folder)

        mmr_docs = store.max_marginal_relevance_search(
            q, k=k, fetch_k=self.mmr_fetch_k, lambda_mult=self.mmr_lambda_mult
        )
        scored = store.similarity_search_with_score(q, k=self.mmr_fetch_k)

        def _doc_key(d) -> str:
            md = getattr(d, "metadata", {}) or {}
            md_items = sorted((str(k2), str(v2)) for k2, v2 in md.items())
            return (getattr(d, "page_content", "") or "") + "||" + "||".join(
                [f"{k2}={v2}" for k2, v2 in md_items]
            )

        score_map: Dict[str, float] = {}
        for d, s in scored:
            key2 = _doc_key(d)
            if key2 not in score_map or float(s) < score_map[key2]:
                score_map[key2] = float(s)

        merged: List[RagChunk] = []
        for d in mmr_docs:
            text = (d.page_content or "").strip()
            md = dict(d.metadata or {})
            md.setdefault("_index_folder", str(local_folder))
            sc = score_map.get(_doc_key(d), 0.0)
            merged.append(RagChunk(rank=0, score=float(sc), text=text, metadata=md))

        merged.sort(key=lambda x: x.score)

        manifest_payload: Dict[str, Any] = {}
        if include_manifest and client_for_manifest:
            m = self.meta.lookup_structured(
                client=client_for_manifest,
                country=scope_for_manifest,
                category=cat,
                include_manifest=True,
                max_docs_per_manifest=max_docs_per_manifest,
            )
            manifest_payload = m.get("manifest") or {}

        return {
            "top_chunks": [{"text": ch.text, "metadata": self._json_sanitize(ch.metadata)} for ch in merged[:k]],
            "group_id": gid,
            "category": cat,
            "manifest": manifest_payload,
        }
                                
    main()
