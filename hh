#!/usr/bin/env python3
"""
rag_queries_wide_excel_llm_per_client.py

Client-level FAISS index (many files) -> per-file RAG extraction -> wide Excel.

✅ Fixes included:
- Uses the SAME FAISS root as your SageMaker indexer by default: /opt/ml/faiss_indices
- Supports non-"finance" category names (e.g., category="gmn") via INDEX_CATEGORY
- Reads per-file hyperparams (k/max_tokens/temperature) from manifest.json
- Ensures "file-wise" context by building an in-memory FAISS sub-index per policy/file
- Robust JSON parsing (handles JSON + semicolon line)
- No cross-file fallback leakage (if file match fails, we fall back to using all chunks of that file only)
"""

from __future__ import annotations

import os
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import pandas as pd

import numpy as np
import faiss
from boto3 import Session

from langchain_aws import BedrockEmbeddings, ChatBedrock
from langchain_community.vectorstores import FAISS as LCFAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ===================== CONFIG =====================

# ✅ IMPORTANT: This should match where indexer.py saves
FAISS_INDEX_DIR = Path(os.getenv("FAISS_INDEX_DIR", "/opt/ml/faiss_indices")).resolve()

# ✅ IMPORTANT: Your manifest says category="gmn". Set this to your actual category folder name.
# Examples: "finance" OR "gmn"
INDEX_CATEGORY = os.getenv("INDEX_CATEGORY", "gmn").strip()

INDEX_ROOT = FAISS_INDEX_DIR / INDEX_CATEGORY
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./finance_query_exports")).resolve()

# Run only for a single client id (relative under INDEX_ROOT/), or "" for ALL
TARGET_CLIENT_ID = os.getenv("TARGET_CLIENT_ID", "").strip()

# 🔹 FIELD_DEFS: put ALL fields here (order matters)
FIELD_DEFS: Dict[str, Dict[str, str]] = {
    "the name of the company": {
        "query": "What is the legal name of the company (contracting party) in this contract?",
        "type": "free_text",
    },
    "contract_period.START DATE": {
        "query": "What is the contract start date or effective date of this agreement? Return only the date.",
        "type": "date",
    },
    "contract_period.END DATE": {
        "query": "What is the contract end date, expiration date, or termination date if specified?",
        "type": "date",
    },
    "renewal_terms:renewal_period": {
        "query": "What are the renewal terms and renewal period for this contract?",
        "type": "free_text",
    },
    "can fee be applied to all countries": {
        "query": (
            "Does the fee or COLA apply to all countries covered by this contract? "
            "Answer Yes/No/NA and explain briefly in free text."
        ),
        "type": "yes_no",
    },
    "extracted_clauses": {
        "query": (
            "Extract all clauses related to COLA or material price adjustments. "
            "For each clause, return an object with: "
            "clause_type (e.g. 'COLA material'), "
            "clause_text (exact text from the contract), and "
            "section_reference (section number or reference if available). "
            "The field 'extracted_clauses' must be a JSON array of such objects."
        ),
        "type": "list_of_dict",
    },
    # 🔻 Add the rest of your fields...
}

FIELD_NAMES: List[str] = list(FIELD_DEFS.keys())

# AWS / Bedrock
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

EMBED_MODEL = os.getenv("EMBED_MODEL", "amazon.titan-embed-text-v2:0")
LLM_MODEL = os.getenv("LLM_MODEL", "anthropic.claude-3-5-sonnet-20240620-v1:0")

# Defaults if manifest does not provide per-file overrides
K_DEFAULT = int(os.getenv("K_DEFAULT", "4"))
MAX_TOKENS_DEFAULT = int(os.getenv("MAX_TOKENS_DEFAULT", "2000"))
TEMP_DEFAULT = float(os.getenv("TEMP_DEFAULT", "0.0"))

MAX_SOURCE_LINES = int(os.getenv("MAX_SOURCE_LINES", "10"))

# How many chunks from the file we allow into prompt in "fallback" mode (no retrieval)
FALLBACK_MAX_FILE_CHUNKS = int(os.getenv("FALLBACK_MAX_FILE_CHUNKS", "40"))


# ===================== PROMPT TEMPLATE =====================

def build_json_schema_example(field_names_subset: List[str]) -> str:
    lines = ["{"]
    for i, field in enumerate(field_names_subset):
        field_type = FIELD_DEFS[field].get("type", "free_text")
        comma = "," if i < len(field_names_subset) - 1 else ""
        lines.append(f'  "{field}": "<{field_type}>"{comma}')
    lines.append("}")
    return "\n".join(lines)


PROMPT_MULTI = ChatPromptTemplate.from_messages([
    (
        "system",
        """
Instructions:
1. You are a precise, client-scoped contract assistant for the given category documents.
2. Use ONLY the Provided Context, which comes from a single client’s single policy/file.
3. Answer only about the given client and the given policy (file).
4. If the context is insufficient to answer something, explicitly say you don’t know and what is missing.
5. Do NOT hallucinate, infer, or invent terms that are not clearly stated in the contract.
6. Do NOT add citations, references, or bracket labels in any of the field values.
7. Follow the output JSON schema exactly: use the exact field names expected by the calling system.
8. Do NOT output anything except:
   - a single JSON object with the required fields, and
   - on the next line, a single semicolon-separated string of all field values for these fields.

Important notes:
- When the contract is silent, do NOT guess. Return "NA" or empty string as appropriate.
- For Yes/No/NA fields, the value MUST be exactly one of: "Yes", "No", or "NA".
- Dates should be in ISO "YYYY-MM-DD" where possible.
- Never change, rename, add, or remove keys from the JSON schema.
- JSON must be on the first line; semicolon string must be on the second line.
- Do NOT wrap anything in backticks or markdown.

You MUST output a single JSON object with EXACTLY {num_fields} keys.
The keys must be exactly these (order matters):

{json_schema_example}

Then output on the NEXT line:
a semicolon-separated string with ALL {num_fields} values in the SAME ORDER.
"""
    ),
    (
        "human",
        """
Client: {client_id}
Policy (file): {policy_id}

Questions / aspects to consider:
{questions_block}

Provided Context (from this client's file only):
{context}

Now produce:
1) Exactly ONE JSON object with these fields.
2) On the next line, exactly ONE semicolon-separated string with the values in the same order.
"""
    ),
])


# ===================== Helpers =====================

def _norm(s: str) -> str:
    return (s or "").strip().lower().replace("\\", "/")

def _file_keys(s: str) -> set[str]:
    s = _norm(s)
    name = s.split("/")[-1]
    stem = Path(name).stem.lower()
    return {s, name, stem}

def sanitize_filename(s: str) -> str:
    s = s.replace("/", "_")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)

def list_clients(root: Path) -> List[str]:
    if not root.exists():
        return []
    clients: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        fs = set(filenames)
        if "index.faiss" in fs and "index.pkl" in fs:
            p = Path(dirpath)
            rel = p.relative_to(root)
            clients.append(rel.as_posix())
    return sorted(set(clients))

def make_session() -> Session:
    if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
        return Session(
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION,
        )
    return Session(region_name=AWS_REGION)

def load_vs_for_client(client_id: str, session: Session) -> LCFAISS:
    emb = BedrockEmbeddings(
        model_id=EMBED_MODEL,
        client=session.client("bedrock-runtime"),
    )
    index_dir = INDEX_ROOT / client_id
    return LCFAISS.load_local(str(index_dir), emb, allow_dangerous_deserialization=True)

def load_manifest_for_client(client_id: str) -> List[Dict[str, Any]]:
    manifest_path = INDEX_ROOT / client_id / "manifest.json"
    if not manifest_path.exists():
        print(f"[WARN] No manifest.json for client '{client_id}' at {manifest_path}")
        return []
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[WARN] Failed to read manifest for '{client_id}': {e}")
        return []
    docs = data.get("docs", [])
    if not isinstance(docs, list):
        print(f"[WARN] manifest.json for '{client_id}' has no 'docs' list")
        return []
    return docs

def format_docs_for_llm(docs: List[Any]) -> str:
    chunks = []
    for d in docs:
        m = d.metadata or {}
        chunk = (
            f"(client_id: {m.get('client_id')}, "
            f"category: {m.get('category')}, "
            f"policy: {m.get('policy_id')}, "
            f"page: {m.get('page', m.get('page_start', '?'))})\n"
            f"{d.page_content}"
        )
        chunks.append(chunk)
    return "\n\n".join(chunks)

def shorten_source_path(source_file: str, client_id: str) -> str:
    if not source_file:
        return ""
    s = str(source_file)
    idx = s.find(client_id)
    if idx != -1:
        return s[idx:]
    return Path(s).name

def build_sources_summary(docs: List[Any], client_id: str, max_lines: int = MAX_SOURCE_LINES) -> str:
    seen = set()
    lines: List[str] = []
    for d in docs:
        m = d.metadata or {}
        policy_id = m.get("policy_id") or Path(m.get("source_file", "")).name or "UNKNOWN_POLICY"
        raw_source_file = m.get("source_file", "")
        short_path = shorten_source_path(raw_source_file, client_id)

        page = m.get("page")
        page_start = m.get("page_start")
        page_end = m.get("page_end")

        if page is not None:
            page_str = str(page)
        elif page_start is not None and page_end is not None:
            page_str = f"{page_start}-{page_end}" if page_start != page_end else str(page_start)
        elif page_start is not None:
            page_str = str(page_start)
        else:
            page_str = "?"

        key = (policy_id, page_str, short_path)
        if key in seen:
            continue
        seen.add(key)

        lines.append(f"Document: {policy_id} ; Page no.: {page_str}")

        if len(lines) >= max_lines:
            break
    return "\n".join(lines)

def parse_first_json_object(text: str) -> Dict[str, Any]:
    """
    Robust: parses the first JSON object from LLM output even if a 2nd line exists (semicolon string).
    """
    text = (text or "").strip()
    try:
        obj, _ = json.JSONDecoder().raw_decode(text)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        # fallback: attempt to find {...} block
        m = re.search(r"(\{.*\})", text, re.DOTALL)
        if not m:
            return {}
        try:
            return json.loads(m.group(1))
        except Exception:
            return {}

def collect_all_docs_for_policy(
    vs: LCFAISS,
    client_id: str,
    policy_id: str,
    source_file: str,
) -> List[Any]:
    """
    Last-resort fallback: return all chunks from the FAISS docstore belonging to this file.
    No similarity search; still file-wise.
    """
    targets = _file_keys(policy_id) | _file_keys(source_file)
    cid_norm = _norm(client_id)

    ids = list(vs.index_to_docstore_id.values()) if isinstance(vs.index_to_docstore_id, dict) else list(vs.index_to_docstore_id)
    out = []
    for doc_id in ids:
        d = vs.docstore.search(doc_id)
        if not d:
            continue
        m = d.metadata or {}
        meta_cid = _norm(str(m.get("client_id", "")))
        meta_pol = _file_keys(str(m.get("policy_id", "")))
        meta_src = _file_keys(str(m.get("source_file", "")))

        if meta_cid and meta_cid != cid_norm:
            continue

        if (meta_pol | meta_src) & targets:
            out.append(d)

        if len(out) >= FALLBACK_MAX_FILE_CHUNKS:
            break
    return out

def build_policy_scoped_vs(
    vs: LCFAISS,
    client_id: str,
    policy_id: str,
    source_file: str,
) -> Optional[LCFAISS]:
    """
    Build an in-memory FAISS store containing ONLY chunks for this policy/file.
    Uses faiss reconstruct => no re-embedding.
    """
    targets = _file_keys(policy_id) | _file_keys(source_file)
    cid_norm = _norm(client_id)

    if isinstance(vs.index_to_docstore_id, dict):
        pairs = list(vs.index_to_docstore_id.items())
    else:
        pairs = list(enumerate(vs.index_to_docstore_id))

    kept_vecs = []
    new_docs: Dict[str, Any] = {}
    new_map: Dict[int, str] = {}
    new_i = 0

    for i, docstore_id in pairs:
        d = vs.docstore.search(docstore_id)
        if not d:
            continue
        m = d.metadata or {}

        meta_cid = _norm(str(m.get("client_id", "")))
        if meta_cid and meta_cid != cid_norm:
            continue

        meta_pol = _file_keys(str(m.get("policy_id", "")))
        meta_src = _file_keys(str(m.get("source_file", "")))

        if (meta_pol | meta_src) & targets:
            vec = vs.index.reconstruct(int(i))
            kept_vecs.append(vec)

            new_id = str(new_i)
            new_docs[new_id] = d
            new_map[new_i] = new_id
            new_i += 1

    if not kept_vecs:
        return None

    dim = vs.index.d
    new_index = faiss.IndexFlatL2(dim)
    new_index.add(np.array(kept_vecs, dtype="float32"))

    # embedding function attr can differ by version; handle both
    emb_fn = getattr(vs, "embedding_function", None) or getattr(vs, "_embedding_function", None)

    return LCFAISS(
        embedding_function=emb_fn,
        index=new_index,
        docstore=InMemoryDocstore(new_docs),
        index_to_docstore_id=new_map,
    )


# ===================== LLM per FILE-GROUP (2 calls per file) =====================

def answer_fields_for_file_group(
    session: Session,
    client_id: str,
    policy_id: str,
    source_file: str,
    field_names_subset: List[str],
    group_label: str,
    k: int,
    max_tokens: int,
    temperature: float,
) -> Tuple[Dict[str, str], str]:
    values_by_field_subset: Dict[str, str] = {}
    if not field_names_subset:
        return values_by_field_subset, ""

    print(
        f"[INFO] client='{client_id}' policy='{policy_id}' group='{group_label}' "
        f"k={k} max_tokens={max_tokens} temperature={temperature}"
    )

    vs = load_vs_for_client(client_id, session)
    print(f"[DEBUG] loaded index ntotal={vs.index.ntotal} from {INDEX_ROOT / client_id}")

    # ✅ File-wise scoping
    policy_vs = build_policy_scoped_vs(vs, client_id=client_id, policy_id=policy_id, source_file=source_file)

    if policy_vs is None or policy_vs.index.ntotal == 0:
        print(f"[WARN] Could not build policy-scoped index for {client_id}/{policy_id}. Falling back to file-only chunks.")
        file_only_docs = collect_all_docs_for_policy(vs, client_id=client_id, policy_id=policy_id, source_file=source_file)
        if not file_only_docs:
            print(f"[WARN] No file-only chunks found for {client_id}/{policy_id}.")
            return {}, ""
        # Use file chunks directly as context (still file-wise)
        unique_docs = file_only_docs
    else:
        print(f"[INFO] policy-scoped ntotal={policy_vs.index.ntotal} for {client_id}/{policy_id}")
        retriever = policy_vs.as_retriever(search_type="similarity", search_kwargs={"k": k})

        all_docs: List[Any] = []
        for field_name in field_names_subset:
            q = FIELD_DEFS[field_name]["query"]
            docs = retriever.get_relevant_documents(q)
            all_docs.extend(docs)

        # Dedup
        seen = set()
        unique_docs = []
        for d in all_docs:
            m = d.metadata or {}
            key = (
                m.get("policy_id"),
                m.get("page"),
                m.get("page_start"),
                m.get("page_end"),
                m.get("source_file"),
            )
            if key in seen:
                continue
            seen.add(key)
            unique_docs.append(d)

        if not unique_docs:
            print(f"[WARN] Similarity retrieved 0 docs inside policy-scoped index; falling back to file-only chunks.")
            unique_docs = collect_all_docs_for_policy(vs, client_id=client_id, policy_id=policy_id, source_file=source_file)
            if not unique_docs:
                return {}, ""

    context_text = format_docs_for_llm(unique_docs)

    num_fields = len(field_names_subset)
    questions_block = "\n".join(f"- {FIELD_DEFS[f]['query']}" for f in field_names_subset)
    json_schema_example = build_json_schema_example(field_names_subset)

    llm = ChatBedrock(
        client=session.client("bedrock-runtime"),
        model_id=LLM_MODEL,
        model_kwargs={
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        },
    )

    chain = PROMPT_MULTI | llm | StrOutputParser()

    try:
        llm_output = chain.invoke({
            "client_id": client_id,
            "policy_id": policy_id,
            "questions_block": questions_block,
            "context": context_text,
            "num_fields": num_fields,
            "json_schema_example": json_schema_example,
        })
    except Exception as e:
        print(f"[WARN] LLM error for {client_id}/{policy_id} group='{group_label}': {type(e).__name__}: {e}")
        llm_output = ""

    # Save raw output
    debug_dir = OUTPUT_DIR / "llm_raw"
    debug_dir.mkdir(parents=True, exist_ok=True)
    debug_file = debug_dir / f"{sanitize_filename(client_id)}__{sanitize_filename(policy_id)}__{group_label}.txt"
    try:
        debug_file.write_text(llm_output or "", encoding="utf-8")
        print(f"[DEBUG] Saved raw LLM output to {debug_file}")
    except Exception as e:
        print(f"[WARN] Could not save raw LLM output: {e}")

    parsed = parse_first_json_object(llm_output or "")
    if not isinstance(parsed, dict):
        parsed = {}

    # Convert values to strings for Excel
    for field in field_names_subset:
        value = parsed.get(field, "")
        if value is None:
            values_by_field_subset[field] = ""
        elif isinstance(value, (dict, list)):
            values_by_field_subset[field] = json.dumps(value, ensure_ascii=False)
        else:
            values_by_field_subset[field] = str(value)

    sources_summary = build_sources_summary(unique_docs, client_id=client_id, max_lines=MAX_SOURCE_LINES)
    return values_by_field_subset, sources_summary


# ===================== Main =====================

def main():
    if not FIELD_NAMES:
        raise SystemExit("FIELD_DEFS is empty. Please define your JSON fields in FIELD_DEFS.")

    # 2 LLM calls per file
    n_fields = len(FIELD_NAMES)
    mid = n_fields // 2
    GROUP_1_FIELDS = FIELD_NAMES[:mid]
    GROUP_2_FIELDS = FIELD_NAMES[mid:]

    print(f"[INFO] CATEGORY='{INDEX_CATEGORY}' INDEX_ROOT='{INDEX_ROOT}'")
    print(f"[INFO] total fields={n_fields} group1={len(GROUP_1_FIELDS)} group2={len(GROUP_2_FIELDS)}")

    all_clients = list_clients(INDEX_ROOT)
    if not all_clients:
        raise SystemExit(f"No clients found under {INDEX_ROOT} (missing index.faiss/index.pkl?)")

    if TARGET_CLIENT_ID:
        clients = [c for c in all_clients if c == TARGET_CLIENT_ID]
        if not clients:
            raise SystemExit(f"TARGET_CLIENT_ID='{TARGET_CLIENT_ID}' not found under {INDEX_ROOT}")
        print(f"[INFO] Running ONLY for client: {TARGET_CLIENT_ID}")
    else:
        clients = all_clients
        print(f"[INFO] Running for ALL clients: {len(clients)}")

    session = make_session()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for client_id in clients:
        print(f"\n=== Client: {client_id} ===")

        safe_name = sanitize_filename(client_id)
        out_path = OUTPUT_DIR / f"queries_{safe_name}.xlsx"

        # Optional skip
        if out_path.exists():
            print(f"  -> Skipping {client_id}: {out_path} already exists.")
            continue

        manifest_docs = load_manifest_for_client(client_id)
        if not manifest_docs:
            print(f"[WARN] No manifest docs for client '{client_id}', skipping.")
            continue

        rows: List[Dict[str, Any]] = []

        for doc_info in manifest_docs:
            policy_id = doc_info.get("policy_id") or Path(doc_info.get("source_file", "")).name
            source_file = doc_info.get("source_file", "")

            # per-file hyperparams from manifest
            doc_k = int(doc_info.get("k", K_DEFAULT))
            doc_max_tokens = int(doc_info.get("max_tokens", MAX_TOKENS_DEFAULT))
            doc_temperature = float(doc_info.get("temperature", TEMP_DEFAULT))

            print(f"  -> File: {policy_id}  [k={doc_k}, max_tokens={doc_max_tokens}, temp={doc_temperature}]")

            vals_1, sources_1 = answer_fields_for_file_group(
                session=session,
                client_id=client_id,
                policy_id=policy_id,
                source_file=source_file,
                field_names_subset=GROUP_1_FIELDS,
                group_label="group1",
                k=doc_k,
                max_tokens=doc_max_tokens,
                temperature=doc_temperature,
            )

            vals_2, sources_2 = answer_fields_for_file_group(
                session=session,
                client_id=client_id,
                policy_id=policy_id,
                source_file=source_file,
                field_names_subset=GROUP_2_FIELDS,
                group_label="group2",
                k=doc_k,
                max_tokens=doc_max_tokens,
                temperature=doc_temperature,
            )

            values_all: Dict[str, str] = {}
            for field in FIELD_NAMES:
                if field in vals_1:
                    values_all[field] = vals_1[field]
                elif field in vals_2:
                    values_all[field] = vals_2[field]
                else:
                    values_all[field] = ""

            flat_string = ";".join(values_all.get(f, "") for f in FIELD_NAMES)

            sources_parts = []
            if sources_1:
                sources_parts.append("[Group 1]\n" + sources_1)
            if sources_2:
                sources_parts.append("[Group 2]\n" + sources_2)
            sources_summary = "\n\n".join(sources_parts)

            row: Dict[str, Any] = {
                "Client": client_id,
                "Policy": policy_id,
                "Source File": source_file,
                "FlatString": flat_string,
                "Sources": sources_summary,
            }
            for field in FIELD_NAMES:
                row[field] = values_all.get(field, "")

            rows.append(row)

        if not rows:
            print(f"[WARN] No rows generated for client '{client_id}'.")
            continue

        df = pd.DataFrame(rows)
        df.to_excel(out_path, index=False)
        print(f"  -> Saved: {out_path}")

    print("\nAll exports completed.")


if __name__ == "__main__":
    main()
