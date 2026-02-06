# Hybrid RAG: BM25 + Embeddings (FAISS) + Table expansion to full TABLE_BLOCK
# Deterministic, per-client indexes.
#
# Prereqs:
# 1) index/by_client/<company>/<client>/faiss.index
# 2) index/by_client/<company>/<client>/evidence.jsonl  (doc_id aligned with FAISS ids)
# 3) env vars:
#    BEDROCK_EMBED_MODEL_ID
#    BEDROCK_LLM_MODEL_ID
#
# Edit COMPANY/CLIENT/QUESTION below and run.

import json, os, re, math
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, List, Tuple, Optional

import boto3
import faiss
import numpy as np

# ----------------- EDIT THESE -----------------
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
FAISS_ROOT = Path("index/by_client")

COMPANY  = "Koch"
CLIENT   = "client1"
QUESTION = "What is the TOUR CODE?"

# Retrieval knobs (tune these first)
TOP_K_VEC   = 80     # embedding retrieval depth
TOP_K_BM25  = 200    # keyword retrieval depth
TOP_K_FINAL = 40     # final hybrid candidates considered for context

W_VEC  = 0.65        # weight for embedding score
W_BM25 = 0.35        # weight for bm25 score

# Context knobs
NON_TABLE_DOCS_IN_CONTEXT = 10
MAX_TABLES_IN_CONTEXT     = 8

# Table expansion knobs (full table -> blocks)
MAX_ROWS_PER_TABLE_BLOCK  = 800      # keep whole table unless huge; raise if needed
MAX_CHARS_PER_TABLE_BLOCK = 25000    # split table block if exceeds this
MAX_CONTEXT_CHARS         = 120000   # hard cap for prompt

# LLM knobs (deterministic)
TEMPERATURE = 0
MAX_TOKENS  = 900
# ---------------------------------------------

EMBED_MODEL = os.environ.get("BEDROCK_EMBED_MODEL_ID")
LLM_MODEL   = os.environ.get("BEDROCK_LLM_MODEL_ID")
if not EMBED_MODEL or not LLM_MODEL:
    raise RuntimeError("Set env vars: BEDROCK_EMBED_MODEL_ID and BEDROCK_LLM_MODEL_ID")

bedrock = boto3.Session(region_name=AWS_REGION).client("bedrock-runtime")


# ----------------- Bedrock helpers -----------------
def embed_query(q: str) -> np.ndarray:
    resp = bedrock.invoke_model(
        modelId=EMBED_MODEL,
        body=json.dumps({"inputText": q}),
        accept="application/json",
        contentType="application/json",
    )
    v = np.array(json.loads(resp["body"].read())["embedding"], dtype="float32")
    n = float(np.linalg.norm(v))
    if n > 0:
        v = v / n
    return v.reshape(1, -1)

def _extract_text_any(raw: dict) -> str:
    if isinstance(raw, dict):
        if "content" in raw and isinstance(raw["content"], list) and raw["content"]:
            return "".join([c.get("text","") for c in raw["content"] if c.get("type")=="text"]).strip()
        out = raw.get("output", {})
        msg = out.get("message", {})
        cont = msg.get("content", [])
        if isinstance(cont, list) and cont and isinstance(cont[0], dict) and "text" in cont[0]:
            return cont[0]["text"]
        if "completion" in raw and isinstance(raw["completion"], str):
            return raw["completion"]
    return ""

def call_llm(prompt: str) -> str:
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "messages": [{"role":"user","content":[{"type":"text","text":prompt}]}],
    }
    resp = bedrock.invoke_model(
        modelId=LLM_MODEL,
        body=json.dumps(body),
        accept="application/json",
        contentType="application/json",
    )
    raw = json.loads(resp["body"].read())
    return _extract_text_any(raw).strip()


# ----------------- Store loading -----------------
def _loc_prefix(loc: Any) -> str:
    s = str(loc or "")
    return s.split("ROW=")[0].strip()

def load_pair(company: str, client: str, root: Path = FAISS_ROOT):
    pair_dir = root / company / client
    ev_path = pair_dir / "evidence.jsonl"
    ix_path = pair_dir / "faiss.index"
    if not ev_path.exists() or not ix_path.exists():
        raise FileNotFoundError(f"Missing store for {company}/{client}: {pair_dir}")

    evidence = []
    with open(ev_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                evidence.append(json.loads(line))

    index = faiss.read_index(str(ix_path))
    if index.ntotal != len(evidence):
        raise RuntimeError(f"FAISS ntotal {index.ntotal} != evidence rows {len(evidence)} for {company}/{client}")

    # Precompute table map: table_key -> sorted list of row doc_ids
    table_map: Dict[Tuple[str,str,str], List[int]] = defaultdict(list)
    for doc_id, it in enumerate(evidence):
        if it.get("type") == "TABLE_ROW":
            key = (str(it.get("file","") or ""), str(it.get("table_id") or ""), _loc_prefix(it.get("loc")))
            table_map[key].append(doc_id)

    for k, ids in table_map.items():
        ids.sort(key=lambda d: (
            evidence[d].get("row_index") if evidence[d].get("row_index") is not None else 10**9,
            d
        ))
        table_map[k] = ids

    return evidence, index, table_map


# ----------------- BM25 (no external deps) -----------------
_TOKEN_RE = re.compile(r"[a-z0-9]+", re.I)

def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall((text or "").lower())

def build_bm25(evidence: List[Dict[str,Any]], k1: float = 1.5, b: float = 0.75):
    """
    Build an in-memory BM25 index.
    Returns a dict with inverted index postings and corpus stats.
    """
    inv: Dict[str, List[Tuple[int,int]]] = defaultdict(list)  # token -> list[(doc_id, tf)]
    doc_len = [0] * len(evidence)

    for doc_id, it in enumerate(evidence):
        txt = it.get("content", "") or ""
        toks = tokenize(txt)
        if not toks:
            continue
        doc_len[doc_id] = len(toks)
        tf_map: Dict[str,int] = defaultdict(int)
        for t in toks:
            tf_map[t] += 1
        for t, tf in tf_map.items():
            inv[t].append((doc_id, tf))

    avgdl = (sum(doc_len) / max(1, sum(1 for x in doc_len if x > 0))) if doc_len else 0.0
    N = len(evidence)

    # store constants
    return {
        "inv": inv,
        "doc_len": doc_len,
        "avgdl": avgdl,
        "N": N,
        "k1": float(k1),
        "b": float(b),
    }

def bm25_search(bm25, query: str, top_k: int = 100) -> List[Tuple[int, float]]:
    inv = bm25["inv"]
    doc_len = bm25["doc_len"]
    avgdl = bm25["avgdl"]
    N = bm25["N"]
    k1 = bm25["k1"]
    b = bm25["b"]

    q_toks = tokenize(query)
    if not q_toks:
        return []

    # query term frequency (optional; we just iterate unique terms)
    q_terms = list(dict.fromkeys(q_toks))

    scores: Dict[int, float] = defaultdict(float)

    for t in q_terms:
        postings = inv.get(t)
        if not postings:
            continue
        df = len(postings)
        # idf (BM25+ style)
        idf = math.log(1.0 + (N - df + 0.5) / (df + 0.5))

        for doc_id, tf in postings:
            dl = doc_len[doc_id] if doc_len[doc_id] > 0 else 0
            denom = tf + k1 * (1.0 - b + b * (dl / max(avgdl, 1e-9)))
            score = idf * (tf * (k1 + 1.0)) / max(denom, 1e-9)
            scores[doc_id] += score

    if not scores:
        return []

    # Deterministic sort: score desc, doc_id asc
    ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    return ranked[:top_k]


# ----------------- TABLE_ROW -> TABLE_BLOCK -----------------
def _parse_headers_values(row_text: str) -> Tuple[str, str]:
    headers = ""
    values = ""
    for ln in (row_text or "").splitlines():
        s = ln.strip()
        if s.upper().startswith("HEADERS:"):
            headers = s.split(":", 1)[1].strip()
        elif s.upper().startswith("VALUES:"):
            values = s.split(":", 1)[1].strip()
    return headers, values

def build_table_blocks_for_tables(
    evidence: List[Dict[str,Any]],
    triggered_table_keys: List[Tuple[str,str,str]],
    table_map: Dict[Tuple[str,str,str], List[int]],
    max_tables: int,
    max_rows_per_table: int,
    max_chars_per_block: int
) -> List[Dict[str,Any]]:
    blocks = []
    table_keys = triggered_table_keys[:max_tables]

    for (file, table_id, loc_pref) in table_keys:
        row_ids = table_map.get((file, table_id, loc_pref), [])
        if not row_ids:
            continue

        # Cap rows if needed (deterministic: take first N rows)
        if max_rows_per_table and len(row_ids) > max_rows_per_table:
            row_ids = row_ids[:max_rows_per_table]

        # Find headers
        headers = ""
        for d in row_ids:
            h, _ = _parse_headers_values(evidence[d].get("content",""))
            if h:
                headers = h
                break
        headers = headers or "(unknown headers)"

        def make_block(chunk_ids: List[int], chunk_idx: int) -> Dict[str,Any]:
            r0 = evidence[chunk_ids[0]].get("row_index")
            r1 = evidence[chunk_ids[-1]].get("row_index")

            vals = []
            for d in chunk_ids:
                _, v = _parse_headers_values(evidence[d].get("content",""))
                vals.append(v if v else (evidence[d].get("content","") or "").strip())

            content = (
                f"TYPE=TABLE_BLOCK\n"
                f"FILE={file}\n"
                f"TABLE_ID={table_id}\n"
                f"LOC={loc_pref}\n"
                f"ROW_RANGE={r0}-{r1}\n"
                f"HEADERS: {headers}\n"
                f"ROWS:\n- " + "\n- ".join(vals)
            )
            return {
                "type": "TABLE_BLOCK",
                "file": file,
                "table_id": table_id,
                "loc": loc_pref if chunk_idx == 1 else f"{loc_pref} CHUNK={chunk_idx}",
                "doc_ids": chunk_ids,
                "content": content,
            }

        blk = make_block(row_ids, 1)
        if len(blk["content"]) <= max_chars_per_block:
            blocks.append(blk)
        else:
            # deterministic split
            chunk_size = 250
            chunk_idx = 1
            for i in range(0, len(row_ids), chunk_size):
                blocks.append(make_block(row_ids[i:i+chunk_size], chunk_idx))
                chunk_idx += 1

    return blocks


# ----------------- Hybrid retrieval + answer -----------------
def _minmax_norm(scores: Dict[int, float]) -> Dict[int, float]:
    if not scores:
        return {}
    vals = list(scores.values())
    lo, hi = min(vals), max(vals)
    if abs(hi - lo) < 1e-12:
        return {k: 1.0 for k in scores.keys()}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}

def rag_answer_hybrid(company: str, client: str, question: str):
    evidence, faiss_index, table_map = load_pair(company, client)
    bm25 = build_bm25(evidence)

    # --- Embedding retrieval ---
    qvec = embed_query(question)
    vec_scores_arr, vec_ids_arr = faiss_index.search(qvec, TOP_K_VEC)

    vec_scores: Dict[int, float] = {}
    for s, doc_id in zip(vec_scores_arr[0].tolist(), vec_ids_arr[0].tolist()):
        if doc_id < 0:
            continue
        # keep best
        if doc_id not in vec_scores or float(s) > vec_scores[doc_id]:
            vec_scores[doc_id] = float(s)

    # --- BM25 retrieval ---
    bm25_ranked = bm25_search(bm25, question, top_k=TOP_K_BM25)
    bm25_scores: Dict[int, float] = {doc_id: float(s) for doc_id, s in bm25_ranked}

    # --- Hybrid merge ---
    vec_norm  = _minmax_norm(vec_scores)
    bm25_norm = _minmax_norm(bm25_scores)

    all_ids = set(vec_norm.keys()) | set(bm25_norm.keys())
    hybrid: List[Tuple[int, float, float, float]] = []  # doc_id, combined, vec_n, bm25_n

    for doc_id in all_ids:
        v = vec_norm.get(doc_id, 0.0)
        b = bm25_norm.get(doc_id, 0.0)
        combined = W_VEC * v + W_BM25 * b
        hybrid.append((doc_id, combined, v, b))

    # Deterministic sort: combined desc, then vec_n desc, bm25_n desc, doc_id asc
    hybrid.sort(key=lambda x: (-x[1], -x[2], -x[3], x[0]))
    hybrid = hybrid[:TOP_K_FINAL]

    # --- Detect triggered tables from top hybrid candidates ---
    triggered_tables: Dict[Tuple[str,str,str], int] = {}  # table_key -> best doc_id (for ordering)

    non_table_hits = []
    for doc_id, combined, v, b in hybrid:
        it = evidence[doc_id]
        if it.get("type") == "TABLE_ROW":
            tkey = (str(it.get("file","") or ""), str(it.get("table_id") or ""), _loc_prefix(it.get("loc")))
            if tkey not in triggered_tables:
                triggered_tables[tkey] = doc_id
        else:
            non_table_hits.append((doc_id, combined, v, b, it))

    # Order tables deterministically by earliest triggering doc_id
    ordered_table_keys = sorted(triggered_tables.keys(), key=lambda k: (triggered_tables[k], k[0], k[1], k[2]))

    # Expand tables to full TABLE_BLOCKS
    table_blocks = build_table_blocks_for_tables(
        evidence=evidence,
        triggered_table_keys=ordered_table_keys,
        table_map=table_map,
        max_tables=MAX_TABLES_IN_CONTEXT,
        max_rows_per_table=MAX_ROWS_PER_TABLE_BLOCK,
        max_chars_per_block=MAX_CHARS_PER_TABLE_BLOCK
    )

    # Keep best non-table hits for context
    non_table_hits.sort(key=lambda x: (-x[1], x[0]))
    non_table_hits = non_table_hits[:NON_TABLE_DOCS_IN_CONTEXT]

    # --- Build context (hard cap) ---
    context_parts = []
    cur_chars = 0

    def add(txt: str):
        nonlocal cur_chars
        if cur_chars + len(txt) > MAX_CONTEXT_CHARS:
            return False
        context_parts.append(txt)
        cur_chars += len(txt)
        return True

    # Non-table first
    for doc_id, combined, v, b, it in non_table_hits:
        txt = (
            f"[DOC {doc_id}] hybrid={combined:.4f} vecN={v:.4f} bm25N={b:.4f} "
            f"type={it.get('type','')} file={it.get('file','')} loc={it.get('loc','')}\n"
            f"{it.get('content','')}\n"
        )
        if not add(txt):
            break

    # Tables next (as blocks)
    for tb in table_blocks:
        txt = (
            f"[TABLE_BLOCK] file={tb['file']} table_id={tb['table_id']} loc={tb['loc']} "
            f"rows_doc_ids={tb['doc_ids'][:5]}{'...' if len(tb['doc_ids'])>5 else ''}\n"
            f"{tb['content']}\n"
        )
        if not add(txt):
            break

    context = "\n---\n".join(context_parts)

    prompt = f"""Answer the question using ONLY the context below.
If the answer is not explicitly present, reply exactly: Not found

Question: {question}

Context:
{context}
"""

    answer = call_llm(prompt)

    return {
        "company": company,
        "client": client,
        "question": question,
        "answer": answer,
        "top_hybrid_debug": [
            {
                "doc_id": doc_id,
                "hybrid": combined,
                "vecN": v,
                "bm25N": b,
                "type": evidence[doc_id].get("type",""),
                "file": evidence[doc_id].get("file",""),
                "loc": evidence[doc_id].get("loc",""),
            }
            for (doc_id, combined, v, b) in hybrid[:20]
        ],
        "tables_used": [
            {"file": t["file"], "table_id": t["table_id"], "loc": t["loc"], "num_rows_in_block": len(t["doc_ids"])}
            for t in table_blocks
        ],
        "context_preview": context[:2000],
    }


# ----------------- RUN -----------------
res = rag_answer_hybrid(COMPANY, CLIENT, QUESTION)

print("\nANSWER:\n", res["answer"])
print("\nTOP HYBRID HITS (first 10):")
for d in res["top_hybrid_debug"][:10]:
    print(" -", d)

print("\nTABLES USED:")
for t in res["tables_used"]:
    print(" -", t)

print("\nCONTEXT PREVIEW (first 2000 chars):\n", res["context_preview"])
