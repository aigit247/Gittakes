import json
import os
from pathlib import Path

import boto3
import faiss
import numpy as np


def load_config(path: str = "config.json") -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_evidence(path: Path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    items.sort(key=lambda x: x["doc_id"])
    return items


def embed_titan(bedrock, model_id: str, text: str) -> np.ndarray:
    resp = bedrock.invoke_model(
        modelId=model_id,
        body=json.dumps({"inputText": text})
    )
    vec = json.loads(resp["body"].read())["embedding"]
    v = np.array(vec, dtype="float32")
    n = np.linalg.norm(v)
    return (v / n) if n > 0 else v


def main():
    cfg = load_config()

    region = cfg.get("aws_region", "us-east-1")
    evidence_path = Path(cfg["evidence_jsonl_path"])
    faiss_path = Path(cfg["faiss_index_path"])
    faiss_path.parent.mkdir(parents=True, exist_ok=True)

    embed_model_id = os.getenv("BEDROCK_EMBED_MODEL_ID")
    if not embed_model_id:
        raise RuntimeError("BEDROCK_EMBED_MODEL_ID not set")

    items = load_evidence(evidence_path)
    total = len(items)
    if total == 0:
        raise RuntimeError("No evidence found")

    print(f"[indexer] Total evidence rows: {total}")

    session = boto3.Session(region_name=region)
    bedrock = session.client("bedrock-runtime")

    vectors = []

    for i, it in enumerate(items, start=1):
        vectors.append(embed_titan(bedrock, embed_model_id, it["content"]))

        # ---- SIMPLE PROGRESS PRINT ----
        if i == 1 or i % 20 == 0 or i == total:
            print(f"[indexer] Embedded {i}/{total}")

    mat = np.vstack(vectors).astype("float32")

    print("[indexer] Building FAISS index...")
    index = faiss.IndexFlatIP(mat.shape[1])
    index.add(mat)
    faiss.write_index(index, str(faiss_path))

    print(f"[indexer] Done. Index saved to {faiss_path}")


if __name__ == "__main__":
    main()
