import argparse
import json
import os
import re
from typing import Dict, Iterable, List, Optional


def normalize_text(text: str) -> str:
    text = (text or "").lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokens(text: str) -> List[str]:
    return [t for t in normalize_text(text).split() if t]


def is_relevant_match(candidate: str, relevant_items: Iterable[str], min_overlap: float = 0.35) -> bool:
    cand_norm = normalize_text(candidate)
    if not cand_norm:
        return False

    cand_set = set(tokens(candidate))
    if not cand_set:
        return False

    for rel in relevant_items:
        rel_norm = normalize_text(rel)
        if not rel_norm:
            continue

        if rel_norm in cand_norm or cand_norm in rel_norm:
            return True

        rel_set = set(tokens(rel))
        if not rel_set:
            continue
        overlap = len(cand_set & rel_set) / max(1, len(rel_set))
        if overlap >= min_overlap:
            return True

    return False


def recall_at_k(retrieved: List[str], relevant_items: List[str], k: int) -> float:
    if not relevant_items:
        return 0.0
    top_k = retrieved[:k]
    hit = any(is_relevant_match(item, relevant_items) for item in top_k)
    return 1.0 if hit else 0.0


def reciprocal_rank(retrieved: List[str], relevant_items: List[str]) -> float:
    for idx, item in enumerate(retrieved, start=1):
        if is_relevant_match(item, relevant_items):
            return 1.0 / idx
    return 0.0


def grounding_hit_rate(response: str, retrieved: List[str], min_overlap_tokens: int = 2) -> float:
    response_set = set(tokens(response))
    if not response_set or not retrieved:
        return 0.0

    for snippet in retrieved:
        snippet_set = set(tokens(snippet))
        overlap = len(response_set & snippet_set)
        if overlap >= min_overlap_tokens:
            return 1.0
    return 0.0


def evaluate_rows(rows: List[Dict], k: int) -> Dict[str, float]:
    if not rows:
        return {
            "num_examples": 0,
            "recall_at_k": 0.0,
            "mrr": 0.0,
            "grounding_hit_rate": 0.0,
        }

    recall_scores = []
    rr_scores = []
    grounding_scores = []

    for row in rows:
        relevant_items = row.get("relevant_memory", []) or []
        retrieved_items = row.get("retrieved", []) or []
        response = row.get("response", "") or ""

        recall_scores.append(recall_at_k(retrieved_items, relevant_items, k=k))
        rr_scores.append(reciprocal_rank(retrieved_items, relevant_items))
        grounding_scores.append(grounding_hit_rate(response, retrieved_items))

    n = len(rows)
    return {
        "num_examples": n,
        "recall_at_k": sum(recall_scores) / n,
        "mrr": sum(rr_scores) / n,
        "grounding_hit_rate": sum(grounding_scores) / n,
    }


def load_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def save_json(path: str, payload: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute retrieval quality metrics: Recall@k, MRR, Grounding hit rate."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to JSONL predictions file with fields: query, relevant_memory, retrieved, response.",
    )
    parser.add_argument("--k", type=int, default=5, help="k for Recall@k.")
    parser.add_argument(
        "--out",
        default="reports/retrieval_metrics.json",
        help="Path to output metrics JSON.",
    )
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    metrics = evaluate_rows(rows, k=args.k)

    payload = {
        "input_file": args.input,
        "k": args.k,
        "metrics": metrics,
    }
    save_json(args.out, payload)

    print("Retrieval evaluation completed")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
