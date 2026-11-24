#!/usr/bin/env python3
"""Offline BGE similarity evaluator for saved LTM benchmark outputs.

This script reads the frozen LTM dataset and generated_memories.json produced by
long_term_memory_benchmark.py. It never imports or calls Groq, the production
LLM provider, or the production LongTermMemory implementation.

Long paragraphs are split into overlapping, tokenizer-aware chunks. Every
chunk is checked against the loaded SentenceTransformer model's actual
max_seq_length before encoding, preventing silent model-side truncation.

For each expected/generated paragraph pair, chunk coverage is symmetric:
  0.5 * (mean(expected-chunk best matches) + mean(generated-chunk best matches))
The resulting paragraph-pair matrix is then evaluated with the same symmetric
paragraph-level best-match aggregation used by the original benchmark.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc


def split_memory_paragraphs(text: str) -> list[str]:
    """Match the original benchmark's paragraph splitting behavior."""
    if not isinstance(text, str) or not text.strip():
        return []
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text.strip()) if b.strip()]
    bullet_re = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
    paragraphs: list[str] = []
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if lines and all(bullet_re.match(line) for line in lines):
            paragraphs.extend(bullet_re.sub("", line).strip() for line in lines)
        else:
            paragraphs.append(" ".join(lines))
    return [p for p in paragraphs if p]


def _extract_memory(value: Any, where: str) -> str:
    if isinstance(value, dict):
        memory = value.get("memory")
    else:
        memory = value
    if not isinstance(memory, str):
        raise ValueError(f"{where}: expected a string memory field")
    return memory


def load_cases(dataset_path: Path, generated_path: Path) -> tuple[list[dict[str, str]], dict[str, Any]]:
    dataset = read_json(dataset_path)
    if not isinstance(dataset, dict):
        raise ValueError("Dataset root must be a JSON object")
    raw_cases = dataset.get("cases")
    if raw_cases is None:
        raw_cases = dataset.get("samples")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError('Dataset must contain a non-empty "cases" or "samples" array')

    cases: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, item in enumerate(raw_cases, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Dataset case #{index} is not an object")
        sample_id = item.get("case_id") or item.get("sample_id") or f"LTM_{index:03d}"
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise ValueError(f"Dataset case #{index} has an invalid ID")
        if sample_id in seen:
            raise ValueError(f"Duplicate dataset case ID: {sample_id}")
        seen.add(sample_id)
        expected = item.get("expected_output")
        if expected is None:
            expected = item.get("expected")
        expected_memory = _extract_memory(expected, f"dataset {sample_id} expected output")
        cases.append({"sample_id": sample_id, "expected_memory": expected_memory})

    generated_document = read_json(generated_path)
    if not isinstance(generated_document, dict):
        raise ValueError("Generated-memory file root must be a JSON object")
    # Standard runner output: {"results": {sample_id: result}}.
    # Also accept checkpoint.json: {"generated": {sample_id: result}}.
    generated = generated_document.get("results")
    if generated is None:
        generated = generated_document.get("generated")
    if generated is None:
        # Also accept a direct sample_id -> result mapping.
        generated = generated_document
    if not isinstance(generated, dict):
        raise ValueError("Could not find a sample_id -> generated result mapping")
    return cases, generated


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False, truncation=False)
    ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    # Fast/slow tokenizers may return a batch dimension only when explicitly
    # requested; normalize any one-item nested sequence defensively.
    if ids and isinstance(ids[0], list):
        if len(ids) != 1:
            raise ValueError("Tokenizer unexpectedly returned multiple sequences")
        ids = ids[0]
    return list(ids)


def _offsets(tokenizer: Any, text: str) -> list[tuple[int, int]]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        return_offsets_mapping=True,
    )
    offsets = encoded["offset_mapping"] if isinstance(encoded, dict) else encoded.offset_mapping
    if offsets and isinstance(offsets[0], list) and offsets and offsets[0] and isinstance(offsets[0][0], (list, tuple)):
        if len(offsets) != 1:
            raise ValueError("Tokenizer unexpectedly returned multiple offset sequences")
        offsets = offsets[0]
    return [(int(start), int(end)) for start, end in offsets]


def _encoded_length_with_specials(tokenizer: Any, text: str) -> int:
    encoded = tokenizer(text, add_special_tokens=True, truncation=False)
    ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    if ids and isinstance(ids[0], list):
        if len(ids) != 1:
            raise ValueError("Tokenizer unexpectedly returned multiple sequences")
        ids = ids[0]
    return len(ids)


def token_aware_chunks(text: str, tokenizer: Any, max_seq_length: int, overlap: int) -> list[dict[str, Any]]:
    """Split a paragraph into overlap-preserving chunks, none over max_seq_length.

    Chunk boundaries come from tokenizer character offsets, so the original
    text is sliced rather than reconstructed from token IDs. Each final text
    chunk is tokenized again with special tokens and must fit the model limit.
    """
    if not text.strip():
        return []
    special_count = int(tokenizer.num_special_tokens_to_add(pair=False))
    payload_limit = max_seq_length - special_count
    if payload_limit < 1:
        raise ValueError(
            f"Invalid model max_seq_length={max_seq_length}; tokenizer needs {special_count} special tokens"
        )
    if overlap < 0 or overlap >= payload_limit:
        raise ValueError(f"chunk overlap must be in [0, {payload_limit - 1}], got {overlap}")

    ids = _token_ids(tokenizer, text)
    offsets = _offsets(tokenizer, text)
    if len(ids) != len(offsets):
        raise RuntimeError("Tokenizer returned different token and offset counts")
    if not ids:
        return [{"text": text, "content_tokens": 0, "model_tokens": _encoded_length_with_specials(tokenizer, text),
                 "start_token": 0, "end_token": 0}]

    chunks: list[dict[str, Any]] = []
    start = 0
    while start < len(ids):
        end = min(start + payload_limit, len(ids))
        while end > start:
            char_start = offsets[start][0]
            char_end = offsets[end - 1][1]
            if char_end <= char_start:
                end -= 1
                continue
            chunk_text = text[char_start:char_end]
            model_tokens = _encoded_length_with_specials(tokenizer, chunk_text)
            if model_tokens <= max_seq_length:
                break
            # Decoding/token boundary effects can very occasionally add a
            # token. Shrink the end until the exact text slice fits.
            end -= 1
        if end <= start:
            raise RuntimeError(
                "Unable to create a non-truncating BGE chunk; check tokenizer offsets and model sequence limit"
            )
        chunks.append({
            "text": chunk_text,
            "content_tokens": len(_token_ids(tokenizer, chunk_text)),
            "model_tokens": model_tokens,
            "start_token": start,
            "end_token": end,
        })
        if end >= len(ids):
            break
        next_start = max(start + 1, end - overlap)
        if next_start <= start:
            raise RuntimeError("Chunking did not advance; refusing an infinite loop")
        start = next_start
    return chunks


def _normalise_vectors(vectors: Any, np: Any) -> Any:
    array = np.asarray(vectors, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D embedding matrix, got shape {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("BGE returned NaN or infinite embedding values")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms <= 0) or not np.isfinite(norms).all():
        raise ValueError("BGE returned a zero-length or invalid embedding vector")
    return array / norms


def pair_chunk_coverage(expected_vectors: Any, generated_vectors: Any, np: Any) -> float:
    """Symmetric chunk coverage for one expected/generated paragraph pair."""
    similarities = expected_vectors @ generated_vectors.T
    similarities = np.clip(similarities, -1.0, 1.0)
    expected_to_generated = float(similarities.max(axis=1).mean())
    generated_to_expected = float(similarities.max(axis=0).mean())
    return (expected_to_generated + generated_to_expected) / 2.0


def score_case(
    expected_paragraphs: list[str],
    generated_paragraphs: list[str],
    vectors_by_key: dict[tuple[str, str, int], Any],
    sample_id: str,
    np: Any,
) -> dict[str, Any]:
    if not expected_paragraphs and not generated_paragraphs:
        return {"score": 1.0, "expected_to_generated": 1.0, "generated_to_expected": 1.0,
                "expected_paragraphs": 0, "generated_paragraphs": 0}
    if not expected_paragraphs or not generated_paragraphs:
        return {"score": 0.0, "expected_to_generated": 0.0, "generated_to_expected": 0.0,
                "expected_paragraphs": len(expected_paragraphs), "generated_paragraphs": len(generated_paragraphs)}

    pair_scores = np.zeros((len(expected_paragraphs), len(generated_paragraphs)), dtype=np.float64)
    for i in range(len(expected_paragraphs)):
        e_vectors = vectors_by_key[(sample_id, "expected", i)]
        for j in range(len(generated_paragraphs)):
            g_vectors = vectors_by_key[(sample_id, "generated", j)]
            pair_scores[i, j] = pair_chunk_coverage(e_vectors, g_vectors, np)

    expected_to_generated = float(pair_scores.max(axis=1).mean())
    generated_to_expected = float(pair_scores.max(axis=0).mean())
    score = (expected_to_generated + generated_to_expected) / 2.0
    return {
        "score": round(float(np.clip(score, -1.0, 1.0)), 6),
        "expected_to_generated": round(float(np.clip(expected_to_generated, -1.0, 1.0)), 6),
        "generated_to_expected": round(float(np.clip(generated_to_expected, -1.0, 1.0)), 6),
        "expected_paragraphs": len(expected_paragraphs),
        "generated_paragraphs": len(generated_paragraphs),
        "paragraph_pair_scores": np.round(pair_scores, 6).tolist(),
    }


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def run(args: argparse.Namespace) -> Path:
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Offline BGE evaluation requires numpy and sentence-transformers. "
            "Install them in the project venv; this script does not require Groq."
        ) from exc

    dataset_path = args.dataset.expanduser().resolve()
    results_dir = args.results_dir.expanduser().resolve()
    generated_path = args.generated_file.expanduser().resolve() if args.generated_file else results_dir / "generated_memories.json"
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    if not generated_path.is_file():
        raise FileNotFoundError(f"Generated-memory file not found: {generated_path}")

    cases, generated = load_cases(dataset_path, generated_path)
    selected = cases[: args.limit] if args.limit else cases
    pending: list[dict[str, Any]] = []
    missing: list[str] = []
    failed: list[dict[str, str]] = []

    for case in selected:
        sid = case["sample_id"]
        record = generated.get(sid)
        if not isinstance(record, dict):
            missing.append(sid)
            continue
        if record.get("status") not in (None, "completed"):
            failed.append({"sample_id": sid, "reason": str(record.get("error") or record.get("status"))})
            continue
        memory = record.get("memory")
        if not isinstance(memory, str):
            failed.append({"sample_id": sid, "reason": "generated result has no string memory field"})
            continue
        pending.append({**case, "generated_memory": memory})

    if not pending:
        raise ValueError("No completed generated memories matched the selected dataset cases")

    print(f"Dataset cases: {len(selected)} | Completed memories: {len(pending)} | Missing: {len(missing)} | Failed: {len(failed)}")
    print(f"Loading BGE once: {args.model} | device={args.device}")
    model = SentenceTransformer(args.model, device=args.device)
    max_seq_length = int(model.max_seq_length)
    tokenizer = model.tokenizer
    if not hasattr(tokenizer, "num_special_tokens_to_add"):
        raise RuntimeError("Loaded tokenizer does not expose num_special_tokens_to_add()")
    special_count = int(tokenizer.num_special_tokens_to_add(pair=False))
    payload_limit = max_seq_length - special_count
    overlap = args.chunk_overlap if args.chunk_overlap is not None else min(48, max(0, payload_limit // 10))
    if overlap < 0 or overlap >= payload_limit:
        raise ValueError(f"--chunk-overlap must be between 0 and {payload_limit - 1}")

    # Plan and tokenize every paragraph first. Encode all chunks in one call,
    # so Sentence Transformers displays one aggregate progress bar, not one per case.
    paragraph_data: dict[tuple[str, str, int], dict[str, Any]] = {}
    all_chunk_texts: list[str] = []
    chunk_owners: list[tuple[str, str, int]] = []
    total_paragraphs = 0
    chunked_paragraphs = 0
    paragraphs_over_limit = 0
    max_observed_chunk_tokens = 0

    for case in pending:
        sid = case["sample_id"]
        for side, text in (("expected", case["expected_memory"]), ("generated", case["generated_memory"])):
            paragraphs = split_memory_paragraphs(text)
            for paragraph_index, paragraph in enumerate(paragraphs):
                chunks = token_aware_chunks(paragraph, tokenizer, max_seq_length, overlap)
                total_paragraphs += 1
                source_tokens = len(_token_ids(tokenizer, paragraph))
                if source_tokens + special_count > max_seq_length:
                    paragraphs_over_limit += 1
                if len(chunks) > 1:
                    chunked_paragraphs += 1
                if not chunks:
                    raise RuntimeError(f"{sid} {side} paragraph {paragraph_index}: tokenizer produced no chunks")
                max_chunk_tokens = max(chunk["model_tokens"] for chunk in chunks)
                max_observed_chunk_tokens = max(max_observed_chunk_tokens, max_chunk_tokens)
                key = (sid, side, paragraph_index)
                paragraph_data[key] = {
                    "source_tokens_without_specials": source_tokens,
                    "chunk_count": len(chunks),
                    "chunked": len(chunks) > 1,
                    "chunks": [
                        {"chunk_index": ci, "content_tokens": c["content_tokens"],
                         "model_tokens_including_specials": c["model_tokens"],
                         "source_token_start": c["start_token"], "source_token_end": c["end_token"]}
                        for ci, c in enumerate(chunks)
                    ],
                }
                for chunk_index, chunk in enumerate(chunks):
                    all_chunk_texts.append(chunk["text"])
                    chunk_owners.append(key)

    print(
        f"Paragraphs: {total_paragraphs} | Long paragraphs chunked: {chunked_paragraphs} "
        f"| Embedding chunks: {len(all_chunk_texts)} | model limit: {max_seq_length} tokens"
    )
    if max_observed_chunk_tokens > max_seq_length:
        raise RuntimeError("Internal safety check failed: an embedding chunk exceeds BGE max_seq_length")

    vectors_by_key: dict[tuple[str, str, int], Any] = {}
    if all_chunk_texts:
        embeddings = model.encode(
            all_chunk_texts,
            batch_size=args.batch_size,
            show_progress_bar=not args.quiet,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        embeddings = _normalise_vectors(embeddings, np)
        if len(embeddings) != len(chunk_owners):
            raise RuntimeError("BGE returned a different number of embeddings than input chunks")

        vectors_by_key_lists: dict[tuple[str, str, int], list[Any]] = defaultdict(list)
        for owner, vector in zip(chunk_owners, embeddings):
            vectors_by_key_lists[owner].append(vector)
        vectors_by_key = {key: np.stack(value, axis=0) for key, value in vectors_by_key_lists.items()}
    else:
        print("No non-empty paragraphs found; similarity follows the empty-memory rules without encoding.")

    per_case: dict[str, Any] = {}
    for case in pending:
        sid = case["sample_id"]
        expected_texts = split_memory_paragraphs(case["expected_memory"])
        generated_texts = split_memory_paragraphs(case["generated_memory"])
        metrics = score_case(
            expected_texts, generated_texts, vectors_by_key, sid, np,
        )
        para_details = {}
        for side, paragraphs in (("expected", expected_texts), ("generated", generated_texts)):
            para_details[side] = [
                {"paragraph_index": i, "text": paragraph, **paragraph_data[(sid, side, i)]}
                for i, paragraph in enumerate(paragraphs)
            ]
        per_case[sid] = {"status": "completed", **metrics, "paragraph_diagnostics": para_details}

    scores = [value["score"] for value in per_case.values()]
    summary = {
        "benchmark": "Offline BGE Long-Term Memory Similarity",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "device": str(getattr(model, "device", args.device)),
        "max_seq_length": max_seq_length,
        "special_tokens_per_input": special_count,
        "content_token_capacity_per_chunk": payload_limit,
        "chunk_overlap_tokens": overlap,
        "total_dataset_cases_selected": len(selected),
        "cases_evaluated": len(per_case),
        "cases_missing_generated_output": missing,
        "cases_with_generation_failure": failed,
        "bge_semantic_similarity": {
            "mean": round(sum(scores) / len(scores), 6) if scores else None,
            "cases": len(scores),
            "aggregation": "Mean of per-case symmetric paragraph best-match scores; each long-paragraph pair uses symmetric chunk-coverage cosine.",
        },
        "truncation_safeguard": {
            "paragraphs_total": total_paragraphs,
            "paragraphs_over_single_input_limit": paragraphs_over_limit,
            "paragraphs_split_into_multiple_chunks": chunked_paragraphs,
            "embedding_chunks_total": len(all_chunk_texts),
            "largest_embedded_chunk_tokens_including_specials": max_observed_chunk_tokens,
            "all_embedding_chunks_within_model_limit": max_observed_chunk_tokens <= max_seq_length,
            "silent_model_truncation_expected": False,
            "method": "Tokenizer offset-based overlapping chunks; every exact chunk is retokenized with special tokens and checked against model.max_seq_length before encoding.",
        },
    }

    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else results_dir / "bge_offline"
    output_dir.mkdir(parents=True, exist_ok=True)
    result_doc = {
        "metadata": {
            "model": args.model,
            "device": summary["device"],
            "max_seq_length": max_seq_length,
            "chunk_overlap_tokens": overlap,
            "dataset": str(dataset_path),
            "generated_memories": str(generated_path),
        },
        "results": per_case,
        "summary": summary,
    }
    atomic_write_json(output_dir / "bge_similarity_offline_results.json", result_doc)
    atomic_write_json(output_dir / "bge_similarity_offline_summary.json", summary)

    csv_path = output_dir / "bge_similarity_offline_per_sample.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "sample_id", "score", "expected_to_generated", "generated_to_expected",
            "expected_paragraphs", "generated_paragraphs", "chunked_paragraphs_expected",
            "chunked_paragraphs_generated",
        ])
        writer.writeheader()
        for sid, result in per_case.items():
            diag = result["paragraph_diagnostics"]
            writer.writerow({
                "sample_id": sid,
                "score": result["score"],
                "expected_to_generated": result["expected_to_generated"],
                "generated_to_expected": result["generated_to_expected"],
                "expected_paragraphs": result["expected_paragraphs"],
                "generated_paragraphs": result["generated_paragraphs"],
                "chunked_paragraphs_expected": sum(x["chunked"] for x in diag["expected"]),
                "chunked_paragraphs_generated": sum(x["chunked"] for x in diag["generated"]),
            })

    print("\nOffline BGE evaluation complete (no Groq calls).")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nSaved results to: {output_dir}")
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recompute LTM semantic similarity from saved outputs using BGE only; no Groq API."
    )
    parser.add_argument("--dataset", required=True, type=Path, help="Original frozen LTM benchmark dataset JSON")
    parser.add_argument("--results-dir", required=True, type=Path, help="Existing run directory containing generated_memories.json")
    parser.add_argument("--generated-file", type=Path, default=None, help="Optional explicit generated_memories.json or checkpoint.json")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory (default: RESULTS_DIR/bge_offline)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"SentenceTransformer model name/path (default: {DEFAULT_MODEL})")
    parser.add_argument("--device", default="cpu", help="Embedding device (default: cpu; use cuda only if your PyTorch supports it)")
    parser.add_argument("--batch-size", type=int, default=32, help="SentenceTransformer encoding batch size")
    parser.add_argument("--chunk-overlap", type=int, default=None, help="Overlap in content tokens (default: about 10%% of chunk capacity)")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for a small offline test")
    parser.add_argument("--quiet", action="store_true", help="Hide the single aggregate BGE batch progress bar")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    try:
        run(args)
        return 0
    except KeyboardInterrupt:
        print("Interrupted; no Groq calls were made.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
