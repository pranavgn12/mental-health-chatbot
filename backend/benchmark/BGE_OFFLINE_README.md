# Offline BGE similarity re-evaluation

This utility recomputes only the semantic-similarity metric for an existing LTM benchmark run. It reads:

- the original frozen benchmark dataset (reference `expected.memory` / `expected_output.memory`), and
- the saved `generated_memories.json` from a completed Groq run.

It does **not** import the Groq SDK, call Groq, use the LLM judge, invoke the production LTM generator, or modify the original benchmark outputs.

## Install

Run inside the same Python virtual environment used for the project:

```bash
pip install sentence-transformers numpy
```

The first run may need to download `BAAI/bge-small-en-v1.5` from the model hub if it is not already cached. This is a model download, not a Groq API call. Later runs reuse the local model cache.

## Run

From the project's `backend/` directory:

```bash
python benchmark/bge_similarity_offline.py \
  --dataset benchmark/datasets/long_term_memory_benchmark.json \
  --results-dir benchmark/results/long_term_memory_YYYYMMDD_HHMMSS
```

The results directory must contain `generated_memories.json`. Alternatively, supply an explicit file:

```bash
python benchmark/bge_similarity_offline.py \
  --dataset benchmark/datasets/long_term_memory_benchmark.json \
  --results-dir benchmark/results/long_term_memory_YYYYMMDD_HHMMSS \
  --generated-file /path/to/generated_memories.json
```

By default it runs BGE on CPU (safe and consistent with the original Chroma embedding-function default). To use a supported CUDA device explicitly:

```bash
--device cuda
```

The script loads BGE once, prepares all expected/generated paragraph chunks, and calls `model.encode()` once for all chunks. Therefore, it shows one aggregate `Batches` progress bar, rather than one per test case.

## Long-paragraph truncation safeguard

The script reads the loaded model's actual `max_seq_length` and the tokenizer's special-token count. Paragraphs exceeding that limit are split using tokenizer character offsets into overlapping text chunks. Each exact chunk is tokenized again **with special tokens and without truncation** and checked against `max_seq_length` before it is passed to Sentence Transformers. If the check fails, the script stops instead of silently accepting truncation.

Long-paragraph similarity is chunk-aware: each expected/generated paragraph pair is scored using symmetric best matching across their chunks. Those paragraph-pair scores are then combined with the same symmetric paragraph-level best-match aggregation used by the original benchmark. When both paragraphs fit in one BGE input, their comparison is ordinary cosine similarity, matching the original approach.

Because long paragraphs are now compared with chunk coverage rather than one possibly truncated embedding, the new aggregate score is **not strictly comparable** to the original `0.882485` if any paragraph was split. The new report includes chunk counts and per-paragraph token diagnostics so the difference is auditable.

## Output

Written to `<results-dir>/bge_offline/` by default:

- `bge_similarity_offline_results.json` — per-case scores and paragraph/chunk diagnostics.
- `bge_similarity_offline_summary.json` — aggregate score and truncation-safety report.
- `bge_similarity_offline_per_sample.csv` — compact per-case scores.

Original Groq outputs, judge results, checkpoints, and summaries are left untouched.

## Tests (no model download or Groq calls)

From the tool root:

```bash
python -m unittest discover -s benchmark/tests -v
```
