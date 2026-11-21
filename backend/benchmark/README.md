# Knowledge RAG benchmark v3

This benchmark calls the project's production `services.ai.rag.local_retriever.LocalRetriever` for the ranked retrieval test. It does not create a second copy of the retrieval algorithm.

## Important changes in v3

- Every dataset question is represented in `per_question_metrics.json` and `.csv`. Missing retrieval records are reported as `retrieval_record_missing`; evaluation loops no longer silently truncate with `zip()`.
- Evidence candidate discovery normalizes Unicode whitespace, newlines, case and filename paths. It applies the exact raw Chroma metadata value in `where` filters and falls back to a global corpus query when a source label is missing or yields no content.
- Existing ground-truth annotations are reused only if their question, reference answer, evidence phrase and source mapping still match. A changed question resets stale relevance and answer judgements for that question.
- Completed candidate reviews and completed production retrieval rankings are cached. Resuming an unchanged run does not repeat those local retrieval/embedding operations.
- Relevance judgements are batched (default 5 chunks per Groq request). Each candidate is still scored independently. This reduces request count and repeated prompt overhead; it does not reduce the number of candidate chunks that the model must read.
- Failed/skipped Groq requests are recorded and are not retried automatically on a later run. Use `--retry-failed` only when you intend to spend more tokens retrying them.
- Context-too-large errors are skipped rather than prompting for a different API key, since changing a key cannot fix an oversized prompt. 429/5xx retries use bounded backoff.
- The initial Groq key check uses the model-list endpoint instead of a chat completion, so it does not spend completion tokens.
- Answer similarity uses one cached CPU SentenceTransformer model per process. The production Chroma embedding function is also explicitly configured for CPU by default through `KNOWLEDGE_EMBEDDING_DEVICE=cpu`.
- Answer generation and answer judging are opt-in. The default run performs candidate judging and retrieval evaluation only.

## Installation

Run in the backend virtual environment with the project's existing dependencies:

```bash
pip install chromadb sentence-transformers groq python-dotenv
```

## Run

From the project `backend/` directory:

```bash
python benchmark/knowledge_rag_benchmark.py \
  --dataset benchmark/results/<existing-run>/ground_truth.json \
  --resume benchmark/results/<existing-run> \
  --k 3 5 10
```

The resume operation re-discovers evidence candidates using the corrected source matching. It keeps valid prior judgements and only sends newly discovered, not-yet-judged chunks to Groq. The previous run's `review_complete` flag alone is not trusted; old candidate pools are rebuilt once under v3.

To also run answer generation and answer judging:

```bash
python benchmark/knowledge_rag_benchmark.py \
  --dataset benchmark/results/<existing-run>/ground_truth.json \
  --resume benchmark/results/<existing-run> \
  --k 3 5 10 \
  --run-answer-benchmark
```

The answer benchmark can require up to two additional completions per question (generation + judging). Already judged answers are reused on resume.

To retry requests that were deliberately skipped or failed earlier:

```bash
... --resume benchmark/results/<existing-run> --retry-failed
```

This can consume additional tokens. Without the flag, failed/skipped candidate and answer requests remain pending and are not repeatedly retried.

## Cost controls and defaults

- `--top-n 15`: maximum evidence candidates per evidence phrase. Keep this fixed when comparing runs; reducing it can make the ground-truth candidate pool less complete.
- `--judge-batch-size 5`: evaluate five chunks in one Groq completion by default. Increase only if the chunks are short and the model reliably returns all candidate IDs; decrease if responses are truncated or malformed.
- `--run-answer-benchmark`: required to run the separate answer-generation benchmark. It is off by default.
- `--retry-failed`: explicit opt-in to retry skipped/failed LLM calls.
- `--embedding-device cpu`: CPU is the default for BGE answer similarity; do not set it to `cuda` on a low-memory GPU.

For the current 41-question dataset (one evidence phrase each), `--top-n 15` gives an upper bound of 615 candidate labels. With a batch size of 5, that is at most 123 relevance requests before accounting for already saved labels or candidate pools that return fewer than 15 chunks. Opting into the answer benchmark adds at most 82 more completions (41 generations + 41 answer judgements), before resume-cache savings. Batching reduces call overhead and repeated prompt text, not candidate-content token volume by a factor of five.

The startup banner prints these upper bounds before calling Groq. Relevance labels already present in the resume state are retained and are not sent again.

## Outputs

The run directory contains:

- `review_state.json`: resumable annotations, prior decisions and failed-request markers.
- `ground_truth.json`: reviewed candidate grades and evidence.
- `retrieval_cache.json`: cache of production `LocalRetriever` results and its signature.
- `retrieval_results.json`: retrieved production rankings.
- `per_question_metrics.json` / `.csv`: one row per dataset question, including non-evaluable statuses.
- `metrics.json`: aggregate retrieval metrics and coverage/status counts.
- `answer_metrics.json`, `answer_per_question.json` / `.csv`, `answer_review.json`: answer results, populated when answer reviews exist.

Retrieval precision, recall and MRR are macro-averaged only across questions with a completed candidate review and at least one grade-2/3 chunk. Graded nDCG also includes completed grade-1-only questions. Questions with no candidates or no positive evidence remain present and are identified by `evaluation_status`; they are not silently removed.

## Tests

Run:

```bash
python benchmark/knowledge_rag_benchmark.py --self-test
```

The included offline tests do not require Chroma or Groq. They validate normalization, source fallback, batch judging, cache reuse, fingerprints, missing-record reporting, and CPU model caching. A live Chroma/Groq run is still required to verify the actual local database and API environment.
