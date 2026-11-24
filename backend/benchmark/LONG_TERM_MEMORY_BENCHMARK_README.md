# Long-Term Memory Benchmark

This benchmark evaluates the **production** `LongTermMemory._generate_memory()` method. It does not run the live database, `LongTermMemory.update()`, the upstream `PatternMemoryAnalyzer`, or profile generation. The dataset supplies already-validated patterns; an in-memory query adapter supplies mood and reflection rows.

## What it uses from production

- `services.ai.long_term_memory.MEMORY_PROMPT`
- `LongTermMemory._generate_memory()` and its response parser/12,000-character validator
- `services.llm_provider.generate_response()`
- The current production request settings: `max_completion_tokens=3000`, `reasoning_effort="low"`, JSON-object response format
- Model selected by `GROQ_MODEL`, defaulting to `openai/gpt-oss-120b`

The benchmark temporarily replaces the LTM module's imported `generate_response` reference with a small wrapper that calls the production provider. It swaps `llm_provider.client` in memory when a replacement API key is entered. It does not edit `llm_provider.py`, `long_term_memory.py`, or `.env`.

## Dataset

Default path:

```text
backend/benchmark/datasets/long_term_memory_benchmark.json
```

The runner accepts both:

1. The new `cases` schema with `case_id`, `input`, `expected_output`, and `evaluation_criteria`.
2. The earlier project's `samples` schema with `sample_id`, `current_episode`, `mood_observations`, `reflections`, `validated_patterns`, `previous_memory`, and `expected.memory`.

For the new schema, the runner expects these input fields:

```json
{
  "episode_summary": "...",
  "mood_observations": [],
  "reflections": [],
  "validated_patterns": [],
  "previous_memory": null
}
```

`expected_output.memory` is the reference string. `evaluation_criteria` may contain `must_preserve`, `must_not_infer`, `temporal_constraints`, and `paragraph_expectations`, each an array of strings. If `must_preserve` is omitted, expected-memory paragraphs are used as the retention targets. If `temporal_constraints` is omitted, temporal accuracy is reported as not applicable for that case.

Validate the dataset without calling Groq or loading BGE:

```bash
python benchmark/long_term_memory_benchmark.py --validate-only
```

Or validate a different file:

```bash
python benchmark/long_term_memory_benchmark.py --dataset /path/to/ltm_dataset.json --validate-only
```

## Metrics

The report contains four main metrics:

1. **Information Retention** — LLM judge labels each retention target `retained`, `partial`, or `missing`; Python assigns 1, 0.5, or 0.
2. **Factual Accuracy** — LLM judge extracts atomic generated claims and labels them supported, partially supported, unsupported, or contradicted. Python assigns 1, 0.5, or 0.
3. **Prohibited-Inference Violation Rate** — reported separately as the number of judge-identified violations divided by the number of `must_not_infer` constraints. It is kept separate from factual accuracy because the same generated claim can be unsupported and violate a prohibited-inference constraint; combining them would double-count it.
4. **Temporal & Update Accuracy** — LLM judge checks each explicit temporal constraint and labels it correct, partially correct, incorrect, or unclear. Python assigns 1, 0.5, or 0.
5. **BGE/Chroma Semantic Similarity** — a temporary in-memory ChromaDB collection embeds each expected/generated paragraph using `SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-small-en-v1.5")`. The collection uses cosine distance. Chroma stores the expected and generated embeddings, and the runner retrieves those exact vectors and calculates all-pairs cosine similarity in NumPy. The reported score is the mean of expected-to-generated and generated-to-expected best-match cosine similarities.

BGE similarity is a semantic diagnostic, not a factuality metric. The evaluator uses the same ChromaDB embedding-function approach and default BGE model as `local_retriever.py`, but creates an isolated ephemeral collection for each case. It does **not** query or modify your production `knowledge` collection, and it does not apply BM25/RRF or identifier boosts. The exact vector comparison avoids approximate HNSW nearest-neighbour ranking. This measures semantic closeness only, independently of the knowledge-retrieval ranking logic.

Structural checks (JSON shape, memory string, 12,000-character limit, list syntax) are reported as diagnostics, not headline research metrics. The production prompt calls for separate paragraphs without bullets or headings.

## Run

Run from `backend/` in the project's virtual environment:

```bash
python benchmark/long_term_memory_benchmark.py
```

Smoke test with two samples:

```bash
python benchmark/long_term_memory_benchmark.py --limit 2
```

Specify a dataset or model:

```bash
python benchmark/long_term_memory_benchmark.py --dataset benchmark/datasets/my_ltm_dataset.json --model openai/gpt-oss-120b
```

Skip LLM judging (generation and BGE similarity still run):

```bash
python benchmark/long_term_memory_benchmark.py --skip-judge
```

## API key replacement during a run

The runner uses `GROQ_API_KEY` from `backend/.env` or the shell. It also accepts a comma-separated pool through `GROQ_API_KEYS` or `--api-keys`:

```bash
python benchmark/long_term_memory_benchmark.py --api-keys "key_one,key_two,key_three"
```

On authentication, permission, quota, or rate-limit errors (HTTP 401/403/429 or recognized quota/rate-limit messages), it first tries the next configured key. When the pool is exhausted, it asks for a replacement key using hidden terminal input. The same interrupted request is retried. Press Enter or Ctrl+C to stop safely; completed work remains checkpointed. Replacement keys stay in process memory and are never written to `.env`, logs, or result files.

A 429 may be imposed at the account/organization level; switching API keys belonging to the same account may not clear that limit.

## Resume

Each run creates a timestamped directory under `benchmark/results/long_term_memory_.../`.

```bash
python benchmark/long_term_memory_benchmark.py --resume benchmark/results/long_term_memory_YYYYMMDD_HHMMSS
```

Retry failed generation/judge samples:

```bash
python benchmark/long_term_memory_benchmark.py --resume benchmark/results/long_term_memory_YYYYMMDD_HHMMSS --retry-failed
```

Resume is allowed only when the dataset fingerprint, model, production prompt hash, embedding model, and script version match the saved checkpoint.

## Outputs

- `generated_memories.json` — raw provider response, generated memory, generation status, structural diagnostics
- `llm_judge_results.json` — item-level verdicts and per-case metric values
- `semantic_similarity_results.json` — BGE paragraph similarity per case
- `per_sample_metrics.csv` — compact per-case metrics
- `summary.json` — aggregate headline metrics
- `run_metadata.json` — dataset/model/prompt/version metadata; no keys
- `checkpoint.json` — resumable progress

The dataset and expected outputs should be frozen before benchmarking. Use a separate, blinded judge prompt and manually inspect a subset because an LLM judge is not independent ground truth, especially when using the same model for generation and judging.


## Offline tests

Run the included unit tests from `backend/`:

```bash
python -m unittest discover -s benchmark/tests -v
```

The tests cover both dataset schemas, metric calculations, judge-output validation, temporary Chroma collection behavior, and API-key rotation. They use mocked provider/Chroma boundaries and do not consume Groq tokens.
