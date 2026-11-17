# Step 2: Orchestrator feature switches

The AI orchestrator now reads the centralized settings from `services.settings.get_settings()` and applies the optional feature flags in both `chat()` and `chat_stream()`.

## Behaviour

- Short-term conversation context remains enabled and is always passed to the response generator.
- Knowledge RAG is only initialized and queried when `ENABLE_KNOWLEDGE_RAG=true`.
- Long-term memory is only retrieved when `ENABLE_LONG_TERM_MEMORY=true`.
- Episode analysis/lifecycle is skipped when episodic memory, mood detection, reflection, and memory consolidation are all disabled.
- Episode creation, message-to-episode assignment, episode summaries, and closure are controlled by `ENABLE_EPISODIC_MEMORY`.
- Mood timeline reads/writes are controlled by `ENABLE_MOOD_DETECTION`.
- Reflection reads/writes and reflection context are controlled by `ENABLE_REFLECTION`.
- Long-term memory consolidation at episode closure is controlled by `ENABLE_MEMORY_CONSOLIDATION`.
- Pattern analysis during consolidation is controlled separately by `ENABLE_PATTERN_DETECTION`.

## Important dependencies / current limitations

1. `ENABLE_MEMORY_CONSOLIDATION=true` requires `ENABLE_EPISODIC_MEMORY=true` to create and close episodes that can be consolidated. Without an active episode, there is nothing to consolidate.
2. `ENABLE_MOOD_DETECTION` and `ENABLE_REFLECTION` use the existing active episode as their persistence scope. They do not create episodes when episodic memory is disabled.
3. `MemoryRAG` is currently an empty placeholder in the supplied project. `ENABLE_MEMORY_RAG` is defined in settings but cannot activate a retrieval implementation yet.
4. `TurnAnalyzer` currently produces episode, mood, and reflection decisions in one combined LLM call. The orchestrator gates the associated retrieval, persistence, and response context independently, but enabling any episode workflow feature still invokes that combined analyzer. Splitting the analyzer into independent prompts is a separate future optimization.
5. Conversation title generation and short-term conversation summarization are existing LLM operations outside these feature switches. This step does not disable them.

## Test

From the backend directory:

```bash
PYTHONPATH=. pytest -q tests/test_settings.py tests/test_orchestrator_flags.py
```

The full backend test suite also contains `test_db.py`, which imports the database engine during collection and requires a valid local database environment. Configure the backend `.env` before running the complete suite.
