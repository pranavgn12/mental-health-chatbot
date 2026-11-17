"""Central application settings loaded from environment variables / .env.

Feature flags are introduced here first; the orchestrator will consume them
in the next implementation step. Short-term memory is intentionally always on.
"""

import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

# Load the backend .env when the application imports this module.
load_dotenv()


def _env_bool(name: str, default: bool) -> bool:
    """Parse common boolean environment values without treating 'false' as true."""
    value = os.getenv(name)
    if value is None:
        return default

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False

    raise ValueError(
        f"Invalid boolean value for {name}: {value!r}. "
        "Use true/false, 1/0, yes/no, or on/off."
    )


@dataclass(frozen=True)
class AppSettings:
    # Model configuration. The benchmark will use GPT-OSS 120B consistently.
    groq_model: str

    # Conversation context: deliberately not configurable/off-switchable.
    enable_short_term_memory: bool = True

    # Knowledge retrieval.
    enable_knowledge_rag: bool = False

    # Persistent memory components.
    enable_episodic_memory: bool = False
    enable_long_term_memory: bool = False
    enable_memory_rag: bool = False
    enable_memory_consolidation: bool = False

    # Additional analysis/generation components.
    enable_pattern_detection: bool = False
    enable_reflection: bool = False
    enable_mood_detection: bool = False

    # Development / benchmark controls.
    benchmark_mode: bool = False


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Return one immutable settings snapshot for the running application."""
    return AppSettings(
        groq_model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b").strip(),
        # Always True by design; no environment variable can disable it.
        enable_short_term_memory=True,
        enable_knowledge_rag=_env_bool("ENABLE_KNOWLEDGE_RAG", False),
        enable_episodic_memory=_env_bool("ENABLE_EPISODIC_MEMORY", False),
        enable_long_term_memory=_env_bool("ENABLE_LONG_TERM_MEMORY", False),
        enable_memory_rag=_env_bool("ENABLE_MEMORY_RAG", False),
        enable_memory_consolidation=_env_bool("ENABLE_MEMORY_CONSOLIDATION", False),
        enable_pattern_detection=_env_bool("ENABLE_PATTERN_DETECTION", False),
        enable_reflection=_env_bool("ENABLE_REFLECTION", False),
        enable_mood_detection=_env_bool("ENABLE_MOOD_DETECTION", False),
        benchmark_mode=_env_bool("BENCHMARK_MODE", False),
    )


def reload_settings() -> AppSettings:
    """Clear the cached snapshot and reload environment values (mainly for tests)."""
    get_settings.cache_clear()
    return get_settings()
