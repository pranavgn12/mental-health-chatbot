#!/usr/bin/env python3
"""
Dump the existing chatbot source files needed to integrate streaming safely.

Run from anywhere:
    python dump_streaming_context.py /path/to/mental-health-chatbot

If the project root is omitted, the script tries the current directory and its
parents. Output defaults to streaming_code_dump.txt in the current directory:
    python dump_streaming_context.py /path/to/mental-health-chatbot -o code_dump.txt

The script only reads source/config text files. It skips virtual environments,
build output, caches, secrets, databases, and vector-store contents.
"""
from __future__ import annotations

import argparse
from pathlib import Path

# Core request/response and LLM path, plus the services invoked by the
# orchestrator. Include optional files when present; missing files are listed.
RELEVANT_FILES = [
    # API / database / model layer
    "backend/main.py",
    "backend/routers/messages.py",
    "backend/routers/chats.py",
    "backend/models.py",
    "backend/database.py",
    "backend/schemas.py",
    "backend/auth.py",
    "backend/auth_utils.py",

    # User-facing generation and provider
    "backend/services/llm_provider.py",
    "backend/services/llm.py",
    "backend/services/prompt_builder.py",
    "backend/services/title_generator.py",
    "backend/services/ai/orchestrator.py",

    # Orchestrator dependencies that may run before/after the reply
    "backend/services/ai/episode_detector.py",
    "backend/services/ai/episode_manager.py",
    "backend/services/ai/episode_summarizer.py",
    "backend/services/ai/json_parser.py",
    "backend/services/ai/mood_detector.py",
    "backend/services/ai/reflection_generator.py",
    "backend/services/ai/pattern_detector.py",
    "backend/services/ai/profile.py",
    "backend/services/ai/short_term_memory.py",
    "backend/services/ai/long_term_memory.py",
    "backend/services/ai/memory_consolidator.py",
    "backend/services/ai/memory_rag.py",
    "backend/services/ai/knowledge_rag.py",
    "backend/services/ai/__init__.py",
    "backend/services/__init__.py",
    "backend/routers/__init__.py",

    # Prompt files used by those services (text only; no secrets expected)
    "backend/services/ai/prompts/episode_detector_prompt.py",
    "backend/services/ai/prompts/episode_summarizer_prompt.py",
    "backend/services/ai/prompts/mood_detector_prompt.py",
    "backend/services/ai/prompts/reflection_prompt.py",
    "backend/services/ai/prompts/memory_consolidator_prompt.py",

    # Frontend path that sends a message and renders the response
    "frontend/src/services/api.js",
    "frontend/src/services/messages.js",
    "frontend/src/services/chat.js",
    "frontend/src/hooks/useChat.js",
    "frontend/src/context/AppContext.jsx",
    "frontend/src/components/ChatWindow.jsx",
    "frontend/src/components/MessageBubble.jsx",
    "frontend/src/components/MessageInput.jsx",
]

SKIP_DIRS = {
    ".git", ".venv", "venv", "env", "node_modules", "__pycache__",
    ".pytest_cache", ".mypy_cache", ".ruff_cache", "dist", "build",
    "vector_db", "chroma_db", ".next", "coverage",
}
SKIP_NAMES = {
    ".env", ".env.local", ".env.development", ".env.production",
    "id_rsa", "id_ed25519",
}
TEXT_SUFFIXES = {".py", ".js", ".jsx", ".ts", ".tsx", ".json", ".toml", ".ini", ".cfg", ".md", ".txt"}


def find_project_root(argument: str | None) -> Path:
    candidates: list[Path] = []
    if argument:
        candidates.append(Path(argument).expanduser().resolve())
    here = Path.cwd().resolve()
    candidates.extend([here, *here.parents])

    for candidate in candidates:
        if (candidate / "backend").is_dir() and (candidate / "frontend").is_dir():
            return candidate
    raise SystemExit(
        "Could not find project root containing both 'backend/' and 'frontend/'.\n"
        "Pass the path explicitly, e.g. python dump_streaming_context.py "
        "/home/your-user/Desktop/mental-health-chatbot"
    )


def safe_read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"[ERROR READING FILE: {exc}]"


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump chatbot files relevant to streaming integration.")
    parser.add_argument("project_root", nargs="?", help="Project root containing backend/ and frontend/")
    parser.add_argument("-o", "--output", default="streaming_code_dump.txt", help="Output text file")
    args = parser.parse_args()

    root = find_project_root(args.project_root)
    output = Path(args.output).expanduser().resolve()

    parts = [
        "CHATBOT STREAMING-INTEGRATION CODE DUMP",
        f"Project root: {root}",
        "Note: .env files, virtual environments, databases, vector stores, and build output are excluded.",
        "=" * 100,
        "",
    ]

    included = 0
    missing: list[str] = []

    for relative in RELEVANT_FILES:
        path = root / relative
        parts.extend(["\n" + "#" * 100, f"FILE: {relative}", "#" * 100, ""])
        if not path.is_file():
            parts.append("[FILE NOT FOUND]")
            missing.append(relative)
            continue
        if path.name in SKIP_NAMES or path.suffix.lower() not in TEXT_SUFFIXES:
            parts.append("[SKIPPED: not an allowed text source file]")
            continue

        content = safe_read(path)
        # Keep the dump bounded and flag truncation rather than silently cutting.
        max_chars = 250_000
        if len(content) > max_chars:
            content = content[:max_chars] + "\n\n[TRUNCATED: file exceeded 250,000 characters]\n"
        parts.append(content)
        included += 1

    parts.extend([
        "\n" + "=" * 100,
        f"SUMMARY: included {included} files; missing {len(missing)} files.",
        "Missing optional paths:",
        *([f"  - {name}" for name in missing] or ["  - none"]),
        "",
    ])

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts), encoding="utf-8")
    print(f"Project root: {root}")
    print(f"Dump written to: {output}")
    print(f"Files included: {included}")
    print(f"Files not found: {len(missing)}")
    print("\nNext: upload the generated dump file in this chat.")
    print("Before sharing, check it does not contain API keys, passwords, tokens, or personal data.")


if __name__ == "__main__":
    main()
