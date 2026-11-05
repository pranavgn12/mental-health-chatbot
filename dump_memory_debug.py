#!/usr/bin/env python3
# Run from the backend or project root: python dump_memory_debug.py
# Creates memory_debug_dump.txt. Review it for private information before sharing.

from pathlib import Path
import re

ROOT = Path.cwd()
OUTPUT = ROOT / "memory_debug_dump.txt"
SKIP = {".git", ".venv", "venv", "__pycache__", "node_modules",
        ".pytest_cache", ".mypy_cache", "dist", "build", "uploads", "instance"}
WORDS = ("memory", "pattern", "episode", "mood", "reflection",
         "orchestrator", "llm", "prompt", "message", "profile")
REDACTIONS = [
    (re.compile(r'(?i)\\b(api[_-]?key|secret[_-]?key|access[_-]?token|refresh[_-]?token|password|passwd|client[_-]?secret)\\b(\\s*[:=]\\s*)([\\\"\\\']?)[^,\\s\\\"\\\']+\\3'),
     r"\\1\\2[REDACTED]"),
    (re.compile(r"(?i)(Bearer\\s+)[A-Za-z0-9._~+/=-]+"), r"\\1[REDACTED]"),
    (re.compile(r"\\b(sk-[A-Za-z0-9_-]{12,}|AIza[A-Za-z0-9_-]{20,})\\b"), "[REDACTED_API_KEY]"),
    (re.compile(r"(?i)(://[^:/\\s]+:)[^@/\\s]+(@)"), r"\\1[REDACTED]\\2"),
]

def redact(text):
    for pattern, replacement in REDACTIONS:
        text = pattern.sub(replacement, text)
    return text

def should_include(path):
    rel_parts = path.relative_to(ROOT).parts
    if any(part in SKIP or part.startswith(".") for part in rel_parts):
        return False
    if not path.is_file() or path == OUTPUT:
        return False
    if path.name.lower() in {".env", ".env.local", ".env.production"}:
        return False
    if path.suffix.lower() not in {".py", ".toml", ".ini", ".cfg", ".txt"}:
        return False
    rel = path.relative_to(ROOT).as_posix().lower()
    name = path.name.lower()
    if "/services/ai/" in "/" + rel + "/":
        return True
    if any(part in {"models", "model"} for part in rel_parts):
        return True
    if any(part in {"tests", "routers", "routes", "api"} for part in rel_parts):
        return any(word in rel for word in WORDS)
    if name in {"requirements.txt", "pyproject.toml", "pytest.ini",
                "config.py", "database.py", "db.py", "llm_provider.py"}:
        return True
    return any(word in name for word in WORDS)

def main():
    files = sorted((p for p in ROOT.rglob("*") if should_include(p)),
                   key=lambda p: p.relative_to(ROOT).as_posix())
    sections = [
        "MEMORY / PATTERN DEBUG SOURCE DUMP",
        "Project root: " + str(ROOT),
        "Files included: " + str(len(files)),
        "Excluded: virtual environments, .git, node_modules, uploads, .env files.",
        "=" * 88,
    ]
    for path in files:
        rel = path.relative_to(ROOT).as_posix()
        try:
            content = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError) as exc:
            sections.append("\\n===== SKIPPED " + rel + ": " + str(exc) + " =====")
            continue
        sections.extend(["\\n\\n" + "=" * 88, "FILE: " + rel,
                         "=" * 88 + "\\n", redact(content)])
    OUTPUT.write_text("\\n".join(sections), encoding="utf-8")
    print("Created:", OUTPUT)
    print("Files included:", len(files))
    print("Review the dump for private information before sharing.")

if __name__ == "__main__":
    main()
