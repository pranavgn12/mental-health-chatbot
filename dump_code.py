
import os
from pathlib import Path

# Project root (directory containing this script)
ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "project_dump.txt"

# Directories to ignore
IGNORE_DIRS = {
    "node_modules",
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".next",
    "dist",
    "build",
    "coverage",
    ".idea",
    ".vscode",
    "site-packages",
    "env",
    ".env",
}

# Files and extensions to ignore
IGNORE_FILES = {
    "project_dump.txt",
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "poetry.lock",
}

IGNORE_EXTENSIONS = {
    ".pyc",
    ".pyo",
    ".so",
    ".dll",
    ".exe",
    ".bin",
    ".db",
    ".sqlite",
    ".sqlite3",
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".webp",
    ".ico",
    ".mp4",
    ".mp3",
    ".pdf",
    ".zip",
    ".tar",
    ".gz",
    ".woff",
    ".woff2",
    ".ttf",
    ".otf",
}

# Avoid dumping secrets
IGNORE_SECRET_NAMES = {
    ".env",
    ".env.local",
    ".env.production",
    "credentials.json",
    "secrets.json",
}

def should_ignore(path):
    if path.name in IGNORE_SECRET_NAMES:
        return True

    if path.name in IGNORE_FILES:
        return True

    if path.suffix.lower() in IGNORE_EXTENSIONS:
        return True

    return False


def dump_project():
    file_count = 0

    with OUTPUT.open("w", encoding="utf-8") as dump:
        dump.write("PROJECT SOURCE CODE DUMP\n")
        dump.write(f"Project root: {ROOT}\n")
        dump.write("=" * 80 + "\n\n")

        for current_root, dirs, files in os.walk(ROOT):

            # Exclude unwanted directories
            dirs[:] = sorted(
                d for d in dirs if d not in IGNORE_DIRS
            )

            for filename in sorted(files):
                filepath = Path(current_root) / filename

                if should_ignore(filepath):
                    continue

                try:
                    # Skip binary files
                    with filepath.open("rb") as f:
                        if b"\0" in f.read(8192):
                            continue

                    content = filepath.read_text(
                        encoding="utf-8",
                        errors="replace"
                    )

                    relative_path = filepath.relative_to(ROOT)

                    dump.write("\n")
                    dump.write("=" * 80 + "\n")
                    dump.write(f"FILE: {relative_path}\n")
                    dump.write("=" * 80 + "\n\n")
                    dump.write(content)
                    dump.write("\n\n")

                    file_count += 1

                except (OSError, UnicodeError) as error:
                    print(f"Skipped {filepath}: {error}")

    print(f"\nDump completed!")
    print(f"Files included: {file_count}")
    print(f"Output: {OUTPUT}")


if __name__ == "__main__":
    dump_project()
