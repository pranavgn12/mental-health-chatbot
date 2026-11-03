
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = PROJECT_ROOT / "ai_code_dump.txt"

EXCLUDED_DIRS = {
    "__pycache__",
    ".venv",
    "venv",
    "node_modules",
}

INCLUDED_EXTENSIONS = {
    ".py",
    ".txt",
    ".md",
    ".json",
    ".yaml",
    ".yml",
}

def dump_directory():
    if not SOURCE_DIR.exists():
        print(f"Directory not found: {SOURCE_DIR}")
        return

    files = sorted(
        path for path in SOURCE_DIR.rglob("*")
        if path.is_file()
        and path.suffix.lower() in INCLUDED_EXTENSIONS
        and not any(part in EXCLUDED_DIRS for part in path.parts)
    )

    with OUTPUT_FILE.open("w", encoding="utf-8") as output:
        output.write("AI SERVICES CODE DUMP\n")
        output.write("=" * 80 + "\n")
        output.write(f"Source: {SOURCE_DIR}\n")
        output.write(f"Total files: {len(files)}\n")
        output.write("=" * 80 + "\n\n")

        for file in files:
            relative_path = file.relative_to(PROJECT_ROOT)

            output.write("\n\n")
            output.write("=" * 80 + "\n")
            output.write(f"FILE: {relative_path}\n")
            output.write("=" * 80 + "\n\n")

            try:
                content = file.read_text(encoding="utf-8")
                output.write(content)
                output.write("\n")
            except (UnicodeDecodeError, OSError) as error:
                output.write(f"[Could not read file: {error}]\n")

    print(f"Dump completed!")
    print(f"Files included: {len(files)}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    dump_directory()
