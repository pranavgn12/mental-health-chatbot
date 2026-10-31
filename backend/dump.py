
from pathlib import Path
import zipfile

# Project root = directory where this script is located.
ROOT = Path(__file__).resolve().parent

FILES = [
    # Backend: episode lifecycle
    "services/ai/orchestrator.py",
    "services/ai/long_term_memory.py",
    "services/ai/memory_consolidator.py",
    "services/ai/episode_manager.py",
    "services/ai/episode_detector.py",
    "services/ai/episode_summarizer.py",

    # Backend: API
    "routers/memory.py",
    "routers/messages.py",
    "main.py",
    "models.py",

    # Frontend: Profile and chat
    "src/pages/Chat.jsx",
    "src/hooks/useChat.js",
    "src/services/messages.js",
]

OUTPUT = ROOT / "profile_debug_dump.zip"

with zipfile.ZipFile(OUTPUT, "w", zipfile.ZIP_DEFLATED) as archive:

    for relative_path in FILES:

        path = ROOT / relative_path

        if not path.is_file():
            print(f"[MISSING] {relative_path}")
            continue

        archive.write(path, arcname=relative_path)
        print(f"[ADDED] {relative_path}")

print("\nDump created:", OUTPUT)
