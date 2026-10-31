
from pathlib import Path
import zipfile

ROOT = Path.cwd()

FILES = [
    # Backend
    "backend/services/ai/orchestrator.py",
    "backend/services/ai/long_term_memory.py",
    "backend/services/ai/memory_consolidator.py",
    "backend/services/ai/prompts/memory_consolidator_prompt.py",
    "backend/routers/memory.py",
    "backend/routers/messages.py",
    "backend/services/profile_service.py",
    "backend/services/memory_service.py",
    "backend/services/ai/episode_detector.py",
    "backend/services/ai/episode_manager.py",
    "backend/services/ai/episode_summarizer.py",
    "backend/models.py",
    "backend/main.py",

    # Frontend
    "frontend/src/services/messages.js",
    "frontend/src/hooks/useChat.js",
    "frontend/src/pages/Chat.jsx",
]

OUTPUT = ROOT / "profile_debug_dump.zip"

with zipfile.ZipFile(OUTPUT, "w", zipfile.ZIP_DEFLATED) as z:

    for relative in FILES:
        path = ROOT / relative

        if path.is_file():
            z.write(path, arcname=relative)
            print("[ADDED]", relative)
        else:
            print("[MISSING]", relative)

print("\nCreated:", OUTPUT)

