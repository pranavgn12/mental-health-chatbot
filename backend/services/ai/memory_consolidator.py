
import logging

from services.llm_provider import generate_response
from services.ai.prompts.memory_consolidator_prompt import PROMPT
from models import LongTermMemory
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


class MemoryConsolidator:

    def consolidate(self, previous_memory, summary, reflections):

        if not summary or not summary.strip():
            raise ValueError("Episode summary is empty.")

        prompt = self.build_prompt(
            previous_memory,
            summary,
            reflections
        )

        response = generate_response(prompt)

        if not isinstance(response, str) or not response.strip():
            raise ValueError("LLM returned an empty memory.")

        memory = response.strip()

        print("\n===== Generated Long-Term Memory =====")
        print(memory)
        print("=======================================\n")

        return memory

    def build_prompt(self, previous_memory, summary, reflections):

        previous_memory = (
            previous_memory.strip()
            if previous_memory
            else "No previous memory exists."
        )

        reflections = reflections or []

        reflections_text = "\n".join(
            f"- {r.reflection}"
            for r in reflections
            if r.reflection
        )

        if not reflections_text:
            reflections_text = "No reflections available."

        return PROMPT.format(
            memory=previous_memory,
            summary=summary.strip(),
            reflections=reflections_text
        )

    def load(self, db: Session, user_id):

        record = db.query(LongTermMemory).filter(
            LongTermMemory.user_id == user_id
        ).first()

        return record.memory if record else None

    def save(self, db: Session, user_id, memory):

        if not memory or not memory.strip():
            raise ValueError("Refusing to save empty long-term memory.")

        try:
            existing = db.query(LongTermMemory).filter(
                LongTermMemory.user_id == user_id
            ).first()

            if existing:
                existing.memory = memory.strip()
            else:
                existing = LongTermMemory(
                    user_id=user_id,
                    memory=memory.strip()
                )
                db.add(existing)

            db.commit()
            db.refresh(existing)

            print("\n===== Long-Term Memory Saved =====")
            print(existing.memory)
            print("==================================\n")

            return existing.memory

        except Exception:
            db.rollback()
            logger.exception("Failed to save long-term memory.")
            raise

