from services.llm_provider import generate_response

from services.ai.prompts.memory_consolidator_prompt import PROMPT

from models import LongTermMemory

from sqlalchemy.orm import Session


class MemoryConsolidator:

    def consolidate(

        self,

        previous_memory,

        summary,

        reflections

    ):

        ##################################################
        # Build Prompt
        ##################################################

        prompt = self.build_prompt(

            previous_memory,

            summary,

            reflections

        )

        ##################################################
        # LLM
        ##################################################

        memory = generate_response(

            prompt

        ).strip()

        print("\n===== Long-Term Memory =====")

        print(

            memory

        )

        print("============================\n")

        return memory

    def build_prompt(

        self,

        previous_memory,

        summary,

        reflections

    ):

        ##################################################
        # Previous Memory
        ##################################################

        if previous_memory is None:

            previous_memory = "None"

        ##################################################
        # Reflections
        ##################################################

        if len(reflections) == 0:

            reflections_text = "None"

        else:

            reflections_text = ""

            for reflection in reflections:

                reflections_text += (

                    f"- {reflection.reflection}\n"

                )

        ##################################################
        # Prompt
        ##################################################

        return PROMPT.format(

            memory=previous_memory,

            summary=summary,

            reflections=reflections_text

        )

    def load(

        self,

        db: Session,

        user_id

    ):

        memory = db.query(

            LongTermMemory

        ).filter(

            LongTermMemory.user_id == user_id

        ).first()

        if memory is None:

            return None

        return memory.memory

    def save(

        self,

        db: Session,

        user_id,

        memory

    ):

        existing = db.query(

            LongTermMemory

        ).filter(

            LongTermMemory.user_id == user_id

        ).first()

        ##################################################
        # Update Existing Memory
        ##################################################

        if existing:

            existing.memory = memory

            db.commit()

            db.refresh(

                existing

            )

            print("\n===== Long-Term Memory Updated =====")

            print(

                existing.memory

            )

            print("====================================\n")

            return

        ##################################################
        # Create New Memory
        ##################################################

        new_memory = LongTermMemory(

            user_id=user_id,

            memory=memory

        )

        db.add(

            new_memory

        )

        db.commit()

        db.refresh(

            new_memory

        )

        print("\n===== Long-Term Memory Created =====")

        print(

            new_memory.memory

        )

        print("====================================\n")