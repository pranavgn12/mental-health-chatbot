import json
import traceback

from services.llm_provider import generate_response
from services.ai.json_parser import JSONParser
from services.ai.prompts.reflection_prompt import PROMPT
from models import EpisodeReflection


class ReflectionGenerator:

    def __init__(self):
        self.json_parser = JSONParser()

    def update(
        self,
        db,
        episode,
        conversation,
        latest_message,
        reply
    ):
        print("\n" + "=" * 60)
        print("REFLECTION DEBUG: UPDATE STARTED")
        print("=" * 60)

        try:
            # 1. Validate episode
            if episode is None:
                print("[ERROR] No active episode. Cannot generate reflection.")
                return

            print(f"[INFO] Episode ID: {episode.episode_id}")
            print(f"[INFO] Episode title: {episode.title}")
            print(f"[INFO] Latest message: {latest_message!r}")
            print(f"[INFO] Conversation messages: {len(conversation)}")

            # 2. Increment counter
            episode.messages_since_last_reflection = (
                (episode.messages_since_last_reflection or 0) + 1
            )

            db.commit()
            db.refresh(episode)

            counter = episode.messages_since_last_reflection

            print(f"[DEBUG] Reflection counter: {counter}/2")

            # 3. Wait until threshold
            if counter < 2:
                print("[INFO] Waiting for another message.")
                return

            # 4. Load previous reflections
            previous_reflections = (
                db.query(EpisodeReflection)
                .filter(
                    EpisodeReflection.episode_id == episode.episode_id
                )
                .order_by(EpisodeReflection.created_at)
                .all()
            )

            print(
                f"[INFO] Previous reflections found: "
                f"{len(previous_reflections)}"
            )

            # 5. Build prompt
            prompt = self.build_prompt(
                episode,
                previous_reflections,
                conversation,
                latest_message,
                reply
            )

            print("\n[DEBUG] Reflection prompt:")
            print(prompt)

            # 6. Call LLM
            print("\n[INFO] Calling reflection LLM...")

            response = generate_response(prompt)

            print("\n[DEBUG] Raw LLM response:")
            print(repr(response))

            if not response or not response.strip():
                print("[ERROR] LLM returned an empty response.")
                return

            # 7. Parse JSON
            print("\n[INFO] Parsing LLM response...")

            decision = self.parse_reflection_response(response)

            print("[DEBUG] Parsed decision:", repr(decision))

            if not isinstance(decision, dict):
                print("[ERROR] Parsed response is not a JSON object.")
                return

            if "reflection" not in decision:
                print("[ERROR] Missing 'reflection' key in response.")
                return

            reflection = decision["reflection"]

            # 8. Handle no reflection
            if reflection is None:
                print(
                    "[INFO] LLM decided there is no meaningful "
                    "new insight."
                )
                self.reset_counter(db, episode)
                return

            if not isinstance(reflection, str) or not reflection.strip():
                print("[ERROR] Reflection is not a valid non-empty string.")
                return

            reflection = reflection.strip()

            # 9. Duplicate detection
            if previous_reflections:
                latest_reflection = previous_reflections[-1].reflection

                if (
                    latest_reflection
                    and latest_reflection.strip().casefold()
                    == reflection.casefold()
                ):
                    print("[INFO] Duplicate reflection detected; not saving.")
                    self.reset_counter(db, episode)
                    return

            # 10. Save reflection
            print("\n[INFO] Saving reflection to database...")

            self.save(db, episode, reflection)

            # Reset only after successful save
            self.reset_counter(db, episode)

            print("[SUCCESS] Reflection pipeline completed.")

        except Exception as exc:
            print("\n[ERROR] Reflection generation failed.")
            print(f"[ERROR] {type(exc).__name__}: {exc}")
            traceback.print_exc()

            try:
                db.rollback()
                print("[DEBUG] Database transaction rolled back.")
            except Exception:
                traceback.print_exc()

        finally:
            print("=" * 60)
            print("REFLECTION DEBUG: UPDATE FINISHED")
            print("=" * 60 + "\n")

    def parse_reflection_response(self, response):
        """
        Parse JSON safely, including responses wrapped in Markdown fences.
        """
        text = response.strip()

        if text.startswith("```"):
            lines = text.splitlines()

            if lines and lines[0].startswith("```"):
                lines = lines[1:]

            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]

            text = "\n".join(lines).strip()

            if text.lower().startswith("json"):
                text = text[4:].strip()

        try:
            result = json.loads(text)
        except json.JSONDecodeError as exc:
            print(f"[ERROR] Invalid JSON: {exc}")
            print(f"[DEBUG] Text being parsed: {text!r}")
            return None

        return result

    def reset_counter(self, db, episode):
        try:
            episode.messages_since_last_reflection = 0
            db.commit()
            db.refresh(episode)

            print("[DEBUG] Reflection counter reset to 0.")

        except Exception:
            db.rollback()
            print("[ERROR] Failed to reset reflection counter.")
            traceback.print_exc()
            raise

    def build_prompt(
        self,
        episode,
        previous_reflections,
        conversation,
        latest_message,
        reply
    ):
        episode_text = f"""
Episode ID: {episode.episode_id}

Title: {episode.title}

Status: {episode.status}

Summary:
{episode.summary if episode.summary else "None"}
"""

        if not previous_reflections:
            reflections_text = "None"
        else:
            reflections_text = "\n".join(
                f"{item.created_at} - {item.reflection}"
                for item in previous_reflections
            )

        conversation_text = "\n".join(
            f"{message['role'].capitalize()}: {message['content']}"
            for message in conversation
        )

        return PROMPT.format(
            episode=episode_text,
            reflections=reflections_text,
            conversation=conversation_text,
            message=latest_message,
            reply=reply
        )

    def save(self, db, episode, reflection):
        new_reflection = EpisodeReflection(
            episode_id=episode.episode_id,
            reflection=reflection
        )

        db.add(new_reflection)
        db.commit()
        db.refresh(new_reflection)

        print("\n===== REFLECTION SAVED =====")
        print(f"ID: {new_reflection.id}")
        print(f"Episode: {new_reflection.episode_id}")
        print(f"Reflection: {new_reflection.reflection}")
        print(f"Created at: {new_reflection.created_at}")
        print("============================\n")
