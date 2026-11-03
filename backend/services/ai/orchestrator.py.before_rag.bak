
from services.llm import (
    chat,
    chat_stream as llm_chat_stream
)

from services.title_generator import generate_title

from services.ai.episode_manager import EpisodeManager
from services.ai.episode_detector import EpisodeDetector

from services.ai.short_term_memory import ShortTermMemory
from services.ai.long_term_memory import LongTermMemory
from services.ai.memory_rag import MemoryRAG
from services.ai.knowledge_rag import KnowledgeRAG
from services.ai.profile import Profile

from services.ai.pattern_detector import PatternDetector

from services.ai.mood_detector import MoodDetector
from services.ai.reflection_generator import ReflectionGenerator

from services.ai.episode_summarizer import EpisodeSummarizer


class AIOrchestrator:

    def __init__(self):

        self.episode_manager = EpisodeManager()
        self.episode_detector = EpisodeDetector()

        self.mood_detector = MoodDetector()
        self.reflection_generator = ReflectionGenerator()

        self.episode_summarizer = EpisodeSummarizer()

        self.short_term_memory = ShortTermMemory()
        self.long_term_memory = LongTermMemory()

        self.memory_rag = MemoryRAG()
        self.knowledge_rag = KnowledgeRAG()

        self.profile = Profile()

        self.pattern_detector = PatternDetector()

    # =====================================================
    # EPISODE LIFECYCLE
    # =====================================================

    def _handle_episode_lifecycle(
        self,
        db,
        user_id,
        conversation,
        latest_message,
        episode,
        decision
    ):

        long_term_memory_updated = False
        closed_episode_id = None

        action = decision.get("action")

        # -------------------------------------------------
        # CREATE EPISODE
        # -------------------------------------------------

        if action == "CREATE":

            episode = self.episode_manager.create_episode(
                db,
                user_id,
                decision["title"]
            )

        # -------------------------------------------------
        # CLOSE EPISODE
        # -------------------------------------------------

        elif action == "CLOSE":

            if episode is not None:

                print("\n========== CLOSING EPISODE ==========")
                print("Episode:", episode.episode_id)

                # 1. Generate episode summary.
                summary = self.episode_summarizer.summarize(
                    conversation
                )

                if not summary or not summary.strip():
                    raise ValueError(
                        "Episode summarization returned an empty summary."
                    )

                # 2. Save episode summary.
                self.episode_summarizer.save(
                    db,
                    episode,
                    summary
                )

                # 3. Update cumulative long-term memory.
                # This method loads previous memory, retrieves
                # reflections, consolidates and saves the result.

                updated_memory = self.long_term_memory.update(
                    db,
                    user_id,
                    episode
                )

                if not updated_memory or not updated_memory.strip():
                    raise ValueError(
                        "Long-term memory update failed."
                    )

                print("\nLong-term memory successfully updated.")

                # 4. Preserve the closed episode ID.
                closed_episode_id = episode.episode_id

                # 5. Close episode only after memory is saved.
                self.episode_manager.close_episode(
                    db,
                    episode
                )

                # 6. Notify the frontend through the response.
                long_term_memory_updated = True

                print("Episode closed:", closed_episode_id)
                print("Long-term memory updated:", True)
                print("====================================\n")

            # No active episode remains after CLOSE.
            episode = None

        elif action == "CONTINUE":

            pass

        elif action == "NONE":

            pass

        return (
            episode,
            long_term_memory_updated,
            closed_episode_id
        )

    # =====================================================
    # NON-STREAMING CHAT
    # =====================================================

    def chat(
        self,
        db,
        user_id,
        conversation,
        latest_message,
        current_title
    ):

        # 1. Generate conversation title.
        title = None

        if current_title == "New Conversation":

            title = generate_title(
                latest_message
            )

        # 2. Get active episode.
        episode = self.episode_manager.get_active_episode(
            db,
            user_id
        )

        # 3. Detect episode action.
        decision = self.episode_detector.detect(
            episode,
            latest_message
        )

        # 4. Handle episode lifecycle.
        (
            episode,
            long_term_memory_updated,
            closed_episode_id
        ) = self._handle_episode_lifecycle(
            db=db,
            user_id=user_id,
            conversation=conversation,
            latest_message=latest_message,
            episode=episode,
            decision=decision
        )

        # 5. Retrieve current long-term memory.
        # This retrieves the newly updated memory when CLOSE occurred.
        long_term_memory_text = self.long_term_memory.retrieve(
            db,
            user_id
        )

        # 6. Retrieve knowledge.
        knowledge = self.knowledge_rag.retrieve(
            latest_message
        )

        # 7. Generate assistant response.
        reply = chat(
            conversation=conversation,
            profile=None,
            memory=long_term_memory_text,
            knowledge=knowledge,
            episode=episode,
            reflection=None,
            mood=None
        )

        # 8. Mood detection.
        if episode is not None:

            self.mood_detector.update(
                db,
                episode,
                conversation,
                latest_message,
                reply
            )

        # 9. Reflection generation.
        if episode is not None:

            self.reflection_generator.update(
                db,
                episode,
                conversation,
                latest_message,
                reply
            )

        # 10. Return response.
        return {

            "reply": reply,

            "title": title,

            "episode": episode,

            "episode_id": (
                episode.episode_id
                if episode is not None
                else None
            ),

            "closed_episode_id": closed_episode_id,

            "long_term_memory_updated": long_term_memory_updated

        }

    # =====================================================
    # STREAMING CHAT
    # =====================================================

    def chat_stream(
        self,
        db,
        user_id,
        conversation,
        latest_message,
        current_title
    ):

        # 1. Generate conversation title.
        title = None

        if current_title == "New Conversation":

            title = generate_title(
                latest_message
            )

        # 2. Get active episode.
        episode = self.episode_manager.get_active_episode(
            db,
            user_id
        )

        # 3. Detect episode action.
        decision = self.episode_detector.detect(
            episode,
            latest_message
        )

        # 4. Handle episode lifecycle.
        (
            episode,
            long_term_memory_updated,
            closed_episode_id
        ) = self._handle_episode_lifecycle(
            db=db,
            user_id=user_id,
            conversation=conversation,
            latest_message=latest_message,
            episode=episode,
            decision=decision
        )

        # 5. Retrieve latest long-term memory.
        long_term_memory_text = self.long_term_memory.retrieve(
            db,
            user_id
        )

        # 6. Retrieve knowledge.
        knowledge = self.knowledge_rag.retrieve(
            latest_message
        )

        # 7. Stream assistant response.
        reply_parts = []

        for chunk in llm_chat_stream(
            conversation=conversation,
            profile=None,
            memory=long_term_memory_text,
            knowledge=knowledge,
            episode=episode,
            reflection=None,
            mood=None
        ):

            reply_parts.append(chunk)

            yield {

                "type": "chunk",

                "content": chunk

            }

        # 8. Combine streamed response.
        reply = "".join(reply_parts)

        # 9. Mood detection.
        if episode is not None:

            self.mood_detector.update(
                db,
                episode,
                conversation,
                latest_message,
                reply
            )

        # 10. Reflection generation.
        if episode is not None:

            self.reflection_generator.update(
                db,
                episode,
                conversation,
                latest_message,
                reply
            )

        # 11. Send completion event.
        # The frontend can now refresh the Profile.
        yield {

            "type": "done",

            "title": title,

            "episode_id": (
                episode.episode_id
                if episode is not None
                else None
            ),

            "closed_episode_id": closed_episode_id,

            "long_term_memory_updated": long_term_memory_updated

        }

