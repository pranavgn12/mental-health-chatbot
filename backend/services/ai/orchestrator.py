
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
from services.ai.memory_consolidator import MemoryConsolidator

from services.ai.mood_detector import MoodDetector
from services.ai.reflection_generator import ReflectionGenerator

from services.ai.episode_summarizer import EpisodeSummarizer

from models import EpisodeReflection


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
        self.memory_consolidator = MemoryConsolidator()

    def chat(
        self,
        db,
        user_id,
        conversation,
        latest_message,
        current_title
    ):
        """
        Existing non-streaming chat method.
        Preserved for backward compatibility.
        """

        # 1. Generate conversation title
        title = None

        if current_title == "New Conversation":
            title = generate_title(latest_message)

        # 2. Episode lifecycle
        episode = self.episode_manager.get_active_episode(
            db,
            user_id
        )

        decision = self.episode_detector.detect(
            episode,
            latest_message
        )

        if decision["action"] == "CREATE":
            episode = self.episode_manager.create_episode(
                db,
                user_id,
                decision["title"]
            )

        elif decision["action"] == "CLOSE":
            if episode is not None:

                # Generate episode summary
                summary = self.episode_summarizer.summarize(
                    conversation
                )

                # Save episode summary
                self.episode_summarizer.save(
                    db,
                    episode,
                    summary
                )

                # Load previous memory
                previous_memory = self.memory_consolidator.load(
                    db,
                    user_id
                )

                # Load episode reflections
                reflections = db.query(
                    EpisodeReflection
                ).filter(
                    EpisodeReflection.episode_id
                    == episode.episode_id
                ).order_by(
                    EpisodeReflection.created_at
                ).all()

                # Build updated memory
                memory = self.memory_consolidator.consolidate(
                    previous_memory,
                    summary,
                    reflections
                )

                # Save long-term memory
                self.memory_consolidator.save(
                    db,
                    user_id,
                    memory
                )

                # Close episode
                self.episode_manager.close_episode(
                    db,
                    episode
                )

            episode = None

        elif decision["action"] == "CONTINUE":
            pass

        elif decision["action"] == "NONE":
            pass

        # 3. Retrieve profile
        profile = None

        # 4. Retrieve long-term memory
        long_term_memory = self.memory_consolidator.load(
            db,
            user_id
        )

        # 5. Retrieve knowledge RAG
        knowledge = self.knowledge_rag.retrieve(
            latest_message
        )

        # 6. Generate complete response
        reply = chat(
            conversation=conversation,
            profile=profile,
            memory=long_term_memory,
            knowledge=knowledge,
            episode=episode,
            reflection=None,
            mood=None
        )

        # 7. Mood detection
        if episode is not None:
            self.mood_detector.update(
                db,
                episode,
                conversation,
                latest_message,
                reply
            )

        # 8. Reflection generation
        if episode is not None:
            self.reflection_generator.update(
                db,
                episode,
                conversation,
                latest_message,
                reply
            )

        # 9. Return existing response format
        return {
            "reply": reply,
            "title": title,
            "episode": episode
        }

    def chat_stream(
        self,
        db,
        user_id,
        conversation,
        latest_message,
        current_title
    ):
        """
        Streaming chat method.

        Yields events in these formats:

        {"type": "chunk", "content": "..."}
        {"type": "done", "title": "...", "episode_id": "..."}
        """

        # 1. Generate conversation title
        title = None

        if current_title == "New Conversation":
            title = generate_title(latest_message)

        # 2. Episode lifecycle
        episode = self.episode_manager.get_active_episode(
            db,
            user_id
        )

        decision = self.episode_detector.detect(
            episode,
            latest_message
        )

        if decision["action"] == "CREATE":
            episode = self.episode_manager.create_episode(
                db,
                user_id,
                decision["title"]
            )

        elif decision["action"] == "CLOSE":
            if episode is not None:

                # Generate episode summary
                summary = self.episode_summarizer.summarize(
                    conversation
                )

                # Save episode summary
                self.episode_summarizer.save(
                    db,
                    episode,
                    summary
                )

                # Load previous memory
                previous_memory = self.memory_consolidator.load(
                    db,
                    user_id
                )

                # Load episode reflections
                reflections = db.query(
                    EpisodeReflection
                ).filter(
                    EpisodeReflection.episode_id
                    == episode.episode_id
                ).order_by(
                    EpisodeReflection.created_at
                ).all()

                # Build updated memory
                memory = self.memory_consolidator.consolidate(
                    previous_memory,
                    summary,
                    reflections
                )

                # Save long-term memory
                self.memory_consolidator.save(
                    db,
                    user_id,
                    memory
                )

                # Close episode
                self.episode_manager.close_episode(
                    db,
                    episode
                )

            episode = None

        elif decision["action"] == "CONTINUE":
            pass

        elif decision["action"] == "NONE":
            pass

        # 3. Retrieve profile
        profile = None

        # 4. Retrieve long-term memory
        long_term_memory = self.memory_consolidator.load(
            db,
            user_id
        )

        # 5. Retrieve knowledge RAG
        knowledge = self.knowledge_rag.retrieve(
            latest_message
        )

        # 6. Stream the assistant response
        reply_parts = []

        for chunk in llm_chat_stream(
            conversation=conversation,
            profile=profile,
            memory=long_term_memory,
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

        # Combine all streamed chunks into the complete response.
        reply = "".join(reply_parts)

        # 7. Mood detection
        if episode is not None:
            self.mood_detector.update(
                db,
                episode,
                conversation,
                latest_message,
                reply
            )

        # 8. Reflection generation
        if episode is not None:
            self.reflection_generator.update(
                db,
                episode,
                conversation,
                latest_message,
                reply
            )

        # 9. Signal that generation and post-processing are complete.
        yield {
            "type": "done",
            "title": title,
            "episode_id": (
                episode.episode_id
                if episode is not None
                else None
            )
        }

