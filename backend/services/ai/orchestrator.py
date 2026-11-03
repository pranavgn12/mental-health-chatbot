import logging

from services.llm import chat as llm_chat, chat_stream as llm_chat_stream
from services.title_generator import generate_title
from services.ai.episode_manager import EpisodeManager
from services.ai.episode_detector import EpisodeDetector
from services.ai.long_term_memory import LongTermMemory
from services.ai.memory_rag import MemoryRAG
from services.ai.knowledge_rag import KnowledgeRAG
from services.ai.pattern_detector import PatternDetector
from services.ai.mood_detector import MoodDetector
from services.ai.reflection_generator import ReflectionGenerator
from services.ai.episode_summarizer import EpisodeSummarizer
from models import Message

logger = logging.getLogger(__name__)


class AIOrchestrator:
    def __init__(self):
        self.episode_manager = EpisodeManager()
        self.episode_detector = EpisodeDetector()
        self.mood_detector = MoodDetector()
        self.reflection_generator = ReflectionGenerator()
        self.episode_summarizer = EpisodeSummarizer()
        self.long_term_memory = LongTermMemory()
        self.memory_rag = MemoryRAG()
        self.knowledge_rag = KnowledgeRAG()
        self.pattern_detector = PatternDetector()

    def _prepare_episode(self, db, user_id, latest_message, conversation=None):
        episode = self.episode_manager.get_active_episode(db, user_id)
        decision = self.episode_detector.detect(
            episode, latest_message, recent_messages=conversation
        )
        action = decision.get("action") if isinstance(decision, dict) else None
        if episode is None and action == "CREATE":
            episode = self.episode_manager.create_episode(
                db, user_id, decision.get("title") or "Untitled Episode"
            )
        elif episode is not None and action not in {"CONTINUE", "CLOSE"}:
            # Never create a second active episode. Fail safe to CONTINUE.
            logger.warning("Unexpected episode action %r for active episode %s", action, episode.episode_id)
            action = "CONTINUE"
        elif episode is None and action != "CREATE":
            action = "NONE"
        return episode, action

    def _assign_message(self, db, message_id, episode):
        if message_id is None:
            return
        message = db.query(Message).filter(Message.id == message_id).one_or_none()
        if message is not None:
            message.episode_id = episode.episode_id if episode else None
            db.commit()

    def _finish_turn(self, db, user_id, episode, action, conversation,
                     latest_message, reply, user_message_id):
        if episode is None:
            return None, False

        self._assign_message(db, user_message_id, episode)
        # Mood records are transition-level and linked to the exact user message.
        self.mood_detector.update(
            db, episode, conversation, latest_message, reply,
            message_id=user_message_id
        )
        self.reflection_generator.update(
            db, episode, conversation, latest_message, reply
        )

        if action != "CLOSE":
            return episode.episode_id, False

        # CLOSE means this turn belongs to the old episode; finalize only
        # after its mood and reflection have been processed.
        rows = (db.query(Message).filter(Message.episode_id == episode.episode_id)
                .order_by(Message.id).all())
        episode_transcript = [
            {"role": row.role, "content": row.content}
            for row in rows if row.role in {"user", "assistant"}
        ]
        # Assistant response is not inserted by the router until after this
        # method returns, so include it in the closing episode summary now.
        episode_transcript.append({"role": "assistant", "content": reply})
        summary = self.episode_summarizer.summarize(episode_transcript)
        if not summary or not summary.strip():
            raise ValueError("Episode summarization returned an empty summary.")
        self.episode_summarizer.save(db, episode, summary)

        # Pattern detection compares this episode with earlier episodes and
        # existing user patterns, then persists versionable pattern records.
        try:
            self.pattern_detector.detect_and_save(db, episode)
        except Exception:
            db.rollback()
            logger.exception("Pattern detection failed for episode %s", episode.episode_id)

        updated_memory = self.long_term_memory.update(db, user_id, episode)
        if not updated_memory or not updated_memory.strip():
            raise ValueError("Long-term memory update failed.")

        closed_id = episode.episode_id
        self.episode_manager.close_episode(db, episode)
        return closed_id, True

    def chat(self, db, user_id, conversation, latest_message, current_title,
             user_message_id=None):
        title = generate_title(latest_message) if current_title == "New Conversation" else None
        episode, action = self._prepare_episode(
            db, user_id, latest_message, conversation=conversation
        )
        self._assign_message(db, user_message_id, episode)
        memory = self.long_term_memory.retrieve(db, user_id)
        knowledge = self.knowledge_rag.retrieve(latest_message)
        reply = llm_chat(conversation=conversation, profile=None, memory=memory,
                     knowledge=knowledge, episode=episode, reflection=None, mood=None)
        episode_id, memory_updated = self._finish_turn(
            db, user_id, episode, action, conversation, latest_message,
            reply, user_message_id
        )
        return {
            "reply": reply, "title": title,
            "episode": episode if action != "CLOSE" else None,
            "episode_id": episode_id if action != "CLOSE" else None,
            "closed_episode_id": episode_id if action == "CLOSE" else None,
            "long_term_memory_updated": memory_updated,
        }

    def chat_stream(self, db, user_id, conversation, latest_message,
                    current_title, user_message_id=None):
        title = generate_title(latest_message) if current_title == "New Conversation" else None
        episode, action = self._prepare_episode(
            db, user_id, latest_message, conversation=conversation
        )
        self._assign_message(db, user_message_id, episode)
        memory = self.long_term_memory.retrieve(db, user_id)
        knowledge = self.knowledge_rag.retrieve(latest_message)
        parts = []
        for chunk in llm_chat_stream(conversation=conversation, profile=None,
                                     memory=memory, knowledge=knowledge,
                                     episode=episode, reflection=None, mood=None):
            parts.append(chunk)
            yield {"type": "chunk", "content": chunk}
        reply = "".join(parts)
        episode_id, memory_updated = self._finish_turn(
            db, user_id, episode, action, conversation, latest_message,
            reply, user_message_id
        )
        yield {
            "type": "done", "title": title,
            "episode_id": episode_id if action != "CLOSE" else None,
            "closed_episode_id": episode_id if action == "CLOSE" else None,
            "long_term_memory_updated": memory_updated,
        }

####################################################################################################
