import logging

from services.llm import chat as llm_chat, chat_stream as llm_chat_stream
from services.title_generator import generate_title
from services.ai.episode_manager import EpisodeManager
from services.ai.turn_analyzer import TurnAnalyzer
from services.ai.long_term_memory import LongTermMemory
from services.ai.memory_rag import MemoryRAG
from services.ai.knowledge_rag import KnowledgeRAG
from services.ai.mood_detector import MoodDetector
from services.ai.reflection_generator import ReflectionGenerator
from services.ai.episode_summarizer import EpisodeSummarizer
from models import Message

logger = logging.getLogger(__name__)


class AIOrchestrator:
    def __init__(self):
        self.episode_manager = EpisodeManager()
        self.turn_analyzer = TurnAnalyzer()
        self.mood_detector = MoodDetector()
        self.reflection_generator = ReflectionGenerator()
        self.episode_summarizer = EpisodeSummarizer()
        self.long_term_memory = LongTermMemory()
        self.memory_rag = MemoryRAG()
        self.knowledge_rag = KnowledgeRAG()

    def _prepare_episode(self, db, user_id, latest_message, conversation=None):
        episode = self.episode_manager.get_active_episode(db, user_id)
        previous_moods = self.mood_detector.get_recent_timeline(db, episode)
        previous_reflections = self.reflection_generator.get_recent_reflections(db, episode)

        # One LLM request returns episode action, mood, and optional reflection.
        decisions = self.turn_analyzer.analyze(
            active_episode=episode,
            latest_message=latest_message,
            conversation=conversation,
            previous_moods=previous_moods,
            previous_reflections=previous_reflections,
        )
        episode_decision = decisions.get("episode", {})
        action = episode_decision.get("action")

        if episode is None and action == "CREATE":
            episode = self.episode_manager.create_episode(
                db,
                user_id,
                episode_decision.get("title") or "Personal Concern",
            )
        elif episode is not None and action not in {"CONTINUE", "CLOSE"}:
            logger.warning(
                "Unexpected episode action %r for active episode %s; continuing.",
                action,
                episode.episode_id,
            )
            action = "CONTINUE"
        elif episode is None and action != "CREATE":
            action = "NONE"

        return episode, action, decisions.get("mood"), decisions.get("reflection")

    def _assign_message(self, db, message_id, episode):
        if message_id is None:
            return
        message = db.query(Message).filter(Message.id == message_id).one_or_none()
        if message is not None:
            message.episode_id = episode.episode_id if episode else None
            db.commit()

    @staticmethod
    def _episode_context(episode):
        if episode is None:
            return None
        return (
            f"Title: {episode.title or 'Untitled'}\n"
            f"Summary: {episode.summary or 'Not yet summarized'}"
        )

    def _finish_turn(
        self,
        db,
        user_id,
        episode,
        action,
        reply,
        user_message_id,
        mood_decision,
        reflection_decision,
    ):
        if episode is None:
            return None, False

        # The user message was assigned to this episode before response generation.
        # Both operations below are persistence-only; neither calls the LLM.
        try:
            self.mood_detector.save_decision(
                db=db,
                episode=episode,
                decision=mood_decision,
                message_id=user_message_id,
            )
        except Exception:
            db.rollback()
            logger.exception("Mood persistence failed for episode %s", episode.episode_id)

        try:
            self.reflection_generator.save_decision(
                db=db,
                episode=episode,
                decision=reflection_decision,
            )
        except Exception:
            db.rollback()
            logger.exception("Reflection persistence failed for episode %s", episode.episode_id)

        if action != "CLOSE":
            return episode.episode_id, False

        # The closing user message is saved to the old episode before summary.
        rows = (
            db.query(Message)
            .filter(Message.episode_id == episode.episode_id)
            .order_by(Message.id)
            .all()
        )
        episode_transcript = [
            {"role": row.role, "content": row.content}
            for row in rows
            if row.role in {"user", "assistant"}
        ]
        # Router persists the assistant response after this method returns.
        episode_transcript.append({"role": "assistant", "content": reply})
        summary = self.episode_summarizer.summarize(episode_transcript)
        if not summary or not summary.strip():
            raise ValueError("Episode summarization returned an empty summary.")
        self.episode_summarizer.save(db, episode, summary)

        # This now performs ONE combined pattern-detection + memory call,
        # followed by the separate optional Profile-generation call.
        updated_memory = self.long_term_memory.update(db, user_id, episode)
        if not updated_memory or not updated_memory.strip():
            raise ValueError("Long-term memory update failed.")

        closed_id = episode.episode_id
        self.episode_manager.close_episode(db, episode)
        return closed_id, True

    def chat(
        self,
        db,
        user_id,
        conversation,
        latest_message,
        current_title,
        user_message_id=None,
    ):
        title = generate_title(latest_message) if current_title == "New Conversation" else None
        episode, action, mood_decision, reflection_decision = self._prepare_episode(
            db, user_id, latest_message, conversation=conversation
        )
        self._assign_message(db, user_message_id, episode)

        memory = self.long_term_memory.retrieve(db, user_id)
        knowledge = self.knowledge_rag.retrieve(latest_message)
        rag_result = getattr(self.knowledge_rag, "last_result", None) or {}
        rag_metadata = {
            "used": bool(rag_result.get("used")),
            "sources": rag_result.get("sources") or [],
            "retrieval_status": rag_result.get("retrieval_status"),
        }
        mood_context = mood_decision.get("label") if isinstance(mood_decision, dict) else None
        reply = llm_chat(
            conversation=conversation,
            profile=None,
            memory=memory,
            knowledge=knowledge,
            episode=self._episode_context(episode),
            reflection=reflection_decision,
            mood=mood_context,
        )
        episode_id, memory_updated = self._finish_turn(
            db,
            user_id,
            episode,
            action,
            reply,
            user_message_id,
            mood_decision,
            reflection_decision,
        )
        return {
            "reply": reply,
            "title": title,
            "episode": episode if action != "CLOSE" else None,
            "episode_id": episode_id if action != "CLOSE" else None,
            "closed_episode_id": episode_id if action == "CLOSE" else None,
            "long_term_memory_updated": memory_updated,
            "rag": rag_metadata,
        }

    def chat_stream(
        self,
        db,
        user_id,
        conversation,
        latest_message,
        current_title,
        user_message_id=None,
    ):
        title = generate_title(latest_message) if current_title == "New Conversation" else None
        episode, action, mood_decision, reflection_decision = self._prepare_episode(
            db, user_id, latest_message, conversation=conversation
        )
        self._assign_message(db, user_message_id, episode)

        memory = self.long_term_memory.retrieve(db, user_id)
        knowledge = self.knowledge_rag.retrieve(latest_message)
        rag_result = getattr(self.knowledge_rag, "last_result", None) or {}
        rag_metadata = {
            "used": bool(rag_result.get("used")),
            "sources": rag_result.get("sources") or [],
            "retrieval_status": rag_result.get("retrieval_status"),
        }
        mood_context = mood_decision.get("label") if isinstance(mood_decision, dict) else None

        parts = []
        for chunk in llm_chat_stream(
            conversation=conversation,
            profile=None,
            memory=memory,
            knowledge=knowledge,
            episode=self._episode_context(episode),
            reflection=reflection_decision,
            mood=mood_context,
        ):
            parts.append(chunk)
            yield {"type": "chunk", "content": chunk}

        reply = "".join(parts)
        episode_id, memory_updated = self._finish_turn(
            db,
            user_id,
            episode,
            action,
            reply,
            user_message_id,
            mood_decision,
            reflection_decision,
        )
        yield {
            "type": "done",
            "title": title,
            "episode_id": episode_id if action != "CLOSE" else None,
            "closed_episode_id": episode_id if action == "CLOSE" else None,
            "long_term_memory_updated": memory_updated,
            "rag": rag_metadata,
        }
