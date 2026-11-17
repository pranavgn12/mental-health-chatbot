import logging
from datetime import datetime

from services.llm import chat as llm_chat, chat_stream as llm_chat_stream
from services.title_generator import generate_title
from services.ai.episode_manager import EpisodeManager
from services.ai.turn_analyzer import TurnAnalyzer
from services.ai.long_term_memory import LongTermMemory
from services.ai.knowledge_rag import KnowledgeRAG
from services.ai.mood_detector import MoodDetector
from services.ai.reflection_generator import ReflectionGenerator
from services.ai.episode_summarizer import EpisodeSummarizer
from services.settings import get_settings
from models import Message, Episode

logger = logging.getLogger(__name__)


class AIOrchestrator:
    """Coordinates optional AI components using the central environment settings.

    Short-term conversation context remains supplied by the message route and is
    always passed to the response generator. Optional persistent components are
    only initialized and executed when their feature flags are enabled.
    """

    def __init__(self, settings=None):
        self.settings = settings or get_settings()

        # Episode lifecycle is needed by episodic memory, mood/reflection
        # persistence, or long-term consolidation. With all three disabled,
        # no episode analysis or database episode operations are performed.
        self.episode_workflow_enabled = any((
            self.settings.enable_episodic_memory,
            self.settings.enable_mood_detection,
            self.settings.enable_reflection,
            self.settings.enable_memory_consolidation,
        ))

        self.episode_manager = EpisodeManager() if self.episode_workflow_enabled else None
        self.turn_analyzer = TurnAnalyzer() if self.episode_workflow_enabled else None
        self.mood_detector = (
            MoodDetector() if self.settings.enable_mood_detection else None
        )
        self.reflection_generator = (
            ReflectionGenerator() if self.settings.enable_reflection else None
        )
        self.episode_summarizer = (
            EpisodeSummarizer()
            if self.settings.enable_episodic_memory or self.settings.enable_memory_consolidation
            else None
        )
        self.long_term_memory = (
            LongTermMemory()
            if self.settings.enable_long_term_memory or self.settings.enable_memory_consolidation
            else None
        )
        self.knowledge_rag = (
            KnowledgeRAG() if self.settings.enable_knowledge_rag else None
        )

    def _prepare_episode(self, db, user_id, latest_message, conversation=None):
        if not self.episode_workflow_enabled:
            return None, "NONE", None, None

        episode = self.episode_manager.get_active_episode(db, user_id)
        previous_moods = (
            self.mood_detector.get_recent_timeline(db, episode)
            if self.mood_detector else []
        )
        previous_reflections = (
            self.reflection_generator.get_recent_reflections(db, episode)
            if self.reflection_generator else []
        )
        decisions = self.turn_analyzer.analyze(
            active_episode=episode,
            latest_message=latest_message,
            conversation=conversation,
            previous_moods=previous_moods,
            previous_reflections=previous_reflections,
        )
        episode_decision = decisions.get("episode", {})
        action = episode_decision.get("action")

        # Only the episodic-memory feature owns episode creation/closure.
        if self.settings.enable_episodic_memory:
            if episode is None and action == "CREATE":
                episode = self.episode_manager.create_episode(
                    db, user_id, episode_decision.get("title") or "Personal Concern",
                )
            elif episode is not None and action not in {"CONTINUE", "CLOSE"}:
                logger.warning("Unexpected episode action %r for %s; continuing.", action, episode.episode_id)
                action = "CONTINUE"
            elif episode is None and action != "CREATE":
                action = "NONE"
        else:
            # Mood/reflection/consolidation can use an already-active episode,
            # but must not create a new episodic-memory record by themselves.
            action = "CONTINUE" if episode is not None else "NONE"

        mood = decisions.get("mood") if self.settings.enable_mood_detection else None
        reflection = decisions.get("reflection") if self.settings.enable_reflection else None
        return episode, action, mood, reflection

    def _assign_message(self, db, message_id, episode):
        if not self.settings.enable_episodic_memory or message_id is None:
            return
        message = db.query(Message).filter(Message.id == message_id).one_or_none()
        if message is not None:
            message.episode_id = episode.episode_id if episode else None
            db.commit()

    @staticmethod
    def _episode_context(episode):
        if episode is None:
            return None
        return f"Title: {episode.title or 'Untitled'}\nSummary: {episode.summary or 'Not yet summarized'}"

    @staticmethod
    def _fallback_summary(episode, transcript):
        lines = []
        for message in transcript:
            role = str(message.get("role", "")).strip().capitalize()
            content = str(message.get("content", "")).strip()
            if role in {"User", "Assistant"} and content:
                lines.append(f"{role}: {content}")
        body = "\n".join(lines)
        if len(body) > 6000:
            body = body[-6000:]
        title = getattr(episode, "title", None) or "Untitled episode"
        return f"Episode: {title}\n\nConversation record (automatic fallback; not AI-summarized):\n{body}"

    @staticmethod
    def _reload_episode(db, episode_id):
        return db.query(Episode).filter(Episode.episode_id == episode_id).one_or_none()

    def _finish_turn(self, db, user_id, episode, action, reply, user_message_id,
                     mood_decision, reflection_decision):
        if episode is None:
            return None, False
        episode_id = episode.episode_id

        if self.mood_detector:
            try:
                self.mood_detector.save_decision(
                    db=db, episode=episode, decision=mood_decision,
                    message_id=user_message_id,
                )
            except Exception:
                db.rollback()
                logger.exception("Mood persistence failed for %s", episode_id)
                episode = self._reload_episode(db, episode_id)

        if self.reflection_generator:
            try:
                self.reflection_generator.save_decision(
                    db=db, episode=episode, decision=reflection_decision,
                )
            except Exception:
                db.rollback()
                logger.exception("Reflection persistence failed for %s", episode_id)
                episode = self._reload_episode(db, episode_id)

        if action != "CLOSE" or not self.settings.enable_episodic_memory:
            return episode_id, False

        try:
            rows = (
                db.query(Message).filter(Message.episode_id == episode_id)
                .order_by(Message.id).all()
            )
            transcript = [
                {"role": row.role, "content": row.content}
                for row in rows if row.role in {"user", "assistant"}
            ]
        except Exception:
            db.rollback()
            logger.exception("Transcript retrieval failed for %s", episode_id)
            transcript = []
        transcript.append({"role": "assistant", "content": reply or ""})

        if self.episode_summarizer:
            try:
                summary = self.episode_summarizer.summarize(transcript)
                if not isinstance(summary, str) or not summary.strip():
                    raise ValueError("Empty episode summary.")
                summary = summary.strip()
            except Exception:
                logger.exception("Summary generation failed for %s; using fallback.", episode_id)
                summary = self._fallback_summary(episode, transcript)
            try:
                episode = self._reload_episode(db, episode_id) or episode
                self.episode_summarizer.save(db, episode, summary)
            except Exception:
                db.rollback()
                logger.exception("Summary persistence failed for %s; continuing to closure.", episode_id)

        memory_updated = False
        if self.settings.enable_memory_consolidation and self.long_term_memory:
            try:
                old_memory = self.long_term_memory.retrieve(db, user_id)
                updated_memory = self.long_term_memory.update(
                    db, user_id, self._reload_episode(db, episode_id) or episode,
                    update_patterns=self.settings.enable_pattern_detection,
                )
                memory_updated = bool(
                    updated_memory and updated_memory.strip() and updated_memory != old_memory
                )
            except Exception:
                db.rollback()
                logger.exception("Memory consolidation failed for %s; continuing to closure.", episode_id)

        try:
            episode = self._reload_episode(db, episode_id) or episode
            self.episode_manager.close_episode(db, episode)
        except Exception:
            db.rollback()
            logger.exception("EpisodeManager closure failed for %s; attempting direct closure.", episode_id)
            try:
                episode = self._reload_episode(db, episode_id)
                if episode is not None:
                    episode.status = "CLOSED"
                    episode.active_user_id = None
                    episode.closed_at = datetime.utcnow()
                    db.commit()
            except Exception:
                db.rollback()
                logger.exception("Direct closure also failed for %s.", episode_id)

        return episode_id, memory_updated

    def _retrieve_context(self, db, user_id, latest_message):
        memory = None
        if self.settings.enable_long_term_memory and self.long_term_memory:
            memory = self.long_term_memory.retrieve(db, user_id)

        knowledge = None
        rag_metadata = {"used": False, "sources": [], "retrieval_status": None}
        if self.settings.enable_knowledge_rag and self.knowledge_rag:
            knowledge = self.knowledge_rag.retrieve(latest_message)
            rag_result = getattr(self.knowledge_rag, "last_result", None) or {}
            rag_metadata = {
                "used": bool(rag_result.get("used")),
                "sources": rag_result.get("sources") or [],
                "retrieval_status": rag_result.get("retrieval_status"),
            }
        return memory, knowledge, rag_metadata

    def _generation_context(self, episode, reflection, mood):
        return {
            "episode": self._episode_context(episode)
            if self.settings.enable_episodic_memory else None,
            "reflection": reflection if self.settings.enable_reflection else None,
            "mood": (
                mood.get("label") if self.settings.enable_mood_detection and isinstance(mood, dict)
                else None
            ),
        }

    def chat(self, db, user_id, conversation, latest_message, current_title, user_message_id=None):
        title = generate_title(latest_message) if current_title == "New Conversation" else None
        episode, action, mood_decision, reflection_decision = self._prepare_episode(
            db, user_id, latest_message, conversation=conversation,
        )
        self._assign_message(db, user_message_id, episode)
        memory, knowledge, rag_metadata = self._retrieve_context(db, user_id, latest_message)
        context = self._generation_context(episode, reflection_decision, mood_decision)
        reply = llm_chat(
            conversation=conversation, profile=None, memory=memory, knowledge=knowledge,
            episode=context["episode"], reflection=context["reflection"], mood=context["mood"],
        )
        episode_id, memory_updated = self._finish_turn(
            db, user_id, episode, action, reply, user_message_id,
            mood_decision, reflection_decision,
        )
        return {
            "reply": reply, "title": title,
            "episode": episode if self.settings.enable_episodic_memory and action != "CLOSE" else None,
            "episode_id": episode_id if self.settings.enable_episodic_memory and action != "CLOSE" else None,
            "closed_episode_id": episode_id if self.settings.enable_episodic_memory and action == "CLOSE" else None,
            "long_term_memory_updated": memory_updated, "rag": rag_metadata,
        }

    def chat_stream(self, db, user_id, conversation, latest_message, current_title, user_message_id=None):
        title = generate_title(latest_message) if current_title == "New Conversation" else None
        episode, action, mood_decision, reflection_decision = self._prepare_episode(
            db, user_id, latest_message, conversation=conversation,
        )
        self._assign_message(db, user_message_id, episode)
        memory, knowledge, rag_metadata = self._retrieve_context(db, user_id, latest_message)
        context = self._generation_context(episode, reflection_decision, mood_decision)
        parts = []
        for chunk in llm_chat_stream(
            conversation=conversation, profile=None, memory=memory, knowledge=knowledge,
            episode=context["episode"], reflection=context["reflection"], mood=context["mood"],
        ):
            parts.append(chunk)
            yield {"type": "chunk", "content": chunk}
        reply = "".join(parts)
        episode_id, memory_updated = self._finish_turn(
            db, user_id, episode, action, reply, user_message_id,
            mood_decision, reflection_decision,
        )
        yield {
            "type": "done", "title": title,
            "episode_id": episode_id if self.settings.enable_episodic_memory and action != "CLOSE" else None,
            "closed_episode_id": episode_id if self.settings.enable_episodic_memory and action == "CLOSE" else None,
            "long_term_memory_updated": memory_updated, "rag": rag_metadata,
        }
