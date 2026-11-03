import logging

from services.llm_provider import generate_response
from services.ai.json_parser import JSONParser
from services.ai.prompts.mood_prompt import PROMPT

from models import EpisodeMood


logger = logging.getLogger(__name__)


# IMPORTANT:
# These labels must match the moods supported by your application
# and your existing mood database conventions.
MOOD_LABELS = {
    "happy",
    "sad",
    "angry",
    "fear",
    "disgust",
    "surprise",
    "neutral",
}

MIN_CONFIDENCE = 0.75
MAX_CONTEXT_MESSAGES = 6
MAX_TIMELINE_ENTRIES = 5


class MoodDetector:

    def __init__(self):
        self.json_parser = JSONParser()

    def update(
        self,
        db,
        episode,
        conversation,
        latest_message,
        reply,
        message_id=None,
    ):
        """
        Call once for each new user turn, usually after the assistant
        has generated its reply.

        The detector predicts the user's mood and saves it only if:
        - The result is valid.
        - Confidence meets the threshold.
        - Supporting evidence is provided.
        - The observation is linked to the exact user message.
        - The mood differs from the latest saved mood in this episode.

        Repeated consecutive labels are intentionally skipped; a later return
        to an earlier mood is saved when it follows a different mood.
        """

        if not latest_message or not latest_message.strip():
            return

        # A mood observation must be tied to one concrete user message.
        # Without the message ID, we cannot make retries idempotent.
        if message_id is None:
            logger.warning("Skipping mood persistence: user message ID was not supplied.")
            return

        try:
            # Idempotency: a retry for the same message must not create a row.
            existing = db.query(EpisodeMood).filter(
                EpisodeMood.message_id == message_id
            ).first()
            if existing is not None:
                return

            # Load the saved mood timeline.
            previous_moods = (
                db.query(EpisodeMood)
                .filter(
                    EpisodeMood.episode_id == episode.episode_id
                )
                .order_by(EpisodeMood.created_at.asc(), EpisodeMood.id.asc())
                .all()
            )

            # Only pass a small amount of recent dialogue to the LLM.
            recent_context = self._get_recent_context(
                conversation=conversation,
                latest_message=latest_message,
                reply=reply,
            )

            # Avoid putting a long mood history into every prompt.
            recent_moods = previous_moods[-MAX_TIMELINE_ENTRIES:]

            prompt = self.build_prompt(
                episode=episode,
                previous_moods=recent_moods,
                conversation=recent_context,
                latest_message=latest_message,
                reply=reply,
            )

            response = generate_response(prompt)

            if not response:
                return

            decision = self.json_parser.parse(response)

            # Reject malformed or unexpected output.
            if not isinstance(decision, dict):
                logger.debug("Mood detection returned invalid JSON.")
                return

            mood = self._normalise_mood(decision.get("mood"))

            # The model may explicitly say it cannot determine a mood.
            if mood is None:
                return

            confidence = self._normalise_confidence(
                decision.get("confidence")
            )

            # Missing or invalid confidence is not accepted.
            if confidence is None:
                return

            if confidence < MIN_CONFIDENCE:
                return

            evidence = decision.get("evidence")

            if not isinstance(evidence, str) or not evidence.strip():
                return

            # Store transitions, not repeated snapshots. A repeated mood on
            # consecutive turns is not a new event and should not clutter the
            # timeline. The next different mood will still be recorded.
            if not self._is_transition(mood, previous_moods):
                logger.debug(
                    "Skipping unchanged mood %s for episode %s",
                    mood, episode.episode_id
                )
                return

            self.save(
                db=db,
                episode=episode,
                mood=mood,
                message_id=message_id,
                confidence=confidence,
                evidence=evidence.strip(),
            )

        except Exception:
            # Avoid printing prompts and full conversation contents,
            # which can expose sensitive information in terminal logs.
            db.rollback()
            logger.exception(
                "Mood detection failed for episode %s",
                getattr(episode, "episode_id", "unknown"),
            )

    @staticmethod
    def _is_transition(mood, previous_moods):
        """True only when the predicted mood differs from the latest saved mood."""
        if not previous_moods:
            return True
        return (previous_moods[-1].mood or "").strip().lower() != mood

    def _get_recent_context(
        self,
        conversation,
        latest_message,
        reply,
    ):
        """
        Return prior dialogue only. The latest user message and reply
        are supplied separately in the prompt.

        This avoids accidentally presenting the current turn twice.
        """

        messages = []

        for item in conversation or []:
            if not isinstance(item, dict):
                continue

            role = str(item.get("role", "")).strip().lower()
            content = item.get("content", "")

            if role not in {"user", "assistant"}:
                continue

            if not isinstance(content, str) or not content.strip():
                continue

            messages.append({
                "role": role,
                "content": content.strip(),
            })

        # Remove the reply if the caller already included it.
        if (
            messages
            and messages[-1]["role"] == "assistant"
            and messages[-1]["content"] == (reply or "").strip()
        ):
            messages.pop()

        # Remove the latest user turn if the caller already included it.
        if (
            messages
            and messages[-1]["role"] == "user"
            and messages[-1]["content"] == latest_message.strip()
        ):
            messages.pop()

        return messages[-MAX_CONTEXT_MESSAGES:]

    def _normalise_mood(self, value):
        """Return a canonical mood label or None."""

        if not isinstance(value, str):
            return None

        mood = value.strip().lower()

        if mood in {"", "none", "null", "unknown", "unclear"}:
            return None

        if mood not in MOOD_LABELS:
            return None

        return mood

    def _normalise_confidence(self, value):
        """Accept numeric confidence values between 0 and 1."""

        if isinstance(value, bool):
            return None

        try:
            confidence = float(value)
        except (TypeError, ValueError):
            return None

        if not 0.0 <= confidence <= 1.0:
            return None

        return confidence

    def build_prompt(
        self,
        episode,
        previous_moods,
        conversation,
        latest_message,
        reply,
    ):
        episode_text = f"""
Episode ID: {episode.episode_id}
Title: {episode.title or "Untitled"}
Status: {episode.status}
Summary: {episode.summary or "None"}
"""

        if not previous_moods:
            moods_text = "None recorded."
        else:
            moods_text = "\n".join(
                f"{m.created_at} - {m.mood}"
                for m in previous_moods
            )

        if not conversation:
            conversation_text = "No prior messages available."
        else:
            conversation_text = "\n".join(
                f"{m['role'].capitalize()}: {m['content']}"
                for m in conversation
            )

        return PROMPT.format(
            allowed_moods=", ".join(sorted(MOOD_LABELS)),
            episode=episode_text,
            moods=moods_text,
            conversation=conversation_text,
            message=latest_message,
            reply=reply or "No assistant reply available.",
        )

    def save(self, db, episode, mood, message_id=None, confidence=None, evidence=None):
        """Persist a message-linked mood observation."""

        new_mood = EpisodeMood(
            episode_id=episode.episode_id,
            mood=mood,
            message_id=message_id,
            confidence=confidence,
            evidence=evidence,
        )

        try:
            db.add(new_mood)
            db.commit()
            db.refresh(new_mood)

            logger.debug(
                "Saved mood transition: episode=%s, mood=%s",
                episode.episode_id,
                mood,
            )

        except Exception:
            db.rollback()
            raise




####################################################################################################
