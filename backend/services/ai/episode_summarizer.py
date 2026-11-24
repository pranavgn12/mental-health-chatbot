import re

from services.llm_provider import generate_response
from services.ai.prompts.episode_summary_prompt import PROMPT


class EpisodeSummarizer:
    MAX_WORDS = 85
    ALLOWED_ROLES = {"user", "human", "assistant"}

    @classmethod
    def _format_conversation(cls, conversation):
        """Keep both speakers so short user replies retain their context."""
        turns = []
        for message in conversation or []:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", "")).strip().lower()
            content = str(message.get("content") or "").strip()
            if role not in cls.ALLOWED_ROLES or not content:
                continue
            # Normalize the role label while preserving the message verbatim.
            speaker = "User" if role in {"user", "human"} else "Assistant"
            turns.append(f"{speaker}: {content}")
        return "\n".join(turns)

    @classmethod
    def _is_synthesized_summary(cls, summary):
        """Validate basic output structure before the summary is persisted."""
        if not isinstance(summary, str) or not summary.strip():
            return False

        text = summary.strip()
        words = re.findall(r"\b[\w'-]+\b", text)
        if len(words) > cls.MAX_WORDS:
            return False

        # Reject dialogue/transcript formatting and generic failure placeholders.
        if re.search(r"(?im)^\s*(user|assistant|human)\s*:", text):
            return False
        if re.search(r"(?im)^\s*(user|assistant)\s*[-—]", text):
            return False
        if re.search(
            r"(?i)(summary could not be generated|unable to generate (?:a )?summary|"
            r"failed to generate (?:a )?summary|no summary available)",
            text,
        ):
            return False
        if text.startswith(("-", "*", "•", "#")) or re.search(r"(?m)^\s*\d+[.)]\s", text):
            return False

        # Count sentence boundaries while avoiding common abbreviations such as e.g.
        normalized = re.sub(
            r"\b(?:e\.g|i\.e|mr|mrs|ms|dr|prof|vs)\.",
            lambda match: match.group(0).replace(".", ""),
            text,
            flags=re.IGNORECASE,
        )
        sentence_parts = re.split(r"(?<=[.!?])\s+(?=[\"'“‘(A-Z0-9])", normalized)
        sentence_count = len([part for part in sentence_parts if part.strip()])
        if sentence_count < 2 or sentence_count > 4:
            return False
        if not re.search(r"[.!?][\"'”’)]?$", text):
            return False
        return True

    def summarize(self, conversation):
        formatted_conversation = self._format_conversation(conversation)
        if not formatted_conversation:
            raise ValueError("Cannot summarize an episode without conversation messages.")
        if not any(
            line.startswith("User:") for line in formatted_conversation.splitlines()
        ):
            raise ValueError("Cannot summarize an episode without user messages.")

        prompt = PROMPT.format(conversation=formatted_conversation)
        raw_summary = generate_response(prompt)
        summary = raw_summary.strip() if isinstance(raw_summary, str) else ""

        # Retry only when the model ignores the summary constraints.
        if not self._is_synthesized_summary(summary):
            retry_prompt = (
                prompt
                + "\n\nIMPORTANT CORRECTION: Return only 2-4 synthesized "
                  "third-person sentences, under 85 words. Do not reproduce "
                  "messages or use speaker labels."
            )
            raw_summary = generate_response(retry_prompt)
            summary = raw_summary.strip() if isinstance(raw_summary, str) else ""

        if not self._is_synthesized_summary(summary):
            raise ValueError("The model did not return a concise synthesized episode summary.")

        print("\n===== Episode Summary =====")
        print(summary)
        print("===========================\n")
        return summary

    def save(self, db, episode, summary):
        episode.summary = summary
        db.commit()
        db.refresh(episode)
        print("\n===== Episode Summary Saved =====")
        print(episode.summary)
        print("===============================\n")
