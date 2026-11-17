import re

from services.llm_provider import generate_response
from services.ai.prompts.episode_summary_prompt import PROMPT


class EpisodeSummarizer:
    MAX_WORDS = 85

    @staticmethod
    def _user_statements(conversation):
        """Keep user-authored content only; assistant turns are not episode facts."""
        statements = []
        for message in conversation or []:
            role = str(message.get("role", "")).strip().lower()
            content = str(message.get("content", "")).strip()
            if role in {"user", "human"} and content:
                statements.append(content)
        return "\n".join(f"- {content}" for content in statements)

    @classmethod
    def _is_synthesized_summary(cls, summary):
        if not isinstance(summary, str) or not summary.strip():
            return False
        text = summary.strip()
        if len(re.findall(r"\b[\w'-]+\b", text)) > cls.MAX_WORDS:
            return False
        if re.search(r"(?im)^\s*(user|assistant|human)\s*:", text):
            return False
        if re.search(r"(?im)^\s*(user|assistant)\s*[-—]", text):
            return False
        return True

    def summarize(self, conversation):
        user_statements = self._user_statements(conversation)
        if not user_statements:
            raise ValueError("Cannot summarize an episode without user messages.")

        prompt = PROMPT.format(conversation=user_statements)
        summary = generate_response(prompt).strip()

        # Retry only when the model ignores the summary constraints.
        if not self._is_synthesized_summary(summary):
            retry_prompt = (
                prompt
                + "\n\nIMPORTANT CORRECTION: Return only 2-4 synthesized "
                  "third-person sentences, under 85 words. Do not reproduce "
                  "messages or use speaker labels."
            )
            summary = generate_response(retry_prompt).strip()

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
