from services.llm_provider import generate_response

from services.ai.prompts.episode_summary_prompt import PROMPT


class EpisodeSummarizer:

    def summarize(

        self,

        conversation

    ):

        ##################################################
        # Build Prompt
        ##################################################

        prompt = self.build_prompt(

            conversation

        )

        ##################################################
        # LLM
        ##################################################

        summary = generate_response(

            prompt

        ).strip()

        print("\n===== Episode Summary =====")
        print(summary)
        print("===========================\n")

        return summary

    def build_prompt(

        self,

        conversation

    ):

        ##################################################
        # Conversation
        ##################################################

        conversation_text = ""

        for message in conversation:

            role = message["role"].capitalize()

            conversation_text += f"{role}: {message['content']}\n"

        ##################################################
        # Prompt
        ##################################################

        return PROMPT.format(

            conversation=conversation_text

        )

    def save(

        self,

        db,

        episode,

        summary

    ):

        ##################################################
        # Save Summary
        ##################################################

        episode.summary = summary

        db.commit()

        db.refresh(

            episode

        )

        print("\n===== Episode Summary Saved =====")

        print(

            episode.summary

        )

        print("===============================\n")