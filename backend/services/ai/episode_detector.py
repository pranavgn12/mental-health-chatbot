from services.llm_provider import generate_response

from services.ai.json_parser import JSONParser
from services.ai.prompts.episode_detector_prompt import PROMPT


class EpisodeDetector:

    def __init__(self):

        self.json_parser = JSONParser()

    def detect(

        self,

        active_episode,

        latest_message

    ):

        prompt = self.build_prompt(

            active_episode,

            latest_message

        )

        response = generate_response(

            prompt

        )

        decision = self.json_parser.parse(

            response

        )

        print("\n===== Episode Detector =====")
        print(prompt)
        print("---------------------------")
        print(response)
        print("===========================\n")

        return decision

    def build_prompt(

        self,

        active_episode,

        latest_message

    ):

        ##################################################
        # No Active Episode
        ##################################################

        if active_episode is None:

            episode = "None"

        ##################################################
        # Active Episode
        ##################################################

        else:

            episode = f"""
Episode ID: {active_episode.episode_id}

Title: {active_episode.title}

Status: {active_episode.status}

Summary:
{active_episode.summary if active_episode.summary else "None"}
"""

        return PROMPT.format(

            episode=episode,

            message=latest_message

        )