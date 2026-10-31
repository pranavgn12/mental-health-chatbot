from services.llm_provider import generate_response


SYSTEM_PROMPT = """
Generate a short title for this conversation.

Rules:
- 2 to 5 words.
- No quotes.
- No punctuation.
- No emojis.
- Return ONLY the title.
"""


def generate_title(first_message: str):

    contents = [

        {

            "role": "user",

            "content": first_message

        }

    ]

    return generate_response(

        contents,

        SYSTEM_PROMPT

    ).strip()