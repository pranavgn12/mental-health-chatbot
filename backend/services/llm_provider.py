
import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-20b"
)


def _build_messages(contents, system_instruction=None):
    """Build the messages sent to the Groq API."""
    messages = []

    if system_instruction:
        messages.append({
            "role": "system",
            "content": system_instruction
        })

    if isinstance(contents, str):
        messages.append({
            "role": "user",
            "content": contents
        })
    else:
        messages.extend(contents)

    return messages


def generate_response(contents, system_instruction=None):
    """
    Generate a complete response.

    Keep this function for existing services such as episode detection,
    mood detection, reflection generation, and title generation.
    """
    messages = _build_messages(
        contents,
        system_instruction
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.3,
        max_completion_tokens=1024,
        top_p=1
    )

    return response.choices[0].message.content or ""


def generate_response_stream(contents, system_instruction=None):
    """
    Generate a response incrementally.

    Yields text chunks as they arrive from Groq instead of waiting
    for the complete response.
    """
    messages = _build_messages(
        contents,
        system_instruction
    )

    stream = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.3,
        max_completion_tokens=8000,
        top_p=1,
        stream=True
    )

    for chunk in stream:
        if not chunk.choices:
            continue

        text = chunk.choices[0].delta.content

        if text:
            yield text

