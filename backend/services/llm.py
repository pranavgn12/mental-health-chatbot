
from services.llm_provider import (
    generate_response,
    generate_response_stream
)

from services.prompt_builder import (
    build_contents,
    build_system_instruction
)


def chat(
    conversation,
    memory=None,
    knowledge=None,
    profile=None,
    episode=None,
    reflection=None,
    mood=None
):
    """
    Generate a complete response.
    Used by existing code that expects a normal string.
    """
    system_instruction = build_system_instruction(
        memory,
        knowledge,
        profile,
        episode,
        reflection,
        mood
    )

    contents = build_contents(conversation)

    return generate_response(
        contents,
        system_instruction
    )


def chat_stream(
    conversation,
    memory=None,
    knowledge=None,
    profile=None,
    episode=None,
    reflection=None,
    mood=None
):
    """
    Stream the response incrementally.
    Yields text chunks as they arrive from the LLM.
    """
    system_instruction = build_system_instruction(
        memory,
        knowledge,
        profile,
        episode,
        reflection,
        mood
    )

    contents = build_contents(conversation)

    yield from generate_response_stream(
        contents,
        system_instruction
    )

