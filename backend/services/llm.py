from services.llm_provider import (
    generate_response,
    generate_response_stream,
)
from services.prompt_builder import (
    SYSTEM_PROMPT,
    build_contents,
    build_system_instruction,
)


def chat(
    conversation,
    memory=None,
    knowledge=None,
    profile=None,
    episode=None,
    reflection=None,
    mood=None,
):
    """Generate a complete response from compact conversation context."""
    stable_context = build_system_instruction(
        memory=memory,
        profile=profile,
        episode=episode,
        reflection=reflection,
        mood=mood,
    )
    contents = build_contents(
        conversation,
        context=stable_context,
        knowledge=knowledge,
    )
    return generate_response(contents, SYSTEM_PROMPT)


def chat_stream(
    conversation,
    memory=None,
    knowledge=None,
    profile=None,
    episode=None,
    reflection=None,
    mood=None,
):
    """Stream the response using the same context layout as chat()."""
    stable_context = build_system_instruction(
        memory=memory,
        profile=profile,
        episode=episode,
        reflection=reflection,
        mood=mood,
    )
    contents = build_contents(
        conversation,
        context=stable_context,
        knowledge=knowledge,
    )
    yield from generate_response_stream(contents, SYSTEM_PROMPT)
