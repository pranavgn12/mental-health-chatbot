"""Conversational LLM entry points."""

from services.llm_provider import generate_response, generate_response_stream
from services.prompt_builder import build_contents, build_system_instruction

# Prompt instructions guide the usual length; this is a hard upper ceiling.
CHAT_MAX_COMPLETION_TOKENS =2000


def _build_chat_request(
    conversation,
    memory=None,
    knowledge=None,
    profile=None,
    episode=None,
    reflection=None,
    mood=None,
):
    system_instruction = build_system_instruction(
        memory=memory,
        knowledge=knowledge,
        profile=profile,
        episode=episode,
        reflection=reflection,
        mood=mood,
    )
    return build_contents(conversation), system_instruction


def chat(
    conversation,
    memory=None,
    knowledge=None,
    profile=None,
    episode=None,
    reflection=None,
    mood=None,
):
    """Generate one concise, non-streaming conversational response."""
    contents, system_instruction = _build_chat_request(
        conversation,
        memory=memory,
        knowledge=knowledge,
        profile=profile,
        episode=episode,
        reflection=reflection,
        mood=mood,
    )
    return generate_response(
        contents,
        system_instruction,
        max_completion_tokens=CHAT_MAX_COMPLETION_TOKENS,
    )


def chat_stream(
    conversation,
    memory=None,
    knowledge=None,
    profile=None,
    episode=None,
    reflection=None,
    mood=None,
):
    """Stream the same concise, safety-aware conversational response."""
    contents, system_instruction = _build_chat_request(
        conversation,
        memory=memory,
        knowledge=knowledge,
        profile=profile,
        episode=episode,
        reflection=reflection,
        mood=mood,
    )
    yield from generate_response_stream(
        contents,
        system_instruction,
        max_completion_tokens=CHAT_MAX_COMPLETION_TOKENS,
    )
