import logging
import os
import time
import traceback

from dotenv import load_dotenv
from groq import Groq

load_dotenv()
logger = logging.getLogger(__name__)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")


def _build_messages(contents, system_instruction=None):
    messages = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    if isinstance(contents, str):
        messages.append({"role": "user", "content": contents})
    else:
        messages.extend(contents or [])
    return messages


def _caller_name():
    for frame in reversed(traceback.extract_stack()[:-2]):
        if frame.name not in {"generate_response", "_caller_name", "<module>"}:
            return f"{frame.name} ({os.path.basename(frame.filename)}:{frame.lineno})"
    return "unknown"


def _log_usage(caller, response, elapsed):
    usage = getattr(response, "usage", None)
    if usage is None:
        logger.info(
            "groq_usage caller=%s model=%s usage=unavailable duration_s=%.2f",
            caller, MODEL, elapsed,
        )
        return

    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0
    total_tokens = getattr(usage, "total_tokens", prompt_tokens + completion_tokens) or 0
    details = getattr(usage, "prompt_tokens_details", None)
    cached_tokens = getattr(details, "cached_tokens", 0) or 0
    cache_pct = (100.0 * cached_tokens / prompt_tokens) if prompt_tokens else 0.0

    logger.info(
        "groq_usage caller=%s model=%s prompt_tokens=%s completion_tokens=%s "
        "cached_tokens=%s cache_hit_pct=%.1f total_tokens=%s duration_s=%.2f",
        caller, MODEL, prompt_tokens, completion_tokens, cached_tokens,
        cache_pct, total_tokens, elapsed,
    )


def generate_response(
    contents,
    system_instruction=None,
    *,
    max_completion_tokens=1024,
    response_format=None,
    reasoning_effort="low"
):
    messages = _build_messages(contents, system_instruction)
    caller = _caller_name()
    request_params = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.3,
        "max_completion_tokens": max_completion_tokens,
        "top_p": 1,
        "reasoning_effort": reasoning_effort
    }
    if response_format is not None:
        request_params["response_format"] = response_format

    started = time.perf_counter()
    try:
        response = client.chat.completions.create(**request_params)
        elapsed = time.perf_counter() - started
        if not response.choices:
            raise RuntimeError("Groq returned no completion choices")

        choice = response.choices[0]
        content = choice.message.content or ""
        _log_usage(caller, response, elapsed)

        if choice.finish_reason == "length":
            raise RuntimeError(
                f"Groq response was truncated at {max_completion_tokens} completion tokens"
            )
        return content
    except Exception:
        logger.exception("Groq request failed (caller=%s, model=%s)", caller, MODEL)
        raise


def generate_response_stream(
    contents,
    system_instruction=None,
    *,
    max_completion_tokens=8000,
):
    """Yield response text chunks; streaming usage is not logged by this helper."""
    messages = _build_messages(contents, system_instruction)
    caller = _caller_name()
    started = time.perf_counter()
    try:
        stream = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.3,
            max_completion_tokens=max_completion_tokens,
            top_p=1,
            stream=True,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            content = chunk.choices[0].delta.content
            if content:
                yield content
        logger.info(
            "groq_stream_finished caller=%s model=%s duration_s=%.2f",
            caller, MODEL, time.perf_counter() - started,
        )
    except Exception:
        logger.exception("Groq streaming request failed (caller=%s, model=%s)", caller, MODEL)
        raise
