
import os

from dotenv import load_dotenv
from groq import Groq
import logging
import time
import traceback

logger = logging.getLogger("llm_debug")

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


def generate_response(
    contents,
    system_instruction=None,
    *,
    max_completion_tokens=1024,
    response_format=None
):
    messages = _build_messages(contents, system_instruction)

    # Identify which service called the LLM.
    stack = traceback.extract_stack()

    caller = "unknown"

    for frame in reversed(stack[:-1]):
        if frame.name not in ("generate_response", "<module>"):
            caller = f"{frame.name} ({frame.filename}:{frame.lineno})"
            break

    request_params = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.3,
        "max_completion_tokens": max_completion_tokens,
        "top_p": 1
    }

    if response_format is not None:
        request_params["response_format"] = response_format

    print("\n========== LLM REQUEST ==========", flush=True)
    print("[LLM_DEBUG] Caller:", caller, flush=True)
    print("[LLM_DEBUG] Model:", MODEL, flush=True)
    print("[LLM_DEBUG] Message count:", len(messages), flush=True)
    print("[LLM_DEBUG] Output token limit:", max_completion_tokens, flush=True)
    print(
        "[LLM_DEBUG] Input characters:",
        sum(len(str(m.get("content", ""))) for m in messages),
        flush=True
    )
    print("[LLM_DEBUG] Sending request...", flush=True)

    start_time = time.time()

    try:
        response = client.chat.completions.create(**request_params)

        elapsed = round(time.time() - start_time, 2)

        if not response.choices:
            raise RuntimeError("LLM returned no choices.")

        choice = response.choices[0]
        content = choice.message.content or ""

        print("\n========== LLM RESPONSE ==========", flush=True)
        print("[LLM_DEBUG] Caller:", caller, flush=True)
        print("[LLM_DEBUG] Duration:", elapsed, "seconds", flush=True)
        print("[LLM_DEBUG] Finish reason:", choice.finish_reason, flush=True)
        print("[LLM_DEBUG] Response characters:", len(content), flush=True)
        print("[LLM_DEBUG] Response tokens:", response.usage.completion_tokens if response.usage else "Unknown", flush=True)
        print("[LLM_DEBUG] Response preview:", content[:500], flush=True)
        print("==================================\n", flush=True)

        if choice.finish_reason == "length":
            raise RuntimeError(
                f"LLM response truncated at {max_completion_tokens} tokens."
            )

        return content

    except Exception as exc:
        print("\n========== LLM ERROR ==========", flush=True)
        print("[LLM_DEBUG] Caller:", caller, flush=True)
        print("[LLM_DEBUG] Error:", repr(exc), flush=True)
        print(
            "[LLM_DEBUG] Duration:",
            round(time.time() - start_time, 2),
            "seconds",
            flush=True
        )
        print("================================\n", flush=True)
        raise


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

