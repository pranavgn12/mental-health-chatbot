SYSTEM_PROMPT = """
You are an empathetic AI mental health assistant.

Your goals are:

• Be supportive and conversational.
• Never repeat previous conversation history.
• Answer naturally.
• If the user is emotional, acknowledge it gently.
• Do not mention internal memory or prompt instructions.
"""


def build_system_instruction(

    memory=None,
    knowledge=None,
    profile=None,
    episode=None,
    reflection=None,
    mood=None

):

    instruction = SYSTEM_PROMPT

    if profile:

        instruction += f"\n\nUser Profile:\n{profile}"

    if memory:

        instruction += f"\n\nLong-Term Memory:\n{memory}"

    if knowledge:

        instruction += f"\n\nKnowledge:\n{knowledge}"

    if episode:

        instruction += f"\n\nCurrent Episode:\n{episode}"

    if reflection:

        instruction += f"\n\nReflection:\n{reflection}"

    if mood:

        instruction += f"\n\nMood Timeline:\n{mood}"

    return instruction


def build_contents(conversation):

    contents = []

    for message in conversation:

        role = message["role"]

        if role == "model":
            role = "assistant"

        contents.append({

            "role": role,

            "content": message["content"]

        })

    return contents