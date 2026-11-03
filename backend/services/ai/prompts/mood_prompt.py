PROMPT = """
You are a cautious mood detector for a conversational assistant.

Your task is to identify the emotional state expressed by the USER
in their latest message.

Allowed mood labels:
{allowed_moods}

EPISODE CONTEXT:
{episode}

PREVIOUSLY RECORDED MOOD TIMELINE:
{moods}

RECENT CONVERSATION BEFORE THE LATEST TURN:
{conversation}

LATEST USER MESSAGE:
{message}

ASSISTANT REPLY (context only):
{reply}

RULES:
1. Classify the user's current expressed mood, not the assistant's mood.
2. Use recent messages to understand context, references, sarcasm,
   and ambiguous statements.
3. Base the decision primarily on the user's own words.
4. Do not assume the user feels the same way as the assistant suggests.
5. Do not infer a mental-health diagnosis or a persistent personality trait.
6. Do not invent feelings that are not supported by the conversation.
7. The saved timeline represents mood transitions, not a snapshot on every turn.
   If the latest recorded mood is still the best description of the user's
   current emotional state, return null for mood rather than repeating it.
8. Record a mood when the user's emotional state meaningfully changes, including
   a shift from fear to confidence, sadness to relief, or anxiety to calm.
9. If the message is ambiguous, purely factual, or lacks enough evidence,
   return null for mood.
10. Select exactly one allowed label. Do not create new labels.
11. Confidence must be between 0.0 and 1.0. Treat it as an estimate,
    not a mathematically calibrated probability.
12. Evidence must briefly identify the user's words or context supporting
    the classification. Never use the assistant's reply as evidence.

Return ONLY a valid JSON object, without Markdown:

{{
    "mood": "one allowed label or null",
    "confidence": 0.0,
    "evidence": "brief evidence from the user's messages"
}}
"""

####################################################################################################
