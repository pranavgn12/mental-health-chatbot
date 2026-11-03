QUERY_GENERATION_PROMPT = """
You create exactly TWO complementary search queries for a mental-health
knowledge retrieval system.

Return ONLY valid JSON:
{
  "queries": [
    "first focused search query",
    "second complementary search query"
  ]
}

Requirements:
- Preserve the user's actual intent; do not invent symptoms or diagnoses.
- Query 1 should retrieve foundational, evidence-based information about
  the main subject.
- Query 2 should retrieve a complementary practical angle, such as coping
  strategies, considerations, limitations, or when to seek professional help.
- Avoid near-duplicates and vague wording.
- Each query should be independently understandable.
- Do not include personal identifiers.
"""
