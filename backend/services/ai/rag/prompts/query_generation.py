QUERY_GENERATION_PROMPT = """
Create ONE complementary query for searching a curated document knowledge
collection. The exact user query is always searched separately by the caller.

Return ONLY valid JSON:
{
  "queries": ["one focused complementary query"]
}

Requirements:
- Preserve the user's actual intent, named entities, population, dates,
  numerical values, abbreviations, and any document-specific identifiers.
- Preserve identifiers exactly as written (for example, code IDs, section
  numbers, table numbers, figure numbers, standard numbers, or paper labels).
- Do not add words such as latest, recent, current, official or practical unless
  they are part of the user's request.
- For a named section, table, figure, recommendation, equation, standard, or
  other document reference, retain its exact label and identifier.
- Do not invent identifiers, facts, symptoms, diagnoses, or treatments.
- Avoid a near-duplicate of the original query.
- The query must be independently understandable.
"""
