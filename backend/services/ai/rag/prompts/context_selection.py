CONTEXT_SELECTION_PROMPT = """
The retrieved context may contain paragraphs, lists and clinical tables.
Use only the supplied evidence. Preserve exact recommendation identifiers,
intervention wording, populations, qualifiers, recommendation strength and
certainty of evidence. Never move a strength or certainty value from one row
or recommendation to another. Do not infer missing cells. If a requested
value is absent, say it was not found in the retrieved sources.

For local references, cite the supplied document title/file and PDF page
number(s). For web references, cite only the supplied URL. Never invent a
source, page, recommendation ID or citation.
"""
