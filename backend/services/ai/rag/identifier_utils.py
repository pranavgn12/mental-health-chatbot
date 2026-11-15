"""Generic document-reference extraction for corpus-aware retrieval.

This module deliberately does not contain a domain-specific prefix list. It
recognizes common labelled references (Table 2, Section 3.1, Recommendation 4)
and code-like references (ABC12, RFC-9110, ISO 27001, Q3). Candidate identifiers
can be intersected with identifiers discovered in the indexed collection to
avoid boosting arbitrary word/number combinations.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

# Common labels used to refer to numbered structures in many document types.
# The aliases normalize to the same canonical label.
_LABEL_ALIASES = {
    "rec": "RECOMMENDATION",
    "recs": "RECOMMENDATION",
    "recommendation": "RECOMMENDATION",
    "recommendations": "RECOMMENDATION",
    "sec": "SECTION",
    "section": "SECTION",
    "sections": "SECTION",
    "subsection": "SUBSECTION",
    "subsections": "SUBSECTION",
    "chapter": "CHAPTER",
    "chapters": "CHAPTER",
    "table": "TABLE",
    "tables": "TABLE",
    "exhibit": "EXHIBIT",
    "exhibits": "EXHIBIT",
    "schedule": "SCHEDULE",
    "schedules": "SCHEDULE",
    "form": "FORM",
    "forms": "FORM",
    "volume": "VOLUME",
    "volumes": "VOLUME",
    "paragraph": "PARAGRAPH",
    "paragraphs": "PARAGRAPH",
    "subparagraph": "SUBPARAGRAPH",
    "subparagraphs": "SUBPARAGRAPH",
    "box": "BOX",
    "boxes": "BOX",
    "panel": "PANEL",
    "panels": "PANEL",
    "study": "STUDY",
    "studies": "STUDY",
    "trial": "TRIAL",
    "trials": "TRIAL",
    "cohort": "COHORT",
    "cohorts": "COHORT",
    "hypothesis": "HYPOTHESIS",
    "hypotheses": "HYPOTHESIS",
    "fig": "FIGURE",
    "figs": "FIGURE",
    "figure": "FIGURE",
    "figures": "FIGURE",
    "eq": "EQUATION",
    "equation": "EQUATION",
    "equations": "EQUATION",
    "appendix": "APPENDIX",
    "appendices": "APPENDIX",
    "annex": "ANNEX",
    "annexes": "ANNEX",
    "article": "ARTICLE",
    "articles": "ARTICLE",
    "clause": "CLAUSE",
    "clauses": "CLAUSE",
    "rule": "RULE",
    "rules": "RULE",
    "requirement": "REQUIREMENT",
    "requirements": "REQUIREMENT",
    "control": "CONTROL",
    "controls": "CONTROL",
    "item": "ITEM",
    "items": "ITEM",
    "question": "QUESTION",
    "questions": "QUESTION",
    "objective": "OBJECTIVE",
    "objectives": "OBJECTIVE",
    "algorithm": "ALGORITHM",
    "algorithms": "ALGORITHM",
    "experiment": "EXPERIMENT",
    "experiments": "EXPERIMENT",
    "dataset": "DATASET",
    "datasets": "DATASET",
    "test": "TEST",
    "tests": "TEST",
    "case": "CASE",
    "cases": "CASE",
    "part": "PART",
    "parts": "PART",
    "step": "STEP",
    "steps": "STEP",
    "protocol": "PROTOCOL",
    "protocols": "PROTOCOL",
    "standard": "STANDARD",
    "standards": "STANDARD",
    "policy": "POLICY",
    "policies": "POLICY",
    "criterion": "CRITERION",
    "criteria": "CRITERION",
    "theorem": "THEOREM",
    "theorems": "THEOREM",
    "lemma": "LEMMA",
    "lemmas": "LEMMA",
    "proposition": "PROPOSITION",
    "propositions": "PROPOSITION",
}

_LABEL_ALTERNATION = "|".join(
    re.escape(label) for label in sorted(_LABEL_ALIASES, key=len, reverse=True)
)

# Labelled numeric/alphabetic refs, e.g. "Section 2.3", "Table 4", "Appendix A".
_LABEL_REF_RE = re.compile(
    rf"(?<![A-Za-z0-9])(?P<label>{_LABEL_ALTERNATION})\s*"
    rf"(?:no\.?\s*|number\s*|#\s*)?"
    rf"(?P<value>[A-Za-z]?\d+(?:\.\d+){{0,4}}[A-Za-z]?|[A-Za-z]{{1,2}})"
    rf"(?![A-Za-z0-9])",
    re.IGNORECASE,
)

# Acronym/code refs, e.g. ABC12, R-12, RFC9110, ISO 27001, Q3.
# Prefix case is deliberately uppercase in source text; query matching is
# case-insensitive after the collection's identifier catalogue is available.
_CODE_REF_RE = re.compile(
    r"(?<![A-Za-z0-9])(?P<prefix>[A-Z]{1,12})\s*[-_ ]?"
    r"(?P<value>\d+(?:\.\d+){0,4}[A-Z]?)(?![A-Za-z0-9])",
    re.IGNORECASE,
)
# Also accept compact IDs typed in lowercase (e.g. dep1, rfc9110). Whitespace
# separated lowercase word-number pairs are deliberately not treated as IDs.
_COMPACT_CODE_RE = re.compile(
    r"(?<![A-Za-z0-9])(?P<code>[A-Za-z]{1,12}\d+(?:\.\d+){0,4}[A-Za-z]?)(?![A-Za-z0-9])",
    re.IGNORECASE,
)

# Ranges such as ABC1-ABC4, Q1 to Q5, or ISO 27001–27005.
_CODE_RANGE_RE = re.compile(
    r"(?<![A-Za-z0-9])(?P<prefix>[A-Za-z]{1,12})\s*[-_ ]?"
    r"(?P<start>\d+)\s*(?:-|–|—|\bto\b|\bthrough\b)\s*"
    r"(?:(?P<end_prefix>[A-Za-z][A-Za-z0-9]{0,11})\s*[-_ ]?)?"
    r"(?P<end>\d+)(?![A-Za-z0-9])",
    re.IGNORECASE,
)

# Labelled ranges such as Sections 2-4 or Table 1 through Table 3.
_LABEL_RANGE_RE = re.compile(
    rf"(?<![A-Za-z0-9])(?P<label>{_LABEL_ALTERNATION})s?\s*"
    r"(?:no\.?\s*|number\s*|#\s*)?(?P<start>\d+)\s*"
    r"(?:-|–|—|\bto\b|\bthrough\b)\s*"
    rf"(?:(?P<end_label>{_LABEL_ALTERNATION})s?\s*)?"
    r"(?P<end>\d+)(?![A-Za-z0-9])",
    re.IGNORECASE,
)


def _canonical_code(prefix: str, value: str) -> str:
    prefix = re.sub(r"[-_\s]", "", prefix or "").upper()
    value = re.sub(r"\s+", "", value or "").upper()
    label = _LABEL_ALIASES.get(prefix.casefold())
    return f"{label}#{value}" if label else f"{prefix}{value}"


def canonicalize_identifier(identifier: str) -> str:
    """Normalize a single identifier supplied by a caller.

    Examples: ``"dep-1"`` -> ``"DEP1"``; ``"Section 2.1"`` ->
    ``"SECTION#2.1"``. An unknown string is returned in normalized form.
    """
    value = (identifier or "").strip()
    if not value:
        return ""

    labelled = _LABEL_REF_RE.search(value)
    if labelled and labelled.group(0).strip().casefold() == value.casefold():
        return _canonical_code(labelled.group("label"), labelled.group("value"))

    code = _CODE_REF_RE.search(value.upper())
    if code and code.group(0).strip().casefold() == value.casefold():
        return _canonical_code(code.group("prefix"), code.group("value"))

    # Accept already-canonical forms such as SECTION#2.1.
    if "#" in value:
        label, ref = value.split("#", 1)
        return f"{label.strip().upper()}#{ref.strip().upper()}"
    return re.sub(r"[-_\s]", "", value).upper()


def _normalize_known(known_identifiers: Iterable[str] | None) -> set[str] | None:
    if known_identifiers is None:
        return None
    return {
        canonicalize_identifier(item)
        for item in known_identifiers
        if isinstance(item, str) and item.strip()
    }


def extract_identifiers(
    text: str,
    known_identifiers: Iterable[str] | None = None,
    *,
    expand_ranges: bool = True,
    max_range_size: int = 30,
) -> list[str]:
    """Extract generic document identifiers from text.

    If ``known_identifiers`` is supplied, candidates are restricted to IDs
    discovered in the indexed corpus. Ranges are expanded only to identifiers
    that actually exist in that corpus. Without a catalogue, ranges are
    expanded conservatively up to ``max_range_size``.
    """
    text = text or ""
    known = _normalize_known(known_identifiers)
    found: list[str] = []

    def add(value: str):
        value = canonicalize_identifier(value)
        if value and (known is None or value in known) and value not in found:
            found.append(value)

    if expand_ranges:
        for match in _CODE_RANGE_RE.finditer(text):
            prefix = match.group("prefix")
            end_prefix = match.group("end_prefix") or prefix
            start = int(match.group("start"))
            end = int(match.group("end"))
            if prefix.casefold() != end_prefix.casefold():
                continue
            if end < start or end - start > max_range_size:
                continue
            values = [
                _canonical_code(prefix, str(number))
                for number in range(start, end + 1)
            ]
            if known is not None:
                values = [value for value in values if value in known]
            found.extend(value for value in values if value not in found)

        for match in _LABEL_RANGE_RE.finditer(text):
            label = match.group("label")
            end_label = match.group("end_label") or label
            if _LABEL_ALIASES.get(label.casefold()) != _LABEL_ALIASES.get(end_label.casefold()):
                continue
            start = int(match.group("start"))
            end = int(match.group("end"))
            if end < start or end - start > max_range_size:
                continue
            values = [
                _canonical_code(label, str(number))
                for number in range(start, end + 1)
            ]
            if known is not None:
                values = [value for value in values if value in known]
            found.extend(value for value in values if value not in found)

    for match in _LABEL_REF_RE.finditer(text):
        add(_canonical_code(match.group("label"), match.group("value")))

    for match in _CODE_REF_RE.finditer(text):
        prefix = match.group("prefix")
        candidate = _canonical_code(prefix, match.group("value"))
        # Uppercase acronyms are self-identifying. A lowercase/title-case
        # spaced or hyphenated prefix is accepted only when it is a known
        # identifier in the indexed corpus (unless it is a structural label).
        if prefix.isupper() or prefix.casefold() in _LABEL_ALIASES or (known is not None and candidate in known):
            add(candidate)

    for match in _COMPACT_CODE_RE.finditer(text):
        code = match.group("code")
        # Split at the first digit; this is a compact code, not ordinary prose.
        split_at = next((i for i, char in enumerate(code) if char.isdigit()), len(code))
        add(_canonical_code(code[:split_at], code[split_at:]))

    return found


def identifier_in_text(identifier: str, text: str) -> bool:
    """Return whether text contains the same normalized identifier."""
    target = canonicalize_identifier(identifier)
    if not target:
        return False
    return target in set(extract_identifiers(text or ""))


def format_identifier(identifier: str) -> str:
    """Return a readable identifier for a retrieval query."""
    canonical = canonicalize_identifier(identifier)
    if "#" in canonical:
        label, value = canonical.split("#", 1)
        return f"{label.title()} {value}"
    return canonical


def expand_identifier_range(text: str, known_identifiers: Iterable[str] | None = None) -> list[str]:
    """Convenience alias for extracting IDs, including recognised ranges."""
    return extract_identifiers(text, known_identifiers=known_identifiers, expand_ranges=True)
