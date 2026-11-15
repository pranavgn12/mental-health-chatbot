Pattern analysis + 10-record limit redesign
===========================================

What changed
------------
1. PatternMemoryAnalyzer uses only episode.summary as pattern evidence. It
   supplies the current episode plus up to eight previous CLOSED episodes.
   It does not read messages, mood observations, or reflections for pattern
   detection.
2. The analyzer supplies at most 10 existing patterns to the LLM.
3. Evidence uses real episode_id values and short quotes copied from the
   corresponding episode summary. Python verifies that each quote occurs in
   the referenced summary, allowing harmless case and whitespace differences.
   The model's optional reason is explanatory text, not evidence.
4. New recurring patterns require evidence from at least two distinct episode
   summaries, including the current episode. Resolution requires an existing
   pattern plus a quote from the current summary. Reactivation requires current
   and prior occurrence evidence.
5. Invalid candidates are skipped individually. Existing pattern descriptions
   and categories remain authoritative when updating by pattern_id.
6. The database is trimmed to at most 10 EpisodePattern records per user.
   Ranking is by most recent last_seen_at first, then ACTIVE status, then
   confidence. Records outside the retained 10 are deleted. Episode summaries
   remain untouched and can support rediscovery later.
7. LongTermMemory now makes a separate memory-generation request using the
   current episode summary, current mood observations, current reflections,
   previous long-term memory, and the final validated pattern list.
8. Pattern changes are committed separately so a memory-generation failure
   does not roll them back. The existing Profile generator is retained.

No database migration is required. The implementation uses the existing
EpisodePattern fields: first_seen_at, last_seen_at, evidence, status, etc.

Files included
--------------
- backend/services/ai/pattern_memory_analyzer.py
- backend/services/ai/long_term_memory.py
- backend/services/ai/pattern_detector.py (updated module description only;
  legacy helper functions remain available)
- backend/tests/test_pattern_memory_analyzer.py
- backend/tests/test_strict_memory_validation.py

Apply
-----
From the project root:

  unzip -o pattern_limit10_patch.zip -d .

Or apply the unified patch:

  patch -p1 < pattern_limit10_patch.patch

Run focused tests from backend/:

  python -m unittest tests.test_pattern_memory_analyzer tests.test_strict_memory_validation -v

Verification
------------
The two focused test modules contain 22 tests and passed in an isolated test
harness. The harness uses stubs for the database models/provider; it does not
run a live MySQL transaction or call your configured LLM provider. Restart the
backend and test by closing an episode. Then verify that the pattern table has
at most 10 records for that user and that long-term memory generation receives
moods, reflections, and the final validated patterns.

LLM call count
--------------
There are now two core calls: one for pattern analysis and one for long-term
memory generation. The existing Profile.generate() step is still retained and
can make an additional LLM call when non-empty long-term memory is available.
