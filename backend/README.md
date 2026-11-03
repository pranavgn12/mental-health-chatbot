# Pattern evidence fix

This patch replaces `services/ai/pattern_detector.py`.

## What changes
- Prompt and persistence now agree: evidence is an object with `episode_id` and `detail`.
- Evidence details must exactly match a supplied source line for that episode.
- Current and up to eight prior closed episodes contribute summaries, mood observations, mood evidence, and reflections.
- At least two grounded observations are required before saving a recurring pattern.
- Duplicate evidence is not appended repeatedly when an existing pattern is updated.
- Existing pattern IDs are preserved when matching descriptions are updated.

## Apply
From the backend root:

```bash
cp services/ai/pattern_detector.py services/ai/pattern_detector.py.before_evidence_fix
unzip -o ~/Downloads/pattern_evidence_fix.zip -d .
```

No database migration is required; this uses the existing `EpisodePattern`, `EpisodeMood`, and `EpisodeReflection` models.

## Test
Run from the backend root:

```bash
python -m unittest discover -s tests -p 'test_pattern_evidence.py' -v
```

The tests cover the evidence schema mismatch, unknown episode IDs, fabricated details, and duplicate evidence. They do not call the live LLM provider or database.
