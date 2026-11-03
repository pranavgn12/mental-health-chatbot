# Episode detection and mood timeline fix

This patch is based on the uploaded `backend_full_dump.txt` (2026-11-03).
It changes the active `services/ai/` implementation only.

## What changed

### Episode detection
- Explicit resolution statements now close an active episode immediately, including:
  - `I am no longer afraid of exams.`
  - `I am no longer feared of exams.`
  - `I don't fear exams anymore.`
  - `I have overcome my fear of exams.`
  - `That issue doesn't bother me anymore.`
- Tentative statements such as `I'm not sure if I'm no longer afraid` do not force closure.
- Explicit resolution of an unrelated named topic does not close the current episode.
- The LLM prompt now receives recent conversation context as well as the active episode title/summary.
- Invalid or incompatible LLM actions are normalized to safe actions.

### Mood detection
- The timeline now stores mood transitions, not the same mood repeatedly on every message.
- A repeated mood is skipped when it matches the latest saved mood for that episode.
- Every saved mood must have a `message_id`.
- Reprocessing the same message is idempotent; the code checks for an existing observation before calling the model, and the model schema already declares `message_id` unique.
- When the mood changes, the new mood is saved with the message that expressed that change.

## Apply

From the backend root, back up the four original files, then extract the ZIP into the backend root so the paths merge:

```bash
cp services/ai/episode_detector.py services/ai/episode_detector.py.bak
cp services/ai/mood_detector.py services/ai/mood_detector.py.bak
cp services/ai/orchestrator.py services/ai/orchestrator.py.bak
cp services/ai/prompts/episode_detector_prompt.py services/ai/prompts/episode_detector_prompt.py.bak
cp services/ai/prompts/mood_prompt.py services/ai/prompts/mood_prompt.py.bak
unzip -o episode_mood_fix.zip -d .
```

## Database migration

The SQLAlchemy model in the dump already declares `EpisodeMood.message_id` as unique, but `Base.metadata.create_all()` does not retrofit a unique index onto an existing MySQL table. Run this once after backing up the database:

```bash
python migrations/20261104_mood_transition_dedupe.py
```

The migration removes duplicate retries that share a non-NULL `message_id`, keeping the oldest row. It also compacts consecutive identical mood labels within each episode, keeping the first observation in each consecutive run. It does not infer or rewrite missing legacy message links.

## Tests

```bash
python -m unittest discover -s tests -p 'test_episode_mood_detection.py' -v
```

The tests cover explicit closure, uncertainty, unrelated topics, mood transitions, missing message IDs, and retry idempotency. They mock the LLM and database; run an end-to-end test with your actual MySQL database and LLM credentials after applying the patch.
