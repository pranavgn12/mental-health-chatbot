# Environment-backed feature settings

1. Copy `backend/.env.example` to `backend/.env`.
2. Fill in the database credentials and provider key for actual model calls.
3. Set the feature flags to `true` or `false`.
4. Import `get_settings` from `services.settings` to read the immutable settings snapshot.

Short-term memory is intentionally always enabled in code and has no environment switch.

The flags are configuration-only in this step. The orchestrator will begin enforcing them in the next implementation step.

Run the settings tests from the backend directory:

```bash
python -m unittest tests.test_settings -v
```
