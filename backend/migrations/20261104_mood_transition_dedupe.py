"""One-time migration: deduplicate message-linked moods and enforce idempotency.

Run from the backend directory:
    python migrations/20261104_mood_transition_dedupe.py

For rows with the same non-NULL message_id, the oldest row (lowest id) is kept.
Legacy rows with NULL message_id are left untouched because their originating
messages cannot be identified safely.
"""
from sqlalchemy import inspect, text
from database import engine


def migrate():
    inspector = inspect(engine)
    if "episode_moods" not in inspector.get_table_names():
        raise RuntimeError("episode_moods table does not exist")

    with engine.begin() as connection:
        # First remove duplicate retries for one exact message, keeping the
        # earliest row so the original evidence and timestamp are preserved.
        duplicates = connection.execute(text("""
            SELECT message_id
            FROM episode_moods
            WHERE message_id IS NOT NULL
            GROUP BY message_id
            HAVING COUNT(*) > 1
        """)).scalars().all()

        removed = 0
        for message_id in duplicates:
            ids = connection.execute(text("""
                SELECT id
                FROM episode_moods
                WHERE message_id = :message_id
                ORDER BY created_at ASC, id ASC
            """), {"message_id": message_id}).scalars().all()
            for duplicate_id in ids[1:]:
                connection.execute(text(
                    "DELETE FROM episode_moods WHERE id = :id"
                ), {"id": duplicate_id})
                removed += 1

        # Then compact consecutive identical mood labels within each episode.
        # This is the new transition-based timeline policy. A later recurrence
        # is retained if another mood occurred in between.
        episode_ids = connection.execute(text(
            "SELECT DISTINCT episode_id FROM episode_moods"
        )).scalars().all()
        for episode_id in episode_ids:
            rows = connection.execute(text("""
                SELECT id, mood
                FROM episode_moods
                WHERE episode_id = :episode_id
                ORDER BY created_at ASC, id ASC
            """), {"episode_id": episode_id}).all()
            previous_mood = None
            for row in rows:
                current_mood = (row.mood or "").strip().lower()
                if current_mood and current_mood == previous_mood:
                    connection.execute(text(
                        "DELETE FROM episode_moods WHERE id = :id"
                    ), {"id": row.id})
                    removed += 1
                else:
                    previous_mood = current_mood

        inspector = inspect(engine)
        unique_columns = set()
        for constraint in inspector.get_unique_constraints("episode_moods"):
            unique_columns.update(constraint.get("column_names") or [])
        for index in inspector.get_indexes("episode_moods"):
            if index.get("unique"):
                unique_columns.update(index.get("column_names") or [])

        if "message_id" not in unique_columns:
            connection.execute(text(
                "CREATE UNIQUE INDEX uq_episode_moods_message_id "
                "ON episode_moods (message_id)"
            ))

    print(f"Migration complete. Removed {removed} duplicate message-linked mood rows.")


if __name__ == "__main__":
    migrate()
