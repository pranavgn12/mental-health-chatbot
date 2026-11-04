"""Add per-assistant-message source references.

Run from the backend directory:
    python migrations/20261104_message_references.py

The column is nullable so existing messages remain valid and display no
References action until they have source metadata.
"""
from sqlalchemy import inspect, text

from database import engine


def migrate():
    inspector = inspect(engine)
    if "messages" not in inspector.get_table_names():
        raise RuntimeError("messages table does not exist")

    columns = {column["name"] for column in inspector.get_columns("messages")}
    if "source_references" not in columns:
        with engine.begin() as connection:
            connection.execute(text(
                "ALTER TABLE messages "
                "ADD COLUMN source_references JSON NULL"
            ))
        print("Added messages.source_references.")
    else:
        print("messages.source_references already exists; nothing to do.")


if __name__ == "__main__":
    migrate()
