# app/services/chat_history.py
"""
SQLite-backed chat history, keyed by session_id.

Shares the same lightweight SQLite database as job tracking — no new
infrastructure to run, and it persists across restarts like everything
else in this single-instance deployment. Swap for Redis/Postgres under
the same conditions you'd swap the job store.
"""
from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager

from app.services.job_store import DB_PATH

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chat_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chat_messages_session ON chat_messages (session_id);
"""


@contextmanager
def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with _connect() as conn:
        conn.executescript(_SCHEMA)


def append_message(session_id: str, role: str, content: str) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT INTO chat_messages (session_id, role, content, created_at) "
            "VALUES (?, ?, ?, ?)",
            (session_id, role, content, time.time()),
        )


def get_history(session_id: str, max_turns: int) -> list[dict]:
    """Returns up to the last `max_turns` (user, assistant) exchanges, oldest first,
    in the {"role": ..., "content": ...} shape the OpenAI-compatible chat API expects."""
    limit = max_turns * 2  # each turn is one user + one assistant message

    with _connect() as conn:
        rows = conn.execute(
            "SELECT role, content FROM chat_messages WHERE session_id = ? "
            "ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()

    return [{"role": row["role"], "content": row["content"]} for row in reversed(rows)]


init_db()
