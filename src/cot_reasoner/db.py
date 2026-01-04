"""SQLite database module for persisting reasoning results."""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

from cot_reasoner.core.chain import ReasoningChain

# Default database path - project folder
# __file__ = src/cot_reasoner/db.py → .parent.parent.parent = cot_reasoner/
DEFAULT_DB_PATH = Path(__file__).parent.parent.parent / "data" / "reasoning.db"


class Database:
    """SQLite database for storing reasoning results."""

    def __init__(self, db_path: Optional[Path] = None):
        """Initialize database connection.

        Args:
            db_path: Path to SQLite database file. Defaults to ~/.cot_reasoner/reasoning.db
        """
        self.db_path = db_path or DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize database schema."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS reasoning_results (
                    id TEXT PRIMARY KEY,
                    query TEXT NOT NULL,
                    answer TEXT,
                    confidence REAL DEFAULT 0.0,
                    provider TEXT,
                    model TEXT,
                    strategy TEXT,
                    total_tokens INTEGER DEFAULT 0,
                    steps_json TEXT,
                    metadata_json TEXT,
                    status TEXT DEFAULT 'completed',
                    error TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create index for faster queries
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_created_at
                ON reasoning_results(created_at DESC)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_status
                ON reasoning_results(status)
            """)

            conn.commit()

    @contextmanager
    def _get_connection(self):
        """Get a database connection context manager."""
        conn = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def save_result(
        self,
        result_id: str,
        chain: Optional[ReasoningChain] = None,
        status: str = "completed",
        error: Optional[str] = None,
        query: Optional[str] = None,
    ) -> str:
        """Save a reasoning result to database.

        Args:
            result_id: Unique identifier for the result
            chain: ReasoningChain object (optional if saving pending/failed status)
            status: Status of the reasoning (pending, completed, failed)
            error: Error message if status is failed
            query: Query text (required if chain is None)

        Returns:
            The result_id
        """
        with self._get_connection() as conn:
            if chain:
                steps_json = json.dumps([s.to_dict() for s in chain.steps])
                metadata_json = json.dumps(chain.metadata)

                conn.execute("""
                    INSERT OR REPLACE INTO reasoning_results
                    (id, query, answer, confidence, provider, model, strategy,
                     total_tokens, steps_json, metadata_json, status, error, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    result_id,
                    chain.query,
                    chain.answer,
                    chain.confidence,
                    chain.provider,
                    chain.model,
                    chain.strategy,
                    chain.total_tokens,
                    steps_json,
                    metadata_json,
                    status,
                    error,
                    datetime.now(),
                ))
            else:
                # Save pending or failed status without full chain
                conn.execute("""
                    INSERT OR REPLACE INTO reasoning_results
                    (id, query, status, error, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    result_id,
                    query or "",
                    status,
                    error,
                    datetime.now(),
                ))

            conn.commit()

        return result_id

    def get_result(self, result_id: str) -> Optional[dict]:
        """Get a reasoning result by ID.

        Args:
            result_id: The result identifier

        Returns:
            Dict with result data or None if not found
        """
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM reasoning_results WHERE id = ?",
                (result_id,)
            ).fetchone()

            if not row:
                return None

            return self._row_to_dict(row)

    def get_recent_results(self, limit: int = 10, status: Optional[str] = None) -> list[dict]:
        """Get recent reasoning results.

        Args:
            limit: Maximum number of results to return
            status: Filter by status (optional)

        Returns:
            List of result dictionaries
        """
        with self._get_connection() as conn:
            if status:
                rows = conn.execute(
                    "SELECT * FROM reasoning_results WHERE status = ? ORDER BY created_at DESC LIMIT ?",
                    (status, limit)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM reasoning_results ORDER BY created_at DESC LIMIT ?",
                    (limit,)
                ).fetchall()

            return [self._row_to_dict(row) for row in rows]

    def search_results(self, query: str, limit: int = 10) -> list[dict]:
        """Search reasoning results by query text.

        Args:
            query: Search term
            limit: Maximum results

        Returns:
            List of matching results
        """
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM reasoning_results WHERE query LIKE ? ORDER BY created_at DESC LIMIT ?",
                (f"%{query}%", limit)
            ).fetchall()

            return [self._row_to_dict(row) for row in rows]

    def delete_result(self, result_id: str) -> bool:
        """Delete a reasoning result.

        Args:
            result_id: The result identifier

        Returns:
            True if deleted, False if not found
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM reasoning_results WHERE id = ?",
                (result_id,)
            )
            conn.commit()
            return cursor.rowcount > 0

    def get_stats(self) -> dict:
        """Get database statistics.

        Returns:
            Dict with counts and stats
        """
        with self._get_connection() as conn:
            total = conn.execute("SELECT COUNT(*) FROM reasoning_results").fetchone()[0]
            completed = conn.execute(
                "SELECT COUNT(*) FROM reasoning_results WHERE status = 'completed'"
            ).fetchone()[0]
            failed = conn.execute(
                "SELECT COUNT(*) FROM reasoning_results WHERE status = 'failed'"
            ).fetchone()[0]
            pending = conn.execute(
                "SELECT COUNT(*) FROM reasoning_results WHERE status = 'pending'"
            ).fetchone()[0]

            avg_tokens = conn.execute(
                "SELECT AVG(total_tokens) FROM reasoning_results WHERE status = 'completed'"
            ).fetchone()[0]

            return {
                "total": total,
                "completed": completed,
                "failed": failed,
                "pending": pending,
                "avg_tokens": round(avg_tokens or 0, 2),
            }

    def _row_to_dict(self, row: sqlite3.Row) -> dict:
        """Convert database row to dictionary."""
        data = dict(row)

        # Parse JSON fields
        if data.get("steps_json"):
            data["steps"] = json.loads(data["steps_json"])
        else:
            data["steps"] = []
        del data["steps_json"]

        if data.get("metadata_json"):
            data["metadata"] = json.loads(data["metadata_json"])
        else:
            data["metadata"] = {}
        del data["metadata_json"]

        # Convert timestamps to ISO format strings
        if data.get("created_at"):
            if isinstance(data["created_at"], str):
                pass  # Already string
            else:
                data["created_at"] = data["created_at"].isoformat()

        if data.get("updated_at"):
            if isinstance(data["updated_at"], str):
                pass
            else:
                data["updated_at"] = data["updated_at"].isoformat()

        return data


# Global database instance
_db: Optional[Database] = None


def get_db() -> Database:
    """Get the global database instance."""
    global _db
    if _db is None:
        _db = Database()
    return _db


def init_db(db_path: Optional[Path] = None) -> Database:
    """Initialize the global database instance.

    Args:
        db_path: Optional custom database path

    Returns:
        Database instance
    """
    global _db
    _db = Database(db_path)
    return _db
