"""User and tenant management for Streamlit app.

Provides a lightweight SQLite-backed user store with:
- roles (admin, org_admin, team_member, individual)
- organizations and teams
- password hashing with PBKDF2-HMAC-SHA256
"""

from __future__ import annotations

import hashlib
import os
import secrets
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def hash_password(password: str, *, salt: Optional[str] = None, rounds: int = 200_000) -> str:
    """Return a PBKDF2 password hash encoded as algo$rounds$salt$hash."""
    if not password:
        raise ValueError("Password cannot be empty")
    salt_hex = salt or secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(salt_hex),
        rounds,
    )
    return f"pbkdf2_sha256${rounds}${salt_hex}${dk.hex()}"


def verify_password(password: str, encoded_hash: str) -> bool:
    """Verify a plain password against a PBKDF2 encoded hash."""
    try:
        algo, rounds_s, salt_hex, hash_hex = encoded_hash.split("$")
        if algo != "pbkdf2_sha256":
            return False
        rounds = int(rounds_s)
    except (ValueError, AttributeError):
        return False

    candidate = hash_password(password, salt=salt_hex, rounds=rounds)
    return secrets.compare_digest(candidate, encoded_hash)


class UserStore:
    """Persistent SQLite store for users, organizations, and teams."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS organizations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS teams (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    organization_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE (organization_id, name),
                    FOREIGN KEY (organization_id) REFERENCES organizations(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL UNIQUE,
                    full_name TEXT NOT NULL,
                    role TEXT NOT NULL,
                    password_hash TEXT NOT NULL,
                    organization_id INTEGER,
                    team_id INTEGER,
                    is_active INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (organization_id) REFERENCES organizations(id) ON DELETE SET NULL,
                    FOREIGN KEY (team_id) REFERENCES teams(id) ON DELETE SET NULL,
                    CHECK (role IN ('admin', 'org_admin', 'team_member', 'individual'))
                );
                """
            )

    def ensure_admin(self, username: str, password: str, full_name: str = "Platform Administrator") -> Dict[str, Any]:
        """Create an admin account if it does not already exist."""
        existing = self.get_user_by_username(username)
        if existing:
            return existing
        return self.create_user(
            username=username,
            password=password,
            full_name=full_name,
            role="admin",
            organization_id=None,
            team_id=None,
        )

    def create_organization(self, name: str) -> int:
        if not name.strip():
            raise ValueError("Organization name is required")
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO organizations (name, created_at) VALUES (?, ?)",
                (name.strip(), _utc_now()),
            )
            return int(cur.lastrowid)

    def create_team(self, organization_id: int, name: str) -> int:
        if not name.strip():
            raise ValueError("Team name is required")
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO teams (organization_id, name, created_at) VALUES (?, ?, ?)",
                (organization_id, name.strip(), _utc_now()),
            )
            return int(cur.lastrowid)

    def create_user(
        self,
        *,
        username: str,
        password: str,
        full_name: str,
        role: str,
        organization_id: Optional[int],
        team_id: Optional[int],
    ) -> Dict[str, Any]:
        username = username.strip().lower()
        full_name = full_name.strip()
        role = role.strip()

        if not username:
            raise ValueError("Username is required")
        if not full_name:
            raise ValueError("Full name is required")
        if role not in {"admin", "org_admin", "team_member", "individual"}:
            raise ValueError("Invalid role")
        if len(password) < 8:
            raise ValueError("Password must be at least 8 characters")

        if role == "admin":
            organization_id = None
            team_id = None
        elif role == "individual":
            team_id = None

        now = _utc_now()
        with self._conn() as conn:
            cur = conn.execute(
                """
                INSERT INTO users (
                    username, full_name, role, password_hash,
                    organization_id, team_id, is_active, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?)
                """,
                (
                    username,
                    full_name,
                    role,
                    hash_password(password),
                    organization_id,
                    team_id,
                    now,
                    now,
                ),
            )
            user_id = int(cur.lastrowid)
        user = self.get_user_by_id(user_id)
        if not user:
            raise RuntimeError("Failed to fetch created user")
        return user

    def list_organizations(self) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, name, created_at FROM organizations ORDER BY name"
            ).fetchall()
        return [dict(row) for row in rows]

    def list_teams(self, organization_id: Optional[int] = None) -> List[Dict[str, Any]]:
        query = (
            "SELECT t.id, t.name, t.organization_id, o.name AS organization_name, t.created_at "
            "FROM teams t JOIN organizations o ON o.id = t.organization_id"
        )
        params: Tuple[Any, ...] = ()
        if organization_id is not None:
            query += " WHERE t.organization_id = ?"
            params = (organization_id,)
        query += " ORDER BY o.name, t.name"

        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def list_users(self) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT
                    u.id,
                    u.username,
                    u.full_name,
                    u.role,
                    u.is_active,
                    u.created_at,
                    o.name AS organization_name,
                    t.name AS team_name
                FROM users u
                LEFT JOIN organizations o ON o.id = u.organization_id
                LEFT JOIN teams t ON t.id = u.team_id
                ORDER BY u.created_at DESC
                """
            ).fetchall()
        users = [dict(row) for row in rows]
        for user in users:
            user["is_active"] = bool(user["is_active"])
        return users

    def get_user_by_id(self, user_id: int) -> Optional[Dict[str, Any]]:
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT
                    u.id,
                    u.username,
                    u.full_name,
                    u.role,
                    u.password_hash,
                    u.is_active,
                    u.organization_id,
                    u.team_id,
                    o.name AS organization_name,
                    t.name AS team_name,
                    u.created_at,
                    u.updated_at
                FROM users u
                LEFT JOIN organizations o ON o.id = u.organization_id
                LEFT JOIN teams t ON t.id = u.team_id
                WHERE u.id = ?
                """,
                (user_id,),
            ).fetchone()
        return dict(row) if row else None

    def get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        username = username.strip().lower()
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT
                    u.id,
                    u.username,
                    u.full_name,
                    u.role,
                    u.password_hash,
                    u.is_active,
                    u.organization_id,
                    u.team_id,
                    o.name AS organization_name,
                    t.name AS team_name,
                    u.created_at,
                    u.updated_at
                FROM users u
                LEFT JOIN organizations o ON o.id = u.organization_id
                LEFT JOIN teams t ON t.id = u.team_id
                WHERE u.username = ?
                """,
                (username,),
            ).fetchone()
        return dict(row) if row else None

    def authenticate(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        user = self.get_user_by_username(username)
        if not user or not bool(user.get("is_active")):
            return None
        if not verify_password(password, user["password_hash"]):
            return None

        # Do not return password hash to caller
        user = dict(user)
        user.pop("password_hash", None)
        user["is_active"] = bool(user["is_active"])
        return user

    def set_user_active(self, user_id: int, active: bool) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE users SET is_active = ?, updated_at = ? WHERE id = ?",
                (1 if active else 0, _utc_now(), user_id),
            )
