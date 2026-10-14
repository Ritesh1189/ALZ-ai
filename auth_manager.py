"""
auth_manager.py - Secure Local Authentication System for ALZ ai Resume Intelligence.
Provides email-based signup, password hashing, and user credential management using SQLite.
"""

import sqlite3
import hashlib
import os
import re
from datetime import datetime
from typing import Tuple, Optional, Dict, Any

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "users.db")


def _get_connection() -> sqlite3.Connection:
    """Returns a connection to the SQLite database, ensuring directory exists."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Initializes the users table and seeds a demo account if empty."""
    conn = _get_connection()
    with conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                full_name TEXT,
                created_at TEXT NOT NULL,
                last_login TEXT
            )
            """
        )

    # Seed default demo account if it doesn't exist
    demo_email = "admin@alzai.com"
    user = get_user_by_email(demo_email)
    if not user:
        register_user(
            email=demo_email,
            password="password123",
            full_name="ALZ ai Administrator",
        )


def _hash_password(password: str, salt: str) -> str:
    """Computes a secure PBKDF2 HMAC SHA-256 hash."""
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt.encode("utf-8"), 100000
    ).hex()


def validate_email(email: str) -> bool:
    """Checks if the given email has a valid email structure."""
    pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
    return bool(re.match(pattern, email.strip()))


def register_user(
    email: str, password: str, full_name: str = ""
) -> Tuple[bool, str]:
    """
    Registers a new user account with email and password.
    Returns (success, message).
    """
    email_clean = email.strip().lower()

    if not validate_email(email_clean):
        return False, "Please enter a valid email address."

    if len(password) < 6:
        return False, "Password must be at least 6 characters long."

    salt = os.urandom(16).hex()
    pwd_hash = _hash_password(password, salt)
    now_iso = datetime.now().isoformat()

    try:
        conn = _get_connection()
        with conn:
            conn.execute(
                """
                INSERT INTO users (email, password_hash, salt, full_name, created_at, last_login)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (email_clean, pwd_hash, salt, full_name.strip(), now_iso, now_iso),
            )
        return True, "Account registered successfully! You can now sign in."
    except sqlite3.IntegrityError:
        return False, "An account with this email address already exists."
    except Exception as e:
        return False, f"Registration failed: {str(e)}"


def authenticate_user(
    email: str, password: str
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Authenticates an existing user by email and password.
    Returns (success, message, user_dict).
    """
    email_clean = email.strip().lower()

    if not email_clean or not password:
        return False, "Email and password are required.", None

    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE email = ?", (email_clean,))
    row = cursor.fetchone()

    if not row:
        return False, "No account found with this email address.", None

    stored_hash = row["password_hash"]
    salt = row["salt"]

    test_hash = _hash_password(password, salt)
    if test_hash != stored_hash:
        return False, "Invalid password. Please check your credentials.", None

    # Update last login
    now_iso = datetime.now().isoformat()
    with conn:
        conn.execute(
            "UPDATE users SET last_login = ? WHERE id = ?", (now_iso, row["id"])
        )

    user_info = {
        "id": row["id"],
        "email": row["email"],
        "full_name": row["full_name"] or row["email"].split("@")[0].title(),
        "created_at": row["created_at"],
    }
    return True, "Login successful!", user_info


def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    """Fetches user details by email without exposing the password hash."""
    email_clean = email.strip().lower()
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, email, full_name, created_at, last_login FROM users WHERE email = ?",
        (email_clean,),
    )
    row = cursor.fetchone()
    if row:
        return dict(row)
    return None


# Initialize database upon import
init_db()
