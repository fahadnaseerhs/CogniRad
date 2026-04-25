"""
auth.py — CogniRad Track B | Task B6
======================================
Handles student authentication: login, token verification, and logout.

Functions
---------
login_student(cms)      → str          Issue a token for a valid student.
verify_token(token)     → Student      Return the Student or raise AuthenticationError.
logout_student(token)   → bool         Invalidate the token; returns True on success.

Raises
------
AuthenticationError     Custom exception for any auth failure (bad CMS, expired
                        token, already logged-out token, etc.).

Dependencies
------------
* database  — get_student_by_cms, create_session, get_cms_from_token,
               delete_session
* secrets   — cryptographically-safe token generation (stdlib)
"""

from __future__ import annotations

import secrets

import database


# ---------------------------------------------------------------------------
# Public exception
# ---------------------------------------------------------------------------

class AuthenticationError(Exception):
    """Raised whenever an authentication or authorisation check fails."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def login_student(cms: str) -> str:
    """
    Verify that *cms* belongs to a known, active student and issue a session
    token.

    Parameters
    ----------
    cms:
        The student's CMS identifier (e.g. ``"CMS001"``).

    Returns
    -------
    str
        A URL-safe random token (32 bytes → 43 hex chars) that the client
        must supply on every subsequent request.

    Raises
    ------
    AuthenticationError
        If *cms* is not found in the database.
    """
    student = await database.get_student_by_cms(cms)
    if student is None:
        raise AuthenticationError(
            f"Student '{cms}' not found. "
            "Please check your CMS ID and try again."
        )

    token = secrets.token_urlsafe(32)

    # Invalidate any pre-existing session for this student so only one
    # active session exists at a time.
    await database.create_session(token, cms, invalidate_existing=True)

    return token


async def verify_token(token: str) -> object:
    """
    Resolve *token* to the corresponding Student ORM object.

    Parameters
    ----------
    token:
        The opaque session token previously returned by :func:`login_student`.

    Returns
    -------
    database.Student
        The authenticated student record.

    Raises
    ------
    AuthenticationError
        If the token is not found (expired, logged out, or never issued).
    """
    cms = await database.get_cms_from_token(token)
    if cms is None:
        raise AuthenticationError(
            "Invalid or expired token. Please log in again."
        )

    student = await database.get_student_by_cms(cms)
    if student is None:
        # Token exists but the student record is gone — clean up and reject.
        await database.delete_session(token)
        raise AuthenticationError(
            "Student record not found for this token. Please log in again."
        )

    return student


async def logout_student(token: str) -> bool:
    """
    Invalidate *token*, ending the student's session.

    Parameters
    ----------
    token:
        The opaque session token to invalidate.

    Returns
    -------
    bool
        ``True`` if the token was found and deleted; ``False`` if it was
        already absent (idempotent — does not raise).
    """
    deleted = await database.delete_session(token)
    return deleted


async def get_student_channel_key(token: str) -> str | None:
    """
    Convenience helper: resolve *token* → student → channel_key string.

    Returns ``"CH-3"`` style key or ``None`` if the student is not on any
    channel.  Raises :class:`AuthenticationError` if the token is invalid.
    """
    student = await verify_token(token)
    if student.channel_id is None:
        return None
    return f"CH-{student.channel_id}"

