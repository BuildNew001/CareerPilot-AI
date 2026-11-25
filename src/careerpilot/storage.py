"""Local JSON persistence for candidate preferences and profile memory."""

from __future__ import annotations

import json
from pathlib import Path

from careerpilot.config import DEFAULT_STATE_PATH
from careerpilot.models import ResumeProfile, UserPreferences


def save_user_state(
    preferences: UserPreferences,
    profile: ResumeProfile | None = None,
    state_path: Path | None = None,
) -> None:
    """Persist user preferences and optional profile to local JSON storage."""
    path = state_path or DEFAULT_STATE_PATH
    payload: dict[str, object] = {
        "preferences": preferences.model_dump(),
        "profile": profile.model_dump() if profile is not None else None,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_user_state(
    state_path: Path | None = None,
) -> tuple[UserPreferences, ResumeProfile | None]:
    """Load persisted preferences and optional profile if available."""
    path = state_path or DEFAULT_STATE_PATH
    if not path.exists():
        return UserPreferences(), None

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        prefs = UserPreferences.model_validate(raw.get("preferences") or {})
        raw_profile = raw.get("profile")
        profile = ResumeProfile.model_validate(raw_profile) if raw_profile else None
        return prefs, profile
    except Exception:
        return UserPreferences(), None


def clear_user_state(state_path: Path | None = None) -> None:
    """Remove saved local state file if it exists."""
    path = state_path or DEFAULT_STATE_PATH
    if path.exists():
        path.unlink()
