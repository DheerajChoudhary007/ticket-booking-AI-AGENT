"""Deterministic, timezone-aware date logic. Nothing here reads a hardcoded date or the system clock
except get_now(), which takes an injectable UTC instant so tests can control "now".

Why this exists: LLMs are unreliable at calendar arithmetic, so relative phrases ("tomorrow",
"next Friday") are resolved here in code, and every date the LLM produces is re-validated here.
"""
import re
from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

from app.domain.issues import IssueCode

MAX_DAYS_AHEAD = 365


class DateError(ValueError):
    def __init__(self, code: IssueCode, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def get_now(tz_name: str, utc_now: datetime | None = None) -> datetime:
    """Current time as a timezone-aware datetime in the given IANA timezone."""
    utc_now = utc_now or datetime.now(UTC)
    if utc_now.tzinfo is None:
        raise ValueError("utc_now must be timezone-aware")
    return utc_now.astimezone(ZoneInfo(tz_name))


_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


def parse_iso_date(value: str) -> date:
    """Strict YYYY-MM-DD only (date.fromisoformat alone also accepts compact/week formats)."""
    v = value.strip()
    if not _ISO_DATE.fullmatch(v):
        raise DateError(IssueCode.INVALID_DATE, f"'{value}' is not a valid date (expected YYYY-MM-DD).")
    try:
        return date.fromisoformat(v)
    except ValueError:
        raise DateError(IssueCode.INVALID_DATE, f"'{value}' is not a real calendar date.") from None


def check_travel_date(d: date, today: date, max_days_ahead: int = MAX_DAYS_AHEAD) -> None:
    """Raise DateError unless today <= d <= today + max_days_ahead."""
    if d < today:
        raise DateError(IssueCode.DATE_IN_PAST, f"{d.isoformat()} is in the past (today is {today.isoformat()}).")
    if (d - today).days > max_days_ahead:
        raise DateError(
            IssueCode.DATE_TOO_FAR,
            f"{d.isoformat()} is more than {max_days_ahead} days away; bookings open up to {max_days_ahead} days ahead.",
        )


_WEEKDAYS = {
    "monday": 0, "mon": 0,
    "tuesday": 1, "tues": 1, "tue": 1,
    "wednesday": 2, "wed": 2,
    "thursday": 3, "thurs": 3, "thur": 3, "thu": 3,
    "friday": 4, "fri": 4,
    "saturday": 5, "sat": 5,
    "sunday": 6, "sun": 6,
}
_WEEKDAY_RE = re.compile(
    r"\b(?:(next|this|coming)\s+)?(" + "|".join(sorted(_WEEKDAYS, key=len, reverse=True)) + r")\b"
)
_IN_DAYS_RE = re.compile(r"\bin\s+(\d{1,3})\s+days?\b")
_DAY_AFTER_TOMORROW_RE = re.compile(r"\bday after tomorrow\b")
_TOMORROW_RE = re.compile(r"\b(?:tomorrow|tmrw)\b")
_TODAY_RE = re.compile(r"\b(?:today|tonight)\b")


def _resolve_weekday(qualifier: str | None, weekday: int, today: date) -> date:
    """Rules: a bare/"this" weekday is its next occurrence after today. "next <weekday>" is that
    weekday in the *following calendar (ISO) week* when the bare occurrence falls in the current week.
    Naming today's own weekday without a qualifier is genuinely ambiguous and is surfaced as such."""
    delta = (weekday - today.weekday()) % 7
    if delta == 0:
        if qualifier == "this":
            return today
        if qualifier == "next":
            return today + timedelta(days=7)
        raise DateError(
            IssueCode.AMBIGUOUS_DATE,
            f"Today ({today.isoformat()}) is already that weekday: do you mean today or "
            f"{(today + timedelta(days=7)).isoformat()}?",
        )
    target = today + timedelta(days=delta)
    if qualifier == "next" and target.isocalendar()[:2] == today.isocalendar()[:2]:
        target += timedelta(days=7)
    return target


def resolve_relative_date(text: str, today: date) -> date | None:
    """Resolve simple relative phrases. Returns None when the text is not a (single, unambiguous)
    relative phrase; the caller then falls back to the LLM-provided ISO date, which is still validated.
    Text containing digits (other than "in N days") is never resolved here: "Friday 28th" must not be
    reduced to "the next Friday"."""
    t = " ".join(text.lower().split())
    if not t:
        return None

    found: list[date] = []
    if m := _IN_DAYS_RE.search(t):
        found.append(today + timedelta(days=int(m.group(1))))
    elif re.search(r"\d", t):
        return None

    if _DAY_AFTER_TOMORROW_RE.search(t):
        found.append(today + timedelta(days=2))
    elif _TOMORROW_RE.search(t):
        found.append(today + timedelta(days=1))
    if _TODAY_RE.search(t):
        found.append(today)
    for m in _WEEKDAY_RE.finditer(t):
        found.append(_resolve_weekday(m.group(1), _WEEKDAYS[m.group(2)], today))

    distinct = set(found)
    return distinct.pop() if len(distinct) == 1 else None
