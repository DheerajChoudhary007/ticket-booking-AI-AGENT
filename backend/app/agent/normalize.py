"""Turns untrusted LLM output (TravelExtraction) into validated TravelRequest fields plus issues.

Each field is validated in isolation: a bad value yields a FieldIssue for that field only and the rest
of the extraction is kept. Dates are resolved deterministically where possible (see app.domain.dates).
"""
import re
from collections.abc import Callable
from datetime import date, time
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ValidationError

from app.agent.schemas import TravelExtraction
from app.domain.dates import DateError, check_travel_date, parse_iso_date, resolve_relative_date
from app.domain.issues import FieldIssue, IssueCode
from app.domain.travel import DepartureTimePreference, SeatPreference, TransportMode, TravelClass, TravelRequest


class NormalizedExtraction(BaseModel):
    updates: TravelRequest  # only the fields found in this message
    issues: list[FieldIssue]


_MODE_SYNONYMS = {
    "plane": "flight", "air": "flight", "airplane": "flight", "aeroplane": "flight", "flights": "flight",
    "rail": "train", "railway": "train", "trains": "train",
    "coach": "bus", "buses": "bus",
}
_CLASS_SYNONYMS = {
    "3a": "ac_3_tier", "3ac": "ac_3_tier", "ac3": "ac_3_tier", "ac_3": "ac_3_tier", "3_tier_ac": "ac_3_tier",
    "2a": "ac_2_tier", "2ac": "ac_2_tier", "ac2": "ac_2_tier", "ac_2": "ac_2_tier", "2_tier_ac": "ac_2_tier",
    "1a": "ac_1_tier", "1ac": "ac_1_tier", "ac1": "ac_1_tier", "ac_1": "ac_1_tier", "1_tier_ac": "ac_1_tier",
    "sl": "sleeper", "cc": "chair_car", "premium": "premium_economy", "eco": "economy",
}
_DEPARTURE_SYNONYMS = {"noon": "afternoon", "late_night": "night", "midnight": "night", "dawn": "early_morning"}
_SEAT_SYNONYMS = {"lower": "lower_berth", "upper": "upper_berth", "lb": "lower_berth", "ub": "upper_berth"}


def _to_enum(enum_cls: type[StrEnum], text: str, synonyms: dict[str, str]) -> StrEnum:
    key = re.sub(r"[\s\-/]+", "_", text.strip().lower())
    for suffix in ("_class", "_seat"):
        key = key.removesuffix(suffix)
    try:
        return enum_cls(synonyms.get(key, key))
    except ValueError:
        raise ValueError(f"'{text}' is not one of: {', '.join(m.value for m in enum_cls)}") from None


_HHMM = re.compile(r"([01]?\d|2[0-3]):([0-5]\d)(?::[0-5]\d)?")


def _parse_hhmm(text: str) -> time:
    m = _HHMM.fullmatch(text.strip())
    if not m:
        raise ValueError(f"'{text}' is not a valid time (expected 24-hour HH:MM)")
    return time(int(m.group(1)), int(m.group(2)))


_CONVERTERS: dict[str, Callable[[Any], Any]] = {
    "origin": lambda v: v,
    "destination": lambda v: v,
    "passenger_count": lambda v: v,
    "budget": lambda v: v,
    "special_requirements": lambda v: v,
    "arrival_deadline": _parse_hhmm,
    "transport_mode": lambda v: _to_enum(TransportMode, v, _MODE_SYNONYMS),
    "travel_class": lambda v: _to_enum(TravelClass, v, _CLASS_SYNONYMS),
    "departure_time_preference": lambda v: _to_enum(DepartureTimePreference, v, _DEPARTURE_SYNONYMS),
    "seat_preference": lambda v: _to_enum(SeatPreference, v, _SEAT_SYNONYMS),
}


def _is_blank(value: object) -> bool:
    return value is None or (isinstance(value, str) and not value.strip()) or value == []


def _error_message(field: str, exc: ValueError) -> str:
    if isinstance(exc, ValidationError):
        detail = exc.errors()[0]["msg"].removeprefix("Value error, ")
    else:
        detail = str(exc)
    return f"Invalid {field.replace('_', ' ')}: {detail}"


def _resolve_date(raw: TravelExtraction, today: date) -> tuple[date | None, FieldIssue | None]:
    """Deterministic resolution of the user's own words wins over the LLM's ISO date; the LLM's
    ISO date is the fallback for explicit dates ('5 Dec'). Either way the result is range-checked."""
    text = (raw.travel_date_text or "").strip()
    iso = (raw.travel_date or "").strip()
    if not text and not iso:
        return None, None
    try:
        resolved = resolve_relative_date(text, today) if text else None
        if resolved is None:
            if raw.date_is_ambiguous:
                return None, FieldIssue(
                    field="travel_date",
                    code=IssueCode.AMBIGUOUS_DATE,
                    message=f"The date '{text or iso}' can be read more than one way.",
                )
            if not iso:
                return None, FieldIssue(
                    field="travel_date",
                    code=IssueCode.AMBIGUOUS_DATE,
                    message=f"Could not work out a specific day from '{text}'.",
                )
            resolved = parse_iso_date(iso)
        check_travel_date(resolved, today)
        return resolved, None
    except DateError as exc:
        return None, FieldIssue(field="travel_date", code=exc.code, message=exc.message)


def normalize_extraction(raw: TravelExtraction, today: date) -> NormalizedExtraction:
    accepted: dict[str, Any] = {}
    issues: list[FieldIssue] = []

    for field, convert in _CONVERTERS.items():
        value = getattr(raw, field)
        if _is_blank(value):
            continue
        try:
            candidate = convert(value)
            validated = TravelRequest(**{field: candidate})  # validates this field in isolation
        except ValueError as exc:  # pydantic's ValidationError is a ValueError subclass
            issues.append(FieldIssue(field=field, code=IssueCode.INVALID_VALUE, message=_error_message(field, exc)))
        else:
            accepted[field] = getattr(validated, field)  # the cleaned value, not the raw one

    resolved_date, date_issue = _resolve_date(raw, today)
    if resolved_date is not None:
        accepted["travel_date"] = resolved_date
    if date_issue is not None:
        issues.append(date_issue)

    return NormalizedExtraction(updates=TravelRequest(**accepted), issues=issues)
