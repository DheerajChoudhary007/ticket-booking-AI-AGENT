"""Core travel domain model: enums and the validated TravelRequest.

Pure Python: no LLM, no I/O. Every field is optional because a request is built up over several
conversation turns; whether it is *complete enough to search* is decided by
app.domain.validation.missing_fields, not by this model.
"""
import string
import unicodedata
from datetime import date, time
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator

MAX_PASSENGERS = 9
MAX_BUDGET = 10_000_000
MAX_SPECIAL_REQUIREMENTS = 10
MAX_REQUIREMENT_LENGTH = 100


class TransportMode(StrEnum):
    TRAIN = "train"
    BUS = "bus"
    FLIGHT = "flight"


class TravelClass(StrEnum):
    # Flights
    ECONOMY = "economy"
    PREMIUM_ECONOMY = "premium_economy"
    BUSINESS = "business"
    FIRST = "first"
    # Trains
    SLEEPER = "sleeper"  # also used by buses
    AC_3_TIER = "ac_3_tier"
    AC_2_TIER = "ac_2_tier"
    AC_1_TIER = "ac_1_tier"
    CHAIR_CAR = "chair_car"
    # Buses
    SEATER = "seater"
    AC = "ac"
    NON_AC = "non_ac"


class DepartureTimePreference(StrEnum):
    EARLY_MORNING = "early_morning"  # roughly 04:00-08:00
    MORNING = "morning"  # 08:00-12:00
    AFTERNOON = "afternoon"  # 12:00-16:00
    EVENING = "evening"  # 16:00-20:00
    NIGHT = "night"  # 20:00-04:00


class SeatPreference(StrEnum):
    WINDOW = "window"
    AISLE = "aisle"
    LOWER_BERTH = "lower_berth"
    UPPER_BERTH = "upper_berth"


_PLACE_PUNCTUATION = frozenset(" .,'-()")


def _is_place_char(ch: str) -> bool:
    # Letters and combining marks (so Devanagari etc. work) plus basic punctuation. No digits.
    return unicodedata.category(ch)[0] in {"L", "M"} or ch in _PLACE_PUNCTUATION


def _clean_place(value: str) -> str:
    cleaned = " ".join(value.split())
    if not cleaned:
        raise ValueError("must not be blank")
    has_letter = any(unicodedata.category(c)[0] == "L" for c in cleaned)
    if not has_letter or not all(_is_place_char(c) for c in cleaned):
        raise ValueError("must contain only letters and basic punctuation")
    return string.capwords(cleaned) if cleaned.islower() else cleaned


class TravelRequest(BaseModel):
    """Validated (possibly partial) travel request. Budget is per passenger, in INR."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    origin: str | None = Field(default=None, max_length=80)
    destination: str | None = Field(default=None, max_length=80)
    travel_date: date | None = None
    transport_mode: TransportMode | None = None
    passenger_count: int | None = Field(default=None, ge=1, le=MAX_PASSENGERS)
    departure_time_preference: DepartureTimePreference | None = None
    arrival_deadline: time | None = None  # latest acceptable arrival, time of day
    travel_class: TravelClass | None = None
    budget: float | None = Field(default=None, gt=0, le=MAX_BUDGET)
    seat_preference: SeatPreference | None = None
    special_requirements: list[str] = Field(default_factory=list)

    @field_validator("origin", "destination")
    @classmethod
    def _validate_place(cls, v: str | None) -> str | None:
        return None if v is None else _clean_place(v)

    @field_validator("special_requirements")
    @classmethod
    def _validate_requirements(cls, items: list[str]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for item in items:
            cleaned = " ".join(item.split())
            if not cleaned:
                continue
            if len(cleaned) > MAX_REQUIREMENT_LENGTH:
                raise ValueError(f"each requirement must be at most {MAX_REQUIREMENT_LENGTH} characters")
            if cleaned.casefold() not in seen:
                seen.add(cleaned.casefold())
                out.append(cleaned)
        if len(out) > MAX_SPECIAL_REQUIREMENTS:
            raise ValueError(f"at most {MAX_SPECIAL_REQUIREMENTS} special requirements are allowed")
        return out
