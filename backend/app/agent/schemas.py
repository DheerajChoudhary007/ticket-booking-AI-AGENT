"""What the LLM is asked to produce. Deliberately loose (strings, not enums) for most fields:
LLM output is untrusted, so app.agent.normalize validates and converts it field by field and a single
bad value never discards the rest of the extraction."""
from enum import StrEnum

from pydantic import BaseModel, Field, field_validator

from app.domain.travel import DepartureTimePreference, SeatPreference, TransportMode, TravelClass


def _values(enum_cls: type[StrEnum]) -> str:
    return ", ".join(m.value for m in enum_cls)


class Intent(StrEnum):
    """The LLM's *suggestion* of what the message is about. It is never authorization for an action:
    the workflow checks it against persisted state, and the booking tool re-validates independently."""

    NEW_TRAVEL_REQUEST = "new_travel_request"
    UPDATE_TRAVEL_REQUEST = "update_travel_request"
    PROVIDE_INFORMATION = "provide_information"  # answering a question the assistant asked
    SELECT_OPTION = "select_option"
    PROVIDE_PASSENGER_DETAILS = "provide_passenger_details"
    CONFIRM_BOOKING = "confirm_booking"
    DECLINE_BOOKING = "decline_booking"
    CHECK_BOOKING_STATUS = "check_booking_status"
    GENERAL_QUESTION = "general_question"
    OTHER = "other"


class TravelExtraction(BaseModel):
    intent: Intent = Field(
        default=Intent.OTHER,
        description=(
            "Best-fit intent. new_travel_request: starts a new trip. update_travel_request: changes or "
            "corrects trip details already known. provide_information: answers a question the assistant "
            "just asked. select_option: picks a listed travel option. provide_passenger_details: gives "
            "traveller names/ages. confirm_booking: clearly agrees to book. decline_booking: refuses or "
            "cancels. check_booking_status: asks about an existing booking. general_question: travel "
            "question not about a booking. other: greetings, small talk, anything else."
        ),
    )
    origin: str | None = Field(default=None, description="City/place the user departs from, exactly as stated.")
    destination: str | None = Field(default=None, description="City/place the user travels to, exactly as stated.")
    travel_date_text: str | None = Field(
        default=None, description="Only the user's own words for the travel date (e.g. 'tomorrow', '5 Dec')."
    )
    travel_date: str | None = Field(
        default=None, description="The travel date as YYYY-MM-DD, resolved against today's date. Null if not stated."
    )
    date_is_ambiguous: bool = Field(
        default=False,
        description="True if the date can reasonably be read more than one way (e.g. '05/06') or cannot be pinned to one day.",
    )
    transport_mode: str | None = Field(default=None, description=f"One of: {_values(TransportMode)}.")
    passenger_count: int | None = Field(
        default=None, description="Total number of travellers ('me and my wife' = 2). Null if not stated."
    )
    departure_time_preference: str | None = Field(
        default=None, description=f"One of: {_values(DepartureTimePreference)}."
    )
    arrival_deadline: str | None = Field(
        default=None, description="Latest acceptable arrival time of day as 24-hour HH:MM (e.g. '18:00')."
    )
    travel_class: str | None = Field(default=None, description=f"One of: {_values(TravelClass)}.")
    budget: float | None = Field(default=None, description="Maximum price per passenger in INR, as a number.")
    seat_preference: str | None = Field(default=None, description=f"One of: {_values(SeatPreference)}.")
    special_requirements: list[str] = Field(
        default_factory=list, description="Other needs (wheelchair, extra luggage, meals...), short phrases."
    )

    @field_validator("intent", mode="before")
    @classmethod
    def _unknown_intent_is_other(cls, v: object) -> object:
        try:
            return Intent(v)
        except (ValueError, TypeError):
            return Intent.OTHER
