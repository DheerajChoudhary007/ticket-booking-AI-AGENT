"""Deterministic validation and completeness checks on a TravelRequest."""
from datetime import date

from app.domain.dates import DateError, check_travel_date
from app.domain.issues import FieldIssue, IssueCode
from app.domain.travel import TransportMode, TravelClass, TravelRequest

# Required before any search is allowed. Order = the order we prefer to ask about them.
MANDATORY_FIELDS: tuple[str, ...] = (
    "origin",
    "destination",
    "transport_mode",
    "travel_date",
    "passenger_count",
)

CLASSES_BY_MODE: dict[TransportMode, frozenset[TravelClass]] = {
    TransportMode.TRAIN: frozenset(
        {TravelClass.SLEEPER, TravelClass.AC_3_TIER, TravelClass.AC_2_TIER, TravelClass.AC_1_TIER, TravelClass.CHAIR_CAR}
    ),
    TransportMode.BUS: frozenset({TravelClass.SEATER, TravelClass.SLEEPER, TravelClass.AC, TravelClass.NON_AC}),
    TransportMode.FLIGHT: frozenset(
        {TravelClass.ECONOMY, TravelClass.PREMIUM_ECONOMY, TravelClass.BUSINESS, TravelClass.FIRST}
    ),
}


def missing_fields(req: TravelRequest) -> list[str]:
    """Mandatory fields that have no value yet, in preferred asking order."""
    return [f for f in MANDATORY_FIELDS if getattr(req, f) is None]


def validate_request(req: TravelRequest, today: date) -> list[FieldIssue]:
    """Cross-field and date-range checks. Safe to run on a merged, persisted request: a date that was
    valid yesterday may be invalid today."""
    issues: list[FieldIssue] = []

    if req.origin and req.destination and req.origin.casefold() == req.destination.casefold():
        issues.append(
            FieldIssue(
                field="destination",
                code=IssueCode.SAME_ORIGIN_DESTINATION,
                message="Origin and destination are the same.",
            )
        )

    if req.travel_date is not None:
        try:
            check_travel_date(req.travel_date, today)
        except DateError as exc:
            issues.append(FieldIssue(field="travel_date", code=exc.code, message=exc.message))

    if req.transport_mode and req.travel_class and req.travel_class not in CLASSES_BY_MODE[req.transport_mode]:
        allowed = ", ".join(sorted(c.value for c in CLASSES_BY_MODE[req.transport_mode]))
        issues.append(
            FieldIssue(
                field="travel_class",
                code=IssueCode.INCOMPATIBLE_CLASS,
                message=f"'{req.travel_class.value}' is not available for {req.transport_mode.value} (options: {allowed}).",
            )
        )
    return issues


def is_ready_for_search(req: TravelRequest, today: date) -> bool:
    return not missing_fields(req) and not validate_request(req, today)
