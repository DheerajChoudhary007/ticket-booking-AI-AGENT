import pytest
from pydantic import ValidationError

from app.domain.travel import MAX_PASSENGERS, TransportMode, TravelClass, TravelRequest


def test_empty_request_is_valid_partial_state():
    assert TravelRequest().model_dump(exclude_defaults=True) == {}


def test_full_request_roundtrip():
    r = TravelRequest(
        origin="Jaipur", destination="Delhi", travel_date="2026-11-22", transport_mode="train",
        passenger_count=2, departure_time_preference="morning", arrival_deadline="18:00",
        travel_class="ac_3_tier", budget=1500, seat_preference="window", special_requirements=["wheelchair"],
    )
    assert r.transport_mode is TransportMode.TRAIN and r.travel_class is TravelClass.AC_3_TIER
    assert r.arrival_deadline.hour == 18


@pytest.mark.parametrize("n", [0, -1, MAX_PASSENGERS + 1])
def test_passenger_count_bounds(n):
    with pytest.raises(ValidationError):
        TravelRequest(passenger_count=n)


@pytest.mark.parametrize("n", [1, MAX_PASSENGERS])
def test_passenger_count_edges_ok(n):
    assert TravelRequest(passenger_count=n).passenger_count == n


@pytest.mark.parametrize("b", [0, -5, 10_000_001])
def test_budget_bounds(b):
    with pytest.raises(ValidationError):
        TravelRequest(budget=b)


@pytest.mark.parametrize("field,value", [("transport_mode", "teleport"), ("travel_class", "luxury"), ("seat_preference", "roof")])
def test_invalid_enums_rejected(field, value):
    with pytest.raises(ValidationError):
        TravelRequest(**{field: value})


def test_unknown_field_rejected():
    with pytest.raises(ValidationError):
        TravelRequest(favourite_colour="blue")


@pytest.mark.parametrize(
    "raw,expected",
    [("  jaipur ", "Jaipur"), ("new   delhi", "New Delhi"), ("New Delhi", "New Delhi"), ("NYC", "NYC"), ("जयपुर", "जयपुर"), ("Delhi (NCR)", "Delhi (NCR)")],
)
def test_place_cleaning(raw, expected):
    assert TravelRequest(origin=raw).origin == expected


@pytest.mark.parametrize("bad", ["", "   ", "123", "Delhi; DROP TABLE", "<script>", "a" * 81])
def test_place_rejects_bad_input(bad):
    with pytest.raises(ValidationError):
        TravelRequest(destination=bad)


def test_special_requirements_cleaned_and_deduped():
    r = TravelRequest(special_requirements=["  Wheelchair  ", "wheelchair", "", "extra   luggage"])
    assert r.special_requirements == ["Wheelchair", "extra luggage"]


def test_special_requirements_limits():
    with pytest.raises(ValidationError):
        TravelRequest(special_requirements=["x" * 101])
    with pytest.raises(ValidationError):
        TravelRequest(special_requirements=[f"req {i}" for i in range(11)])


def test_validate_assignment_enforced():
    r = TravelRequest()
    with pytest.raises(ValidationError):
        r.passenger_count = 50
