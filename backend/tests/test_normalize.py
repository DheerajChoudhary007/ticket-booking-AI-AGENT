from datetime import date, time

import pytest

from app.agent.normalize import normalize_extraction
from app.agent.schemas import Intent, TravelExtraction
from app.domain.issues import IssueCode
from app.domain.travel import DepartureTimePreference, TransportMode, TravelClass

SAT = date(2026, 11, 21)


def norm(**fields):
    return normalize_extraction(TravelExtraction(**fields), SAT)


def issue_map(result):
    return {i.field: i.code for i in result.issues}


def test_complete_request():
    r = norm(
        origin="jaipur", destination="Delhi", travel_date_text="tomorrow", travel_date="2026-11-22",
        transport_mode="Train", passenger_count=2, departure_time_preference="morning",
    )
    u = r.updates
    assert (u.origin, u.destination, u.travel_date) == ("Jaipur", "Delhi", date(2026, 11, 22))
    assert u.transport_mode is TransportMode.TRAIN and u.passenger_count == 2
    assert u.departure_time_preference is DepartureTimePreference.MORNING
    assert r.issues == []


def test_nothing_extracted_gives_empty_update_and_no_issues():
    r = norm()
    assert r.updates.model_dump(exclude_defaults=True) == {} and r.issues == []


def test_blank_strings_and_empty_lists_are_treated_as_not_provided():
    r = norm(origin="  ", destination="", transport_mode=" ", special_requirements=[], travel_date_text=" ")
    assert r.updates.model_dump(exclude_defaults=True) == {} and r.issues == []


def test_deterministic_date_overrides_wrong_llm_arithmetic():
    r = norm(travel_date_text="tomorrow", travel_date="2026-11-30")  # LLM got it wrong
    assert r.updates.travel_date == date(2026, 11, 22) and r.issues == []


def test_llm_iso_date_used_for_explicit_dates():
    r = norm(travel_date_text="5 December", travel_date="2026-12-05")
    assert r.updates.travel_date == date(2026, 12, 5)


def test_iso_date_without_text_is_accepted_and_validated():
    assert norm(travel_date="2026-12-05").updates.travel_date == date(2026, 12, 5)
    assert issue_map(norm(travel_date="2026-11-01")) == {"travel_date": IssueCode.DATE_IN_PAST}


def test_explicit_date_text_with_digits_not_hijacked_by_weekday_logic():
    r = norm(travel_date_text="Friday 4 December", travel_date="2026-12-04")
    assert r.updates.travel_date == date(2026, 12, 4)


def test_past_date_rejected_and_not_stored():
    r = norm(travel_date_text="20 November", travel_date="2026-11-20")
    assert r.updates.travel_date is None
    assert issue_map(r) == {"travel_date": IssueCode.DATE_IN_PAST}


def test_far_future_date_rejected():
    assert issue_map(norm(travel_date="2028-01-01")) == {"travel_date": IssueCode.DATE_TOO_FAR}


def test_hallucinated_invalid_iso_date_rejected():
    assert issue_map(norm(travel_date="2026-02-30")) == {"travel_date": IssueCode.INVALID_DATE}
    assert issue_map(norm(travel_date="soon")) == {"travel_date": IssueCode.INVALID_DATE}


def test_llm_ambiguity_flag_blocks_date():
    r = norm(travel_date_text="05/06", travel_date="2026-06-05", date_is_ambiguous=True)
    assert r.updates.travel_date is None and issue_map(r) == {"travel_date": IssueCode.AMBIGUOUS_DATE}


def test_relative_phrase_beats_spurious_ambiguity_flag():
    r = norm(travel_date_text="tomorrow", date_is_ambiguous=True)
    assert r.updates.travel_date == date(2026, 11, 22) and r.issues == []


def test_same_weekday_as_today_is_ambiguous():
    r = norm(travel_date_text="saturday", travel_date="2026-11-21")
    assert r.updates.travel_date is None and issue_map(r) == {"travel_date": IssueCode.AMBIGUOUS_DATE}


def test_date_text_that_cannot_be_pinned_down_is_flagged_not_silently_dropped():
    assert issue_map(norm(travel_date_text="next month")) == {"travel_date": IssueCode.AMBIGUOUS_DATE}


@pytest.mark.parametrize(
    "mode,expected", [("plane", "flight"), ("Rail", "train"), ("coach", "bus"), ("FLIGHT", "flight"), ("bus", "bus")]
)
def test_mode_synonyms(mode, expected):
    assert norm(transport_mode=mode).updates.transport_mode.value == expected


@pytest.mark.parametrize(
    "cls,expected",
    [("3A", "ac_3_tier"), ("AC 2 tier", "ac_2_tier"), ("1AC", "ac_1_tier"), ("SL", "sleeper"), ("CC", "chair_car"),
     ("Economy class", "economy"), ("premium economy", "premium_economy"), ("non-ac", "non_ac")],
)
def test_class_synonyms(cls, expected):
    assert norm(travel_class=cls).updates.travel_class is TravelClass(expected)


def test_invalid_field_is_isolated_and_others_survive():
    r = norm(origin="Jaipur", destination="Delhi", transport_mode="teleport", passenger_count=2)
    assert r.updates.origin == "Jaipur" and r.updates.passenger_count == 2
    assert r.updates.transport_mode is None
    assert issue_map(r) == {"transport_mode": IssueCode.INVALID_VALUE}
    assert "teleport" in r.issues[0].message


@pytest.mark.parametrize("n", [0, -2, 10])
def test_passenger_count_out_of_range_flagged(n):
    r = norm(passenger_count=n)
    assert r.updates.passenger_count is None and issue_map(r) == {"passenger_count": IssueCode.INVALID_VALUE}


def test_bad_place_flagged():
    r = norm(origin="12345", destination="Delhi")
    assert r.updates.origin is None and r.updates.destination == "Delhi"
    assert issue_map(r) == {"origin": IssueCode.INVALID_VALUE}


@pytest.mark.parametrize("t,expected", [("18:00", time(18, 0)), ("6:30", time(6, 30)), ("23:59", time(23, 59)), ("07:05:00", time(7, 5))])
def test_arrival_deadline_parsing(t, expected):
    assert norm(arrival_deadline=t).updates.arrival_deadline == expected


@pytest.mark.parametrize("t", ["6pm", "24:00", "18:60", "evening", "1800"])
def test_arrival_deadline_invalid_flagged(t):
    assert issue_map(norm(arrival_deadline=t)) == {"arrival_deadline": IssueCode.INVALID_VALUE}


def test_budget_seat_and_requirements():
    r = norm(budget=1500, seat_preference="Window seat", special_requirements=["wheelchair", "Wheelchair"])
    assert r.updates.budget == 1500 and r.updates.seat_preference.value == "window"
    assert r.updates.special_requirements == ["wheelchair"]
    assert issue_map(norm(budget=-10)) == {"budget": IssueCode.INVALID_VALUE}


def test_unknown_intent_falls_back_to_other():
    assert TravelExtraction(intent="make_me_a_sandwich").intent is Intent.OTHER
    assert TravelExtraction(intent=["x"]).intent is Intent.OTHER
    assert TravelExtraction(intent="confirm_booking").intent is Intent.CONFIRM_BOOKING
