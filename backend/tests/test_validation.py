from datetime import date

from app.domain.issues import IssueCode
from app.domain.travel import TravelRequest
from app.domain.validation import MANDATORY_FIELDS, is_ready_for_search, missing_fields, validate_request

TODAY = date(2026, 11, 21)


def complete(**over) -> TravelRequest:
    base = dict(origin="Jaipur", destination="Delhi", travel_date="2026-11-22", transport_mode="train", passenger_count=2)
    return TravelRequest(**{**base, **over})


def test_empty_request_is_missing_everything_in_asking_order():
    assert missing_fields(TravelRequest()) == list(MANDATORY_FIELDS)


def test_partial_request_reports_only_what_is_missing():
    r = TravelRequest(origin="Jaipur", destination="Delhi")
    assert missing_fields(r) == ["transport_mode", "travel_date", "passenger_count"]


def test_optional_fields_are_never_required():
    assert missing_fields(complete()) == []
    assert is_ready_for_search(complete(), TODAY)


def test_complete_request_has_no_issues():
    assert validate_request(complete(), TODAY) == []


def test_same_origin_and_destination_case_insensitive():
    issues = validate_request(complete(origin="Delhi", destination="delhi"), TODAY)
    assert [i.code for i in issues] == [IssueCode.SAME_ORIGIN_DESTINATION]
    assert not is_ready_for_search(complete(origin="Delhi", destination="delhi"), TODAY)


def test_past_date_flagged_on_persisted_state():
    issues = validate_request(complete(travel_date="2026-11-20"), TODAY)
    assert [(i.field, i.code) for i in issues] == [("travel_date", IssueCode.DATE_IN_PAST)]


def test_date_far_future_flagged():
    assert validate_request(complete(travel_date="2028-01-01"), TODAY)[0].code is IssueCode.DATE_TOO_FAR


def test_class_must_match_mode():
    issues = validate_request(complete(transport_mode="flight", travel_class="sleeper"), TODAY)
    assert [i.code for i in issues] == [IssueCode.INCOMPATIBLE_CLASS]
    assert validate_request(complete(transport_mode="flight", travel_class="business"), TODAY) == []
    assert validate_request(complete(transport_mode="bus", travel_class="sleeper"), TODAY) == []
    assert validate_request(complete(transport_mode="train", travel_class="ac_3_tier"), TODAY) == []


def test_class_without_mode_is_not_an_error_yet():
    assert validate_request(TravelRequest(travel_class="business"), TODAY) == []


def test_incomplete_request_not_ready_even_without_issues():
    assert not is_ready_for_search(TravelRequest(origin="Jaipur"), TODAY)
