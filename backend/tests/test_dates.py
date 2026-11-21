from datetime import UTC, date, datetime, timedelta

import pytest

from app.domain.dates import DateError, check_travel_date, get_now, parse_iso_date, resolve_relative_date
from app.domain.issues import IssueCode

SAT = date(2026, 11, 21)  # Saturday
MON = date(2026, 11, 16)  # Monday


def test_fixture_dates_are_the_weekdays_assumed():
    assert SAT.strftime("%A") == "Saturday" and MON.strftime("%A") == "Monday"


def test_now_is_timezone_aware_and_crosses_midnight():
    now = get_now("Asia/Kolkata", datetime(2026, 11, 21, 20, 0, tzinfo=UTC))
    assert now.tzinfo is not None
    assert now.date() == date(2026, 11, 22)  # 20:00 UTC is already the next day in India
    assert get_now("UTC", datetime(2026, 11, 21, 20, 0, tzinfo=UTC)).date() == date(2026, 11, 21)


def test_get_now_rejects_naive_datetime():
    with pytest.raises(ValueError):
        get_now("UTC", datetime(2026, 11, 21, 20, 0))


def test_get_now_default_clock_is_aware():
    assert get_now("Asia/Kolkata").utcoffset() == timedelta(hours=5, minutes=30)


@pytest.mark.parametrize("s", ["2026-13-01", "2026-02-30", "21-11-2026", "20261121", "2026-1-5", "tomorrow", "", "2026-11-21T10:00"])
def test_parse_iso_rejects(s):
    with pytest.raises(DateError) as e:
        parse_iso_date(s)
    assert e.value.code is IssueCode.INVALID_DATE


def test_parse_iso_accepts_valid_and_leap_day():
    assert parse_iso_date(" 2026-11-21 ") == SAT
    assert parse_iso_date("2028-02-29") == date(2028, 2, 29)
    with pytest.raises(DateError):
        parse_iso_date("2027-02-29")


def test_check_travel_date_boundaries():
    check_travel_date(SAT, SAT)  # today is fine
    check_travel_date(SAT + timedelta(days=365), SAT)  # last allowed day
    with pytest.raises(DateError) as past:
        check_travel_date(SAT - timedelta(days=1), SAT)
    assert past.value.code is IssueCode.DATE_IN_PAST
    with pytest.raises(DateError) as far:
        check_travel_date(SAT + timedelta(days=366), SAT)
    assert far.value.code is IssueCode.DATE_TOO_FAR


@pytest.mark.parametrize(
    "text,today,expected",
    [
        ("today", SAT, SAT),
        ("tonight", SAT, SAT),
        ("tomorrow", SAT, date(2026, 11, 22)),
        ("Tomorrow morning", SAT, date(2026, 11, 22)),
        ("tmrw", SAT, date(2026, 11, 22)),
        ("day after tomorrow", SAT, date(2026, 11, 23)),
        ("in 3 days", SAT, date(2026, 11, 24)),
        ("in 1 day", SAT, date(2026, 11, 22)),
        ("friday", SAT, date(2026, 11, 27)),
        ("fri", SAT, date(2026, 11, 27)),
        ("this friday", MON, date(2026, 11, 20)),
        ("friday", MON, date(2026, 11, 20)),
        ("next friday", MON, date(2026, 11, 27)),  # bare Friday is this week -> next week's
        ("next friday", SAT, date(2026, 11, 27)),  # bare Friday already falls in the next ISO week
        ("next monday", SAT, date(2026, 11, 23)),
        ("this saturday", SAT, SAT),
        ("next saturday", SAT, date(2026, 11, 28)),
        ("   TOMORROW  ", SAT, date(2026, 11, 22)),
    ],
)
def test_relative_resolution(text, today, expected):
    assert resolve_relative_date(text, today) == expected


@pytest.mark.parametrize("text", ["", "   ", "sometime soon", "next month", "friday 28th", "28 november", "tomorrow or friday", "05/06", "month"])
def test_relative_resolution_declines(text):
    assert resolve_relative_date(text, SAT) is None


def test_bare_weekday_equal_to_today_is_ambiguous():
    with pytest.raises(DateError) as e:
        resolve_relative_date("saturday", SAT)
    assert e.value.code is IssueCode.AMBIGUOUS_DATE
    assert "2026-11-28" in e.value.message


def test_relative_resolution_across_year_boundary():
    assert resolve_relative_date("tomorrow", date(2026, 12, 31)) == date(2027, 1, 1)
    assert resolve_relative_date("in 3 days", date(2028, 2, 27)) == date(2028, 3, 1)  # leap year
