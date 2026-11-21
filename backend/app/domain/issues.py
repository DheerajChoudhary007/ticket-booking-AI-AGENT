"""Machine-readable problems found while interpreting or validating a travel request."""
from enum import StrEnum

from pydantic import BaseModel


class IssueCode(StrEnum):
    INVALID_VALUE = "invalid_value"
    INVALID_DATE = "invalid_date"
    DATE_IN_PAST = "date_in_past"
    DATE_TOO_FAR = "date_too_far"
    AMBIGUOUS_DATE = "ambiguous_date"
    SAME_ORIGIN_DESTINATION = "same_origin_destination"
    INCOMPATIBLE_CLASS = "incompatible_class"


class FieldIssue(BaseModel):
    field: str
    code: IssueCode
    message: str
