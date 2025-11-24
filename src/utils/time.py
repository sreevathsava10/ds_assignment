from datetime import datetime
import pandas as pd

DATE_FMT = "%Y-%m-%d %H:%M:%S.%f %Z"


def iso_to_datetime(iso_str: str, date_format: str = DATE_FMT) -> datetime:
    return datetime.strptime(iso_str, date_format)


def hour_of_iso_date(iso_str: str, date_format: str = DATE_FMT) -> int:
    return iso_to_datetime(iso_str, date_format).hour


def robust_hour_of_iso_date(date_str: str) -> int:
    if not isinstance(date_str, str):
        raise ValueError("Input must be a string")

    # Reject strings without timezone
    if "UTC" not in date_str and "Z" not in date_str and "+" not in date_str:
        raise ValueError(f"Invalid ISO timestamp: {date_str}")

    try:
        dt = datetime.fromisoformat(date_str.replace(" UTC", "+00:00"))
        return dt.hour

    except Exception:
        raise ValueError(f"Invalid ISO timestamp: {date_str}")


