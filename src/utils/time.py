from datetime import datetime
import pandas as pd

DATE_FMT = "%Y-%m-%d %H:%M:%S.%f %Z"


def iso_to_datetime(iso_str: str, date_format: str = DATE_FMT) -> datetime:
    return datetime.strptime(iso_str, date_format)


def hour_of_iso_date(iso_str: str, date_format: str = DATE_FMT) -> int:
    return iso_to_datetime(iso_str, date_format).hour


def robust_hour_of_iso_date(iso_str: str, date_format: str = DATE_FMT) -> int:
    try:
        dt = pd.to_datetime(iso_str, format=date_format, errors="raise")
    except:
        dt = pd.to_datetime(iso_str, format="%Y-%m-%d %H:%M:%S %Z", errors="coerce")
    
    if pd.isna(dt):
        return None  # or -1 if you prefer
    return dt.hour
