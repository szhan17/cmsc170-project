# this keeps the original weather columns for the knn, naive bayes, and perceptron steps.

from __future__ import annotations

import csv
import math
from datetime import date, timedelta
from pathlib import Path


DATA_FIELDS = (
    "T2M",
    "T2M_MAX",
    "T2M_MIN",
    "T2MDEW",
    "RH2M",
    "QV2M",
    "PS",
    "WS2M",
    "WS10M",
    "WD10M",
    "CLOUD_AMT",
    "ALLSKY_SFC_SW_DWN",
    "CLRSKY_SFC_SW_DWN",
    "PRECTOTCORR",
)
DATE_FIELDS = ("YEAR", "MO", "DY")
DEFAULT_DATA_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "nasa_power_los_banos_2001_2025.csv"
)


def _read_header(reader, source: Path) -> list[str]:
    # this skips the nasa metadata and looks for the row naming the csv columns.
    for row in reader:
        # this trims spaces and makes column names consistent.
        fields = [field.strip().upper() for field in row]
        if not fields or all(not field for field in fields):
            continue
        if fields[0] == "-END HEADER-":
            continue
        if not set(DATE_FIELDS).issubset(fields):
            continue

        # this catches repeated column names before they can select the wrong values.
        duplicates = sorted({field for field in fields if fields.count(field) > 1})
        if duplicates:
            raise ValueError(
                f"{source}: duplicate CSV column(s): {', '.join(duplicates)}"
            )

        # this checks that every weather measurement needed by the project is present.
        missing = [field for field in DATA_FIELDS if field not in fields]
        if missing:
            raise ValueError(
                f"{source}: missing required CSV column(s): {', '.join(missing)}"
            )
        return fields

    raise ValueError(
        f"{source}: could not find a CSV header containing YEAR, MO, DY "
        "and all required NASA POWER measurements"
    )


def _parse_day(values: dict[str, str], source: Path, line_number: int) -> date:
    # this combines nasa's year, month, and day columns into one date.
    try:
        year, month, day = (int(values[field]) for field in DATE_FIELDS)
        return date(year, month, day)
    except (KeyError, ValueError) as error:
        raise ValueError(
            f"{source}: line {line_number}: invalid date in YEAR, MO, DY"
        ) from error


def _load_records(reader, header: list[str], source: Path) -> list[dict[str, float]]:
    # this maps column names to positions, so column order does not matter.
    fields = (*DATE_FIELDS, *DATA_FIELDS)
    positions = {field: header.index(field) for field in fields}
    records: list[dict[str, float]] = []
    previous_day: date | None = None

    for row in reader:
        # this ignores blank lines without treating them as weather records.
        if not row or all(not field.strip() for field in row):
            continue

        line_number = reader.line_num
        # this catches truncated or extra data before reading column values.
        if len(row) != len(header):
            raise ValueError(
                f"{source}: line {line_number}: expected {len(header)} columns, "
                f"found {len(row)}"
            )

        row_values = {field: row[index].strip() for field, index in positions.items()}
        current_day = _parse_day(row_values, source, line_number)
        numeric_values: dict[str, float] = {}
        for field in fields:
            try:
                numeric_values[field] = float(row_values[field])
            except ValueError as error:
                raise ValueError(
                    f"{source}: line {line_number}: invalid numeric value in "
                    f"column {field} for {current_day.isoformat()}: "
                    f"{row_values[field]!r}"
                ) from error

        # this checks consecutive dates so later next-day labels stay aligned.
        if previous_day is not None and current_day != previous_day + timedelta(days=1):
            raise ValueError(
                f"{source}: line {line_number}: expected the day after "
                f"{previous_day.isoformat()}, found {current_day.isoformat()}"
            )

        # this rejects missing markers and invalid numbers in any of the 17 columns.
        for field, value in numeric_values.items():
            if not math.isfinite(value) or value == -999.0:
                raise ValueError(
                    f"{source}: line {line_number}: missing or non-finite value "
                    f"in column {field} for {current_day.isoformat()}: "
                    f"{row_values[field]!r}"
                )

        # this keeps the date parts and weather values together as 17 floats.
        records.append(numeric_values)
        previous_day = current_day

    if not records:
        raise ValueError(f"{source}: no weather records found")
    # this returns the validated records in their original date order.
    return records


def load_weather_csv(path: str | Path = DEFAULT_DATA_PATH) -> list[dict[str, float]]:
    # this opens nasa's csv and returns its validated daily records.
    source = Path(path)
    # this handles nasa's optional byte-order mark and reads quoted csv fields.
    with source.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.reader(file)
        header = _read_header(reader, source)
        return _load_records(reader, header, source)


def main() -> None:
    # this prints the record count and date range when the script is run.
    records = load_weather_csv()
    print(
        f"Loaded {len(records):,} daily records "
        f"({date(int(records[0]['YEAR']), int(records[0]['MO']), int(records[0]['DY'])).isoformat()} "
        f"to {date(int(records[-1]['YEAR']), int(records[-1]['MO']), int(records[-1]['DY'])).isoformat()})."
    )


if __name__ == "__main__":
    main()