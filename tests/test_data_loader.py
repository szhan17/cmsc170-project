import csv
import tempfile
import unittest
from io import StringIO
from pathlib import Path

from src.data_loader import DATA_FIELDS, DATE_FIELDS, load_weather_csv


class LoadWeatherCsvTests(unittest.TestCase):
    def write_csv(self, content: str) -> Path:
        # this writes test data to a temporary file for the loader to read.
        temporary_file = tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", suffix=".csv", delete=False
        )
        with temporary_file:
            temporary_file.write(content)
        self.addCleanup(Path(temporary_file.name).unlink, missing_ok=True)
        return Path(temporary_file.name)

    def csv_content(self, rows: list[list[str]]) -> str:
        # this formats rows as csv, including commas and quoted fields.
        output = StringIO()
        writer = csv.writer(output)
        writer.writerows(rows)
        return output.getvalue()

    def row(self, year: int, month: int, day: int) -> list[str]:
        # this makes a valid day of data with one value for every measurement.
        return [str(year), str(month), str(day), *["1.25"] * len(DATA_FIELDS)]

    def test_loads_nasa_metadata_and_preserves_canonical_measurement_order(self) -> None:
        # this checks metadata handling and correct values when columns are reordered.
        measurement_columns = list(reversed(DATA_FIELDS))
        columns = ["YEAR", "MO", "DY", *measurement_columns]
        expected = {
            field: float(index) for index, field in enumerate(measurement_columns)
        }
        row = [
            "2001",
            "1",
            "1",
            *(str(expected[field]) for field in measurement_columns),
        ]
        path = self.write_csv(
            self.csv_content(
                [
                    ["NASA POWER daily data"],
                    ["-END HEADER-"],
                    columns,
                    row,
                ]
            )
        )

        records = load_weather_csv(path)

        # this checks the date, 17 numeric columns, and consistent field order.
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["YEAR"], 2001.0)
        self.assertEqual(records[0]["MO"], 1.0)
        self.assertEqual(records[0]["DY"], 1.0)
        self.assertEqual(tuple(records[0]), (*DATE_FIELDS, *DATA_FIELDS))
        self.assertEqual(len(records[0]), 17)
        self.assertTrue(all(isinstance(value, float) for value in records[0].values()))
        self.assertEqual(
            {field: records[0][field] for field in DATA_FIELDS},
            {field: expected[field] for field in DATA_FIELDS},
        )

    def test_rejects_missing_required_columns(self) -> None:
        # this makes sure a file missing weather columns reports a clear error.
        path = self.write_csv(
            self.csv_content([["YEAR", "MO", "DY"], self.row(2001, 1, 1)[:3]])
        )

        with self.assertRaisesRegex(ValueError, "missing required CSV column"):
            load_weather_csv(path)

    def test_rejects_missing_measurements(self) -> None:
        # this makes sure blank measurements are not accepted as real data.
        path = self.write_csv(
            self.csv_content(
                [
                    ["YEAR", "MO", "DY", *DATA_FIELDS],
                    ["2001", "1", "1", *[""] * len(DATA_FIELDS)],
                ]
            )
        )

        with self.assertRaisesRegex(ValueError, "invalid numeric value in column T2M"):
            load_weather_csv(path)

    def test_rejects_missing_dates_in_the_daily_sequence(self) -> None:
        # this makes sure gaps in daily dates are caught by the loader.
        path = self.write_csv(
            self.csv_content(
                [
                    ["YEAR", "MO", "DY", *DATA_FIELDS],
                    self.row(2001, 1, 1),
                    self.row(2001, 1, 3),
                ]
            )
        )

        with self.assertRaisesRegex(ValueError, "expected the day after"):
            load_weather_csv(path)

    def test_rejects_nasa_missing_value_sentinel(self) -> None:
        # this checks that nasa's -999 marker is treated as missing data.
        row = self.row(2001, 1, 1)
        row[3] = "-999.0"
        path = self.write_csv(
            self.csv_content([["YEAR", "MO", "DY", *DATA_FIELDS], row])
        )

        with self.assertRaisesRegex(
            ValueError, "column T2M for 2001-01-01.*-999"
        ):
            load_weather_csv(path)

    def test_rejects_empty_dataset(self) -> None:
        # this makes sure a header without data rows is rejected.
        path = self.write_csv(self.csv_content([["YEAR", "MO", "DY", *DATA_FIELDS]]))

        with self.assertRaisesRegex(ValueError, "no weather records found"):
            load_weather_csv(path)


if __name__ == "__main__":
    unittest.main()