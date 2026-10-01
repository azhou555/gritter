"""Offline tests for report accounting and failure handling."""
from datetime import date, datetime, timedelta, timezone
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

spec = importlib.util.spec_from_file_location(
    "download_report", Path(__file__).resolve().parents[1] / "scripts/download_report.py"
)
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def payload(rows):
    return {"package": "gritter", "type": "overall_downloads", "data": rows}


def row(day="2026-09-29", count=10, category="without_mirrors"):
    return {"date": day, "downloads": count, "category": category}


class DownloadReportTests(unittest.TestCase):
    def test_mirrors_excluded(self):
        history = report.parse_history(payload([row(), row(count=999, category="with_mirrors")]))
        self.assertEqual(history, {date(2026, 9, 29): 10})

    def test_invalid_history(self):
        for rows in [[], [row(count=-1)], [row(count=True)], [row(), row()]]:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                report.parse_history(payload(rows))

    def test_windows_and_missing_days(self):
        latest = date(2026, 9, 29)
        history = {latest - timedelta(days=n): n + 1 for n in range(60)}
        self.assertEqual(report.window_total(history, latest, 30), "465")
        self.assertEqual(report.window_total(history, latest - timedelta(days=30), 30), "1,365")
        del history[latest - timedelta(days=5)]
        self.assertEqual(report.window_total(history, latest, 7), "22 recorded downloads · 6/7 days reported (incomplete)")

    def test_empty_window_is_not_zero(self):
        latest = date(2026, 9, 29)
        self.assertEqual(report.window_total({latest: 10}, latest - timedelta(days=1), 30),
                         "No data · 0/30 days reported")

    def test_explicit_zero_counts_as_reported(self):
        latest = date(2026, 9, 29)
        self.assertEqual(report.window_total({latest: 0}, latest, 1), "0")
        self.assertEqual(report.window_total({latest: 0}, latest, 7),
                         "0 recorded downloads · 1/7 days reported (incomplete)")

    def test_stale_data_is_visible(self):
        result = report.render_report({date(2026, 9, 20): 0}, datetime(2026, 9, 30, tzinfo=timezone.utc))
        self.assertIn("more than three days old", result)
        self.assertIn("Latest day | 0", result)
        self.assertIn("Not reported", result)
        self.assertIn("0 recorded downloads · 1/7 days reported (incomplete)", result)

    def test_offline_artifacts_and_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "input.json"
            source.write_text(json.dumps(payload([row()])))
            summary = root / "summary.md"
            with patch.dict("os.environ", {"GITHUB_STEP_SUMMARY": str(summary)}):
                self.assertEqual(report.main(["--input", str(source), "--output-dir", str(root / "out")]), 0)
            self.assertEqual(summary.read_text(), (root / "out/report.md").read_text())
            self.assertIn("2026-09-29,10", (root / "out/downloads.csv").read_text())
            self.assertEqual(json.loads((root / "out/source.json").read_text()), payload([row()]))

    def test_api_failure_creates_no_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "out"
            with patch.object(report, "urlopen", side_effect=HTTPError(report.URL, 429, "Rate limited", {}, None)):
                self.assertEqual(report.main(["--output-dir", str(output)]), 1)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
