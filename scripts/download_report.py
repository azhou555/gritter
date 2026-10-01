"""Generate download reports from PyPI Stats using only the standard library."""
from __future__ import annotations

import argparse
import csv
from datetime import date, datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

URL = "https://pypistats.org/api/packages/gritter/overall?mirrors=false"


def parse_history(payload: dict) -> dict[date, int]:
    if payload.get("package") != "gritter" or payload.get("type") != "overall_downloads":
        raise ValueError("Unexpected package or response type")
    history = {}
    for row in payload["data"]:
        if row["category"] != "without_mirrors":
            continue
        day = date.fromisoformat(row["date"])
        count = row["downloads"]
        if type(count) is not int or count < 0 or day in history:
            raise ValueError("Invalid count or duplicate date in download history")
        history[day] = count
    if not history:
        raise ValueError("No download history available; this is not a zero download count")
    return dict(sorted(history.items()))


def window_total(history: dict[date, int], end: date, days: int) -> str:
    dates = [end - timedelta(days=n) for n in range(days)]
    missing = sum(day not in history for day in dates)
    if missing:
        return f"Unavailable ({missing} missing days)"
    return f"{sum(history[day] for day in dates):,}"


def render_report(history: dict[date, int], generated: datetime) -> str:
    latest = max(history)
    lines = [
        "# Gritter download report", "",
        f"Generated: {generated.isoformat()} | Latest source date: {latest}", "",
        "Source: [PyPI Stats](https://pypistats.org/project/gritter/). Known mirrors excluded.",
        "Counts measure downloads, not unique users or successful installations. CI and repeat downloads count; cached installs may not.", "",
        "Windows end on the latest source date; the month metric is a rolling 30 days.", "",
        "| Window | Downloads |", "| --- | ---: |",
    ]
    for label, days in [("Latest day", 1), ("Last 7 days", 7), ("Last 30 days", 30)]:
        lines.append(f"| {label} | {window_total(history, latest, days)} |")
    lines.append(f"| Previous 30 days | {window_total(history, latest - timedelta(days=30), 30)} |")
    if (generated.date() - latest).days > 3:
        lines.extend(["", "**Source data is more than three days old.**"])
    lines.extend(["", "## Daily history (last 30 calendar days)", "", "| Date | Downloads |", "| --- | ---: |"])
    for offset in range(30):
        day = latest - timedelta(days=offset)
        value = f"{history[day]:,}" if day in history else "Unavailable"
        lines.append(f"| {day} | {value} |")
    lines.extend(["", "Full available history is in `downloads.csv`; the original response is in `source.json`.",
                  "Release-level counts require the separate BigQuery query in `docs/analytics/downloads-by-version.sql`.", ""])
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("download-report"))
    parser.add_argument("--input", type=Path, help="Use a saved API response for offline reporting")
    args = parser.parse_args(argv)
    try:
        if args.input:
            payload = json.loads(args.input.read_text())
        else:
            request = Request(URL, headers={"User-Agent": "gritter-download-report/1.0"})
            with urlopen(request, timeout=30) as response:
                payload = json.load(response)
        history = parse_history(payload)
        report = render_report(history, datetime.now(timezone.utc))
    except (HTTPError, URLError, TimeoutError, OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Download report failed: {exc}. No zero counts were substituted.", file=sys.stderr)
        return 1
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "source.json").write_text(json.dumps(payload, indent=2) + "\n")
    (args.output_dir / "report.md").write_text(report)
    with (args.output_dir / "downloads.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["date", "downloads_without_mirrors"])
        writer.writerows((day.isoformat(), count) for day, count in history.items())
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as handle:
            handle.write(report)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
