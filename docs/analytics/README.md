# Download tracking

## Implementation plan

1. Fetch the public PyPI Stats daily series for `gritter`, excluding known mirrors.
2. Generate daily, 7-day, and rolling 30-day totals, plus the preceding 30-day total for comparison. Mark incomplete windows unavailable.
3. Run daily and manually in GitHub Actions; display a summary and save Markdown, CSV, and raw JSON artifacts.
4. Validate accounting and failures offline. Provide a separate SQL query for release-level analysis.

## Viewing reports

After this workflow reaches the default branch, open **Actions → PyPI download report**. It is scheduled for 14:23 UTC daily; GitHub may delay scheduled runs. Use **Run workflow** for a manual report. No secrets or package installation are required.

Each successful run has totals in its summary and a `pypi-download-report-…` artifact containing:

- `report.md`: totals and the last 30 calendar days.
- `downloads.csv`: the full daily history returned by the service.
- `source.json`: the original response for reproducibility.

Artifacts are retained for 90 days, subject to repository retention limits. Download snapshots for longer-term storage. These are rolling snapshots, not a permanent archive: do not add totals across snapshots because their date ranges overlap.

## Local usage

```bash
python3 scripts/download_report.py --output-dir /tmp/gritter-download-report
# Re-create a report without another API request:
python3 scripts/download_report.py --input /tmp/gritter-download-report/source.json --output-dir /tmp/gritter-report-replay
# Run focused tests without installing Gritter's dependencies:
python3 -m unittest discover -s tests -p test_download_report.py
```

The [PyPI Stats API](https://pypistats.org/api/) updates daily and retains 180 days of time series. Avoid fetching the endpoint more than once per day; use the saved response for reruns. API failures (including rate limits or a package with no available history) fail the job rather than recording zero downloads. Missing dates remain unavailable. Data more than three days old is flagged in the report.

These counts measure downloads, not unique users or successful installs. CI and repeated downloads contribute; caches and private mirrors affect counts. No analytics is added to the installed application.

## Counts by package release

PyPI Stats does not expose package-version breakdowns. Run [downloads-by-version.sql](downloads-by-version.sql) in the BigQuery console to obtain daily counts by release for the last 30 completed UTC days. This is a separate, manual analysis and is not included in the scheduled report.

It requires a Google Cloud project with BigQuery enabled. Preview the query's estimated bytes and project quota before executing it. See the [PyPA setup and schema documentation](https://packaging.python.org/en/latest/guides/analyzing-pypi-package-downloads/). Its named mirror exclusions may differ from PyPI Stats, so totals need not match exactly. The query has not been executed against a cloud project as part of this implementation.
