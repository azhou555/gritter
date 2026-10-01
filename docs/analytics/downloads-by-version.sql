-- GoogleSQL. Run in a Google Cloud project with BigQuery enabled.
-- Inspect estimated bytes before running; query costs depend on your project.
-- Completed UTC days only. Excludes known mirrors, but includes CI and repeats.
SELECT
  DATE(timestamp) AS day,
  file.version AS version,
  COUNT(*) AS downloads
FROM `bigquery-public-data.pypi.file_downloads`
WHERE timestamp >= TIMESTAMP(DATE_SUB(CURRENT_DATE('UTC'), INTERVAL 30 DAY))
  AND timestamp < TIMESTAMP(CURRENT_DATE('UTC'))
  AND file.project = 'gritter'
  AND COALESCE(details.installer.name, '') NOT IN ('bandersnatch', 'z3c.pypimirror')
GROUP BY day, version
ORDER BY day DESC, downloads DESC;
