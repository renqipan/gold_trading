# GitHub Actions production seeds

The two compressed files in this directory are deterministic bootstrap inputs for a fresh
GitHub-hosted runner:

- `gold_eastmoney.csv.gz`: canonical COMEX proxy history through the last confirmed session
  available when automation was introduced.
- `dgs3mo_cash_yield.csv.gz`: FRED DGS3MO history used for conservative cash accrual.

The workflow expands these files into the ignored `data/raw/` runtime directory. The production
pipeline then appends newer COMEX rows and refreshes DGS3MO when due. Runtime cache changes are
not committed; the existing release script continues to stage only the four public website JSON
files.

Keeping a fixed seed prevents a new runner from silently rebuilding old backtest history from a
different vendor series. Confirmed published history remains immutable, while the previously
published provisional tail may be refreshed once it settles.
