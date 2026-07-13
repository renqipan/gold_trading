from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = ROOT / "data" / "raw"
LOCAL_LOGS = ROOT / "local_logs"


def probe_gold() -> dict[str, Any]:
    started = time.time()
    try:
        import akshare as ak

        frame = ak.futures_foreign_hist(symbol="GC").copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
        valid = frame.dropna(subset=["date", "close"]).sort_values("date")
        if valid.empty:
            raise RuntimeError("Sina COMEX GC returned no usable rows")
        return {
            "name": "gold",
            "ok": True,
            "source": "AkShare/Sina COMEX GC",
            "rows": int(len(valid)),
            "lastDate": str(valid.iloc[-1]["date"].date()),
            "lastValue": float(valid.iloc[-1]["close"]),
            "elapsedSeconds": round(time.time() - started, 2),
        }
    except Exception as exc:  # source probe should report a compact failure.
        return {
            "name": "gold",
            "ok": False,
            "source": "AkShare/Sina COMEX GC",
            "error": " ".join(str(exc).split())[:220],
            "elapsedSeconds": round(time.time() - started, 2),
        }


def probe_cash_cache() -> dict[str, Any]:
    path = RAW_DATA / "dgs3mo_cash_yield.csv"
    if not path.exists():
        return {
            "name": "cash_yield",
            "ok": False,
            "source": "FRED DGS3MO local cache",
            "error": "cache file is missing",
        }
    frame = pd.read_csv(path, parse_dates=["date"])
    values = pd.to_numeric(frame.get("cash_yield_pct"), errors="coerce")
    valid = frame.loc[values.notna()].copy()
    if valid.empty:
        return {
            "name": "cash_yield",
            "ok": False,
            "source": "FRED DGS3MO local cache",
            "error": "cache has no usable values",
        }
    latest_date = pd.Timestamp(valid.iloc[-1]["date"])
    reference_date = pd.Timestamp.now().normalize()
    age_days = int((reference_date - latest_date.normalize()).days)
    valid_age = 0 <= age_days <= 10
    return {
        "name": "cash_yield",
        "ok": valid_age,
        "source": "FRED DGS3MO local cache",
        "rows": int(len(valid)),
        "lastDate": str(latest_date.date()),
        "lastValue": float(values.loc[valid.index[-1]]),
        "ageDays": age_days,
        "maxAgeDays": 10,
        "error": "" if valid_age else "cache is stale or future-dated and must be refreshed",
    }


def main() -> None:
    LOCAL_LOGS.mkdir(parents=True, exist_ok=True)
    results = [probe_gold(), probe_cash_cache()]
    (LOCAL_LOGS / "data_source_probe.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    for result in results:
        status = "OK" if result["ok"] else "FAIL"
        detail = result.get("lastDate") or result.get("error", "")
        print(f"{status:4} {result['name']}: {detail}")
    if not all(result["ok"] for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
