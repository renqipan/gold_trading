from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import random
import re
import subprocess
import time
import urllib.parse
import zipfile
from dataclasses import asdict, dataclass, replace
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from hmmlearn.hmm import GaussianHMM
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_DATA = ROOT / "public" / "data"
LOCAL_LOGS = ROOT / "local_logs"
RAW_DATA = ROOT / "data" / "raw"
MARKET_SOURCE_METADATA = RAW_DATA / "market_source_metadata.json"
FORWARD_LEDGER = PUBLIC_DATA / "gold_forward_ledger.json"
OFFLINE_MODE = False

FORMAL_STRATEGY_VERSION = "2026-07-13-v1"
EXECUTION_ENGINE_VERSION = "strict-next-open-v2"

EASTMONEY_KLINE = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
EASTMONEY_SEARCH = "https://searchapi.eastmoney.com/api/suggest/get"
EASTMONEY_TOKEN = "D43BF722C8E33F0689C5A6D47D64A2D0"

REQUEST_USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
]


@dataclass(frozen=True)
class RiskConfig:
    train_end_date: str = "2021-02-23"
    validation_end_date: str = "2023-02-27"
    forward_holdout_start_date: str = "2026-07-13"
    max_position: float = 1.0
    max_leverage: float = 1.0
    long_only: bool = True
    max_single_loss: float = 0.14
    dynamic_trend_risk_enabled: bool = True
    normal_trend_risk_budget: float = 0.10
    strong_trend_risk_budget: float = 0.14
    strong_trend_ret_120_threshold: float = 0.12
    max_drawdown_soft: float = 0.18
    max_drawdown_hard: float = 0.30
    hard_drawdown_cooldown_days: int = 63
    atr_window: int = 14
    meta_event_gap_days: int = 5
    cusum_threshold_mult: float = 0.8
    hmm_feature_policy: str = "gold_core"
    hmm_retrain_every_days: int = 252
    hmm_exit_confirmation_days: int = 20
    profit_atr_multiple: float = 10.0
    stop_atr_multiple: float = 6.0
    atr_stop_enabled: bool = True
    atr_profit_enabled: bool = True
    hmm_exit_enabled: bool = True
    realistic_cost_bps: float = 8.0
    cash_yield_enabled: bool = True
    cash_yield_haircut_bps: float = 50.0
    live_soft_drawdown_position: float = 0.5
    live_hard_drawdown_position: float = 0.0

    def __post_init__(self) -> None:
        if not self.long_only:
            raise ValueError("The formal gold strategy is long-only")
        if not 0 < self.max_position <= 1.0:
            raise ValueError("max_position must be in (0, 1] for an unlevered strategy")
        if not 0 < self.max_leverage <= 1.0:
            raise ValueError("max_leverage must be in (0, 1] for an unlevered strategy")
        if self.hmm_feature_policy != "gold_core":
            raise ValueError("The formal HMM feature policy is fixed to gold_core")
        if self.cash_yield_haircut_bps < 0:
            raise ValueError("cash_yield_haircut_bps must be non-negative")


STATE_TO_CODE = {
    "牛市": "s1",
    "熊市": "s2",
    "震荡": "s3",
    "恐慌": "s4",
}


def ensure_dirs() -> None:
    for path in [PUBLIC_DATA, LOCAL_LOGS, RAW_DATA]:
        path.mkdir(parents=True, exist_ok=True)


def browser_headers(referer: str = "https://quote.eastmoney.com/") -> dict[str, str]:
    return {
        "User-Agent": random.choice(REQUEST_USER_AGENTS),
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "Connection": "keep-alive",
        "Referer": referer,
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-site",
    }


def compact_error(text: str, max_length: int = 220) -> str:
    line = " ".join(text.strip().split())
    if len(line) <= max_length:
        return line
    return f"{line[:max_length].rstrip()}..."


def json_safe(value: Any) -> Any:
    """Convert numpy scalars and non-finite values to strict JSON types."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_json_atomic(path: Path, payload: Any, *, indent: int | None = None) -> None:
    """Replace a published JSON file atomically after strict serialization succeeds."""
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(
            json_safe(payload),
            ensure_ascii=False,
            indent=indent,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def build_price_status(latest_date: pd.Timestamp) -> dict[str, Any]:
    """Describe whether the newest quote is a settled daily bar or a live intraday snapshot."""
    session_date = pd.Timestamp(latest_date).date()
    shanghai_now = pd.Timestamp.now(tz="Asia/Shanghai")
    settlement_lag_days = 1 if shanghai_now.hour >= 8 else 2
    confirmed_through = (shanghai_now.normalize() - pd.Timedelta(days=settlement_lag_days)).date()
    is_final = session_date <= confirmed_through
    return {
        "kind": "confirmed_daily_close" if is_final else "intraday_snapshot",
        "isFinal": is_final,
        "sessionDate": str(session_date),
        "observedAt": pd.Timestamp.now(tz="UTC").isoformat(),
        "timezone": "Asia/Shanghai",
        "settlementRule": "COMEX session is treated as final after 08:00 Asia/Shanghai on the next calendar day",
    }


def random_request_pause(attempt: int) -> None:
    if attempt == 0:
        time.sleep(random.uniform(0.20, 0.70))
        return
    backoff = min(6.0, 0.75 * (2 ** (attempt - 1)))
    time.sleep(backoff + random.uniform(0.25, 1.10))


def curl_json(full_url: str, headers: dict[str, str], timeout: int) -> dict[str, Any]:
    command = [
        "curl",
        "-q",
        "-L",
        "--silent",
        "--show-error",
        "--fail",
        "--compressed",
        "--http1.1",
        "--retry",
        "2",
        "--retry-delay",
        "1",
        "--max-time",
        str(timeout),
        "--noproxy",
        "*",
    ]
    for name, value in headers.items():
        command.extend(["-H", f"{name}: {value}"])
    command.append(full_url)

    result = subprocess.run(command, text=True, capture_output=True, timeout=timeout + 5)
    if result.returncode != 0:
        stderr = compact_error(result.stderr)
        raise RuntimeError(f"curl exit {result.returncode}: {stderr}")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        preview = compact_error(result.stdout)
        raise RuntimeError(f"curl returned invalid JSON: {preview}") from exc


def curl_text(full_url: str, headers: dict[str, str], timeout: int) -> str:
    command = [
        "curl",
        "-q",
        "-L",
        "--silent",
        "--show-error",
        "--fail",
        "--compressed",
        "--http1.1",
        "--retry",
        "2",
        "--retry-delay",
        "1",
        "--max-time",
        str(timeout),
        "--noproxy",
        "*",
    ]
    for name, value in headers.items():
        command.extend(["-H", f"{name}: {value}"])
    command.append(full_url)

    result = subprocess.run(command, text=True, capture_output=True, timeout=timeout + 5)
    if result.returncode != 0:
        stderr = compact_error(result.stderr)
        raise RuntimeError(f"curl exit {result.returncode}: {stderr}")
    return result.stdout


def curl_bytes(full_url: str, headers: dict[str, str], timeout: int) -> bytes:
    command = [
        "curl",
        "-q",
        "-L",
        "--silent",
        "--show-error",
        "--fail",
        "--compressed",
        "--http1.1",
        "--retry",
        "2",
        "--retry-delay",
        "1",
        "--max-time",
        str(timeout),
        "--noproxy",
        "*",
    ]
    for name, value in headers.items():
        command.extend(["-H", f"{name}: {value}"])
    command.append(full_url)

    result = subprocess.run(command, capture_output=True, timeout=timeout + 5)
    if result.returncode != 0:
        stderr = compact_error(result.stderr.decode("utf-8", errors="ignore"))
        raise RuntimeError(f"curl exit {result.returncode}: {stderr}")
    return result.stdout


def request_json(url: str, params: dict[str, Any], timeout: int = 12, attempts: int = 4) -> dict[str, Any]:
    if OFFLINE_MODE:
        raise RuntimeError("offline mode")
    last_error: Exception | None = None
    with requests.Session() as session:
        session.trust_env = False
        for attempt in range(attempts):
            random_request_pause(attempt)
            headers = browser_headers()
            try:
                response = session.get(url, params=params, headers=headers, timeout=(4, timeout))
                response.raise_for_status()
                return response.json()
            except Exception as exc:
                last_error = exc

    full_url = f"{url}?{urllib.parse.urlencode(params)}"
    try:
        return curl_json(full_url, browser_headers(), timeout)
    except Exception as exc:
        raise RuntimeError(f"request failed after retries: {last_error}; curl fallback: {exc}") from exc


def request_text(url: str, timeout: int = 20, attempts: int = 3) -> str:
    if OFFLINE_MODE:
        raise RuntimeError("offline mode")
    last_error: Exception | None = None
    with requests.Session() as session:
        session.trust_env = False
        for attempt in range(attempts):
            random_request_pause(attempt)
            headers = browser_headers(referer=url)
            try:
                response = session.get(url, headers=headers, timeout=(5, timeout))
                response.raise_for_status()
                return response.text
            except Exception as exc:
                last_error = exc
    try:
        return curl_text(url, browser_headers(referer=url), timeout)
    except Exception as exc:
        raise RuntimeError(f"text request failed after retries: {last_error}; curl fallback: {exc}") from exc


def request_bytes(url: str, timeout: int = 30, attempts: int = 3) -> bytes:
    if OFFLINE_MODE:
        raise RuntimeError("offline mode")
    last_error: Exception | None = None
    with requests.Session() as session:
        session.trust_env = False
        for attempt in range(attempts):
            random_request_pause(attempt)
            headers = browser_headers(referer=url)
            try:
                response = session.get(url, headers=headers, timeout=(5, timeout))
                response.raise_for_status()
                return response.content
            except Exception as exc:
                last_error = exc
    try:
        return curl_bytes(url, browser_headers(referer=url), timeout)
    except Exception as exc:
        raise RuntimeError(f"binary request failed after retries: {last_error}; curl fallback: {exc}") from exc


def fetch_eastmoney_kline(secid: str, name: str, limit: int = 6600) -> pd.DataFrame:
    params = {
        "secid": secid,
        "klt": "101",
        "fqt": "1",
        "lmt": str(limit),
        "end": "20500000",
        "iscca": "1",
        "fields1": "f1,f2,f3,f4,f5,f6,f7,f8",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63,f64",
        "ut": "f057cbcbce2a86e2866ab8877db1d059",
        "forcect": "1",
    }
    payload = request_json(EASTMONEY_KLINE, params)
    data = payload.get("data") or {}
    klines = data.get("klines") or []
    if not klines:
        raise RuntimeError(f"No kline data for {name} ({secid})")

    rows = [line.split(",") for line in klines]
    columns = [
        "date",
        "open",
        "close",
        "high",
        "low",
        "volume",
        "amount",
        "amplitude",
        "pct_change",
        "change",
        "turnover",
        "extra1",
        "extra2",
        "extra3",
    ]
    frame = pd.DataFrame(rows, columns=columns[: len(rows[0])])
    frame["date"] = pd.to_datetime(frame["date"])
    for column in frame.columns:
        if column != "date":
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.sort_values("date").drop_duplicates("date").set_index("date")
    frame = frame.rename(columns={column: f"{name}_{column}" for column in frame.columns})
    return frame


def load_cached_eastmoney_kline(name: str) -> pd.DataFrame | None:
    cache_path = RAW_DATA / f"{name}_eastmoney.csv"
    if not cache_path.exists():
        return None
    frame = pd.read_csv(cache_path, parse_dates=["date"])
    frame = frame.set_index("date").sort_index()
    return frame


def repair_ohlc(frame: pd.DataFrame, name: str) -> tuple[pd.DataFrame, int]:
    required = [f"{name}_open", f"{name}_close", f"{name}_high", f"{name}_low"]
    if not all(column in frame for column in required):
        return frame, 0
    repaired = frame.copy()
    before_high = repaired[f"{name}_high"].copy()
    before_low = repaired[f"{name}_low"].copy()
    ohlc = repaired[required]
    repaired[f"{name}_high"] = ohlc.max(axis=1)
    repaired[f"{name}_low"] = ohlc.min(axis=1)
    changed = ((before_high != repaired[f"{name}_high"]) | (before_low != repaired[f"{name}_low"])).sum()
    return repaired, int(changed)


def fetch_search_quote_id(term: str) -> list[dict[str, Any]]:
    payload = request_json(
        EASTMONEY_SEARCH,
        {
            "input": term,
            "type": "14",
            "token": EASTMONEY_TOKEN,
            "count": "20",
        },
    )
    table = payload.get("QuotationCodeTable") or {}
    return table.get("Data") or []


def fetch_cpi_yoy() -> pd.DataFrame:
    if OFFLINE_MODE:
        return pd.DataFrame(columns=["us_cpi_yoy"], index=pd.DatetimeIndex([], name="date"))
    try:
        import akshare as ak

        frame = ak.macro_usa_cpi_yoy()
        frame = frame.rename(columns={"时间": "date", "现值": "us_cpi_yoy"})
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame["us_cpi_yoy"] = pd.to_numeric(frame["us_cpi_yoy"], errors="coerce")
        frame = frame[["date", "us_cpi_yoy"]].dropna(subset=["date"]).set_index("date")
        return frame.sort_index()
    except Exception as exc:
        print(f"[warn] CPI data unavailable: {exc}")
        return pd.DataFrame(columns=["us_cpi_yoy"], index=pd.DatetimeIndex([], name="date"))


def fetch_cot_gold() -> pd.DataFrame:
    if OFFLINE_MODE:
        return pd.DataFrame(
            columns=["cot_gold_long", "cot_gold_short", "cot_gold_net"],
            index=pd.DatetimeIndex([], name="date"),
        )
    try:
        import akshare as ak

        frame = ak.macro_usa_cftc_merchant_goods_holding()
        frame = frame.rename(
            columns={
                "日期": "date",
                "黄金-多头仓位": "cot_gold_long",
                "黄金-空头仓位": "cot_gold_short",
                "黄金-净仓位": "cot_gold_net",
            }
        )
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        for column in ["cot_gold_long", "cot_gold_short", "cot_gold_net"]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = frame[["date", "cot_gold_long", "cot_gold_short", "cot_gold_net"]]
        return frame.dropna(subset=["date"]).sort_values("date").set_index("date")
    except Exception as exc:
        print(f"[warn] COT data unavailable: {exc}")
        return pd.DataFrame(
            columns=["cot_gold_long", "cot_gold_short", "cot_gold_net"],
            index=pd.DatetimeIndex([], name="date"),
        )


def fetch_cftc_managed_money_gold(start_year: int = 2013, end_year: int | None = None) -> pd.DataFrame:
    end_year = end_year or pd.Timestamp.now(tz="UTC").year
    rows: list[pd.DataFrame] = []
    for year in range(start_year, end_year + 1):
        zip_path = RAW_DATA / f"cftc_fut_disagg_{year}.zip"
        cache_is_fresh = zip_path.exists() and (OFFLINE_MODE or (
            year < end_year or (time.time() - zip_path.stat().st_mtime) < 3 * 24 * 60 * 60
        ))
        if not cache_is_fresh:
            url = f"https://www.cftc.gov/files/dea/history/fut_disagg_txt_{year}.zip"
            try:
                zip_path.write_bytes(request_bytes(url, timeout=12, attempts=1))
            except Exception as exc:
                if zip_path.exists():
                    print(f"[warn] CFTC managed money {year} refresh unavailable; using cached file")
                else:
                    print(f"[warn] CFTC managed money {year} unavailable: {compact_error(str(exc))}")
                    continue
        try:
            with zipfile.ZipFile(zip_path) as archive:
                with archive.open(archive.namelist()[0]) as handle:
                    year_frame = pd.read_csv(handle, low_memory=False)
            gold = year_frame[
                year_frame["Market_and_Exchange_Names"].astype(str).str.contains(
                    "GOLD - COMMODITY EXCHANGE", na=False
                )
            ].copy()
            if gold.empty:
                continue
            gold["date"] = pd.to_datetime(gold["Report_Date_as_YYYY-MM-DD"], errors="coerce")
            for column in [
                "M_Money_Positions_Long_All",
                "M_Money_Positions_Short_All",
                "M_Money_Positions_Spread_All",
                "Open_Interest_All",
            ]:
                gold[column] = pd.to_numeric(gold[column], errors="coerce")
            gold["cftc_mm_gold_long"] = gold["M_Money_Positions_Long_All"]
            gold["cftc_mm_gold_short"] = gold["M_Money_Positions_Short_All"]
            gold["cftc_mm_gold_spread"] = gold["M_Money_Positions_Spread_All"]
            gold["cftc_mm_gold_net"] = gold["cftc_mm_gold_long"] - gold["cftc_mm_gold_short"]
            gold["cftc_gold_open_interest"] = gold["Open_Interest_All"]
            gold["cftc_mm_gold_net_pct_oi"] = gold["cftc_mm_gold_net"] / gold["cftc_gold_open_interest"].replace(0, np.nan)
            rows.append(
                gold[
                    [
                        "date",
                        "cftc_mm_gold_long",
                        "cftc_mm_gold_short",
                        "cftc_mm_gold_spread",
                        "cftc_mm_gold_net",
                        "cftc_gold_open_interest",
                        "cftc_mm_gold_net_pct_oi",
                    ]
                ]
            )
        except Exception as exc:
            print(f"[warn] CFTC managed money {year} parse failed: {compact_error(str(exc))}")
    if not rows:
        return pd.DataFrame(
            columns=[
                "cftc_mm_gold_long",
                "cftc_mm_gold_short",
                "cftc_mm_gold_spread",
                "cftc_mm_gold_net",
                "cftc_gold_open_interest",
                "cftc_mm_gold_net_pct_oi",
            ],
            index=pd.DatetimeIndex([], name="date"),
        )
    frame = pd.concat(rows, ignore_index=True)
    return frame.dropna(subset=["date"]).drop_duplicates("date").set_index("date").sort_index()


def fetch_macro_surprise(fn_name: str, prefix: str) -> pd.DataFrame:
    if OFFLINE_MODE:
        return pd.DataFrame(
            columns=[
                f"{prefix}_actual",
                f"{prefix}_forecast",
                f"{prefix}_previous",
                f"{prefix}_surprise",
            ],
            index=pd.DatetimeIndex([], name="date"),
        )
    try:
        import akshare as ak

        frame = getattr(ak, fn_name)()
        frame = frame.rename(
            columns={
                "日期": "date",
                "今值": f"{prefix}_actual",
                "预测值": f"{prefix}_forecast",
                "前值": f"{prefix}_previous",
            }
        )
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        for column in [f"{prefix}_actual", f"{prefix}_forecast", f"{prefix}_previous"]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame[f"{prefix}_surprise"] = frame[f"{prefix}_actual"] - frame[f"{prefix}_forecast"]
        return frame[
            [
                "date",
                f"{prefix}_actual",
                f"{prefix}_forecast",
                f"{prefix}_previous",
                f"{prefix}_surprise",
            ]
        ].dropna(subset=["date"]).set_index("date").sort_index()
    except Exception as exc:
        print(f"[warn] {prefix} surprise data unavailable: {exc}")
        return pd.DataFrame(
            columns=[
                f"{prefix}_actual",
                f"{prefix}_forecast",
                f"{prefix}_previous",
                f"{prefix}_surprise",
            ],
            index=pd.DatetimeIndex([], name="date"),
        )


def load_cached_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(index=pd.DatetimeIndex([], name="date"))
    frame = pd.read_csv(path, parse_dates=["date"])
    return frame.set_index("date").sort_index()


def append_new_market_rows(
    cached: pd.DataFrame,
    fresh: pd.DataFrame,
    *,
    refresh_provisional_tail: bool = False,
) -> pd.DataFrame:
    """Append new dates and optionally refresh only the still-provisional tail date."""
    if cached.empty or fresh.empty:
        return cached if fresh.empty else fresh
    merged = cached.copy()
    tail_date = cached.index.max()
    if (
        refresh_provisional_tail
        and tail_date in fresh.index
        and build_price_status(tail_date)["isFinal"] is False
    ):
        common_columns = cached.columns.intersection(fresh.columns)
        merged.loc[tail_date, common_columns] = fresh.loc[tail_date, common_columns]
    new_rows = fresh.loc[fresh.index > cached.index.max()].copy()
    if new_rows.empty:
        return merged
    all_columns = merged.columns.union(new_rows.columns)
    return pd.concat(
        [merged.reindex(columns=all_columns), new_rows.reindex(columns=all_columns)]
    ).sort_index()


def load_market_source_metadata() -> dict[str, str]:
    if not MARKET_SOURCE_METADATA.exists():
        return {}
    try:
        return json.loads(MARKET_SOURCE_METADATA.read_text(encoding="utf-8"))
    except Exception:
        return {}


def fetch_alternative_market_history(name: str, cached: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Fetch current rows from independent Sina/AkShare endpoints when Eastmoney is down."""
    import akshare as ak

    if name == "gold":
        fresh = ak.futures_foreign_hist(symbol="GC").copy()
        source = "AkShare/Sina COMEX gold continuous GC extension"
    elif name in {"gld", "vixy"}:
        fresh = ak.stock_us_daily(symbol=name.upper(), adjust="").copy()
        source = f"AkShare/Sina {name.upper()} daily extension"
    elif name == "spx":
        fresh = ak.index_us_stock_sina(symbol=".INX").copy()
        source = "AkShare/Sina S&P 500 .INX extension"
    elif name == "us10y":
        start = (cached.index.max() - pd.Timedelta(days=14)).strftime("%Y%m%d")
        fresh = ak.bond_zh_us_rate(start_date=start).rename(
            columns={"日期": "date", "美国国债收益率10年": "close"}
        )
        fresh["open"] = fresh["close"]
        fresh["high"] = fresh["close"]
        fresh["low"] = fresh["close"]
        source = "AkShare US Treasury 10Y daily yield extension"
    elif name == "dxy":
        uup = ak.stock_us_daily(symbol="UUP", adjust="").copy()
        uup["date"] = pd.to_datetime(uup["date"], errors="coerce")
        uup = uup.dropna(subset=["date"]).set_index("date").sort_index()
        anchor_rows = uup.loc[uup.index <= cached.index.max()]
        if anchor_rows.empty:
            return pd.DataFrame(), ""
        anchor_uup = float(anchor_rows["close"].iloc[-1])
        anchor_dxy = float(cached["dxy_close"].dropna().iloc[-1])
        fresh = uup.reset_index()
        for column in ["open", "high", "low", "close"]:
            fresh[column] = anchor_dxy * fresh[column] / anchor_uup
        source = "AkShare/Sina UUP return proxy extension for DXY"
    else:
        return pd.DataFrame(), ""

    fresh["date"] = pd.to_datetime(fresh["date"], errors="coerce")
    fresh = fresh.dropna(subset=["date"]).set_index("date").sort_index()
    keep = [column for column in ["open", "close", "high", "low", "volume", "amount"] if column in fresh]
    fresh = fresh[keep].rename(columns={column: f"{name}_{column}" for column in keep})
    return fresh, source


def apply_cpi_release_lag(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    out.index = pd.DatetimeIndex(out.index) + pd.DateOffset(months=1) + pd.Timedelta(days=14)
    out.index.name = "date"
    return out.sort_index()


def apply_cot_release_lag(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    out.index = pd.DatetimeIndex(out.index) + pd.Timedelta(days=3)
    out.index.name = "date"
    return out.sort_index()


def apply_monthly_release_lag(frame: pd.DataFrame, days_after_month_end: int = 7) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    month_start = pd.DatetimeIndex(out.index)
    out.index = month_start + pd.offsets.MonthEnd(0) + pd.Timedelta(days=days_after_month_end)
    out.index.name = "date"
    return out.sort_index()


def fetch_fred_series(series_id: str, column_name: str) -> pd.DataFrame:
    if OFFLINE_MODE:
        raise RuntimeError(f"offline mode: use cached FRED {series_id}")
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    text = request_text(url, timeout=8, attempts=1)
    frame = pd.read_csv(StringIO(text))
    date_column = "observation_date" if "observation_date" in frame.columns else frame.columns[0]
    value_column = series_id if series_id in frame.columns else frame.columns[-1]
    frame = frame.rename(columns={date_column: "date", value_column: column_name})
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame[column_name] = pd.to_numeric(frame[column_name].replace(".", np.nan), errors="coerce")
    return frame[["date", column_name]].dropna(subset=["date"]).set_index("date").sort_index()


def fetch_cboe_vix_history() -> pd.DataFrame:
    url = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
    text = request_text(url, timeout=12, attempts=1)
    frame = pd.read_csv(StringIO(text))
    frame = frame.rename(
        columns={
            "DATE": "date",
            "OPEN": "vix_open",
            "HIGH": "vix_high",
            "LOW": "vix_low",
            "CLOSE": "vix_close",
        }
    )
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for column in ["vix_open", "vix_high", "vix_low", "vix_close"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame[["date", "vix_open", "vix_high", "vix_low", "vix_close"]].dropna(subset=["date"]).set_index("date").sort_index()


def fetch_gpr_index() -> pd.DataFrame:
    url = "https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls"
    xls_path = RAW_DATA / "data_gpr_export.xls"
    cache_is_fresh = xls_path.exists() and (time.time() - xls_path.stat().st_mtime) < 14 * 24 * 60 * 60
    if not cache_is_fresh:
        try:
            xls_path.write_bytes(request_bytes(url, timeout=45, attempts=2))
        except Exception:
            if not xls_path.exists():
                raise
    frame = pd.read_excel(xls_path)
    frame = frame.rename(
        columns={
            "month": "date",
            "GPR": "gpr_index",
            "GPRT": "gpr_threats",
            "GPRA": "gpr_acts",
        }
    )
    keep = ["date", "gpr_index", "gpr_threats", "gpr_acts"]
    frame = frame[[column for column in keep if column in frame.columns]].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for column in ["gpr_index", "gpr_threats", "gpr_acts"]:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["date"]).set_index("date").sort_index()
    return frame


def fetch_fomc_decision_dates() -> pd.DataFrame:
    url = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
    html = request_text(url, timeout=20, attempts=2)
    (RAW_DATA / "fomc_calendars.html").write_text(html, encoding="utf-8")
    month_map = {
        "January": 1,
        "February": 2,
        "March": 3,
        "April": 4,
        "May": 5,
        "June": 6,
        "July": 7,
        "August": 8,
        "September": 9,
        "October": 10,
        "November": 11,
        "December": 12,
    }
    current_year: int | None = None
    current_month: int | None = None
    rows: list[dict[str, Any]] = []
    for line in html.splitlines():
        year_match = re.search(r">(\d{4}) FOMC Meetings<", line)
        if year_match:
            current_year = int(year_match.group(1))
            current_month = None
            continue
        month_match = re.search(r"<strong>(January|February|March|April|May|June|July|August|September|October|November|December)</strong>", line)
        if month_match:
            current_month = month_map[month_match.group(1)]
            continue
        date_match = re.search(r'fomc-meeting__date[^>]*>([^<]+)<', line)
        if not date_match or current_year is None or current_month is None:
            continue
        day_numbers = re.findall(r"\d{1,2}", date_match.group(1))
        if not day_numbers:
            continue
        decision_day = int(day_numbers[-1])
        try:
            rows.append({"date": pd.Timestamp(current_year, current_month, decision_day), "fomc_decision": 1.0})
        except ValueError:
            continue
    if not rows:
        return pd.DataFrame(columns=["fomc_decision"], index=pd.DatetimeIndex([], name="date"))
    return pd.DataFrame(rows).drop_duplicates("date").set_index("date").sort_index()


def load_market_data() -> tuple[pd.DataFrame, dict[str, str]]:
    sources: dict[str, str] = {}
    series: list[pd.DataFrame] = []
    source_metadata = load_market_source_metadata()

    instruments = {
        "gold": ("101.QO00Y", "COMEX mini gold continuous proxy"),
        "dxy": ("100.UDI", "US Dollar Index"),
        "us10y": ("171.US10Y", "US 10Y Treasury yield"),
        "gld": ("107.GLD", "SPDR Gold Shares ETF"),
        "vixy": ("107.VIXY", "VIX futures ETF proxy"),
        "spx": ("100.SPX", "S&P 500 index"),
    }

    for name, (secid, description) in instruments.items():
        try:
            frame = fetch_eastmoney_kline(secid, name)
            frame, repaired_rows = repair_ohlc(frame, name)
            frame.to_csv(RAW_DATA / f"{name}_eastmoney.csv")
            series.append(frame)
            sources[name] = f"Eastmoney {secid}: {description}"
            source_metadata.pop(name, None)
            if repaired_rows:
                sources[name] += f"; OHLC repaired rows={repaired_rows}"
        except Exception as exc:
            cached = load_cached_eastmoney_kline(name)
            if cached is not None and len(cached):
                alternative_source = ""
                if not OFFLINE_MODE:
                    try:
                        fresh, alternative_source = fetch_alternative_market_history(name, cached)
                        cached = append_new_market_rows(
                            cached,
                            fresh,
                            refresh_provisional_tail=name == "gold",
                        )
                        if alternative_source:
                            source_metadata[name] = alternative_source
                    except Exception as alternative_exc:
                        print(f"[warn] {name} alternative refresh unavailable: {compact_error(str(alternative_exc))}")
                cached, repaired_rows = repair_ohlc(cached, name)
                if alternative_source:
                    cached.to_csv(RAW_DATA / f"{name}_eastmoney.csv")
                series.append(cached)
                sources[name] = f"Eastmoney {secid}: {description} (cached fallback after refresh failure)"
                persisted_alternative = alternative_source or source_metadata.get(name, "")
                if persisted_alternative:
                    sources[name] += f"; refreshed with {persisted_alternative}"
                if repaired_rows:
                    sources[name] += f"; OHLC repaired rows={repaired_rows}"
                print(f"[warn] {name} refresh unavailable; using cached Eastmoney data")
            else:
                sources[name] = f"unavailable: {exc}"
                print(f"[warn] {name} unavailable: {exc}")

    MARKET_SOURCE_METADATA.write_text(
        json.dumps(source_metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if not any(column.startswith("gold_close") for frame in series for column in frame.columns):
        raise RuntimeError(
            "COMEX gold data is unavailable; refusing to substitute an asset with a different currency or unit"
        )

    raw_cpi = fetch_cpi_yoy()
    if len(raw_cpi):
        raw_cpi.to_csv(RAW_DATA / "us_cpi_yoy_raw.csv")
        cpi = apply_cpi_release_lag(raw_cpi)
        cpi.to_csv(RAW_DATA / "us_cpi_yoy.csv")
        sources["cpi"] = "AkShare Jin10 US CPI YoY; shifted to next month +14 days as point-in-time approximation"
        series.append(cpi)
    else:
        cpi = load_cached_csv(RAW_DATA / "us_cpi_yoy.csv")
        if len(cpi):
            sources["cpi"] = "cached CPI YoY with point-in-time release lag"
            series.append(cpi)
        else:
            sources["cpi"] = "unavailable"

    raw_cot = fetch_cot_gold()
    if len(raw_cot):
        raw_cot.to_csv(RAW_DATA / "cot_gold_raw.csv")
        cot = apply_cot_release_lag(raw_cot)
        cot.to_csv(RAW_DATA / "cot_gold.csv")
        sources["cot"] = "AkShare Jin10 CFTC gold positioning; shifted +3 calendar days to approximate Friday release"
        series.append(cot)
    else:
        cot = load_cached_csv(RAW_DATA / "cot_gold.csv")
        if len(cot):
            sources["cot"] = "cached CFTC gold positioning with release lag"
            series.append(cot)
        else:
            sources["cot"] = "unavailable"

    raw_cftc_mm = fetch_cftc_managed_money_gold(start_year=2013)
    if len(raw_cftc_mm):
        raw_cftc_mm.to_csv(RAW_DATA / "cftc_gold_managed_money_raw.csv")
        cftc_mm = apply_cot_release_lag(raw_cftc_mm)
        cftc_mm.to_csv(RAW_DATA / "cftc_gold_managed_money.csv")
        sources["cftc_managed_money"] = "CFTC official disaggregated futures-only COT; gold managed money; shifted +3 calendar days"
        series.append(cftc_mm)
    else:
        cftc_mm = load_cached_csv(RAW_DATA / "cftc_gold_managed_money.csv")
        if len(cftc_mm):
            sources["cftc_managed_money"] = "cached CFTC official disaggregated gold managed money with release lag"
            series.append(cftc_mm)
        else:
            sources["cftc_managed_money"] = "unavailable"

    for fn_name, prefix, description in [
        ("macro_usa_cpi_monthly", "cpi_mom", "AkShare US CPI MoM actual minus forecast; release-date aligned"),
        ("macro_usa_core_cpi_monthly", "core_cpi_mom", "AkShare US core CPI MoM actual minus forecast; release-date aligned"),
        ("macro_usa_non_farm", "nfp", "AkShare US nonfarm payroll actual minus forecast; release-date aligned"),
    ]:
        surprise = fetch_macro_surprise(fn_name, prefix)
        if len(surprise):
            surprise.to_csv(RAW_DATA / f"{prefix}_surprise.csv")
            sources[prefix] = description
            series.append(surprise)
        else:
            surprise = load_cached_csv(RAW_DATA / f"{prefix}_surprise.csv")
            if len(surprise):
                sources[prefix] = f"{description} (cached fallback)"
                series.append(surprise)
            else:
                sources[prefix] = "unavailable"

    try:
        tips = fetch_fred_series("DFII10", "tips_10y_real_yield")
        tips.to_csv(RAW_DATA / "tips_10y_real_yield.csv")
        sources["tips_real_yield"] = "FRED DFII10 10Y TIPS real yield"
        series.append(tips)
    except Exception as exc:
        tips = load_cached_csv(RAW_DATA / "tips_10y_real_yield.csv")
        if len(tips):
            sources["tips_real_yield"] = "FRED DFII10 10Y TIPS real yield (cached fallback after refresh failure)"
            series.append(tips)
        else:
            sources["tips_real_yield"] = f"unavailable: {compact_error(str(exc))}"
            print("[warn] tips real yield unavailable; falling back to CPI-adjusted proxy")

    try:
        cash_yield = fetch_fred_series("DGS3MO", "cash_yield_pct")
        cash_yield.to_csv(RAW_DATA / "dgs3mo_cash_yield.csv")
        sources["cash_yield"] = "FRED DGS3MO 3-month Treasury secondary-market yield"
        series.append(cash_yield)
    except Exception as exc:
        cash_yield = load_cached_csv(RAW_DATA / "dgs3mo_cash_yield.csv")
        if len(cash_yield):
            sources["cash_yield"] = (
                "FRED DGS3MO 3-month Treasury secondary-market yield "
                "(cached fallback after refresh failure)"
            )
            series.append(cash_yield)
        else:
            sources["cash_yield"] = f"unavailable: {compact_error(str(exc))}"
            print("[warn] cash yield unavailable; cash earns zero in the backtest")

    try:
        vix = fetch_cboe_vix_history()
        vix.to_csv(RAW_DATA / "vix_cboe.csv")
        sources["vix"] = "Cboe official VIX historical index CSV"
        series.append(vix)
    except Exception as exc:
        vix = load_cached_csv(RAW_DATA / "vix_cboe.csv")
        if len(vix):
            sources["vix"] = "Cboe official VIX historical index CSV (cached fallback)"
            series.append(vix)
        else:
            sources["vix"] = f"unavailable: {compact_error(str(exc))}; using VIXY ETF proxy if available"

    try:
        gpr = apply_monthly_release_lag(fetch_gpr_index(), days_after_month_end=7)
        gpr.to_csv(RAW_DATA / "gpr_index.csv")
        sources["gpr"] = "Caldara-Iacoviello Geopolitical Risk Index; month-end +7 days release lag"
        series.append(gpr)
    except Exception as exc:
        gpr = load_cached_csv(RAW_DATA / "gpr_index.csv")
        if len(gpr):
            sources["gpr"] = "Caldara-Iacoviello Geopolitical Risk Index (cached fallback)"
            series.append(gpr)
        else:
            sources["gpr"] = f"unavailable: {compact_error(str(exc))}"
            print("[warn] GPR index unavailable")

    event_columns: set[str] = set()
    try:
        fomc = fetch_fomc_decision_dates()
        fomc.to_csv(RAW_DATA / "fomc_decision_dates.csv")
        sources["fomc"] = "Federal Reserve FOMC calendar decision dates"
        series.append(fomc)
        event_columns.update(fomc.columns)
    except Exception as exc:
        fomc = load_cached_csv(RAW_DATA / "fomc_decision_dates.csv")
        if len(fomc):
            sources["fomc"] = "Federal Reserve FOMC calendar decision dates (cached fallback)"
            series.append(fomc)
            event_columns.update(fomc.columns)
        else:
            sources["fomc"] = f"unavailable: {compact_error(str(exc))}"
            print("[warn] FOMC calendar unavailable")

    data = pd.concat(series, axis=1, sort=False).sort_index().replace([np.inf, -np.inf], np.nan)
    gold_index = data.loc[data["gold_close"].notna()].index
    raw_last_observed = {
        column: str(data[column].dropna().index.max().date())
        for column in data.columns
        if data[column].notna().any()
    }
    existing_event_columns = [column for column in event_columns if column in data]
    event_source = data[existing_event_columns].fillna(0.0)
    event_frame = pd.DataFrame(0.0, index=gold_index, columns=existing_event_columns)
    for column in existing_event_columns:
        releases = event_source.loc[event_source[column] != 0, column]
        for release_date, value in releases.items():
            target_pos = int(gold_index.searchsorted(release_date, side="left"))
            if target_pos < len(gold_index):
                target_date = gold_index[target_pos]
                event_frame.loc[target_date, column] = max(
                    float(event_frame.loc[target_date, column]),
                    float(value),
                )
    non_fill_columns = [
        column
        for column in data.columns
        if (
            column.endswith(("_volume", "_amount"))
            or column in {"gold_open", "gold_high", "gold_low"}
        )
        and column not in event_columns
    ]
    non_fill_frame = data[non_fill_columns].reindex(gold_index)
    data = data.drop(
        columns=existing_event_columns + non_fill_columns,
        errors="ignore",
    ).ffill().reindex(gold_index)
    if non_fill_columns:
        data = data.join(non_fill_frame)
    if len(event_frame.columns):
        data = data.join(event_frame)

    if "us10y_close" in data and "us_cpi_yoy" in data:
        data["real_rate_proxy"] = data["us10y_close"] - data["us_cpi_yoy"]
        sources["real_rate_proxy"] = "US10Y yield minus point-in-time lagged US CPI YoY"
    else:
        data["real_rate_proxy"] = np.nan
        sources["real_rate_proxy"] = "unavailable"
    if "tips_10y_real_yield" in data and data["tips_10y_real_yield"].notna().any():
        data["real_rate"] = data["tips_10y_real_yield"]
        sources["real_rate"] = "FRED DFII10 10Y TIPS real yield"
    else:
        data["real_rate"] = data["real_rate_proxy"]
        sources["real_rate"] = "fallback to US10Y minus lagged US CPI YoY proxy"
    sources["move_index"] = "unavailable: no stable free point-in-time MOVE history source in current environment"
    sources["fed_funds_futures"] = "unavailable: Yahoo/CME futures endpoints blocked or not point-in-time stable in current environment"
    sources["gld_holdings"] = "unavailable: SPDR public endpoint currently exposes bar-list PDF, not a stable daily holdings CSV"
    sources["gold_etf_net_flow"] = "proxy only: GLD signed dollar volume flow from traded price and amount; not official ETF creations/redemptions"

    data.attrs["raw_last_observed"] = raw_last_observed

    return data, sources


def verify_data_quality(data: pd.DataFrame, sources: dict[str, str]) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    instrument_names = ["gold", "dxy", "us10y", "gld", "vix", "vixy", "spx"]
    for name in instrument_names:
        close_column = f"{name}_close"
        if close_column not in data:
            checks[name] = {"available": False, "source": sources.get(name, "unavailable")}
            continue
        subset_columns = [column for column in data.columns if column.startswith(f"{name}_")]
        subset = data[subset_columns].copy()
        high = subset.get(f"{name}_high")
        low = subset.get(f"{name}_low")
        open_ = subset.get(f"{name}_open")
        close = subset.get(close_column)
        invalid_ohlc = 0
        if high is not None and low is not None and open_ is not None and close is not None:
            invalid_ohlc = int(((high < low) | (high < open_) | (high < close) | (low > open_) | (low > close)).sum())
        checks[name] = {
            "available": True,
            "source": sources.get(name, ""),
            "rows": int(close.notna().sum()),
            "start": str(close.dropna().index.min().date()) if close.notna().any() else None,
            "end": str(close.dropna().index.max().date()) if close.notna().any() else None,
            "latestClose": None if close.dropna().empty else float(close.dropna().iloc[-1]),
            "missingClose": int(close.isna().sum()),
            "invalidOhlcRows": invalid_ohlc,
        }

    for name, columns in {
        "cpi": ["us_cpi_yoy"],
        "cot": ["cot_gold_long", "cot_gold_short", "cot_gold_net"],
        "cftc_managed_money": ["cftc_mm_gold_net", "cftc_mm_gold_net_pct_oi"],
        "cpi_mom": ["cpi_mom_surprise"],
        "core_cpi_mom": ["core_cpi_mom_surprise"],
        "nfp": ["nfp_surprise"],
        "tips_real_yield": ["tips_10y_real_yield"],
        "cash_yield": ["cash_yield_pct"],
        "real_rate_proxy": ["real_rate_proxy"],
        "real_rate": ["real_rate"],
        "gpr": ["gpr_index", "gpr_threats", "gpr_acts"],
        "fomc": ["fomc_decision"],
        "move_index": ["move_index"],
        "fed_funds_futures": ["fed_funds_futures_implied_rate"],
        "gld_holdings": ["gld_holdings"],
    }.items():
        available_columns = [column for column in columns if column in data]
        if not available_columns:
            checks[name] = {"available": False, "source": sources.get(name, "unavailable")}
            continue
        valid = data[available_columns].dropna(how="all")
        checks[name] = {
            "available": len(valid) > 0,
            "source": sources.get(name, ""),
            "rows": int(len(valid)),
            "start": str(valid.index.min().date()) if len(valid) else None,
            "end": str(valid.index.max().date()) if len(valid) else None,
            "missingRows": int(data[available_columns].isna().all(axis=1).sum()),
        }

    latest_date = data.index.max()
    reference_date = max(latest_date.normalize(), pd.Timestamp.now().normalize())
    raw_last_observed = data.attrs.get("raw_last_observed", {})
    staleness_limits = {
        "gold_close": 7,
        "dxy_close": 7,
        "gld_close": 7,
        "vix_close": 7,
        "tips_10y_real_yield": 10,
        "cash_yield_pct": 10,
        "cot_gold_net": 14,
        "cftc_mm_gold_net": 14,
        "cpi_mom_surprise": 62,
        "core_cpi_mom_surprise": 62,
        "nfp_surprise": 62,
        "gpr_index": 62,
    }
    raw_availability: dict[str, Any] = {}
    stale_warnings: list[str] = []
    for column, max_age_days in staleness_limits.items():
        last_text = raw_last_observed.get(column)
        if not last_text:
            continue
        last_date = pd.Timestamp(last_text)
        age_days = int((reference_date - last_date).days)
        stale = age_days > max_age_days
        raw_availability[column] = {
            "lastObserved": last_text,
            "ageDays": age_days,
            "maxAgeDays": max_age_days,
            "stale": stale,
        }
        if stale:
            stale_warnings.append(f"{column} stale by {age_days} days")

    ohlc_passed = all(
        (not item.get("available", False)) or item.get("invalidOhlcRows", 0) == 0
        for item in checks.values()
    )
    critical_columns = ["gold_close", "cash_yield_pct"]
    critical_stale = any(
        column not in raw_availability or bool(raw_availability[column].get("stale", False))
        for column in critical_columns
    )
    gold_source = sources.get("gold", "")
    asset_identity_passed = any(
        marker in gold_source
        for marker in ["101.QO00Y", "COMEX", "GC extension"]
    )
    quality = {
        "generatedAt": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
        "referenceDate": str(reference_date.date()),
        "passed": bool(ohlc_passed and not critical_stale and asset_identity_passed),
        "assetIdentity": {
            "expected": "COMEX gold continuous proxy in USD per troy ounce",
            "source": gold_source,
            "passed": asset_identity_passed,
        },
        "checks": checks,
        "rawAvailability": raw_availability,
        "staleWarnings": stale_warnings,
    }
    (LOCAL_LOGS / "data_quality_report.json").write_text(
        json.dumps(quality, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return quality


def compute_atr(frame: pd.DataFrame, prefix: str = "gold", window: int = 14) -> pd.Series:
    high = frame[f"{prefix}_high"]
    low = frame[f"{prefix}_low"]
    close = frame[f"{prefix}_close"]
    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(window).mean()


def risk_position_size(
    entry_price: float,
    atr: float,
    config: RiskConfig,
    risk_budget: float | None = None,
) -> float:
    """Cap portfolio loss at the configured fraction if the initial ATR stop is hit."""
    if not np.isfinite(entry_price) or entry_price <= 0 or not np.isfinite(atr) or atr <= 0:
        return 0.0
    stop_risk_fraction = config.stop_atr_multiple * atr / entry_price
    budget = config.max_single_loss if risk_budget is None else float(risk_budget)
    risk_limited = budget / stop_risk_fraction if stop_risk_fraction > 0 else 0.0
    return float(max(0.0, min(config.max_position, config.max_leverage, risk_limited)))


def trend_risk_budget(row: Any, config: RiskConfig) -> float:
    if not config.dynamic_trend_risk_enabled:
        return config.max_single_loss
    ret_120 = row.get("ret_120", np.nan) if isinstance(row, pd.Series) else getattr(row, "ret_120", np.nan)
    close = row.get("gold_close", np.nan) if isinstance(row, pd.Series) else getattr(row, "gold_close", np.nan)
    sma_120 = row.get("sma_120", np.nan) if isinstance(row, pd.Series) else getattr(row, "sma_120", np.nan)
    strong = (
        np.isfinite(ret_120)
        and np.isfinite(close)
        and np.isfinite(sma_120)
        and ret_120 >= config.strong_trend_ret_120_threshold
        and close > sma_120
    )
    return config.strong_trend_risk_budget if strong else config.normal_trend_risk_budget


def build_features(data: pd.DataFrame, config: RiskConfig) -> pd.DataFrame:
    """Build only the features consumed by the adopted strategy and website."""
    frame = data.copy()
    close = frame["gold_close"]
    frame["ret_1"] = close.pct_change()
    for window in [5, 20, 60, 120]:
        frame[f"ret_{window}"] = close.pct_change(window)
        frame[f"sma_{window}"] = close.rolling(window).mean()
    frame["sma_gap_60"] = close / frame["sma_60"] - 1
    frame["vol_20"] = frame["ret_1"].rolling(20).std() * math.sqrt(252)
    frame["atr"] = compute_atr(frame, window=config.atr_window)
    frame["atr_pct"] = frame["atr"] / close
    frame["drawdown_120"] = close / close.rolling(120).max() - 1
    return frame


def hmm_feature_columns(frame: pd.DataFrame, policy: str = "gold_core") -> list[str]:
    if policy != "gold_core":
        raise ValueError(f"unknown HMM feature policy: {policy}")
    return [
        column
        for column in ["ret_20", "vol_20", "sma_gap_60", "drawdown_120"]
        if column in frame.columns
    ]


def fit_hmm(
    frame: pd.DataFrame,
    train_mask: pd.Series,
    feature_cols: list[str] | None = None,
    feature_policy: str = "gold_core",
) -> tuple[Pipeline, dict[int, str], pd.DataFrame]:
    cols = feature_cols or [
        column
        for column in hmm_feature_columns(frame, feature_policy)
        if frame.loc[train_mask, column].notna().sum() >= 300
    ]
    if len(cols) < 4:
        raise RuntimeError(f"Not enough usable HMM features: {cols}")
    hmm_frame = frame[cols].copy()
    train_data = hmm_frame.loc[train_mask].dropna()
    if len(train_data) < 300:
        raise RuntimeError("Not enough data for HMM training")

    model = GaussianHMM(
        n_components=4,
        covariance_type="diag",
        n_iter=200,
        tol=1e-4,
        random_state=42,
    )
    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("hmm", model),
        ]
    )
    transformed = pipe[:-1].fit_transform(train_data)
    pipe.named_steps["hmm"].fit(transformed)

    all_transformed = pipe[:-1].transform(hmm_frame)
    hmm = pipe.named_steps["hmm"]
    covars = hmm.covars_
    if covars.ndim == 3:
        covars = np.array([np.diag(covar) for covar in covars])
    covars = np.maximum(covars, 1e-6)
    log_emissions = []
    for state in range(hmm.n_components):
        diff = all_transformed - hmm.means_[state]
        log_det = np.log(covars[state]).sum()
        quad = (diff * diff / covars[state]).sum(axis=1)
        log_emissions.append(-0.5 * (log_det + quad))
    log_emissions_array = np.vstack(log_emissions).T
    log_emissions_array = log_emissions_array - log_emissions_array.max(axis=1, keepdims=True)
    emissions = np.exp(log_emissions_array)
    emissions = np.maximum(emissions, 1e-12)

    posterior = np.zeros_like(emissions)
    previous = np.maximum(hmm.startprob_, 1e-12)
    previous = previous / previous.sum()
    transition = np.maximum(hmm.transmat_, 1e-12)
    transition = transition / transition.sum(axis=1, keepdims=True)
    for i, emission in enumerate(emissions):
        prior = previous if i == 0 else previous @ transition
        filtered = prior * emission
        total = filtered.sum()
        if not np.isfinite(total) or total <= 0:
            filtered = np.full(hmm.n_components, 1.0 / hmm.n_components)
        else:
            filtered = filtered / total
        posterior[i] = filtered
        previous = filtered
    hidden = posterior.argmax(axis=1)

    state_stats = []
    temp = frame.copy()
    temp["hmm_raw_state"] = hidden
    usable_train_mask = train_mask & hmm_frame.notna().all(axis=1)
    for state in range(4):
        subset = temp.loc[usable_train_mask & (temp["hmm_raw_state"] == state)]
        state_stats.append(
            {
                "state": state,
                "ret20": subset["ret_20"].mean(),
                "vol20": subset["vol_20"].mean(),
                "drawdown": subset["drawdown_120"].mean(),
                "trend": subset["sma_gap_60"].mean(),
                "count": int(len(subset)),
            }
        )
    stats = pd.DataFrame(state_stats).fillna(0)

    panic_state = stats.sort_values("vol20", ascending=False).iloc[0]["state"].item()
    remaining = stats[stats["state"] != panic_state].copy()
    bull_state = remaining.sort_values(["trend", "ret20"], ascending=False).iloc[0]["state"].item()
    remaining = remaining[remaining["state"] != bull_state]
    bear_state = remaining.sort_values(["trend", "ret20"], ascending=True).iloc[0]["state"].item()
    range_state = remaining[remaining["state"] != bear_state].iloc[0]["state"].item()

    mapping = {
        int(bull_state): "牛市",
        int(bear_state): "熊市",
        int(range_state): "震荡",
        int(panic_state): "恐慌",
    }

    state_frame = pd.DataFrame(index=frame.index)
    state_frame["hmm_raw_state"] = hidden
    state_frame["market_state"] = state_frame["hmm_raw_state"].map(mapping)
    state_frame["market_state_code"] = state_frame["market_state"].map(STATE_TO_CODE)
    for raw_state in range(4):
        label = mapping[raw_state]
        state_frame[f"hmm_prob_{STATE_TO_CODE[label]}"] = posterior[:, raw_state]

    return pipe, mapping, state_frame


def fit_hmm_walk_forward(
    frame: pd.DataFrame,
    initial_train_end: pd.Timestamp,
    retrain_every_days: int,
    feature_policy: str = "gold_core",
) -> tuple[Pipeline, dict[int, str], pd.DataFrame]:
    """Generate HMM states using only information available before each prediction block."""
    initial_mask = pd.Series(frame.index <= initial_train_end, index=frame.index)
    fixed_feature_cols = [
        column
        for column in hmm_feature_columns(frame, feature_policy)
        if frame.loc[initial_mask, column].notna().sum() >= 300
    ]
    latest_pipe, latest_mapping, initial_states = fit_hmm(
        frame,
        initial_mask,
        fixed_feature_cols,
        feature_policy,
    )
    diagnostics: list[dict[str, Any]] = []

    def record_diagnostic(
        cutoff: pd.Timestamp,
        pipe: Pipeline,
        mapping: dict[int, str],
        states: pd.DataFrame,
        mask: pd.Series,
    ) -> None:
        usable_index = frame.loc[mask, fixed_feature_cols].dropna().index
        occupancy = states.loc[usable_index, "market_state"].value_counts(normalize=True)
        hmm_model = pipe.named_steps["hmm"]
        diagnostics.append(
            {
                "cutoff": str(cutoff.date()),
                "feature_policy": feature_policy,
                "feature_count": len(fixed_feature_cols),
                "features": "|".join(fixed_feature_cols),
                "train_rows": int(len(usable_index)),
                "train_start": str(usable_index.min().date()) if len(usable_index) else None,
                "train_end": str(usable_index.max().date()) if len(usable_index) else None,
                "converged": bool(hmm_model.monitor_.converged),
                "bull_occupancy": float(occupancy.get("牛市", 0.0)),
                "bear_occupancy": float(occupancy.get("熊市", 0.0)),
                "range_occupancy": float(occupancy.get("震荡", 0.0)),
                "panic_occupancy": float(occupancy.get("恐慌", 0.0)),
                "state_mapping": json.dumps(mapping, ensure_ascii=False, sort_keys=True),
                "transition_matrix": json.dumps(hmm_model.transmat_.round(8).tolist()),
            }
        )

    record_diagnostic(initial_train_end, latest_pipe, latest_mapping, initial_states, initial_mask)
    state_columns = list(initial_states.columns)
    result = initial_states.copy()
    result.loc[frame.index > initial_train_end, state_columns] = np.nan
    start_pos = frame.index.get_indexer([initial_train_end], method="nearest")[0] + 1

    for pred_start in range(start_pos, len(frame.index), retrain_every_days):
        pred_end = min(pred_start + retrain_every_days, len(frame.index))
        cutoff = frame.index[pred_start - 1]
        if pred_start == start_pos:
            candidate_states = initial_states
        else:
            expanding_mask = pd.Series(frame.index <= cutoff, index=frame.index)
            latest_pipe, latest_mapping, candidate_states = fit_hmm(
                frame,
                expanding_mask,
                fixed_feature_cols,
                feature_policy,
            )
            record_diagnostic(cutoff, latest_pipe, latest_mapping, candidate_states, expanding_mask)
        block_index = frame.index[pred_start:pred_end]
        result.loc[block_index, state_columns] = candidate_states.loc[block_index, state_columns]

    result["hmm_raw_state"] = pd.to_numeric(result["hmm_raw_state"], errors="coerce").astype("Int64")
    result.attrs["hmm_diagnostics"] = diagnostics
    result.attrs["hmm_feature_columns"] = fixed_feature_cols
    return latest_pipe, latest_mapping, result


def primary_long_signal(frame: pd.DataFrame) -> pd.Series:
    """Return the single adopted 120-day trend entry regime."""
    return (
        (frame["gold_close"] > frame["sma_120"])
        | ((frame["sma_20"] > frame["sma_60"]) & (frame["sma_60"] > frame["sma_120"]))
    )


def make_cusum_events(
    frame: pd.DataFrame,
    signal: pd.Series,
    threshold_mult: float,
    min_gap: int,
) -> pd.Series:
    """Sample absolute price shocks while the adopted trend regime is active."""
    returns = frame["gold_close"].pct_change().fillna(0)
    daily_vol = returns.ewm(span=50, adjust=False).std().replace(0, np.nan).ffill()
    events = pd.Series(False, index=frame.index)
    s_pos = 0.0
    s_neg = 0.0
    last_pos = -10_000
    for pos, date in enumerate(frame.index):
        if not bool(signal.loc[date]) or not np.isfinite(daily_vol.loc[date]):
            s_pos = 0.0
            s_neg = 0.0
            continue
        ret = float(returns.loc[date])
        threshold = float(threshold_mult * daily_vol.loc[date])
        s_pos = max(0.0, s_pos + ret)
        s_neg = min(0.0, s_neg + ret)
        if (s_pos > threshold or abs(s_neg) > threshold) and pos - last_pos >= min_gap:
            events.iloc[pos] = True
            last_pos = pos
            s_pos = 0.0
            s_neg = 0.0
    return events


def make_meta_events(frame: pd.DataFrame, signal: pd.Series, config: RiskConfig) -> pd.Series:
    return make_cusum_events(
        frame,
        signal,
        config.cusum_threshold_mult,
        config.meta_event_gap_days,
    )


def generate_signals(
    frame: pd.DataFrame,
    config: RiskConfig,
    *,
    start_at: pd.Timestamp | None = None,
) -> pd.DataFrame:
    signal_frame = frame.copy()
    primary_signal = primary_long_signal(signal_frame)
    events = make_meta_events(signal_frame, primary_signal, config)
    signal_frame["tb_event"] = events
    signal_frame["tb_accepted_event"] = events
    signal_frame["primary_trend_signal"] = primary_signal
    signal_frame["atr_stop_enabled"] = bool(config.atr_stop_enabled)
    signal_frame["atr_profit_enabled"] = bool(config.atr_profit_enabled)
    signal_frame["hmm_exit_enabled"] = bool(config.hmm_exit_enabled)
    signal_frame["target_position_override"] = np.nan
    signal_frame["payoff_ratio"] = config.profit_atr_multiple / config.stop_atr_multiple
    strong_trend = (
        config.dynamic_trend_risk_enabled
        & (signal_frame["gold_close"] > signal_frame["sma_120"])
        & (signal_frame["ret_120"] >= config.strong_trend_ret_120_threshold)
    )
    signal_frame["trend_risk_budget"] = np.where(
        strong_trend,
        config.strong_trend_risk_budget,
        config.normal_trend_risk_budget if config.dynamic_trend_risk_enabled else config.max_single_loss,
    )

    positions = []
    stop_prices = []
    take_profit_prices = []
    execution_actions = []
    exit_reasons = []
    raw_signals = []
    guides = []
    fill_prices = []
    in_position = False
    current_position = 0.0
    stop_price = np.nan
    take_profit_price = np.nan
    hmm_exit_streak = 0
    accepted_arr = events.reindex(signal_frame.index).fillna(False).to_numpy()

    for i, row in enumerate(signal_frame.itertuples()):
        action = "持有/观望"
        raw_signal = "hold"
        guide = "持有/观望"
        exit_reason = ""
        fill_price = np.nan
        exited_this_bar = False
        close = row.gold_close
        high = row.gold_high
        low = row.gold_low
        atr = row.atr
        flat_condition = (not bool(row.primary_trend_signal)) or (
            (row.gold_close < row.sma_60) and row.market_state in ["熊市", "恐慌"]
        )

        if start_at is not None and row.Index < start_at:
            positions.append(0.0)
            stop_prices.append(np.nan)
            take_profit_prices.append(np.nan)
            execution_actions.append("持有/观望")
            exit_reasons.append("")
            raw_signals.append("flat")
            guides.append("卖出/空仓")
            fill_prices.append(np.nan)
            continue

        if in_position:
            hit_profit = (
                config.atr_profit_enabled
                and np.isfinite(take_profit_price)
                and high >= take_profit_price
            )
            hit_stop = (
                config.atr_stop_enabled
                and np.isfinite(stop_price)
                and low <= stop_price
            )
            hmm_exit_setup = row.market_state in ["熊市", "恐慌"] and close < row.sma_60
            hmm_exit_streak = hmm_exit_streak + 1 if hmm_exit_setup else 0
            hmm_exit = config.hmm_exit_enabled and hmm_exit_streak >= config.hmm_exit_confirmation_days
            if hit_stop or hit_profit or hmm_exit:
                if hit_stop:
                    fill_price = min(float(row.gold_open), stop_price) if np.isfinite(row.gold_open) else stop_price
                elif hit_profit:
                    fill_price = max(float(row.gold_open), take_profit_price) if np.isfinite(row.gold_open) else take_profit_price
                else:
                    fill_price = close
                in_position = False
                current_position = 0.0
                stop_price = np.nan
                take_profit_price = np.nan
                hmm_exit_streak = 0
                exited_this_bar = True
                action = "卖出"
                raw_signal = "flat"
                guide = "卖出/空仓"
                if hit_stop:
                    exit_reason = "atr_stop"
                elif hit_profit:
                    exit_reason = "atr_take_profit"
                else:
                    exit_reason = "hmm_trend_exit"
            else:
                action = "持有"
                guide = "持有"

        if not in_position and not exited_this_bar and accepted_arr[i] and np.isfinite(atr):
            in_position = True
            current_position = risk_position_size(close, atr, config, trend_risk_budget(row, config))
            fill_price = close
            stop_price = (
                close - config.stop_atr_multiple * atr
                if config.atr_stop_enabled
                else np.nan
            )
            take_profit_price = (
                close + config.profit_atr_multiple * atr
                if config.atr_profit_enabled
                else np.nan
            )
            hmm_exit_streak = 0
            action = "买入"
            raw_signal = "long"
            guide = "买入"
        elif not in_position and action != "卖出":
            current_position = 0.0
            raw_signal = "flat" if flat_condition else "hold"
            guide = "卖出/空仓" if flat_condition else "持有/观望"

        positions.append(current_position if in_position else 0.0)
        stop_prices.append(stop_price if in_position else np.nan)
        take_profit_prices.append(take_profit_price if in_position else np.nan)
        execution_actions.append(action)
        exit_reasons.append(exit_reason)
        raw_signals.append(raw_signal)
        guides.append(guide)
        fill_prices.append(fill_price)

    signal_frame["position"] = positions
    signal_frame["atr_stop"] = stop_prices
    signal_frame["tb_take_profit"] = take_profit_prices
    signal_frame["execution_action"] = execution_actions
    signal_frame["exit_reason"] = exit_reasons
    signal_frame["raw_signal"] = raw_signals
    signal_frame["guide"] = guides
    signal_frame["fill_price"] = fill_prices
    return signal_frame

def backtest_next_open(
    signal_frame: pd.DataFrame,
    test_mask: pd.Series,
    config: RiskConfig,
    *,
    cost_bps: float | None = None,
    write_log: bool = True,
) -> tuple[pd.DataFrame, dict[str, float]]:
    bt = signal_frame.loc[test_mask].copy()
    required_ohlc = ["gold_open", "gold_high", "gold_low", "gold_close"]
    missing_ohlc = [column for column in required_ohlc if column not in bt]
    if missing_ohlc:
        raise RuntimeError(f"Strict next-open backtest requires OHLC columns: {missing_ohlc}")
    if config.cash_yield_enabled and (
        "cash_yield_pct" not in bt or bt["cash_yield_pct"].notna().sum() < 2
    ):
        raise RuntimeError("Cash-yield-enabled backtest requires point-in-time cash_yield_pct data")
    for column in required_ohlc:
        invalid_price = ~np.isfinite(bt[column]) | (bt[column] <= 0)
        if invalid_price.any():
            bad_dates = ", ".join(str(date.date()) for date in bt.index[invalid_price][:5])
            label = "opening prices" if column == "gold_open" else f"{column} prices"
            raise RuntimeError(
                f"Strict next-open backtest requires valid {label}; invalid dates: {bad_dates}"
            )
    invalid_range = (
        (bt["gold_high"] < bt[["gold_open", "gold_close", "gold_low"]].max(axis=1))
        | (bt["gold_low"] > bt[["gold_open", "gold_close", "gold_high"]].min(axis=1))
    )
    if invalid_range.any():
        bad_dates = ", ".join(str(date.date()) for date in bt.index[invalid_range][:5])
        raise RuntimeError(f"Strict next-open backtest requires internally valid OHLC ranges: {bad_dates}")

    bt["benchmark_ret"] = bt["gold_close"].pct_change().fillna(0)
    equity = 1.0
    peak = 1.0
    risk_peak = 1.0
    cash = 1.0
    cash_interest_earned = 0.0
    previous_date: pd.Timestamp | None = None
    previous_cash_yield_pct = np.nan
    units = 0.0
    stop_price = np.nan
    take_profit_price = np.nan
    entry_price = np.nan
    entry_target_fraction = 0.0
    pending_entry_atr = np.nan
    pending_entry_risk_budget = config.max_single_loss
    pending_entry_target_override = np.nan
    pending_entry = False
    pending_exit = False
    pending_exit_reason = ""
    hmm_exit_streak = 0
    hard_drawdown_cooldown = 0
    rows: list[dict[str, Any]] = []
    trading_cost_bps = config.realistic_cost_bps if cost_bps is None else float(cost_bps)

    for date, row in bt.iterrows():
        open_price = float(row["gold_open"])
        close_price = float(row["gold_close"])
        equity_before = equity
        cash_interest = 0.0
        risk_free_return = 0.0
        applied_cash_yield_pct = np.nan
        if (
            previous_date is not None
            and np.isfinite(previous_cash_yield_pct)
        ):
            calendar_days = max(int((pd.Timestamp(date) - previous_date).days), 0)
            applied_cash_yield_pct = max(
                float(previous_cash_yield_pct) - config.cash_yield_haircut_bps / 100.0,
                0.0,
            )
            if calendar_days > 0 and applied_cash_yield_pct > 0:
                risk_free_return = (
                    (1.0 + applied_cash_yield_pct / 100.0) ** (calendar_days / 365.0) - 1.0
                )
                if config.cash_yield_enabled and cash > 0:
                    cash_interest = cash * risk_free_return
                    cash += cash_interest
                    cash_interest_earned += cash_interest
        turnover = 0.0
        exit_reason = ""
        execution_actions: list[str] = []
        entry_fill_price = np.nan
        exit_fill_price = np.nan

        cost_rate = trading_cost_bps / 10000
        open_equity = cash + units * open_price

        if pending_exit and units > 0:
            proceeds = units * open_price
            turnover += proceeds / equity_before if equity_before > 0 else 0.0
            cash += proceeds * (1 - cost_rate)
            units = 0.0
            entry_price = np.nan
            entry_target_fraction = 0.0
            stop_price = np.nan
            take_profit_price = np.nan
            hmm_exit_streak = 0
            exit_reason = pending_exit_reason or "next_open_rule_exit"
            exit_fill_price = open_price
            execution_actions.append("卖出")

        if units > 0 and not pending_exit:
            gap_stop = np.isfinite(stop_price) and open_price <= stop_price
            gap_profit = np.isfinite(take_profit_price) and open_price >= take_profit_price
            if gap_stop or gap_profit:
                proceeds = units * open_price
                turnover += proceeds / equity_before if equity_before > 0 else 0.0
                cash += proceeds * (1 - cost_rate)
                units = 0.0
                entry_price = np.nan
                entry_target_fraction = 0.0
                stop_price = np.nan
                take_profit_price = np.nan
                hmm_exit_streak = 0
                exit_reason = "atr_stop_gap" if gap_stop else "atr_take_profit_gap"
                exit_fill_price = open_price
                execution_actions.append("卖出")

        open_equity = cash + units * open_price
        drawdown_before = open_equity / risk_peak - 1
        drawdown_scale = 1.0
        if hard_drawdown_cooldown == 0 and drawdown_before <= -config.max_drawdown_hard:
            hard_drawdown_cooldown = config.hard_drawdown_cooldown_days
        if hard_drawdown_cooldown > 0:
            drawdown_scale = config.live_hard_drawdown_position
            hard_drawdown_cooldown -= 1
            if hard_drawdown_cooldown == 0:
                risk_peak = max(open_equity, 1e-12)
        elif drawdown_before <= -config.max_drawdown_soft:
            drawdown_scale = config.live_soft_drawdown_position
        if units > 0 and drawdown_scale < 1.0:
            target_notional_after_drawdown = entry_target_fraction * drawdown_scale * open_equity
            current_notional = units * open_price
            reduction_notional = max(0.0, current_notional - target_notional_after_drawdown)
            if reduction_notional > 1e-12:
                reduction_units = min(units, reduction_notional / open_price)
                proceeds = reduction_units * open_price
                turnover += proceeds / equity_before if equity_before > 0 else 0.0
                cash += proceeds * (1 - cost_rate)
                units -= reduction_units
                exit_fill_price = open_price
                execution_actions.append("卖出（回撤减仓）")
                exit_reason = "drawdown_risk_reduction"
                if units <= 1e-12:
                    units = 0.0
                    entry_price = np.nan
                    entry_target_fraction = 0.0
                    stop_price = np.nan
                    take_profit_price = np.nan
                    hmm_exit_streak = 0
        if pending_entry and units <= 0 and drawdown_scale > 0:
            if np.isfinite(pending_entry_target_override):
                base_target_fraction = float(pending_entry_target_override)
            else:
                base_target_fraction = risk_position_size(
                    open_price,
                    pending_entry_atr,
                    config,
                    pending_entry_risk_budget,
                )
            base_target_fraction = min(
                config.max_position,
                config.max_leverage,
                max(0.0, base_target_fraction),
            )
            target_fraction = base_target_fraction * drawdown_scale
            available_equity = max(cash, 0.0)
            target_notional = target_fraction * available_equity
            notional = min(target_notional, available_equity / (1 + cost_rate))
            entry_cost = notional * cost_rate
            units = notional / open_price if open_price > 0 else 0.0
            entry_price = open_price if units > 0 else np.nan
            entry_target_fraction = base_target_fraction if units > 0 else 0.0
            entry_fill_price = open_price if units > 0 else np.nan
            cash -= notional + entry_cost
            turnover += notional / equity_before if equity_before > 0 else 0.0
            execution_actions.append("买入")
            if np.isfinite(pending_entry_atr):
                stop_price = (
                    open_price - config.stop_atr_multiple * pending_entry_atr
                    if bool(row.get("atr_stop_enabled", config.atr_stop_enabled))
                    else np.nan
                )
                take_profit_price = (
                    open_price + config.profit_atr_multiple * pending_entry_atr
                    if bool(row.get("atr_profit_enabled", config.atr_profit_enabled))
                    else np.nan
                )

        if units > 0:
            hit_stop = np.isfinite(stop_price) and float(row["gold_low"]) <= stop_price
            hit_profit = np.isfinite(take_profit_price) and float(row["gold_high"]) >= take_profit_price
            if hit_stop:
                end_price = stop_price
                exit_reason = "atr_stop"
            elif hit_profit:
                end_price = take_profit_price
                exit_reason = "atr_take_profit"
            else:
                end_price = close_price
            if hit_stop or hit_profit:
                proceeds = units * end_price
                turnover += proceeds / equity_before if equity_before > 0 else 0.0
                cash += proceeds * (1 - cost_rate)
                units = 0.0
                entry_price = np.nan
                entry_target_fraction = 0.0
                stop_price = np.nan
                take_profit_price = np.nan
                hmm_exit_streak = 0
                exit_fill_price = end_price
                execution_actions.append("卖出")

        equity = cash + units * close_price
        if cash < -1e-10 or units < -1e-12 or units * close_price > equity + 1e-9:
            raise RuntimeError(
                f"Long-only cash accounting invariant failed on {date}: cash={cash}, units={units}, equity={equity}"
            )

        strategy_ret = equity / equity_before - 1
        peak = max(peak, equity)
        if hard_drawdown_cooldown == 0:
            risk_peak = max(risk_peak, equity)
        pending_entry = False
        pending_exit = False
        pending_exit_reason = ""
        pending_entry_target_override = np.nan
        target_override = np.nan
        entry_ready = False
        entry_blocked_by_drawdown = False
        pending_entry_drawdown_scale = 1.0

        if units > 0:
            hmm_exit_setup = row["market_state"] in ["熊市", "恐慌"] and close_price < float(row["sma_60"])
            hmm_exit_enabled = bool(row.get("hmm_exit_enabled", config.hmm_exit_enabled))
            hmm_exit_streak = hmm_exit_streak + 1 if hmm_exit_enabled and hmm_exit_setup else 0
            hmm_exit = hmm_exit_enabled and hmm_exit_streak >= config.hmm_exit_confirmation_days
            pending_exit = bool(hmm_exit)
            if hmm_exit:
                pending_exit_reason = "next_open_hmm_exit"
        else:
            target_override = float(row.get("target_position_override", np.nan))
            entry_ready = np.isfinite(row.get("atr", np.nan)) or np.isfinite(target_override)
        entry_signal = units <= 0 and bool(row.get("tb_accepted_event", False)) and entry_ready
        entry_blocked_by_drawdown = bool(entry_signal and hard_drawdown_cooldown > 0)
        if entry_signal and not entry_blocked_by_drawdown:
            pending_entry = True
            pending_entry_atr = float(row.get("atr", np.nan))
            pending_entry_risk_budget = trend_risk_budget(row, config)
            pending_entry_target_override = target_override
            pending_entry_drawdown_scale = drawdown_scale if drawdown_scale > 0 else 1.0

        position = units * close_price / equity if units > 0 and equity > 0 else 0.0
        if position < -1e-12 or position > 1.0 + 1e-9:
            raise RuntimeError(f"Long-only unlevered position invariant failed on {date}: {position}")
        desired_position = position
        if pending_entry:
            if np.isfinite(pending_entry_target_override):
                desired_position = (
                    min(1.0, max(0.0, pending_entry_target_override))
                    * pending_entry_drawdown_scale
                )
            else:
                desired_position = (
                    risk_position_size(
                        close_price,
                        pending_entry_atr,
                        config,
                        pending_entry_risk_budget,
                    )
                    * pending_entry_drawdown_scale
                )
        elif pending_exit:
            desired_position = 0.0
        if entry_blocked_by_drawdown:
            guide = f"回撤冷却（剩余 {hard_drawdown_cooldown} 日）"
            raw_signal = "risk_halted"
        elif pending_exit:
            guide = "卖出（下一交易日开盘）"
            raw_signal = "flat_pending"
        elif pending_entry:
            guide = "买入（下一交易日开盘）"
            raw_signal = "long_pending"
        elif position > 0:
            guide = "持有"
            raw_signal = "long"
        else:
            guide = "卖出/空仓" if not bool(row.get("primary_trend_signal", False)) else "持有/观望"
            raw_signal = "flat" if not bool(row.get("primary_trend_signal", False)) else "hold"

        rows.append(
            {
                "date": date,
                "strategy_ret": strategy_ret,
                "benchmark_ret": row["benchmark_ret"],
                "position": position,
                "desired_position": desired_position,
                "turnover": turnover,
                "cash": cash,
                "cash_interest": cash_interest,
                "cash_interest_earned": cash_interest_earned,
                "cash_yield_pct_used": applied_cash_yield_pct,
                "risk_free_return": risk_free_return,
                "units": units,
                "entry_price": entry_price,
                "entry_target_fraction": entry_target_fraction,
                "entry_fill_price": entry_fill_price,
                "exit_fill_price": exit_fill_price,
                "equity": equity,
                "benchmark_equity": np.nan,
                "drawdown": equity / peak - 1,
                "drawdown_scale": drawdown_scale,
                "hard_drawdown_cooldown": hard_drawdown_cooldown,
                "execution_action": "/".join(execution_actions) if execution_actions else "持有/观望",
                "exit_reason": exit_reason,
                "pending_exit_reason": pending_exit_reason,
                "pending_entry": pending_entry,
                "pending_exit": pending_exit,
                "entry_blocked_by_drawdown": entry_blocked_by_drawdown,
                "guide": guide,
                "raw_signal": raw_signal,
                "stop_price": stop_price,
                "take_profit_price": take_profit_price,
                "atr_stop": stop_price,
                "tb_take_profit": take_profit_price,
            }
        )
        previous_date = pd.Timestamp(date)
        current_cash_yield = float(row.get("cash_yield_pct", np.nan))
        if np.isfinite(current_cash_yield):
            previous_cash_yield_pct = current_cash_yield
    live = pd.DataFrame(rows).set_index("date")
    live["benchmark_equity"] = (1 + live["benchmark_ret"].fillna(0)).cumprod()
    days = max(len(live), 1)
    total_return = live["equity"].iloc[-1] - 1
    benchmark_return = live["benchmark_equity"].iloc[-1] - 1
    annual_return = live["equity"].iloc[-1] ** (252 / days) - 1
    annual_vol = live["strategy_ret"].std() * math.sqrt(252)
    excess_ret = live["strategy_ret"] - live["risk_free_return"]
    sharpe = (
        excess_ret.mean() / excess_ret.std() * math.sqrt(252)
        if excess_ret.std() and np.isfinite(excess_ret.std())
        else 0.0
    )
    active_days = float((live["position"] > 0).mean())
    win_days = live.loc[live["strategy_ret"] != 0, "strategy_ret"]
    downside = excess_ret.clip(upper=0)
    downside_vol = downside.std() * math.sqrt(252)
    sortino = (
        excess_ret.mean() * 252 / downside_vol
        if downside_vol and np.isfinite(downside_vol)
        else 0.0
    )
    max_drawdown = float(live["drawdown"].min())
    calmar = annual_return / abs(max_drawdown) if max_drawdown < 0 else 0.0
    benchmark_drawdown = live["benchmark_equity"] / live["benchmark_equity"].cummax() - 1
    benchmark_vol = live["benchmark_ret"].std() * math.sqrt(252)
    benchmark_excess_ret = live["benchmark_ret"] - live["risk_free_return"]
    benchmark_sharpe = (
        benchmark_excess_ret.mean() / benchmark_excess_ret.std() * math.sqrt(252)
        if benchmark_excess_ret.std()
        else 0.0
    )
    metrics = {
        "execution_model": "t_close_signal_t_plus_1_open_with_intraday_atr_fills",
        "cost_bps": trading_cost_bps,
        "sharpe_definition": "annualized_excess_return_over_lagged_haircut_cash_yield",
        "total_return": float(total_return),
        "benchmark_return": float(benchmark_return),
        "annual_return": float(annual_return),
        "annual_vol": float(annual_vol),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "calmar": float(calmar),
        "max_drawdown": max_drawdown,
        "benchmark_annual_vol": float(benchmark_vol),
        "benchmark_sharpe": float(benchmark_sharpe),
        "benchmark_max_drawdown": float(benchmark_drawdown.min()),
        "active_day_ratio": active_days,
        "daily_win_rate_when_active": float((win_days > 0).mean()) if len(win_days) else 0.0,
        "test_trades": int((live["turnover"] > 0).sum()),
        "entries": int(live["execution_action"].str.contains("买入").sum()),
        "exits": int(live["execution_action"].str.contains("卖出").sum()),
        "turnover": float(live["turnover"].sum()),
        "drawdown_scaled_days": int((live["drawdown_scale"] < 1).sum()),
        "cash_interest_earned": float(live["cash_interest"].sum()),
    }
    if write_log:
        live.to_csv(LOCAL_LOGS / "gold_live_execution.csv", encoding="utf-8-sig")
    return live, metrics


def build_yearly_backtest_report(
    backtest_frame: pd.DataFrame,
    *,
    cost_bps: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for year, subset in backtest_frame.groupby(backtest_frame.index.year):
        strategy_ret = subset["strategy_ret"].fillna(0)
        risk_free_ret = subset.get("risk_free_return", pd.Series(0.0, index=subset.index)).fillna(0)
        excess_ret = strategy_ret - risk_free_ret
        benchmark_ret = subset["benchmark_ret"].fillna(0)
        equity = (1 + strategy_ret).cumprod()
        benchmark = (1 + benchmark_ret).cumprod()
        drawdown = equity / equity.cummax() - 1
        volatility = excess_ret.std()
        rows.append(
            {
                "year": int(year),
                "cost_bps": float(cost_bps),
                "total_return": float(equity.iloc[-1] - 1),
                "benchmark_return": float(benchmark.iloc[-1] - 1),
                "sharpe": float(excess_ret.mean() / volatility * math.sqrt(252)) if volatility else 0.0,
                "max_drawdown": float(drawdown.min()),
                "average_position": float(subset["position"].mean()),
                "turnover": float(subset["turnover"].sum()),
            }
        )
    pd.DataFrame(rows).to_csv(LOCAL_LOGS / "gold_backtest_yearly.csv", index=False, encoding="utf-8-sig")
    return rows


def moving_block_bootstrap_summary(
    backtest_frame: pd.DataFrame,
    *,
    block_length: int = 20,
    simulations: int = 2_000,
    seed: int = 20260713,
) -> dict[str, Any]:
    """Quantify path uncertainty without treating shuffled daily returns as independent."""
    returns = backtest_frame["strategy_ret"].fillna(0.0).to_numpy(dtype=float)
    risk_free_returns = backtest_frame.get(
        "risk_free_return",
        pd.Series(0.0, index=backtest_frame.index),
    ).fillna(0.0).to_numpy(dtype=float)
    if len(returns) < block_length:
        raise RuntimeError("Backtest is too short for the configured moving-block bootstrap")
    rng = np.random.default_rng(seed)
    blocks_needed = math.ceil(len(returns) / block_length)
    max_start = len(returns) - block_length
    total_returns = np.empty(simulations)
    sharpes = np.empty(simulations)
    max_drawdowns = np.empty(simulations)
    for simulation in range(simulations):
        starts = rng.integers(0, max_start + 1, size=blocks_needed)
        sample_indices = np.concatenate(
            [np.arange(start, start + block_length) for start in starts]
        )[: len(returns)]
        sample = returns[sample_indices]
        sample_risk_free = risk_free_returns[sample_indices]
        sample_excess = sample - sample_risk_free
        equity = np.cumprod(1.0 + sample)
        drawdown = equity / np.maximum.accumulate(equity) - 1.0
        volatility = sample_excess.std(ddof=1)
        total_returns[simulation] = equity[-1] - 1.0
        sharpes[simulation] = (
            sample_excess.mean() / volatility * math.sqrt(252)
            if volatility and np.isfinite(volatility)
            else 0.0
        )
        max_drawdowns[simulation] = drawdown.min()

    def percentiles(values: np.ndarray) -> dict[str, float]:
        low, median, high = np.quantile(values, [0.05, 0.50, 0.95])
        return {"p05": float(low), "median": float(median), "p95": float(high)}

    return {
        "method": "moving_block_bootstrap_on_realized_8bps_daily_returns",
        "development_sample_only": True,
        "multiple_testing_correction": False,
        "block_length_days": block_length,
        "simulations": simulations,
        "seed": seed,
        "total_return": percentiles(total_returns),
        "sharpe": percentiles(sharpes),
        "max_drawdown": percentiles(max_drawdowns),
        "probability_positive_total_return": float((total_returns > 0).mean()),
        "probability_positive_sharpe": float((sharpes > 0).mean()),
    }


def overlay_execution_state(
    signal_frame: pd.DataFrame,
    execution_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Use the strict next-open ledger as the sole public position and order state."""
    out = signal_frame.copy()
    if execution_frame.empty:
        raise RuntimeError("Public execution state requires a non-empty strict ledger")
    pre_execution = out.index < execution_frame.index.min()
    public_state_columns = [
        "position",
        "atr_stop",
        "tb_take_profit",
        "guide",
        "raw_signal",
        "execution_action",
        "exit_reason",
        "desired_position",
        "pending_entry",
        "pending_exit",
        "entry_blocked_by_drawdown",
    ]
    for column in public_state_columns:
        if column in out:
            out.loc[pre_execution, column] = np.nan
    common_index = out.index.intersection(execution_frame.index)
    mapping = {
        "position": "position",
        "atr_stop": "atr_stop",
        "tb_take_profit": "tb_take_profit",
        "guide": "guide",
        "raw_signal": "raw_signal",
        "execution_action": "execution_action",
        "exit_reason": "exit_reason",
    }
    for execution_column, signal_column in mapping.items():
        if execution_column in execution_frame:
            out.loc[common_index, signal_column] = execution_frame.loc[common_index, execution_column]
    for column in [
        "desired_position",
        "pending_entry",
        "pending_exit",
        "entry_blocked_by_drawdown",
    ]:
        if column in execution_frame:
            out.loc[common_index, column] = execution_frame.loc[common_index, column]
    return out


def formal_strategy_fingerprint(config: RiskConfig) -> str:
    formal_functions = [
        risk_position_size,
        trend_risk_budget,
        build_features,
        fit_hmm,
        fit_hmm_walk_forward,
        primary_long_signal,
        make_cusum_events,
        make_meta_events,
        generate_signals,
        backtest_next_open,
    ]
    formal_source = "\n\n".join(inspect.getsource(function) for function in formal_functions)
    payload = {
        "strategyVersion": FORMAL_STRATEGY_VERSION,
        "executionEngineVersion": EXECUTION_ENGINE_VERSION,
        "config": asdict(config),
        "formalLogicSha256": hashlib.sha256(formal_source.encode("utf-8")).hexdigest(),
    }
    encoded = json.dumps(json_safe(payload), ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def update_forward_ledger(
    execution_frame: pd.DataFrame,
    config: RiskConfig,
    price_status: dict[str, Any],
) -> dict[str, Any]:
    """Validate immutable forward records and append only completed trading days."""
    ensure_dirs()
    fingerprint = formal_strategy_fingerprint(config)
    forward_start = pd.Timestamp(config.forward_holdout_start_date)
    candidate_frame = execution_frame.loc[execution_frame.index >= forward_start]
    if price_status["isFinal"] is False:
        provisional_date = pd.Timestamp(price_status["sessionDate"])
        candidate_frame = candidate_frame.loc[candidate_frame.index < provisional_date]
    candidate_records: list[dict[str, Any]] = []
    for date, row in candidate_frame.iterrows():
        candidate_records.append(
            {
                "date": str(pd.Timestamp(date).date()),
                "strategyReturn": float(row["strategy_ret"]),
                "benchmarkReturn": float(row["benchmark_ret"]),
                "position": float(row["position"]),
                "recommendedPosition": float(row["desired_position"]),
                "turnover": float(row["turnover"]),
                "cashInterest": float(row.get("cash_interest", 0.0)),
                "executionAction": str(row["execution_action"]),
                "guide": str(row["guide"]),
                "pendingEntry": bool(row["pending_entry"]),
                "pendingExit": bool(row["pending_exit"]),
            }
        )

    ledger = {
        "strategyVersion": FORMAL_STRATEGY_VERSION,
        "executionEngineVersion": EXECUTION_ENGINE_VERSION,
        "configFingerprint": fingerprint,
        "start": config.forward_holdout_start_date,
        "appendOnly": True,
        "records": [],
    }
    if FORWARD_LEDGER.exists():
        existing = json.loads(FORWARD_LEDGER.read_text(encoding="utf-8"))
        for key in ["strategyVersion", "executionEngineVersion", "start"]:
            if existing.get(key) != ledger[key]:
                raise RuntimeError(
                    f"Forward ledger {key} changed; create a new declared strategy version instead of rewriting history"
                )
        if existing.get("appendOnly") is not True or not isinstance(existing.get("records"), list):
            raise RuntimeError("Forward ledger schema is invalid")
        if existing.get("configFingerprint") != fingerprint:
            if existing["records"]:
                raise RuntimeError(
                    "Forward ledger configFingerprint changed; create a new declared strategy version instead of rewriting history"
                )
            existing["configFingerprint"] = fingerprint
        ledger = existing

    existing_by_date = {record["date"]: record for record in ledger["records"]}
    candidate_by_date = {record["date"]: record for record in candidate_records}
    for date, existing_record in existing_by_date.items():
        candidate_record = candidate_by_date.get(date)
        if candidate_record is None:
            raise RuntimeError(f"Forward ledger date {date} disappeared from the recomputed execution history")
        for key, expected in existing_record.items():
            actual = candidate_record.get(key)
            if isinstance(expected, (int, float)) and not isinstance(expected, bool):
                if not np.isfinite(float(actual)) or abs(float(actual) - float(expected)) > 1e-10:
                    raise RuntimeError(f"Forward ledger history changed on {date}: {key}")
            elif actual != expected:
                raise RuntimeError(f"Forward ledger history changed on {date}: {key}")

    last_existing_date = max(existing_by_date, default="")
    missing_historical = [
        record["date"]
        for record in candidate_records
        if record["date"] not in existing_by_date and record["date"] <= last_existing_date
    ]
    if missing_historical:
        raise RuntimeError(f"Forward ledger has non-append gaps: {missing_historical[:3]}")
    new_records = [record for record in candidate_records if record["date"] > last_existing_date]
    ledger["records"].extend(new_records)
    write_json_atomic(FORWARD_LEDGER, ledger, indent=2)

    records = ledger["records"]
    strategy_returns = pd.Series([record["strategyReturn"] for record in records], dtype=float)
    benchmark_returns = pd.Series([record["benchmarkReturn"] for record in records], dtype=float)
    return {
        "start": config.forward_holdout_start_date,
        "end": records[-1]["date"] if records else None,
        "days": int(len(records)),
        "total_return": float((1.0 + strategy_returns).prod() - 1.0) if records else 0.0,
        "benchmark_return": float((1.0 + benchmark_returns).prod() - 1.0) if records else 0.0,
        "average_position": (
            float(np.mean([record["position"] for record in records])) if records else 0.0
        ),
        "trade_days": int(sum(record["turnover"] > 0 for record in records)),
        "append_only": True,
        "strategy_version": FORMAL_STRATEGY_VERSION,
        "execution_engine_version": EXECUTION_ENGINE_VERSION,
        "config_fingerprint": fingerprint,
    }


def build_outputs(
    signal_frame: pd.DataFrame,
    backtest_frame: pd.DataFrame,
    backtest_metrics: dict[str, float],
    live_execution_metrics: dict[str, float],
    forward_holdout_metrics: dict[str, Any],
    yearly_backtest_metrics: list[dict[str, Any]],
    backtest_robustness: dict[str, Any],
    sources: dict[str, str],
    state_mapping: dict[int, str],
    data_quality: dict[str, Any],
    config: RiskConfig,
    price_status: dict[str, Any],
) -> None:
    ensure_dirs()

    log_columns = [
        "gold_close",
        "market_state_code",
        "market_state",
        "raw_signal",
        "position",
        "atr_stop",
        "tb_take_profit",
        "tb_event",
        "tb_accepted_event",
        "primary_trend_signal",
        "guide",
        "execution_action",
        "exit_reason",
        "atr_pct",
        "payoff_ratio",
        "trend_risk_budget",
        "cash_yield_pct",
        "dxy_close",
        "us10y_close",
        "tips_10y_real_yield",
        "real_rate",
        "real_rate_proxy",
        "vix_close",
        "vixy_close",
        "gld_close",
        "cot_gold_net",
        "cftc_mm_gold_net",
        "cftc_mm_gold_net_pct_oi",
        "cpi_mom_surprise",
        "core_cpi_mom_surprise",
        "nfp_surprise",
        "gpr_index",
        "fomc_decision",
    ]
    existing = [column for column in log_columns if column in signal_frame.columns]
    signal_frame[existing].to_csv(LOCAL_LOGS / "gold_signals.csv", encoding="utf-8-sig")

    latest = signal_frame.dropna(subset=["gold_close"]).iloc[-1]
    previous = signal_frame.dropna(subset=["gold_close"]).iloc[-2]
    public_risk = asdict(config)

    latest_json = {
        "asOf": str(latest.name.date()),
        "asset": "COMEX 迷你黄金连续合约 QO00Y",
        "assetDetail": "东方财富 101.QO00Y 为历史主序列；主接口失败时，仅用同一 COMEX 黄金 GC 行情刷新尚未定稿的盘中尾部",
        "price": float(latest["gold_close"]),
        "priceStatus": price_status,
        "dailyChange": float(latest["gold_close"] / previous["gold_close"] - 1),
        "isMetaEvent": bool(latest["tb_event"]),
        "isAcceptedEvent": bool(latest["tb_accepted_event"]),
        "marketStateCode": str(latest["market_state_code"]),
        "marketState": str(latest["market_state"]),
        "guide": str(latest["guide"]),
        "rawSignal": str(latest["raw_signal"]),
        "position": float(latest["position"]),
        "recommendedPosition": float(latest.get("desired_position", latest["position"])),
        "pendingEntry": bool(latest.get("pending_entry", False)),
        "pendingExit": bool(latest.get("pending_exit", False)),
        "riskHalted": bool(latest.get("entry_blocked_by_drawdown", False)),
        "atrStop": None if pd.isna(latest["atr_stop"]) else float(latest["atr_stop"]),
        "takeProfit": None if pd.isna(latest["tb_take_profit"]) else float(latest["tb_take_profit"]),
        "atrPct": float(latest["atr_pct"]),
        "risk": public_risk,
        "backtestMetrics": backtest_metrics,
        "liveExecutionMetrics": live_execution_metrics,
        "forwardHoldoutMetrics": forward_holdout_metrics,
        "backtestYearly": yearly_backtest_metrics,
        "backtestRobustness": backtest_robustness,
        "stateMapping": {str(k): v for k, v in state_mapping.items()},
        "sources": sources,
        "dataQuality": data_quality,
        "notes": [
            "VIX 优先使用 Cboe 官方 VIX 历史 CSV；若不可用，风险特征仍可回退参考 VIXY ETF proxy。",
            "实际利率优先使用 FRED DFII10 10Y TIPS；不可用时回退为 US10Y 减滞后 CPI 同比 proxy。",
            "CPI 同比按下一月中旬、COT 持仓按报告日后 3 天做 point-in-time 滞后近似。",
            "CFTC managed money 净多头使用官方 disaggregated futures-only COT，并按报告日后 3 天进入模型。",
            "CPI、核心 CPI、非农 surprise 使用公布日数据的 actual - forecast，不按数据所属月份提前使用。",
            "ETF 资金流使用 GLD 量价方向签名成交额 proxy，并不等同于真实 ETF 净申赎。",
            "MOVE、Fed funds futures implied rate、GLD 官方持仓和黄金 ETF 官方净流入当前没有稳定免费 point-in-time 接口，已在 sources/dataQuality 标为不可用或 proxy。",
            "GPR 使用 Caldara-Iacoviello 月度地缘政治风险指数，并做月末后 7 天滞后近似。",
            "FOMC 事件来自美联储会议日历，作为事件日和 proximity 特征。",
            "正式算法只运行已采纳的趋势、CUSUM、ATR 和 walk-forward HMM 规则，不包含候选策略或分类预测路径。",
            "网站持仓、止损、止盈和待执行动作统一来自正式次日开盘执行账本，不再使用独立的收盘成交状态机。",
            "若最新交易日尚未收盘，网站使用程序运行时取得的最新价格计算当日模型快照，并明确标注为盘中数据；前瞻账本仍只记录已结束交易日。",
            f"HMM 固定使用初始训练期可用的 {config.hmm_feature_policy} 特征集合，后续重训不允许因数据源变长而改变模型维度。",
            f"训练和验证截止日已经冻结；{config.forward_holdout_start_date} 之后的新数据作为 forward holdout，不回流到历史参数选择。",
            f"HMM 使用 expanding walk-forward，并约每 {config.hmm_retrain_every_days} 个交易日重训一次，所有状态概率只使用当时可得数据。",
            f"入场仓位按 ATR 止损距离缩放：普通趋势风险预算 {config.normal_trend_risk_budget:.0%}，120 日收益达到 {config.strong_trend_ret_120_threshold:.0%} 的强趋势预算 {config.strong_trend_risk_budget:.0%}；隔夜跳空可能使实际损失超过计划值。",
            "实盘模拟使用 t 日收盘信号、t+1 日开盘成交、盘中 ATR 障碍、双边交易成本和回撤降仓约束。",
            "网站主回测采用次日开盘事件驱动口径；盘中障碍按障碍价成交，隔夜跳空穿越障碍按开盘价成交。",
            "部分仓位回测使用现金与黄金持仓单位逐日盯市，不假设开盘免费再平衡。",
            f"未投资现金按上一可得日 FRED DGS3MO 三个月美债收益率计息，并保守扣减 {config.cash_yield_haircut_bps:.0f}bps；现金利息与黄金择时收益分开披露。",
            f"HMM 退出需要熊市/恐慌且跌破 60 日均线连续确认 {config.hmm_exit_confirmation_days} 天。",
            "研究结果不构成投资建议。",
        ],
    }
    write_json_atomic(PUBLIC_DATA / "gold_research_latest.json", latest_json, indent=2)

    price_columns = [
        "gold_close",
        "sma_5",
        "sma_20",
        "sma_60",
        "sma_120",
        "market_state",
        "market_state_code",
        "position",
        "desired_position",
        "pending_entry",
        "pending_exit",
        "entry_blocked_by_drawdown",
        "guide",
        "atr_stop",
        "tb_take_profit",
        "tb_event",
        "tb_accepted_event",
    ]
    price = signal_frame[[column for column in price_columns if column in signal_frame.columns]].tail(900)
    price = price.reset_index().rename(
        columns={
            "date": "date",
            "gold_close": "close",
            "market_state": "state",
            "market_state_code": "stateCode",
            "desired_position": "recommendedPosition",
            "pending_entry": "pendingEntry",
            "pending_exit": "pendingExit",
            "entry_blocked_by_drawdown": "riskHalted",
            "atr_stop": "atrStop",
            "tb_take_profit": "takeProfit",
            "tb_event": "event",
            "tb_accepted_event": "acceptedEvent",
        }
    )
    price["date"] = price["date"].dt.strftime("%Y-%m-%d")
    write_json_atomic(
        PUBLIC_DATA / "gold_price_series.json",
        price.replace({np.nan: None}).to_dict("records"),
    )

    bt = backtest_frame[["equity", "benchmark_equity", "drawdown", "position"]].reset_index()
    bt["date"] = bt["date"].dt.strftime("%Y-%m-%d")
    write_json_atomic(
        PUBLIC_DATA / "gold_backtest.json",
        bt.replace({np.nan: None}).to_dict("records"),
    )


def run_pipeline() -> dict[str, Any]:
    ensure_dirs()
    config = RiskConfig()
    market_data, sources = load_market_data()
    data_quality = verify_data_quality(market_data, sources)
    features = build_features(market_data, config)
    features = features.replace([np.inf, -np.inf], np.nan)

    usable = features.dropna(subset=["gold_close"]).copy()
    usable = usable.iloc[220:].copy()
    train_end = pd.Timestamp(config.train_end_date)
    validation_end = pd.Timestamp(config.validation_end_date)
    if train_end not in usable.index or validation_end not in usable.index:
        raise RuntimeError("Configured frozen train/validation split dates are not present in the gold dataset")
    test_mask = features.index > validation_end

    _, state_mapping, state_frame = fit_hmm_walk_forward(
        features,
        train_end,
        config.hmm_retrain_every_days,
        config.hmm_feature_policy,
    )
    hmm_diagnostics = state_frame.attrs.get("hmm_diagnostics", [])
    pd.DataFrame(hmm_diagnostics).to_csv(
        LOCAL_LOGS / "gold_hmm_stability.csv",
        index=False,
        encoding="utf-8-sig",
    )
    features = features.join(state_frame)

    signals = generate_signals(features, config)
    test_start = features.index[test_mask].min()
    evaluation_signals = generate_signals(
        features,
        config,
        start_at=test_start,
    )
    backtest_frame, backtest_metrics = backtest_next_open(
        evaluation_signals,
        test_mask,
        config,
        cost_bps=0.0,
        write_log=False,
    )
    _, net_5bps_metrics = backtest_next_open(
        evaluation_signals,
        test_mask,
        config,
        cost_bps=5.0,
        write_log=False,
    )
    _, net_5bps_zero_cash_metrics = backtest_next_open(
        evaluation_signals,
        test_mask,
        replace(config, cash_yield_enabled=False),
        cost_bps=5.0,
        write_log=False,
    )
    live_execution_frame, live_execution_metrics = backtest_next_open(
        evaluation_signals,
        test_mask,
        config,
        cost_bps=config.realistic_cost_bps,
        write_log=True,
    )
    cost_stress: list[dict[str, float]] = []
    for cost_bps, metrics in [
        (0.0, backtest_metrics),
        (5.0, net_5bps_metrics),
        (config.realistic_cost_bps, live_execution_metrics),
    ]:
        cost_stress.append(
            {
                "cost_bps": float(cost_bps),
                "total_return": float(metrics["total_return"]),
                "sharpe": float(metrics["sharpe"]),
                "max_drawdown": float(metrics["max_drawdown"]),
                "turnover": float(metrics["turnover"]),
            }
        )
    for cost_bps in [15.0, 25.0]:
        _, stress_metrics = backtest_next_open(
            evaluation_signals,
            test_mask,
            config,
            cost_bps=cost_bps,
            write_log=False,
        )
        cost_stress.append(
            {
                "cost_bps": cost_bps,
                "total_return": float(stress_metrics["total_return"]),
                "sharpe": float(stress_metrics["sharpe"]),
                "max_drawdown": float(stress_metrics["max_drawdown"]),
                "turnover": float(stress_metrics["turnover"]),
            }
        )
    backtest_robustness = {
        "sample_role": "reviewed_development_history_not_independent_out_of_sample",
        "cost_stress": cost_stress,
        "moving_block_bootstrap": moving_block_bootstrap_summary(live_execution_frame),
    }
    (LOCAL_LOGS / "gold_backtest_robustness.json").write_text(
        json.dumps(json_safe(backtest_robustness), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    public_signals = overlay_execution_state(signals, live_execution_frame)
    latest_signal_date = public_signals.dropna(subset=["gold_close"]).index[-1]
    price_status = build_price_status(latest_signal_date)
    forward_holdout_metrics = update_forward_ledger(live_execution_frame, config, price_status)
    backtest_metrics.update(
        {
            "net_total_return_5bps": net_5bps_metrics["total_return"],
            "net_sharpe_5bps": net_5bps_metrics["sharpe"],
            "net_max_drawdown_5bps": net_5bps_metrics["max_drawdown"],
            "net_total_return_5bps_without_cash_yield": net_5bps_zero_cash_metrics["total_return"],
            "cash_yield_return_uplift_5bps": (
                net_5bps_metrics["total_return"] - net_5bps_zero_cash_metrics["total_return"]
            ),
        }
    )
    yearly_backtest_metrics = build_yearly_backtest_report(
        live_execution_frame,
        cost_bps=config.realistic_cost_bps,
    )

    build_outputs(
        public_signals,
        live_execution_frame,
        backtest_metrics,
        live_execution_metrics,
        forward_holdout_metrics,
        yearly_backtest_metrics,
        backtest_robustness,
        sources,
        state_mapping,
        data_quality,
        config,
        price_status,
    )

    latest = public_signals.dropna(subset=["gold_close"]).iloc[-1]
    return {
        "as_of": str(latest.name.date()),
        "price": float(latest["gold_close"]),
        "market_state": str(latest["market_state"]),
        "market_state_code": str(latest["market_state_code"]),
        "guide": str(latest["guide"]),
        "position": float(latest["position"]),
        "backtest_metrics": backtest_metrics,
        "live_execution_metrics": live_execution_metrics,
        "forward_holdout_metrics": forward_holdout_metrics,
        "backtest_yearly": yearly_backtest_metrics,
        "backtest_robustness": backtest_robustness,
        "outputs": {
            "signals_csv": str(LOCAL_LOGS / "gold_signals.csv"),
            "live_execution_csv": str(LOCAL_LOGS / "gold_live_execution.csv"),
            "backtest_yearly_csv": str(LOCAL_LOGS / "gold_backtest_yearly.csv"),
            "backtest_robustness_json": str(LOCAL_LOGS / "gold_backtest_robustness.json"),
            "hmm_stability_csv": str(LOCAL_LOGS / "gold_hmm_stability.csv"),
            "latest_json": str(PUBLIC_DATA / "gold_research_latest.json"),
            "price_json": str(PUBLIC_DATA / "gold_price_series.json"),
            "backtest_json": str(PUBLIC_DATA / "gold_backtest.json"),
            "forward_ledger_json": str(FORWARD_LEDGER),
        },
    }


def main() -> None:
    global OFFLINE_MODE
    parser = argparse.ArgumentParser(description="Run the local gold trend + HMM research pipeline.")
    parser.add_argument("--json", action="store_true", help="Print a JSON summary.")
    parser.add_argument("--offline", action="store_true", help="Use cached market and macro data without network refresh.")
    args = parser.parse_args()
    OFFLINE_MODE = bool(args.offline)
    summary = run_pipeline()
    if args.json:
        print(json.dumps(json_safe(summary), ensure_ascii=False, indent=2, allow_nan=False))
    else:
        print(
            f"{summary['as_of']} {summary['market_state_code']}={summary['market_state']} "
            f"guide={summary['guide']} position={summary['position']:.1%}"
        )


if __name__ == "__main__":
    main()
