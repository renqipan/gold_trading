from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import time
import urllib.parse
import warnings
from dataclasses import asdict, dataclass
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from hmmlearn.hmm import GaussianHMM
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_DATA = ROOT / "public" / "data"
LOCAL_LOGS = ROOT / "local_logs"
RAW_DATA = ROOT / "data" / "raw"

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
    prediction_horizon_days: int = 60
    up_threshold: float = 0.60
    down_threshold: float = 0.36
    max_position: float = 1.0
    max_leverage: float = 1.0
    max_single_loss: float = 0.06
    max_drawdown_soft: float = 0.18
    max_drawdown_hard: float = 0.30
    atr_window: int = 14
    atr_multiple: float = 4.0
    kelly_fraction: float = 1.0
    retrain_every_days: int = 21
    label_purge_days: int = 60
    meta_event_gap_days: int = 3
    meta_event_kind: str = "cusum_abs"
    cusum_threshold_mult: float = 0.8
    primary_signal_mode: str = "hmm_quality"
    hmm_exit_confirmation_days: int = 10
    profit_atr_multiple: float = 10.0
    stop_atr_multiple: float = 4.0


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
        "-k",
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

    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode != 0:
        stderr = compact_error(result.stderr)
        raise RuntimeError(f"curl exit {result.returncode}: {stderr}")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        preview = compact_error(result.stdout)
        raise RuntimeError(f"curl returned invalid JSON: {preview}") from exc


def request_json(url: str, params: dict[str, Any], timeout: int = 12, attempts: int = 4) -> dict[str, Any]:
    last_error: Exception | None = None
    with requests.Session() as session:
        session.trust_env = False
        for attempt in range(attempts):
            random_request_pause(attempt)
            headers = browser_headers()
            try:
                response = session.get(url, params=params, headers=headers, timeout=(4, timeout), verify=False)
                response.raise_for_status()
                return response.json()
            except Exception as exc:
                last_error = exc

    full_url = f"{url}?{urllib.parse.urlencode(params)}"
    try:
        return curl_json(full_url, browser_headers(), timeout)
    except Exception as exc:
        raise RuntimeError(f"request failed after retries: {last_error}; curl fallback: {exc}") from exc


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


def load_market_data() -> tuple[pd.DataFrame, dict[str, str]]:
    sources: dict[str, str] = {}
    series: list[pd.DataFrame] = []

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
            if repaired_rows:
                sources[name] += f"; OHLC repaired rows={repaired_rows}"
        except Exception as exc:
            cached = load_cached_eastmoney_kline(name)
            if cached is not None and len(cached):
                cached, repaired_rows = repair_ohlc(cached, name)
                series.append(cached)
                sources[name] = f"Eastmoney {secid}: {description} (cached fallback after refresh failure)"
                if repaired_rows:
                    sources[name] += f"; OHLC repaired rows={repaired_rows}"
                print(f"[warn] {name} refresh unavailable; using cached Eastmoney data")
            else:
                sources[name] = f"unavailable: {exc}"
                print(f"[warn] {name} unavailable: {exc}")

    if not any(column.startswith("gold_close") for frame in series for column in frame.columns):
        try:
            import akshare as ak

            fallback = ak.spot_hist_sge(symbol="Au99.99")
            fallback["date"] = pd.to_datetime(fallback["date"])
            fallback = fallback.set_index("date").sort_index()
            fallback = fallback.rename(columns={column: f"gold_{column}" for column in fallback.columns})
            fallback["gold_volume"] = np.nan
            fallback["gold_amount"] = np.nan
            series.append(fallback)
            sources["gold"] = "Shanghai Gold Exchange Au99.99 fallback"
        except Exception as exc:
            raise RuntimeError("Gold price data is required but unavailable") from exc

    cpi = fetch_cpi_yoy()
    cot = fetch_cot_gold()
    if len(cpi):
        cpi.to_csv(RAW_DATA / "us_cpi_yoy.csv")
        sources["cpi"] = "AkShare Jin10 US CPI YoY"
        series.append(cpi)
    else:
        sources["cpi"] = "unavailable"

    if len(cot):
        cot.to_csv(RAW_DATA / "cot_gold.csv")
        sources["cot"] = "AkShare Jin10 CFTC gold positioning"
        series.append(cot)
    else:
        sources["cot"] = "unavailable"

    data = pd.concat(series, axis=1).sort_index()
    gold_index = data.loc[data["gold_close"].notna()].index
    data = data.reindex(gold_index)
    data = data.replace([np.inf, -np.inf], np.nan).ffill()

    if "us10y_close" in data and "us_cpi_yoy" in data:
        data["real_rate_proxy"] = data["us10y_close"] - data["us_cpi_yoy"]
        sources["real_rate_proxy"] = "US10Y yield minus US CPI YoY"
    else:
        data["real_rate_proxy"] = np.nan
        sources["real_rate_proxy"] = "unavailable"

    return data, sources


def verify_data_quality(data: pd.DataFrame, sources: dict[str, str]) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    instrument_names = ["gold", "dxy", "us10y", "gld", "vixy", "spx"]
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
        "real_rate_proxy": ["real_rate_proxy"],
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

    quality = {
        "generatedAt": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "passed": all(
            (not item.get("available", False)) or item.get("invalidOhlcRows", 0) == 0
            for item in checks.values()
        ),
        "checks": checks,
    }
    (LOCAL_LOGS / "data_quality_report.json").write_text(
        json.dumps(quality, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return quality


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std.replace(0, np.nan)


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


def add_trend_quality_features(frame: pd.DataFrame) -> pd.DataFrame:
    high = frame["gold_high"]
    low = frame["gold_low"]
    close = frame["gold_close"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=frame.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=frame.index)
    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr_sum = true_range.rolling(14).sum().replace(0, np.nan)
    plus_di = 100 * plus_dm.rolling(14).sum() / atr_sum
    minus_di = 100 * minus_dm.rolling(14).sum() / atr_sum
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
    frame["adx_14"] = dx.rolling(14).mean()

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    frame["rsi_14"] = 100 - (100 / (1 + rs))

    frame["donchian_120"] = close / close.rolling(120).max() - 1
    frame["price_to_high_252"] = close / close.rolling(252).max() - 1
    frame["tsmom_score"] = (
        frame["ret_60"].fillna(0)
        + frame["ret_120"].fillna(0)
        + 0.5 * frame["ret_180"].fillna(0)
    )
    frame["trend_quality_score"] = (
        frame["trend_strength"].fillna(0)
        + frame["ret_vol_adj_20"].fillna(0).clip(-2, 2) * 0.02
        - frame["vol_ratio_20_60"].fillna(1).clip(0, 3) * 0.01
    )
    return frame


def build_features(data: pd.DataFrame, config: RiskConfig) -> pd.DataFrame:
    frame = data.copy()
    close = frame["gold_close"]
    frame["ret_1"] = close.pct_change()
    for window in [3, 5, 10, 20, 30, 60, 120, 180]:
        frame[f"ret_{window}"] = close.pct_change(window)
        frame[f"mom_{window}"] = close / close.shift(window) - 1
        frame[f"sma_{window}"] = close.rolling(window).mean()
        frame[f"sma_gap_{window}"] = close / frame[f"sma_{window}"] - 1
    for window in [10, 20, 30, 60]:
        frame[f"vol_{window}"] = frame["ret_1"].rolling(window).std() * math.sqrt(252)

    frame["atr"] = compute_atr(frame, window=config.atr_window)
    frame["atr_pct"] = frame["atr"] / close
    frame["drawdown_120"] = close / close.rolling(120).max() - 1
    frame["range_pct"] = (frame["gold_high"] - frame["gold_low"]) / close
    frame["ma_cross_5_20"] = frame["sma_5"] / frame["sma_20"] - 1
    frame["ma_cross_20_60"] = frame["sma_20"] / frame["sma_60"] - 1
    frame["ma_cross_60_120"] = frame["sma_60"] / frame["sma_120"] - 1
    frame["vol_ratio_20_60"] = frame["vol_20"] / frame["vol_60"].replace(0, np.nan)
    frame["ret_vol_adj_20"] = frame["ret_20"] / frame["vol_20"].replace(0, np.nan)
    frame["high_breakout_120"] = close / close.rolling(120).max() - 1
    frame["low_breakdown_120"] = close / close.rolling(120).min() - 1
    frame["trend_strength"] = (frame["sma_gap_20"] + frame["ma_cross_20_60"] + frame["ma_cross_60_120"]) / 3

    if "gold_volume" in frame:
        frame["gold_volume_z20"] = rolling_zscore(frame["gold_volume"], 20)
    if "gold_amount" in frame:
        frame["gold_amount_z20"] = rolling_zscore(frame["gold_amount"], 20)

    for name in ["dxy", "us10y", "gld", "vixy", "spx"]:
        close_column = f"{name}_close"
        if close_column in frame:
            frame[f"{name}_ret_5"] = frame[close_column].pct_change(5)
            frame[f"{name}_ret_20"] = frame[close_column].pct_change(20)
            frame[f"{name}_z60"] = rolling_zscore(frame[close_column], 60)
            frame[f"{name}_change_20"] = frame[close_column].diff(20)

    if "spx_close" in frame:
        frame["spx_realized_vol_20"] = frame["spx_close"].pct_change().rolling(20).std() * math.sqrt(252)

    if "gld_amount" in frame:
        signed_flow = np.sign(frame["gld_close"].pct_change()).fillna(0) * frame["gld_amount"]
        frame["gld_flow_proxy_20"] = signed_flow.rolling(20).sum()
        frame["gld_flow_proxy_z60"] = rolling_zscore(frame["gld_flow_proxy_20"], 60)

    if "cot_gold_net" in frame:
        frame["cot_gold_net_chg_4w"] = frame["cot_gold_net"].diff(20)
        frame["cot_gold_net_z52w"] = rolling_zscore(frame["cot_gold_net"], 252)
        frame["cot_gold_long_short_ratio"] = frame["cot_gold_long"] / frame["cot_gold_short"].replace(0, np.nan)

    if "real_rate_proxy" in frame:
        frame["real_rate_proxy_change_20"] = frame["real_rate_proxy"].diff(20)
        frame["real_rate_proxy_z60"] = rolling_zscore(frame["real_rate_proxy"], 60)

    frame = add_trend_quality_features(frame)

    return frame


def feature_columns(frame: pd.DataFrame) -> list[str]:
    forbidden: set[str] = set()
    raw_level_suffixes = ("_open", "_high", "_low", "_close", "_volume", "_amount", "_extra1", "_extra2", "_extra3")
    raw_indicators = {"atr"}
    allowed_prefixes = (
        "ret_",
        "mom_",
        "sma_gap_",
        "vol_",
        "drawdown_",
        "range_",
        "ma_cross_",
        "vol_ratio_",
        "ret_vol_adj_",
        "high_breakout_",
        "low_breakdown_",
        "trend_strength",
        "adx_",
        "rsi_",
        "donchian_",
        "price_to_high_",
        "tsmom_score",
        "trend_quality_score",
        "cot_",
        "real_rate_proxy",
        "hmm_prob_",
        "state_",
    )
    allowed_suffixes = ("_ret_5", "_ret_20", "_z60", "_change_20", "_z20", "_z52w", "_chg_4w")
    cols: list[str] = []
    for column in frame.columns:
        if column in forbidden:
            continue
        if column in raw_indicators:
            continue
        if column.endswith(raw_level_suffixes):
            continue
        allowed = column.startswith(allowed_prefixes) or column.endswith(allowed_suffixes) or "flow_proxy" in column
        if allowed and pd.api.types.is_numeric_dtype(frame[column]):
            cols.append(column)
    return cols


def hmm_feature_columns(frame: pd.DataFrame) -> list[str]:
    candidates = [
        "ret_5",
        "ret_20",
        "vol_20",
        "sma_gap_20",
        "sma_gap_60",
        "ma_cross_20_60",
        "atr_pct",
        "drawdown_120",
        "trend_strength",
        "dxy_ret_20",
        "us10y_change_20",
        "real_rate_proxy_change_20",
        "vixy_ret_20",
        "gld_flow_proxy_z60",
        "cot_gold_net_z52w",
    ]
    return [column for column in candidates if column in frame.columns]


def fit_hmm(frame: pd.DataFrame, train_mask: pd.Series) -> tuple[Pipeline, dict[int, str], pd.DataFrame]:
    cols = [
        column
        for column in hmm_feature_columns(frame)
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
    log_probs = []
    for state in range(hmm.n_components):
        diff = all_transformed - hmm.means_[state]
        log_det = np.log(covars[state]).sum()
        quad = (diff * diff / covars[state]).sum(axis=1)
        log_prior = np.log(max(hmm.startprob_[state], 1e-12))
        log_probs.append(log_prior - 0.5 * (log_det + quad))
    log_probs_array = np.vstack(log_probs).T
    log_probs_array = log_probs_array - log_probs_array.max(axis=1, keepdims=True)
    posterior = np.exp(log_probs_array)
    posterior = posterior / posterior.sum(axis=1, keepdims=True)
    hidden = posterior.argmax(axis=1)

    state_stats = []
    temp = frame.copy()
    temp["hmm_raw_state"] = hidden
    for state in range(4):
        subset = temp.loc[train_mask & (temp["hmm_raw_state"] == state)]
        state_stats.append(
            {
                "state": state,
                "ret20": subset["ret_20"].mean(),
                "vol20": subset["vol_20"].mean(),
                "drawdown": subset["drawdown_120"].mean(),
                "trend": subset["sma_gap_60"].mean(),
                "risk": subset.get("vixy_ret_20", pd.Series(dtype=float)).mean(),
                "count": int(len(subset)),
            }
        )
    stats = pd.DataFrame(state_stats).fillna(0)

    panic_state = stats.sort_values(["vol20", "risk"], ascending=False).iloc[0]["state"].item()
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


def primary_long_signal(frame: pd.DataFrame, mode: str = "trend_slow") -> pd.Series:
    if mode == "trend_slow":
        return (
            (frame["gold_close"] > frame["sma_120"])
            | ((frame["sma_20"] > frame["sma_60"]) & (frame["sma_60"] > frame["sma_120"]))
        )
    if mode == "hmm_trend":
        return (
            (frame["market_state"].isin(["牛市", "震荡"]) & (frame["gold_close"] > frame["sma_60"]))
            | ((frame["gold_close"] > frame["sma_120"]) & (frame["trend_strength"] > -0.015))
        )
    if mode == "hmm_quality":
        trend_slow = primary_long_signal(frame, "trend_slow")
        return trend_slow & ((frame["market_state"] != "恐慌") | (frame["ret_60"] > 0.08))
    raise ValueError(f"unknown primary signal mode: {mode}")


def make_repeated_events(signal: pd.Series, min_gap: int) -> pd.Series:
    events = pd.Series(False, index=signal.index)
    last_pos = -10_000
    for pos, value in enumerate(signal.fillna(False).to_numpy()):
        if value and pos - last_pos >= min_gap:
            events.iloc[pos] = True
            last_pos = pos
    return events


def make_cusum_events(frame: pd.DataFrame, signal: pd.Series, threshold_mult: float, min_gap: int) -> pd.Series:
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
    if config.meta_event_kind == "repeated":
        return make_repeated_events(signal, config.meta_event_gap_days)
    if config.meta_event_kind == "cusum_abs":
        return make_cusum_events(frame, signal, config.cusum_threshold_mult, config.meta_event_gap_days)
    raise ValueError(f"unknown meta event kind: {config.meta_event_kind}")


def triple_barrier_labels(
    frame: pd.DataFrame,
    events: pd.Series,
    config: RiskConfig,
) -> pd.DataFrame:
    event_dates = frame.index[events.fillna(False)]
    rows: list[dict[str, Any]] = []
    close = frame["gold_close"]
    high = frame["gold_high"]
    low = frame["gold_low"]
    atr = frame["atr"]
    for date in event_dates:
        loc = frame.index.get_loc(date)
        if loc + config.prediction_horizon_days >= len(frame) or not np.isfinite(atr.loc[date]):
            continue
        entry = close.loc[date]
        upper = entry + config.profit_atr_multiple * atr.loc[date]
        lower = entry - config.stop_atr_multiple * atr.loc[date]
        end_loc = loc + config.prediction_horizon_days
        label = 0.0
        exit_date = frame.index[end_loc]
        exit_reason = "vertical"
        for future_loc in range(loc + 1, end_loc + 1):
            future_date = frame.index[future_loc]
            hit_upper = high.iloc[future_loc] >= upper
            hit_lower = low.iloc[future_loc] <= lower
            if hit_upper and hit_lower:
                label = 0.0
                exit_date = future_date
                exit_reason = "both_stop_first"
                break
            if hit_upper:
                label = 1.0
                exit_date = future_date
                exit_reason = "profit"
                break
            if hit_lower:
                label = 0.0
                exit_date = future_date
                exit_reason = "stop"
                break
        rows.append(
            {
                "date": date,
                "tb_label": label,
                "tb_exit_date": exit_date,
                "tb_exit_reason": exit_reason,
                "tb_forward_return": close.loc[exit_date] / entry - 1,
            }
        )
    if not rows:
        return pd.DataFrame(columns=["tb_label", "tb_exit_date", "tb_exit_reason", "tb_forward_return"])
    return pd.DataFrame(rows).set_index("date").sort_index()


def fit_meta_xgb_model() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=120,
        max_depth=2,
        learning_rate=0.04,
        subsample=0.80,
        colsample_bytree=0.76,
        reg_lambda=8.0,
        reg_alpha=0.30,
        min_child_weight=8,
        gamma=0.05,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=7,
        n_jobs=4,
        tree_method="hist",
    )


def train_triple_barrier_meta_model(
    frame: pd.DataFrame,
    events: pd.Series,
    labels: pd.DataFrame,
    cols: list[str],
    initial_train_end: pd.Timestamp,
    validation_end: pd.Timestamp,
    test_mask: pd.Series,
    config: RiskConfig,
) -> tuple[Pipeline, pd.Series, dict[str, float], pd.DataFrame]:
    cols = [
        column
        for column in cols
        if frame.loc[frame.index <= initial_train_end, column].notna().sum() >= 100
    ]
    event_frame = frame.join(labels, how="left")
    event_frame["is_meta_event"] = events.reindex(frame.index).fillna(False)
    all_index = frame.index
    start_pos = all_index.get_indexer([initial_train_end], method="nearest")[0] + 1
    probabilities = pd.Series(np.nan, index=frame.index, name="p_profit_first_raw")
    importances: list[pd.DataFrame] = []
    last_pipe: Pipeline | None = None

    for pred_start in range(start_pos, len(all_index), config.retrain_every_days * 2):
        pred_end = min(pred_start + config.retrain_every_days * 2, len(all_index))
        train_cut = all_index[max(0, pred_start - config.prediction_horizon_days)]
        train_events = event_frame.loc[(event_frame.index <= train_cut) & event_frame["tb_label"].notna()]
        predict_index = all_index[pred_start:pred_end]
        predict_events = event_frame.loc[predict_index]
        predict_events = predict_events[predict_events["is_meta_event"]]
        if len(train_events) < 80 or len(predict_events) == 0 or train_events["tb_label"].nunique() < 2:
            continue
        pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", fit_meta_xgb_model())])
        pipe.fit(train_events[cols], train_events["tb_label"])
        probabilities.loc[predict_events.index] = pipe.predict_proba(predict_events[cols])[:, 1]
        importances.append(
            pd.DataFrame(
                {
                    "feature": cols,
                    "importance": pipe.named_steps["model"].feature_importances_,
                }
            )
        )
        last_pipe = pipe

    if last_pipe is None:
        raise RuntimeError("Triple-barrier meta model failed to train any fold")

    validation_events = event_frame.loc[
        (event_frame.index > initial_train_end)
        & (event_frame.index <= validation_end)
        & event_frame["tb_label"].notna()
        & probabilities.notna()
    ]
    test_events = event_frame.loc[
        test_mask
        & event_frame["tb_label"].notna()
        & probabilities.notna()
    ]

    raw_validation_auc = float("nan")
    raw_test_auc = float("nan")
    use_inverse = False
    if len(validation_events) > 20 and validation_events["tb_label"].nunique() == 2:
        raw_validation_auc = float(
            roc_auc_score(validation_events["tb_label"], probabilities.loc[validation_events.index])
        )
        use_inverse = raw_validation_auc < 0.48
    oriented = 1 - probabilities if use_inverse else probabilities
    oriented.name = "p_profit_first"

    if len(test_events) > 20 and test_events["tb_label"].nunique() == 2:
        raw_test_auc = float(roc_auc_score(test_events["tb_label"], probabilities.loc[test_events.index]))

    test_probability_index = test_events.index
    metrics = {
        "target": "triple_barrier_profit_first",
        "raw_validation_auc": raw_validation_auc,
        "raw_test_auc": raw_test_auc,
        "probability_orientation": "inverted_by_validation" if use_inverse else "raw",
        "walk_forward_folds": int(len(importances)),
        "meta_events": int(labels["tb_label"].notna().sum()),
        "meta_positive_rate": float(labels["tb_label"].mean()) if len(labels) else float("nan"),
        "event_gap_days": config.meta_event_gap_days,
        "event_kind": config.meta_event_kind,
        "cusum_threshold_mult": config.cusum_threshold_mult,
        "primary_signal_mode": config.primary_signal_mode,
        "profit_atr_multiple": config.profit_atr_multiple,
        "stop_atr_multiple": config.stop_atr_multiple,
        "vertical_barrier_days": config.prediction_horizon_days,
    }
    if len(test_probability_index) > 20 and event_frame.loc[test_probability_index, "tb_label"].nunique() == 2:
        y_test = event_frame.loc[test_probability_index, "tb_label"]
        p_test = oriented.loc[test_probability_index]
        try:
            metrics["test_auc"] = float(roc_auc_score(y_test, p_test))
            metrics["test_brier"] = float(brier_score_loss(y_test, p_test))
            metrics["test_accuracy_0_5"] = float(accuracy_score(y_test, p_test > 0.5))
        except ValueError:
            metrics["test_auc"] = float("nan")

    importance_frame = (
        pd.concat(importances)
        .groupby("feature", as_index=False)["importance"]
        .mean()
        .sort_values("importance", ascending=False)
    )
    return last_pipe, oriented, metrics, importance_frame


def generate_signals(
    frame: pd.DataFrame,
    probabilities: pd.Series,
    config: RiskConfig,
) -> pd.DataFrame:
    signal_frame = frame.copy()
    primary_signal = primary_long_signal(signal_frame, config.primary_signal_mode)
    events = make_meta_events(signal_frame, primary_signal, config)
    accepted_events = events & (probabilities >= config.up_threshold)
    signal_frame["p_profit_first_event"] = probabilities
    signal_frame["p_profit_first"] = probabilities.ffill()
    signal_frame["p_up_30d"] = signal_frame["p_profit_first"]
    signal_frame["tb_event"] = events
    signal_frame["tb_accepted_event"] = accepted_events
    signal_frame["primary_trend_signal"] = primary_signal
    signal_frame["payoff_ratio"] = config.profit_atr_multiple / config.stop_atr_multiple
    signal_frame["historical_train_win_rate"] = np.nan

    positions = []
    stop_prices = []
    take_profit_prices = []
    execution_actions = []
    exit_reasons = []
    raw_signals = []
    guides = []
    in_position = False
    current_position = 0.0
    entry = np.nan
    stop_price = np.nan
    take_profit_price = np.nan
    hmm_exit_streak = 0
    accepted_arr = accepted_events.reindex(signal_frame.index).fillna(False).to_numpy()

    for i, row in enumerate(signal_frame.itertuples()):
        action = "持有/观望"
        raw_signal = "hold"
        guide = "持有/观望"
        exit_reason = ""
        close = row.gold_close
        high = row.gold_high
        low = row.gold_low
        atr = row.atr
        flat_condition = (not bool(row.primary_trend_signal)) or (
            (row.gold_close < row.sma_60) and row.market_state in ["熊市", "恐慌"]
        )

        if in_position:
            hit_profit = np.isfinite(take_profit_price) and high >= take_profit_price
            hit_stop = np.isfinite(stop_price) and low <= stop_price
            model_exit = (
                bool(row.tb_event)
                and np.isfinite(row.p_profit_first_event)
                and row.p_profit_first_event <= config.down_threshold
            )
            hmm_exit_setup = (
                row.market_state in ["熊市", "恐慌"]
                and close < row.sma_60
            )
            hmm_exit_streak = hmm_exit_streak + 1 if hmm_exit_setup else 0
            hmm_exit = hmm_exit_streak >= config.hmm_exit_confirmation_days
            if hit_stop or hit_profit or model_exit or hmm_exit:
                in_position = False
                current_position = 0.0
                stop_price = np.nan
                take_profit_price = np.nan
                hmm_exit_streak = 0
                action = "卖出"
                raw_signal = "flat"
                guide = "卖出/空仓"
                if hit_stop:
                    exit_reason = "atr_stop"
                elif hit_profit:
                    exit_reason = "atr_take_profit"
                elif model_exit:
                    exit_reason = "xgboost_down_threshold"
                else:
                    exit_reason = "hmm_trend_exit"
            else:
                action = "持有"
                guide = "持有"

        if not in_position and accepted_arr[i] and np.isfinite(atr):
            in_position = True
            current_position = min(config.max_position, config.max_leverage)
            entry = close
            stop_price = entry - config.stop_atr_multiple * atr
            take_profit_price = entry + config.profit_atr_multiple * atr
            hmm_exit_streak = 0
            action = "买入"
            raw_signal = "long"
            guide = "买入"
            exit_reason = ""
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

    signal_frame["position"] = positions
    signal_frame["atr_stop"] = stop_prices
    signal_frame["tb_take_profit"] = take_profit_prices
    signal_frame["execution_action"] = execution_actions
    signal_frame["exit_reason"] = exit_reasons
    signal_frame["raw_signal"] = raw_signals
    signal_frame["guide"] = guides
    return signal_frame


def backtest(signal_frame: pd.DataFrame, test_mask: pd.Series) -> tuple[pd.DataFrame, dict[str, float]]:
    bt = signal_frame.loc[test_mask].copy()
    bt["strategy_ret"] = bt["position"].shift(1).fillna(0) * bt["gold_close"].pct_change().fillna(0)
    bt["benchmark_ret"] = bt["gold_close"].pct_change().fillna(0)
    bt["turnover"] = bt["position"].diff().abs().fillna(bt["position"].abs())
    bt["equity"] = (1 + bt["strategy_ret"]).cumprod()
    bt["benchmark_equity"] = (1 + bt["benchmark_ret"]).cumprod()
    bt["drawdown"] = bt["equity"] / bt["equity"].cummax() - 1

    days = max(len(bt), 1)
    total_return = bt["equity"].iloc[-1] - 1
    benchmark_return = bt["benchmark_equity"].iloc[-1] - 1
    annual_return = (bt["equity"].iloc[-1]) ** (252 / days) - 1
    annual_vol = bt["strategy_ret"].std() * math.sqrt(252)
    sharpe = annual_return / annual_vol if annual_vol and np.isfinite(annual_vol) else 0.0
    max_drawdown = bt["drawdown"].min()
    active_days = float((bt["position"].shift(1).fillna(0) > 0).mean())
    win_days = bt.loc[bt["strategy_ret"] != 0, "strategy_ret"]
    win_rate = float((win_days > 0).mean()) if len(win_days) else 0.0

    metrics = {
        "total_return": float(total_return),
        "benchmark_return": float(benchmark_return),
        "annual_return": float(annual_return),
        "annual_vol": float(annual_vol),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_drawdown),
        "active_day_ratio": active_days,
        "daily_win_rate_when_active": win_rate,
        "test_trades": int(bt["execution_action"].isin(["买入", "卖出"]).sum()) if "execution_action" in bt else 0,
        "turnover": float(bt["turnover"].sum()),
    }
    for cost_bps in [2, 5]:
        net_ret = bt["strategy_ret"] - bt["turnover"] * (cost_bps / 10000)
        net_equity = (1 + net_ret).cumprod()
        net_drawdown = net_equity / net_equity.cummax() - 1
        net_annual_return = net_equity.iloc[-1] ** (252 / days) - 1
        net_annual_vol = net_ret.std() * math.sqrt(252)
        metrics[f"net_total_return_{cost_bps}bps"] = float(net_equity.iloc[-1] - 1)
        metrics[f"net_sharpe_{cost_bps}bps"] = float(net_annual_return / net_annual_vol) if net_annual_vol else 0.0
        metrics[f"net_max_drawdown_{cost_bps}bps"] = float(net_drawdown.min())
    return bt, metrics


def build_outputs(
    signal_frame: pd.DataFrame,
    backtest_frame: pd.DataFrame,
    model_metrics: dict[str, float],
    backtest_metrics: dict[str, float],
    importances: pd.DataFrame,
    sources: dict[str, str],
    state_mapping: dict[int, str],
    data_quality: dict[str, Any],
    config: RiskConfig,
) -> None:
    ensure_dirs()

    log_columns = [
        "gold_close",
        "p_profit_first",
        "p_profit_first_event",
        "p_up_30d",
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
        "historical_train_win_rate",
        "dxy_close",
        "us10y_close",
        "real_rate_proxy",
        "vixy_close",
        "gld_close",
        "cot_gold_net",
    ]
    existing = [column for column in log_columns if column in signal_frame.columns]
    signal_frame[existing].to_csv(LOCAL_LOGS / "gold_signals.csv", encoding="utf-8-sig")

    latest = signal_frame.dropna(subset=["gold_close", "p_up_30d"]).iloc[-1]
    previous = signal_frame.dropna(subset=["gold_close", "p_up_30d"]).iloc[-2]
    top_features = importances.head(10).to_dict("records")

    latest_json = {
        "asOf": str(latest.name.date()),
        "asset": "COMEX 迷你黄金连续合约 QO00Y",
        "assetDetail": "东方财富国际期货 secid=101.QO00Y，作为黄金价格主序列",
        "price": float(latest["gold_close"]),
        "dailyChange": float(latest["gold_close"] / previous["gold_close"] - 1),
        "pUp30d": float(latest["p_up_30d"]),
        "pUpHorizon": float(latest["p_up_30d"]),
        "pProfitFirst": float(latest["p_profit_first"]),
        "predictionHorizonDays": config.prediction_horizon_days,
        "predictionTarget": (
            f"{config.primary_signal_mode} + {config.meta_event_kind} 候选交易在 {config.prediction_horizon_days} 个交易日训练标签窗口内，"
            f"是否先触发 {config.profit_atr_multiple:g} ATR 止盈而不是 "
            f"{config.stop_atr_multiple:g} ATR 止损"
        ),
        "marketStateCode": str(latest["market_state_code"]),
        "marketState": str(latest["market_state"]),
        "guide": str(latest["guide"]),
        "rawSignal": str(latest["raw_signal"]),
        "position": float(latest["position"]),
        "atrStop": None if pd.isna(latest["atr_stop"]) else float(latest["atr_stop"]),
        "takeProfit": None if pd.isna(latest["tb_take_profit"]) else float(latest["tb_take_profit"]),
        "atrPct": float(latest["atr_pct"]),
        "thresholds": {
            "buyAbove": config.up_threshold,
            "sellBelow": config.down_threshold,
        },
        "risk": asdict(config),
        "modelMetrics": model_metrics,
        "backtestMetrics": backtest_metrics,
        "topFeatures": top_features,
        "stateMapping": {str(k): v for k, v in state_mapping.items()},
        "sources": sources,
        "dataQuality": data_quality,
        "notes": [
            "VIX 使用 VIXY ETF 作为风险代理。",
            "实际利率使用 US10Y 减美国 CPI 同比作为 proxy。",
            "ETF 资金流使用 GLD 成交额的量价方向 proxy。",
            "XGBoost 当前预测的是 triple-barrier meta-label：HMM quality + CUSUM 候选交易是否先触发止盈。",
            f"{config.prediction_horizon_days} 日窗口仅用于训练标签和防止标签泄漏，不作为真实持仓的强制退出时间。",
            f"HMM 退出需要熊市/恐慌且跌破 60 日均线连续确认 {config.hmm_exit_confirmation_days} 天。",
            "研究结果不构成投资建议。",
        ],
    }
    (PUBLIC_DATA / "gold_research_latest.json").write_text(
        json.dumps(latest_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    price_columns = [
        "gold_close",
        "sma_5",
        "sma_20",
        "sma_60",
        "sma_120",
        "market_state",
        "market_state_code",
        "p_up_30d",
        "p_profit_first",
        "position",
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
            "p_up_30d": "pUp30d",
            "p_profit_first": "pProfitFirst",
            "atr_stop": "atrStop",
            "tb_take_profit": "takeProfit",
            "tb_event": "event",
            "tb_accepted_event": "acceptedEvent",
        }
    )
    price["date"] = price["date"].dt.strftime("%Y-%m-%d")
    (PUBLIC_DATA / "gold_price_series.json").write_text(
        json.dumps(price.replace({np.nan: None}).to_dict("records"), ensure_ascii=False),
        encoding="utf-8",
    )

    bt = backtest_frame[["equity", "benchmark_equity", "drawdown", "position"]].reset_index()
    bt["date"] = bt["date"].dt.strftime("%Y-%m-%d")
    (PUBLIC_DATA / "gold_backtest.json").write_text(
        json.dumps(bt.replace({np.nan: None}).to_dict("records"), ensure_ascii=False),
        encoding="utf-8",
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
    train_split_at = int(len(usable) * 0.55)
    validation_split_at = int(len(usable) * 0.72)
    train_end = usable.index[train_split_at]
    validation_end = usable.index[validation_split_at]
    train_mask = features.index <= train_end
    test_mask = features.index > validation_end

    _, state_mapping, state_frame = fit_hmm(features, train_mask)
    features = features.join(state_frame)
    state_dummies = pd.get_dummies(features["market_state_code"], prefix="state", dtype=float)
    features = features.join(state_dummies)

    cols = feature_columns(features)
    cols = [column for column in cols if column not in {"hmm_raw_state"}]
    primary_signal = primary_long_signal(features, config.primary_signal_mode)
    meta_events = make_meta_events(features, primary_signal, config)
    meta_labels = triple_barrier_labels(features, meta_events, config)
    model, probabilities, model_metrics, importances = train_triple_barrier_meta_model(
        features,
        meta_events,
        meta_labels,
        cols,
        train_end,
        validation_end,
        test_mask,
        config,
    )
    signals = generate_signals(features, probabilities, config)
    backtest_frame, backtest_metrics = backtest(signals, test_mask)

    build_outputs(
        signals,
        backtest_frame,
        model_metrics,
        backtest_metrics,
        importances,
        sources,
        state_mapping,
        data_quality,
        config,
    )

    latest = signals.dropna(subset=["gold_close", "p_up_30d"]).iloc[-1]
    return {
        "as_of": str(latest.name.date()),
        "price": float(latest["gold_close"]),
        "market_state": str(latest["market_state"]),
        "market_state_code": str(latest["market_state_code"]),
        "p_up_30d": float(latest["p_up_30d"]),
        "guide": str(latest["guide"]),
        "position": float(latest["position"]),
        "model_metrics": model_metrics,
        "backtest_metrics": backtest_metrics,
        "outputs": {
            "signals_csv": str(LOCAL_LOGS / "gold_signals.csv"),
            "latest_json": str(PUBLIC_DATA / "gold_research_latest.json"),
            "price_json": str(PUBLIC_DATA / "gold_price_series.json"),
            "backtest_json": str(PUBLIC_DATA / "gold_backtest.json"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local gold HMM + XGBoost research pipeline.")
    parser.add_argument("--json", action="store_true", help="Print a JSON summary.")
    args = parser.parse_args()
    summary = run_pipeline()
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(
            f"{summary['as_of']} {summary['market_state_code']}={summary['market_state']} "
            f"P(profit first)={summary['p_up_30d']:.2%} guide={summary['guide']} "
            f"position={summary['position']:.1%}"
        )


if __name__ == "__main__":
    main()
