from __future__ import annotations

import csv
import io
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCAL_LOGS = ROOT / "local_logs"

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
)


def compact(text: str, limit: int = 180) -> str:
    value = " ".join(text.strip().split())
    return value if len(value) <= limit else f"{value[:limit].rstrip()}..."


def curl_text(url: str, timeout: int = 10) -> tuple[bool, str, str, float]:
    start = time.time()
    result = subprocess.run(
        [
            "curl",
            "-q",
            "-L",
            "--silent",
            "--show-error",
            "--fail",
            "--compressed",
            "--http1.1",
            "--max-time",
            str(timeout),
            "--noproxy",
            "*",
            "-H",
            f"User-Agent: {USER_AGENT}",
            "-H",
            "Accept: application/json,text/csv,text/plain,*/*",
            "-H",
            "Accept-Language: zh-CN,zh;q=0.9,en;q=0.8",
            url,
        ],
        text=True,
        capture_output=True,
    )
    elapsed = round(time.time() - start, 2)
    if result.returncode != 0:
        return False, result.stdout, compact(result.stderr), elapsed
    return True, result.stdout, "", elapsed


def parse_eastmoney(text: str) -> dict[str, object]:
    payload = json.loads(text)
    klines = ((payload.get("data") or {}).get("klines") or [])
    close = None
    if klines:
        close = float(klines[-1].split(",")[2])
    return {"rows": len(klines), "last": close}


def parse_yahoo_chart(text: str) -> dict[str, object]:
    payload = json.loads(text)
    result = ((payload.get("chart") or {}).get("result") or [])
    quote = (((result[0].get("indicators") or {}).get("quote") or [{}])[0]) if result else {}
    closes = [value for value in (quote.get("close") or []) if value is not None]
    return {"rows": len(closes), "last": closes[-1] if closes else None}


def parse_csv(text: str, value_column: str = "Close") -> dict[str, object]:
    rows = list(csv.DictReader(io.StringIO(text)))
    values = []
    for row in rows:
        raw = row.get(value_column)
        if raw in (None, "", "."):
            continue
        try:
            values.append(float(raw))
        except ValueError:
            continue
    return {"rows": len(values), "last": values[-1] if values else None}


def probe() -> list[dict[str, object]]:
    tests = [
        {
            "name": "eastmoney_gold_qo00y",
            "url": "https://push2his.eastmoney.com/api/qt/stock/kline/get?secid=101.QO00Y&klt=101&fqt=1&lmt=20&end=20500000&iscca=1&fields1=f1,f2,f3,f4,f5,f6,f7,f8&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63,f64&ut=f057cbcbce2a86e2866ab8877db1d059&forcect=1",
            "parser": parse_eastmoney,
        },
        {
            "name": "yahoo_gold_gc_f",
            "url": "https://query1.finance.yahoo.com/v8/finance/chart/GC%3DF?range=1mo&interval=1d",
            "parser": parse_yahoo_chart,
        },
        {
            "name": "stooq_gld_us",
            "url": "https://stooq.com/q/d/l/?s=gld.us&i=d",
            "parser": parse_csv,
        },
        {
            "name": "fred_us10y_dgs10",
            "url": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10",
            "parser": lambda text: parse_csv(text, "DGS10"),
        },
    ]
    results = []
    for item in tests:
        ok, text, error, elapsed = curl_text(item["url"])
        parsed = {"rows": 0, "last": None}
        parse_error = ""
        if ok:
            try:
                parsed = item["parser"](text)
                ok = bool(parsed["rows"])
            except Exception as exc:  # noqa: BLE001 - probe should keep testing other sources.
                ok = False
                parse_error = compact(str(exc))
        results.append(
            {
                "name": item["name"],
                "ok": ok,
                "rows": parsed["rows"],
                "last": parsed["last"],
                "elapsed": elapsed,
                "error": error or parse_error,
                "preview": compact(text[:260]) if not ok and text else "",
            }
        )
    return results


def main() -> None:
    LOCAL_LOGS.mkdir(parents=True, exist_ok=True)
    results = probe()
    (LOCAL_LOGS / "data_source_probe.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
