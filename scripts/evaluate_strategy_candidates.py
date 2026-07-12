"""Pre-registered robustness audit for long-only, unlevered gold candidates.

The already-reviewed development period after 2023-02-27 is deliberately
excluded from selection. Outputs are local research logs and never website data.
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research import gold_research_pipeline as pipeline


SOURCE = ROOT / "research" / "gold_research_pipeline.py"
OUTPUT_CSV = ROOT / "local_logs" / "gold_candidate_pseudo_oos.csv"
YEARLY_CSV = ROOT / "local_logs" / "gold_candidate_pseudo_oos_yearly.csv"
SUMMARY_JSON = ROOT / "local_logs" / "gold_candidate_pseudo_oos_summary.json"

# Fixed anchored windows. Each evaluation starts strictly after its HMM cutoff.
FOLDS = (
    ("wf_2017_2018", "2016-12-30", "2018-12-31"),
    ("wf_2019_2021", "2018-12-31", "2021-02-23"),
    ("frozen_validation", "2021-02-23", "2023-02-27"),
)

# Materiality thresholds prevent unchanged folds from being counted as wins.
MIN_FOLD_RETURN_DELTA = 0.005
MIN_FOLD_SHARPE_DELTA = 0.05
MIN_COMPOUNDED_RETURN_DELTA = 0.02
MAX_DRAWDOWN_DETERIORATION = 0.02
MAX_TURNOVER_MULTIPLE = 1.25


def candidates(base: pipeline.RiskConfig):
    """Small, declared one-factor challenge set; no parameter grid search."""
    return (
        ("formal_baseline", base, True, False),
        ("profit_12atr", replace(base, profit_atr_multiple=12.0), True, False),
        ("stop_7atr", replace(base, stop_atr_multiple=7.0), True, False),
        ("hmm_confirm_30d", replace(base, hmm_exit_confirmation_days=30), True, False),
        ("trend_exit_only", base, False, True),
        ("hybrid_hmm_and_trend_exit", base, True, True),
    )


def source_sha256() -> str:
    return hashlib.sha256(SOURCE.read_bytes()).hexdigest()


def yearly_metrics(frame: pd.DataFrame) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for year, subset in frame.groupby(frame.index.year):
        returns = subset["strategy_ret"].fillna(0.0)
        risk_free = subset.get("risk_free_return", pd.Series(0.0, index=subset.index)).fillna(0.0)
        excess_returns = returns - risk_free
        equity = (1.0 + returns).cumprod()
        drawdown = equity / equity.cummax() - 1.0
        volatility = excess_returns.std()
        rows.append(
            {
                "year": int(year),
                "return_5bps": float(equity.iloc[-1] - 1.0),
                "sharpe_5bps": (
                    float(excess_returns.mean() / volatility * np.sqrt(252))
                    if volatility and np.isfinite(volatility)
                    else 0.0
                ),
                "max_drawdown_5bps": float(drawdown.min()),
                "turnover": float(subset["turnover"].sum()),
            }
        )
    return rows


def main() -> None:
    pipeline.OFFLINE_MODE = True
    pipeline.ensure_dirs()
    base = pipeline.RiskConfig()
    market, _ = pipeline.load_market_data()
    features = pipeline.build_features(market, base).replace([np.inf, -np.inf], np.nan)
    empty_probability = pd.Series(np.nan, index=features.index)
    rows: list[dict[str, object]] = []
    annual_rows: list[dict[str, object]] = []

    for fold, cutoff_text, end_text in FOLDS:
        cutoff = pd.Timestamp(cutoff_text)
        end = pd.Timestamp(end_text)
        fold_features = features.loc[:end].copy()
        _, _, states = pipeline.fit_hmm_walk_forward(
            fold_features,
            cutoff,
            base.hmm_retrain_every_days,
            base.hmm_feature_policy,
        )
        fold_features = fold_features.join(states)
        mask = (fold_features.index > cutoff) & (fold_features.index <= end)
        start = fold_features.index[mask].min()

        for name, config, use_hmm_exit, use_trend_exit in candidates(base):
            signals = pipeline.generate_signals(
                fold_features,
                empty_probability.reindex(fold_features.index),
                config,
                use_xgboost=False,
                use_atr=True,
                start_at=start,
                use_hmm_exit=use_hmm_exit,
                use_trend_exit=use_trend_exit,
            )
            live, metrics = pipeline.backtest_next_open(
                signals,
                mask,
                config,
                cost_bps=5.0,
                write_log=False,
            )
            rows.append(
                {
                    "fold": fold,
                    "hmm_cutoff": cutoff_text,
                    "evaluation_end": end_text,
                    "candidate": name,
                    "return_5bps": metrics["total_return"],
                    "sharpe_5bps": metrics["sharpe"],
                    "max_drawdown_5bps": metrics["max_drawdown"],
                    "entries": metrics["entries"],
                    "exits": metrics["exits"],
                    "turnover": metrics["turnover"],
                }
            )
            for annual in yearly_metrics(live):
                annual_rows.append({"fold": fold, "candidate": name, **annual})

    result = pd.DataFrame(rows)
    annual_result = pd.DataFrame(annual_rows)
    result.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    annual_result.to_csv(YEARLY_CSV, index=False, encoding="utf-8-sig")

    fold_order = [fold[0] for fold in FOLDS]
    baseline = result[result["candidate"] == "formal_baseline"].set_index("fold").loc[fold_order]
    baseline_compounded = float((1.0 + baseline["return_5bps"]).prod() - 1.0)
    baseline_worst_drawdown = float(baseline["max_drawdown_5bps"].min())
    baseline_turnover = float(baseline["turnover"].sum())
    baseline_validation_return = float(baseline.loc["frozen_validation", "return_5bps"])
    aggregate: list[dict[str, object]] = []

    for name, group in result.groupby("candidate", sort=False):
        aligned = group.set_index("fold").loc[fold_order]
        return_delta = aligned["return_5bps"] - baseline["return_5bps"]
        sharpe_delta = aligned["sharpe_5bps"] - baseline["sharpe_5bps"]
        compounded = float((1.0 + aligned["return_5bps"]).prod() - 1.0)
        worst_drawdown = float(aligned["max_drawdown_5bps"].min())
        turnover = float(aligned["turnover"].sum())
        validation_return = float(aligned.loc["frozen_validation", "return_5bps"])
        material_return_wins = int((return_delta >= MIN_FOLD_RETURN_DELTA).sum())
        material_sharpe_wins = int((sharpe_delta >= MIN_FOLD_SHARPE_DELTA).sum())
        passes = bool(
            name != "formal_baseline"
            and material_return_wins >= 2
            and material_sharpe_wins >= 2
            and compounded >= baseline_compounded + MIN_COMPOUNDED_RETURN_DELTA
            and validation_return >= baseline_validation_return
            and worst_drawdown >= baseline_worst_drawdown - MAX_DRAWDOWN_DETERIORATION
            and turnover <= baseline_turnover * MAX_TURNOVER_MULTIPLE
        )
        aggregate.append(
            {
                "candidate": name,
                "compounded_return_5bps": compounded,
                "median_sharpe_5bps": float(aligned["sharpe_5bps"].median()),
                "worst_max_drawdown_5bps": worst_drawdown,
                "material_return_wins": material_return_wins,
                "material_sharpe_wins": material_sharpe_wins,
                "frozen_validation_return_5bps": validation_return,
                "total_entries": int(aligned["entries"].sum()),
                "total_exits": int(aligned["exits"].sum()),
                "total_turnover": turnover,
                "passes_predeclared_gate": passes,
            }
        )

    payload = {
        "source_sha256": source_sha256(),
        "data_as_of": str(features.dropna(subset=["gold_close"]).index.max().date()),
        "development_period_used_for_selection": False,
        "folds": [dict(name=fold, hmm_cutoff=cutoff, evaluation_end=end) for fold, cutoff, end in FOLDS],
        "materiality": {
            "minimum_fold_return_delta": MIN_FOLD_RETURN_DELTA,
            "minimum_fold_sharpe_delta": MIN_FOLD_SHARPE_DELTA,
            "minimum_compounded_return_delta": MIN_COMPOUNDED_RETURN_DELTA,
            "maximum_drawdown_deterioration": MAX_DRAWDOWN_DETERIORATION,
            "maximum_turnover_multiple": MAX_TURNOVER_MULTIPLE,
            "frozen_validation_must_not_underperform": True,
        },
        "aggregate": aggregate,
    }
    SUMMARY_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(pd.DataFrame(aggregate).to_string(index=False))
    print(f"\nFold results: {OUTPUT_CSV}")
    print(f"Yearly results: {YEARLY_CSV}")
    print(f"Summary: {SUMMARY_JSON}")


if __name__ == "__main__":
    main()
