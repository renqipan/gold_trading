from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

from gold_research_pipeline import RiskConfig, backtest_next_open, primary_long_signal, trend_risk_budget


def test_dynamic_trend_risk_budget() -> None:
    config = RiskConfig()
    normal = pd.Series({"ret_120": 0.11, "gold_close": 2100.0, "sma_120": 2000.0})
    strong = pd.Series({"ret_120": 0.12, "gold_close": 2100.0, "sma_120": 2000.0})
    assert trend_risk_budget(normal, config) == config.normal_trend_risk_budget
    assert trend_risk_budget(strong, config) == config.strong_trend_risk_budget


def test_partial_position_uses_cash_and_units_without_free_rebalancing() -> None:
    index = pd.date_range("2024-01-01", periods=3, freq="D")
    frame = pd.DataFrame(
        {
            "gold_open": [100.0, 100.0, 110.0],
            "gold_close": [100.0, 110.0, 121.0],
            "gold_high": [100.0, 110.0, 121.0],
            "gold_low": [100.0, 100.0, 110.0],
            "market_state": ["牛市", "牛市", "牛市"],
            "sma_60": [90.0, 90.0, 90.0],
            "atr": [1.0, 1.0, 1.0],
            "tb_event": [True, False, False],
            "tb_accepted_event": [True, False, False],
            "p_profit_first_event": [0.8, np.nan, np.nan],
        },
        index=index,
    )
    config = replace(
        RiskConfig(),
        max_position=0.5,
        max_leverage=0.5,
        max_single_loss=1.0,
        dynamic_trend_risk_enabled=False,
        profit_atr_multiple=100.0,
        stop_atr_multiple=100.0,
    )
    result, metrics = backtest_next_open(
        frame,
        pd.Series(True, index=index),
        config,
        cost_bps=0.0,
        write_log=False,
    )
    assert abs(metrics["total_return"] - 0.105) < 1e-12
    assert abs(result.iloc[-1]["position"] - (0.005 * 121 / 1.105)) < 1e-12


def test_low_price_modes_are_distinct_from_the_formal_trend() -> None:
    frame = pd.DataFrame(
        {
            "gold_close": [90.0],
            "sma_20": [85.0],
            "sma_60": [100.0],
            "sma_120": [110.0],
            "ret_5": [0.02],
            "ret_20": [0.04],
            "new_low_240": [1.0],
            "recent_new_low_240": [1.0],
        },
        index=[pd.Timestamp("2024-01-01")],
    )
    assert not bool(primary_long_signal(frame, "trend_slow").iloc[0])
    assert bool(primary_long_signal(frame, "below_sma_60").iloc[0])
    assert bool(primary_long_signal(frame, "below_sma_120").iloc[0])
    assert bool(primary_long_signal(frame, "dip_recovery_60").iloc[0])
    assert bool(primary_long_signal(frame, "new_low_240").iloc[0])
    assert bool(primary_long_signal(frame, "dip_recovery_240").iloc[0])
    assert bool(primary_long_signal(frame, "trend_or_dip_240").iloc[0])


if __name__ == "__main__":
    test_dynamic_trend_risk_budget()
    test_partial_position_uses_cash_and_units_without_free_rebalancing()
    test_low_price_modes_are_distinct_from_the_formal_trend()
    print("strategy engine tests passed")
