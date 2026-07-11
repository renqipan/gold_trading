from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

from gold_research_pipeline import RiskConfig, backtest_next_open, purged_training_events


def test_purge_uses_realized_label_exit() -> None:
    prediction_start = pd.Timestamp("2024-01-10")
    events = pd.DataFrame(
        {
            "tb_label": [1.0, 0.0, 1.0],
            "tb_exit_date": ["2024-01-09", "2024-01-10", "2024-01-11"],
        },
        index=pd.to_datetime(["2023-11-01", "2023-11-02", "2023-11-03"]),
    )
    purged = purged_training_events(events, prediction_start)
    assert list(purged.index) == [pd.Timestamp("2023-11-01")]


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


if __name__ == "__main__":
    test_purge_uses_realized_label_exit()
    test_partial_position_uses_cash_and_units_without_free_rebalancing()
    print("strategy engine tests passed")
