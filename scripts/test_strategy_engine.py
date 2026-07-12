from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

import gold_research_pipeline as pipeline
from gold_research_pipeline import (
    RiskConfig,
    backtest_next_open,
    overlay_execution_state,
    primary_long_signal,
    trend_risk_budget,
    update_forward_ledger,
)


def execution_frame(periods: int = 6) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=periods, freq="D")
    return pd.DataFrame(
        {
            "gold_open": 100.0,
            "gold_close": 100.0,
            "gold_high": 100.5,
            "gold_low": 99.5,
            "cash_yield_pct": 0.0,
            "market_state": "牛市",
            "sma_60": 90.0,
            "sma_120": 90.0,
            "ret_120": 0.05,
            "atr": 1.0,
            "tb_event": False,
            "tb_accepted_event": False,
            "primary_trend_signal": True,
            "p_profit_first_event": np.nan,
            "atr_stop_enabled": False,
            "atr_profit_enabled": False,
            "hmm_exit_enabled": True,
            "trend_exit_enabled": False,
            "target_position_override": 1.0,
        },
        index=index,
    )


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
            "cash_yield_pct": [0.0, 0.0, 0.0],
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


def test_long_only_unlevered_config_is_enforced() -> None:
    for kwargs in [
        {"long_only": False},
        {"max_position": 1.01},
        {"max_leverage": 1.01},
    ]:
        try:
            replace(RiskConfig(), **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid long-only configuration accepted: {kwargs}")


def test_signal_executes_only_at_next_open_and_last_signal_remains_pending() -> None:
    frame = execution_frame(3)
    frame.loc[frame.index[0], ["tb_event", "tb_accepted_event"]] = True
    result, _ = backtest_next_open(
        frame,
        pd.Series(True, index=frame.index),
        RiskConfig(),
        cost_bps=0.0,
        write_log=False,
    )
    assert result.iloc[0]["position"] == 0.0
    assert bool(result.iloc[0]["pending_entry"])
    assert result.iloc[1]["position"] > 0.0
    assert "买入" in result.iloc[1]["execution_action"]

    last_only = execution_frame(3)
    last_only.loc[last_only.index[-1], ["tb_event", "tb_accepted_event"]] = True
    last_result, _ = backtest_next_open(
        last_only,
        pd.Series(True, index=last_only.index),
        RiskConfig(),
        cost_bps=0.0,
        write_log=False,
    )
    assert last_result.iloc[-1]["position"] == 0.0
    assert bool(last_result.iloc[-1]["pending_entry"])
    assert last_result.iloc[-1]["guide"] == "买入（下一交易日开盘）"


def test_hmm_confirmation_resets_after_exit_and_can_be_disabled() -> None:
    frame = execution_frame(7)
    frame["market_state"] = "熊市"
    frame["sma_60"] = 101.0
    frame.loc[frame.index[[0, 3]], ["tb_event", "tb_accepted_event"]] = True
    frame.loc[frame.index[5]:, "market_state"] = "牛市"
    config = replace(RiskConfig(), hmm_exit_confirmation_days=2)
    result, _ = backtest_next_open(
        frame,
        pd.Series(True, index=frame.index),
        config,
        cost_bps=0.0,
        write_log=False,
    )
    assert result.iloc[2]["pending_exit"]
    assert result.iloc[3]["exit_reason"] == "next_open_hmm_exit"
    assert result.iloc[4]["position"] > 0.0
    assert not bool(result.iloc[4]["pending_exit"])
    assert result.iloc[-1]["position"] > 0.0

    disabled = frame.copy()
    disabled["hmm_exit_enabled"] = False
    disabled_result, _ = backtest_next_open(
        disabled,
        pd.Series(True, index=disabled.index),
        config,
        cost_bps=0.0,
        write_log=False,
    )
    assert not disabled_result["pending_exit"].any()


def test_double_barrier_is_conservatively_stopped_and_gap_uses_open() -> None:
    frame = execution_frame(3)
    frame.loc[frame.index[0], ["tb_event", "tb_accepted_event"]] = True
    frame["atr_stop_enabled"] = True
    frame["atr_profit_enabled"] = True
    frame.loc[frame.index[1], ["gold_high", "gold_low"]] = [103.0, 97.0]
    config = replace(RiskConfig(), stop_atr_multiple=2.0, profit_atr_multiple=2.0)
    result, metrics = backtest_next_open(
        frame,
        pd.Series(True, index=frame.index),
        config,
        cost_bps=0.0,
        write_log=False,
    )
    assert result.iloc[1]["exit_reason"] == "atr_stop"
    assert abs(metrics["total_return"] + 0.02) < 1e-12
    assert metrics["entries"] == 1 and metrics["exits"] == 1

    gap = execution_frame(4)
    gap.loc[gap.index[0], ["tb_event", "tb_accepted_event"]] = True
    gap["atr_stop_enabled"] = True
    gap.loc[gap.index[2], ["gold_open", "gold_close", "gold_high", "gold_low"]] = [95.0, 95.0, 96.0, 94.0]
    gap_result, gap_metrics = backtest_next_open(
        gap,
        pd.Series(True, index=gap.index),
        config,
        cost_bps=0.0,
        write_log=False,
    )
    assert gap_result.iloc[2]["exit_reason"] == "atr_stop_gap"
    assert abs(gap_metrics["total_return"] + 0.05) < 1e-12


def test_invalid_open_fails_instead_of_using_impossible_intraday_fill() -> None:
    frame = execution_frame(3)
    frame.loc[frame.index[1], "gold_open"] = np.nan
    try:
        backtest_next_open(
            frame,
            pd.Series(True, index=frame.index),
            RiskConfig(),
            cost_bps=0.0,
            write_log=False,
        )
    except RuntimeError as exc:
        assert "opening prices" in str(exc)
    else:
        raise AssertionError("strict execution silently accepted a missing open")

    invalid_range = execution_frame(3)
    invalid_range.loc[invalid_range.index[1], "gold_high"] = 99.0
    try:
        backtest_next_open(
            invalid_range,
            pd.Series(True, index=invalid_range.index),
            RiskConfig(),
            cost_bps=0.0,
            write_log=False,
        )
    except RuntimeError as exc:
        assert "OHLC ranges" in str(exc)
    else:
        raise AssertionError("strict execution silently accepted an invalid OHLC range")


def test_cash_yield_uses_prior_observation_and_calendar_days() -> None:
    frame = execution_frame(2)
    frame.index = pd.DatetimeIndex(["2024-01-05", "2024-01-08"])
    frame["cash_yield_pct"] = [5.0, 99.0]
    config = replace(RiskConfig(), cash_yield_haircut_bps=0.0)
    result, metrics = backtest_next_open(
        frame,
        pd.Series(True, index=frame.index),
        config,
        cost_bps=0.0,
        write_log=False,
    )
    expected = (1.05 ** (3.0 / 365.0)) - 1.0
    assert abs(metrics["total_return"] - expected) < 1e-12
    assert abs(result.iloc[-1]["cash_yield_pct_used"] - 5.0) < 1e-12
    assert abs(result.iloc[-1]["cash_interest"] - expected) < 1e-12

    disabled, disabled_metrics = backtest_next_open(
        frame,
        pd.Series(True, index=frame.index),
        replace(config, cash_yield_enabled=False),
        cost_bps=0.0,
        write_log=False,
    )
    assert disabled_metrics["total_return"] == 0.0
    assert disabled["cash_interest"].sum() == 0.0

    missing = frame.drop(columns=["cash_yield_pct"])
    try:
        backtest_next_open(
            missing,
            pd.Series(True, index=missing.index),
            config,
            cost_bps=0.0,
            write_log=False,
        )
    except RuntimeError as exc:
        assert "cash_yield_pct" in str(exc)
    else:
        raise AssertionError("cash-yield-enabled backtest silently accepted missing rate data")


def test_hard_drawdown_cooldown_recovers_without_leverage_or_shorting() -> None:
    frame = execution_frame(7)
    frame[["tb_event", "tb_accepted_event"]] = True
    frame["atr_stop_enabled"] = True
    frame.loc[frame.index[2], ["gold_open", "gold_close", "gold_high", "gold_low"]] = [60.0, 60.0, 61.0, 59.0]
    frame.loc[frame.index[3]:, ["gold_open", "gold_close", "gold_high", "gold_low"]] = [60.0, 60.0, 60.5, 59.5]
    config = replace(
        RiskConfig(),
        stop_atr_multiple=1.0,
        hard_drawdown_cooldown_days=2,
    )
    result, _ = backtest_next_open(
        frame,
        pd.Series(True, index=frame.index),
        config,
        cost_bps=0.0,
        write_log=False,
    )
    assert result.iloc[2]["exit_reason"] == "atr_stop_gap"
    assert bool(result.iloc[2]["entry_blocked_by_drawdown"])
    assert not bool(result.iloc[2]["pending_entry"])
    assert result.iloc[2]["desired_position"] == 0.0
    assert "回撤冷却" in result.iloc[2]["guide"]
    assert (result["position"] >= 0).all()
    assert (result["position"] <= 1.0 + 1e-12).all()
    assert result.iloc[-1]["position"] > 0.0


def test_soft_drawdown_reduces_existing_position_at_next_open() -> None:
    frame = execution_frame(4)
    frame.loc[frame.index[0], ["tb_event", "tb_accepted_event"]] = True
    frame.loc[frame.index[2]:, ["gold_open", "gold_close", "gold_high", "gold_low"]] = [80.0, 80.0, 80.5, 79.5]
    result, _ = backtest_next_open(
        frame,
        pd.Series(True, index=frame.index),
        RiskConfig(),
        cost_bps=0.0,
        write_log=False,
    )
    assert abs(result.iloc[2]["position"] - 0.5) < 1e-12
    assert "回撤减仓" in result.iloc[2]["execution_action"]


def test_soft_drawdown_scales_pending_target() -> None:
    frame = execution_frame(5)
    frame[["tb_event", "tb_accepted_event"]] = True
    frame["atr_stop_enabled"] = True
    frame.loc[frame.index[2]:, ["gold_open", "gold_close", "gold_high", "gold_low"]] = [80.0, 80.0, 80.5, 79.5]
    config = replace(RiskConfig(), stop_atr_multiple=1.0)
    result, _ = backtest_next_open(
        frame,
        pd.Series(True, index=frame.index),
        config,
        cost_bps=0.0,
        write_log=False,
    )
    assert result.iloc[2]["exit_reason"] == "atr_stop_gap"
    assert bool(result.iloc[2]["pending_entry"])
    assert abs(result.iloc[2]["desired_position"] - 0.5) < 1e-12
    assert abs(result.iloc[3]["position"] - 0.5) < 1e-12


def test_public_state_overlay_uses_strict_execution_ledger() -> None:
    frame = execution_frame(3)
    frame["position"] = 1.0
    frame["atr_stop"] = 80.0
    frame["tb_take_profit"] = 120.0
    frame["guide"] = "持有"
    frame["raw_signal"] = "long"
    frame["execution_action"] = "买入"
    frame["exit_reason"] = ""
    ledger = pd.DataFrame(
        {
            "position": [0.0],
            "desired_position": [0.0],
            "atr_stop": [np.nan],
            "tb_take_profit": [np.nan],
            "guide": ["卖出/空仓"],
            "raw_signal": ["flat"],
            "execution_action": ["持有/观望"],
            "exit_reason": [""],
            "pending_entry": [False],
            "pending_exit": [False],
        },
        index=[frame.index[-1]],
    )
    public = overlay_execution_state(frame, ledger)
    assert pd.isna(public.iloc[0]["position"])
    assert public.iloc[-1]["position"] == 0.0
    assert public.iloc[-1]["guide"] == "卖出/空仓"


def test_forward_ledger_is_append_only() -> None:
    index = pd.DatetimeIndex(["2026-07-13", "2026-07-14"])
    execution = pd.DataFrame(
        {
            "strategy_ret": [0.01, -0.005],
            "benchmark_ret": [0.008, -0.003],
            "position": [1.0, 1.0],
            "desired_position": [1.0, 1.0],
            "turnover": [1.0, 0.0],
            "cash_interest": [0.0, 0.0],
            "execution_action": ["买入", "持有/观望"],
            "guide": ["持有", "持有"],
            "pending_entry": [False, False],
            "pending_exit": [False, False],
        },
        index=index,
    )
    original_path = pipeline.FORWARD_LEDGER
    try:
        with tempfile.TemporaryDirectory() as directory:
            pipeline.FORWARD_LEDGER = Path(directory) / "forward.json"
            metrics = update_forward_ledger(execution, RiskConfig())
            assert metrics["days"] == 2
            replay = update_forward_ledger(execution.copy(), RiskConfig())
            assert replay["days"] == 2

            changed = execution.copy()
            changed.loc[index[0], "strategy_ret"] = 0.02
            try:
                update_forward_ledger(changed, RiskConfig())
            except RuntimeError as exc:
                assert "history changed" in str(exc)
            else:
                raise AssertionError("forward ledger silently rewrote an existing record")
    finally:
        pipeline.FORWARD_LEDGER = original_path


if __name__ == "__main__":
    test_dynamic_trend_risk_budget()
    test_partial_position_uses_cash_and_units_without_free_rebalancing()
    test_low_price_modes_are_distinct_from_the_formal_trend()
    test_long_only_unlevered_config_is_enforced()
    test_signal_executes_only_at_next_open_and_last_signal_remains_pending()
    test_hmm_confirmation_resets_after_exit_and_can_be_disabled()
    test_double_barrier_is_conservatively_stopped_and_gap_uses_open()
    test_invalid_open_fails_instead_of_using_impossible_intraday_fill()
    test_cash_yield_uses_prior_observation_and_calendar_days()
    test_hard_drawdown_cooldown_recovers_without_leverage_or_shorting()
    test_soft_drawdown_reduces_existing_position_at_next_open()
    test_soft_drawdown_scales_pending_target()
    test_public_state_overlay_uses_strict_execution_ledger()
    test_forward_ledger_is_append_only()
    print("strategy engine tests passed")
