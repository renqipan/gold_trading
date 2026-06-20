import { readFileSync } from "node:fs";

function readJson(path) {
  return JSON.parse(readFileSync(path, "utf8"));
}

function assertClose(name, actual, expected, tolerance = 1e-9) {
  if (!Number.isFinite(actual) || !Number.isFinite(expected) || Math.abs(actual - expected) > tolerance) {
    throw new Error(`${name} mismatch: actual=${actual}, expected=${expected}`);
  }
}

function assertEqual(name, actual, expected) {
  if (actual !== expected) {
    throw new Error(`${name} mismatch: actual=${actual}, expected=${expected}`);
  }
}

const latest = readJson("public/data/gold_research_latest.json");
const prices = readJson("public/data/gold_price_series.json");
const backtest = readJson("public/data/gold_backtest.json");

if (!prices.length) throw new Error("gold_price_series.json is empty");
if (!backtest.length) throw new Error("gold_backtest.json is empty");
if (!Array.isArray(latest.ablation) || latest.ablation.length < 5) {
  throw new Error("latest.ablation is missing or too short");
}
if (!Array.isArray(latest.modelValidation) || latest.modelValidation.length < 2) {
  throw new Error("latest.modelValidation is missing or too short");
}
if (!latest.liveExecutionMetrics || !Number.isFinite(latest.liveExecutionMetrics.total_return)) {
  throw new Error("latest.liveExecutionMetrics is missing");
}
if (typeof latest.isMetaEvent !== "boolean" || typeof latest.isAcceptedEvent !== "boolean") {
  throw new Error("latest meta-event flags are missing");
}
if (typeof latest.xgboostEnabled !== "boolean") {
  throw new Error("latest.xgboostEnabled is missing");
}
if (typeof latest.modelMetrics.xgboost_statistical_valid !== "boolean") {
  throw new Error("latest.modelMetrics.xgboost_statistical_valid is missing");
}
if (typeof latest.modelMetrics.xgboost_model_gate_pass !== "boolean") {
  throw new Error("latest.modelMetrics.xgboost_model_gate_pass is missing");
}
if (typeof latest.modelMetrics.xgboost_strategy_gate_pass !== "boolean") {
  throw new Error("latest.modelMetrics.xgboost_strategy_gate_pass is missing");
}
if (latest.modelMetrics.feature_policy !== latest.risk.xgboost_feature_policy) {
  throw new Error("XGBoost feature policy mismatch between modelMetrics and risk");
}

const lastPrice = prices.at(-1);
const previousPrice = prices.at(-2);
const lastBacktest = backtest.at(-1);

assertEqual("latest.asOf", latest.asOf, lastPrice.date);
assertClose("latest.price", latest.price, lastPrice.close);

if (previousPrice) {
  assertClose("latest.dailyChange", latest.dailyChange, lastPrice.close / previousPrice.close - 1);
}

assertClose("backtest total_return", latest.backtestMetrics.total_return, lastBacktest.equity - 1);
assertClose("backtest benchmark_return", latest.backtestMetrics.benchmark_return, lastBacktest.benchmark_equity - 1);
assertClose("live benchmark_return", latest.liveExecutionMetrics.benchmark_return, lastBacktest.benchmark_equity - 1);
assertEqual("predictionHorizonDays", latest.predictionHorizonDays, latest.risk.prediction_horizon_days);
assertClose("buy threshold", latest.thresholds.buyAbove, latest.risk.up_threshold);
assertClose("sell threshold", latest.thresholds.sellBelow, latest.risk.down_threshold);
assertEqual("xgboostEnabled", latest.xgboostEnabled, latest.modelMetrics.xgboost_enabled);

console.log(
  `site data ok: asOf=${latest.asOf}, price=${latest.price}, strategy=${latest.backtestMetrics.total_return}, benchmark=${latest.backtestMetrics.benchmark_return}`,
);
