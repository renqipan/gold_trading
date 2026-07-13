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

function assertStrictlyIncreasingDates(name, rows) {
  let previous = "";
  const seen = new Set();
  for (const row of rows) {
    if (typeof row.date !== "string" || !/^\d{4}-\d{2}-\d{2}$/.test(row.date)) {
      throw new Error(`${name} has an invalid date: ${row.date}`);
    }
    if (seen.has(row.date) || (previous && row.date <= previous)) {
      throw new Error(`${name} dates must be unique and strictly increasing: ${row.date}`);
    }
    seen.add(row.date);
    previous = row.date;
  }
}

const latest = readJson("public/data/gold_research_latest.json");
const prices = readJson("public/data/gold_price_series.json");
const backtest = readJson("public/data/gold_backtest.json");
const forwardLedger = readJson("public/data/gold_forward_ledger.json");
const shanghaiToday = new Date().toLocaleDateString("sv-SE", { timeZone: "Asia/Shanghai" });
const shanghaiHour = Number(
  new Intl.DateTimeFormat("en-US", {
    timeZone: "Asia/Shanghai",
    hour: "2-digit",
    hourCycle: "h23",
  }).format(new Date()),
);
const confirmedThroughDate = new Date(`${shanghaiToday}T00:00:00Z`);
confirmedThroughDate.setUTCDate(confirmedThroughDate.getUTCDate() - (shanghaiHour >= 8 ? 1 : 2));
const confirmedThrough = confirmedThroughDate.toISOString().slice(0, 10);

if (!prices.length) throw new Error("gold_price_series.json is empty");
if (!backtest.length) throw new Error("gold_backtest.json is empty");
if (prices.length < 360) throw new Error("gold_price_series.json must contain at least 360 rows");
assertStrictlyIncreasingDates("gold_price_series.json", prices);
assertStrictlyIncreasingDates("gold_backtest.json", backtest);
assertStrictlyIncreasingDates("gold_forward_ledger.json records", forwardLedger.records || []);
for (const row of prices.slice(-360)) {
  if (!Number.isFinite(row.sma_120)) {
    throw new Error(`120-day moving average is missing in the visible chart window: ${row.date}`);
  }
}
if (!latest.liveExecutionMetrics || !Number.isFinite(latest.liveExecutionMetrics.total_return)) {
  throw new Error("latest.liveExecutionMetrics is missing");
}
if (!latest.priceStatus || typeof latest.priceStatus.isFinal !== "boolean") {
  throw new Error("latest.priceStatus is missing");
}
assertEqual("latest.priceStatus.sessionDate", latest.priceStatus.sessionDate, latest.asOf);
assertEqual("latest.priceStatus.timezone", latest.priceStatus.timezone, "Asia/Shanghai");
assertEqual("latest.priceStatus.isFinal", latest.priceStatus.isFinal, latest.asOf <= confirmedThrough);
if (latest.asOf > shanghaiToday) {
  throw new Error(`latest.asOf cannot be in the future: ${latest.asOf}`);
}
if (!Number.isFinite(Date.parse(latest.priceStatus.observedAt))) {
  throw new Error("latest.priceStatus.observedAt is invalid");
}
for (const field of ["ablation", "entryModeComparison", "modelMetrics", "modelValidation"]) {
  if (field in latest) throw new Error(`retired research field must not be published: ${field}`);
}
if (typeof latest.isMetaEvent !== "boolean" || typeof latest.isAcceptedEvent !== "boolean") {
  throw new Error("latest meta-event flags are missing");
}
if (latest.risk.long_only !== true || latest.risk.max_position > 1 || latest.risk.max_leverage > 1) {
  throw new Error("formal strategy must remain long-only and unlevered");
}
if (!Array.isArray(latest.dataQuality?.staleWarnings)) {
  throw new Error("raw data staleness audit is missing");
}
if (latest.dataQuality?.passed !== true) {
  throw new Error("site release is blocked because the data-quality gate did not pass");
}
if (latest.dataQuality?.assetIdentity?.passed !== true) {
  throw new Error("site release is blocked because the gold asset identity is not verified");
}
assertEqual("latest.asset", latest.asset, "COMEX 迷你黄金连续合约 QO00Y");
const goldSource = String(latest.dataQuality.assetIdentity.source || "");
if (/Au99\.99|Shanghai Gold Exchange/i.test(goldSource)) {
  throw new Error(`site release is blocked because a different gold asset was substituted: ${goldSource}`);
}
if (latest.risk.cash_yield_enabled !== true || latest.risk.cash_yield_haircut_bps < 0) {
  throw new Error("cash-yield policy is missing or invalid");
}
if (latest.backtestRobustness?.sample_role !== "reviewed_development_history_not_independent_out_of_sample") {
  throw new Error("backtest sample role is missing or misleading");
}
const eightBpsStress = latest.backtestRobustness.cost_stress?.find((row) => row.cost_bps === 8);
if (!eightBpsStress) throw new Error("8bps cost stress is missing");
assertClose("8bps stress return", eightBpsStress.total_return, latest.liveExecutionMetrics.total_return);
if (forwardLedger.appendOnly !== true || !Array.isArray(forwardLedger.records)) {
  throw new Error("append-only forward ledger is missing or invalid");
}
assertEqual("forward strategy version", latest.forwardHoldoutMetrics.strategy_version, forwardLedger.strategyVersion);
assertEqual(
  "forward config fingerprint",
  latest.forwardHoldoutMetrics.config_fingerprint,
  forwardLedger.configFingerprint,
);
assertEqual("forward record count", latest.forwardHoldoutMetrics.days, forwardLedger.records.length);

const lastPrice = prices.at(-1);
const previousPrice = prices.at(-2);
const lastBacktest = backtest.at(-1);

assertEqual("latest.asOf", latest.asOf, lastPrice.date);
assertClose("latest.price", latest.price, lastPrice.close);
assertClose("latest.position", latest.position, lastPrice.position);
assertClose("latest.recommendedPosition", latest.recommendedPosition, lastPrice.recommendedPosition);
assertEqual("latest.pendingEntry", latest.pendingEntry, lastPrice.pendingEntry);
assertEqual("latest.pendingExit", latest.pendingExit, lastPrice.pendingExit);
assertEqual("latest.riskHalted", latest.riskHalted, lastPrice.riskHalted);

if (previousPrice) {
  assertClose("latest.dailyChange", latest.dailyChange, lastPrice.close / previousPrice.close - 1);
}

assertClose("8bps backtest total_return", latest.liveExecutionMetrics.total_return, lastBacktest.equity - 1);
assertClose("backtest benchmark_return", latest.backtestMetrics.benchmark_return, lastBacktest.benchmark_equity - 1);
assertClose("live benchmark_return", latest.liveExecutionMetrics.benchmark_return, lastBacktest.benchmark_equity - 1);
console.log(
  `site data ok: asOf=${latest.asOf}, price=${latest.price}, strategy8bps=${latest.liveExecutionMetrics.total_return}, benchmark=${latest.backtestMetrics.benchmark_return}`,
);
