import latest from "../public/data/gold_research_latest.json";
import priceSeries from "../public/data/gold_price_series.json";
import backtestSeries from "../public/data/gold_backtest.json";

type Point = {
  date: string;
  close: number;
  sma_5?: number | null;
  sma_20?: number | null;
  sma_60?: number | null;
  sma_120?: number | null;
  state?: string;
  stateCode?: string;
  pUp30d?: number;
  pUpHorizon?: number;
  position?: number;
  guide?: string;
  atrStop?: number | null;
};

type BacktestPoint = {
  date: string;
  equity: number;
  benchmark_equity: number;
  drawdown: number;
  position: number;
};

const prices = priceSeries as Point[];
const backtest = backtestSeries as BacktestPoint[];

function pct(value: number, digits = 1) {
  return `${(value * 100).toFixed(digits)}%`;
}

function num(value: number, digits = 2) {
  return new Intl.NumberFormat("zh-CN", {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  }).format(value);
}

function linePath(values: number[], width: number, height: number, padding = 12, domain?: [number, number]) {
  const clean = values.filter((value) => Number.isFinite(value));
  const min = domain ? domain[0] : Math.min(...clean);
  const max = domain ? domain[1] : Math.max(...clean);
  const spread = max - min || 1;
  return values
    .map((value, index) => {
      const x = padding + (index / Math.max(values.length - 1, 1)) * (width - padding * 2);
      const y = height - padding - ((value - min) / spread) * (height - padding * 2);
      return `${index === 0 ? "M" : "L"}${x.toFixed(2)},${y.toFixed(2)}`;
    })
    .join(" ");
}

function areaPath(values: number[], width: number, height: number, padding = 12, domain?: [number, number]) {
  const path = linePath(values, width, height, padding, domain);
  return `${path} L${width - padding},${height - padding} L${padding},${height - padding} Z`;
}

function stateClass(stateCode?: string) {
  if (stateCode === "s1") return "stateBull";
  if (stateCode === "s2") return "stateBear";
  if (stateCode === "s4") return "statePanic";
  return "stateRange";
}

function actionClass(action: string) {
  if (action.includes("买入")) return "buy";
  if (action.includes("卖出")) return "sell";
  return "watch";
}

function Gauge({
  label,
  value,
  threshold,
  tone,
}: {
  label: string;
  value: number;
  threshold?: number;
  tone: "buy" | "sell" | "watch";
}) {
  const score = Math.max(0, Math.min(100, value));
  const cx = 86;
  const cy = 88;
  const radius = 56;
  const angle = Math.PI - (score / 100) * Math.PI;
  const pointerX = cx + Math.cos(angle) * radius;
  const pointerY = cy - Math.sin(angle) * radius;
  const thresholdPct = threshold == null ? null : Math.max(0, Math.min(100, threshold));
  const thresholdAngle = thresholdPct == null ? null : Math.PI - (thresholdPct / 100) * Math.PI;
  const thresholdX1 = thresholdAngle == null ? null : cx + Math.cos(thresholdAngle) * 48;
  const thresholdY1 = thresholdAngle == null ? null : cy - Math.sin(thresholdAngle) * 48;
  const thresholdX2 = thresholdAngle == null ? null : cx + Math.cos(thresholdAngle) * 68;
  const thresholdY2 = thresholdAngle == null ? null : cy - Math.sin(thresholdAngle) * 68;
  const gradientId = `gauge-${Array.from(label)
    .map((char) => char.charCodeAt(0).toString(16))
    .join("")}`;

  return (
    <div className={`gauge gauge-${tone}`}>
      <p>{label}</p>
      <svg viewBox="0 0 172 118" role="img" aria-label={`${label} 仪表盘`}>
        <defs>
          <linearGradient id={gradientId} x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#FF4E5B" />
            <stop offset="56%" stopColor="#F0C040" />
            <stop offset="100%" stopColor="#3DDC84" />
          </linearGradient>
        </defs>
        <path className="gaugeTrack" d="M22 88 A64 64 0 0 1 150 88" pathLength="100" />
        <path
          className="gaugeFill"
          d="M22 88 A64 64 0 0 1 150 88"
          pathLength="100"
          style={{ stroke: `url(#${gradientId})`, strokeDasharray: `${score} 100` }}
        />
        {thresholdPct != null && thresholdX1 != null && thresholdY1 != null && thresholdX2 != null && thresholdY2 != null ? (
          <>
            <line className="gaugeThreshold" x1={thresholdX1} y1={thresholdY1} x2={thresholdX2} y2={thresholdY2} />
            <text className="gaugeThresholdText" x={Math.min(thresholdX2 + 2, 124)} y={thresholdY2 - 4}>
              &gt; {thresholdPct.toFixed(0)}%
            </text>
          </>
        ) : null}
        <line className="gaugePointer" x1={cx} y1={cy} x2={pointerX} y2={pointerY} />
        <circle className="gaugePivot" cx={cx} cy={cy} r="3.5" />
      </svg>
      <strong>{score.toFixed(1)}%</strong>
    </div>
  );
}

function PriceChart() {
  const recent = prices.slice(-360);
  const priceDomainValues = recent.flatMap((point) => [
    point.close,
    point.sma_5,
    point.sma_20,
    point.sma_60,
  ]).filter((value): value is number => typeof value === "number" && Number.isFinite(value));
  const domain: [number, number] = [Math.min(...priceDomainValues), Math.max(...priceDomainValues)];
  const closePath = linePath(
    recent.map((point) => point.close),
    960,
    360,
    24,
    domain,
  );
  const area = areaPath(
    recent.map((point) => point.close),
    960,
    360,
    24,
    domain,
  );
  const sma5Path = linePath(recent.map((point) => point.sma_5 as number), 960, 360, 24, domain);
  const sma20Path = linePath(recent.map((point) => point.sma_20 as number), 960, 360, 24, domain);
  const sma60Path = linePath(recent.map((point) => point.sma_60 as number), 960, 360, 24, domain);
  const min = Math.min(...recent.map((point) => point.close));
  const max = Math.max(...recent.map((point) => point.close));

  return (
    <section className="panel wide">
      <div className="sectionHead">
        <div>
          <p className="eyebrow">价格走势</p>
          <h2>黄金近 360 个交易日</h2>
        </div>
        <div className="chartScale">
          <span>高 {num(max, 0)}</span>
          <span>低 {num(min, 0)}</span>
        </div>
      </div>
      <svg className="priceChart" viewBox="0 0 960 360" role="img" aria-label="黄金价格走势图">
        <path d={area} className="chartArea" />
        <path d={closePath} className="chartLine" />
        <path d={sma5Path} className="maLine ma5Line" />
        <path d={sma20Path} className="maLine ma20Line" />
        <path d={sma60Path} className="maLine ma60Line" />
        {recent.map((point, index) => {
          if (index % 28 !== 0) return null;
          return (
            <line
              key={point.date}
              x1={24 + (index / Math.max(recent.length - 1, 1)) * (960 - 48)}
              x2={24 + (index / Math.max(recent.length - 1, 1)) * (960 - 48)}
              y1="24"
              y2="336"
              className="gridLine"
            />
          );
        })}
      </svg>
      <div className="legend">
        <span><i className="legendClose" />收盘价</span>
        <span><i className="legendMa5" />5 日均线</span>
        <span><i className="legendMa20" />20 日均线</span>
        <span><i className="legendMa60" />60 日均线</span>
        <span>数据截至 {latest.asOf}</span>
      </div>
    </section>
  );
}

function EquityChart() {
  const recent = backtest;
  const start = recent[0];
  const end = recent[recent.length - 1];
  const domainValues = recent.flatMap((point) => [point.equity, point.benchmark_equity]);
  const domain: [number, number] = [Math.min(...domainValues), Math.max(...domainValues)];
  const strategyPath = linePath(
    recent.map((point) => point.equity),
    720,
    260,
    22,
    domain,
  );
  const benchmarkPath = linePath(
    recent.map((point) => point.benchmark_equity),
    720,
    260,
    22,
    domain,
  );

  return (
    <section className="panel">
      <div className="sectionHead">
        <div>
          <p className="eyebrow">样本外回测</p>
          <h2>策略净值 vs 买入持有</h2>
        </div>
        <div className="backtestSummary" aria-label="样本外回测摘要">
          <span>{start.date} 至 {end.date}</span>
          <strong className="backtestStrategy" data-series="strategy">
            策略 {pct(latest.backtestMetrics.total_return, 1)}
          </strong>
          <strong className="backtestBenchmark" data-series="benchmark">
            买入持有 {pct(latest.backtestMetrics.benchmark_return, 1)}
          </strong>
        </div>
      </div>
      <svg className="equityChart" viewBox="0 0 720 260" role="img" aria-label="回测净值曲线">
        <path d={benchmarkPath} className="benchmarkLine" />
        <path d={strategyPath} className="equityLine" />
      </svg>
      <div className="legend">
        <span><i className="legendEquity" />策略</span>
        <span><i className="legendBench" />买入持有</span>
      </div>
    </section>
  );
}

function AblationPanel() {
  const rows = latest.ablation ?? [];
  if (!rows.length) return null;

  return (
    <section className="panel wide">
      <div className="sectionHead">
        <div>
          <p className="eyebrow">策略归因</p>
          <h2>扩展消融实验</h2>
        </div>
        <div className="chartScale">
          <span>用于判断收益来自趋势、HMM、XGBoost 还是 ATR</span>
        </div>
      </div>
      <div className="ablationTable" aria-label="策略消融实验">
        <div className="ablationRow ablationHead">
          <span>模块</span>
          <span>总收益</span>
          <span>Sharpe</span>
          <span>最大回撤</span>
          <span>活跃天数</span>
        </div>
        {rows.map((row) => (
          <div key={row.name} className="ablationRow">
            <strong>{row.label}</strong>
            <em className={row.total_return >= latest.backtestMetrics.benchmark_return ? "positive" : "negative"}>
              {pct(row.total_return, 1)}
            </em>
            <em>{num(row.sharpe, 2)}</em>
            <em className="negative">{pct(row.max_drawdown, 1)}</em>
            <em>{pct(row.active_day_ratio, 1)}</em>
          </div>
        ))}
      </div>
    </section>
  );
}

function StateTape() {
  const recent = prices.slice(-120);
  const start = recent[0];
  const end = recent[recent.length - 1];
  return (
    <section className="panel">
      <div className="sectionHead">
        <div>
          <p className="eyebrow">HMM 状态</p>
          <h2>近 120 日状态带</h2>
        </div>
      </div>
      <div className="stateTapeFrame">
        <div className="stateTape" aria-label="HMM 市场状态时间轴">
          {recent.map((point) => (
            <span
              key={point.date}
              className={`stateBlock ${stateClass(point.stateCode)}`}
              title={`${point.date} ${point.state}`}
            />
          ))}
        </div>
        <div className="stateTapeAxis" aria-label="状态带时间范围">
          <span>{start.date}</span>
          <span>{end.date}</span>
        </div>
      </div>
      <div className="stateLegend">
        <span><i className="stateBull" />牛市</span>
        <span><i className="stateBear" />熊市</span>
        <span><i className="stateRange" />震荡</span>
        <span><i className="statePanic" />恐慌</span>
      </div>
    </section>
  );
}

function MetricCard({
  label,
  value,
  detail,
}: {
  label: string;
  value: string;
  detail: string;
}) {
  return (
    <div className="metric">
      <p>{label}</p>
      <strong>{value}</strong>
      <span>{detail}</span>
    </div>
  );
}

export default function Home() {
  const guide = latest.guide;
  const latestPoint = prices[prices.length - 1];
  const previousPoint = prices[prices.length - 2];
  const oneDay = latestPoint.close / previousPoint.close - 1;
  const horizonDays = latest.predictionHorizonDays ?? 30;
  const modelProbability = latest.pProfitFirst ?? latest.pUpHorizon ?? latest.pUp30d;
  const rawAuc = latest.modelMetrics.raw_test_auc ?? latest.modelMetrics.test_auc;
  const signalTone = actionClass(guide);
  const modelScore = modelProbability * 100;
  const featureMax = Math.max(0.000001, ...latest.topFeatures.map((item) => item.importance));
  const modelContext = latest.isMetaEvent ? "当前候选交易" : "最近候选交易延续";
  const gateReason = latest.modelMetrics.xgboost_gate_reason;
  const modelGateText = latest.xgboostEnabled
    ? "XGBoost 已通过模型和策略验证闸门，可参与入场/退出判断。"
    : latest.modelMetrics.xgboost_model_gate_pass && gateReason === "model_gate_pass_strategy_uplift_below_threshold"
      ? "XGBoost 高分信号已通过模型闸门，但验证段硬过滤未带来策略增益，今日操作仍由 HMM/CUSUM/ATR 风控决定。"
      : latest.modelMetrics.xgboost_statistical_valid
      ? "XGBoost 排序验证已修复，但高阈值交易质量未达闸门，今日操作仍由 HMM/CUSUM/ATR 风控决定。"
      : "XGBoost 排序验证未通过，今日操作由 HMM/CUSUM/ATR 风控决定。";

  return (
    <main>
      <section className="hero">
        <nav className="topbar">
          <div className="brand">
            <span>Au</span>
            黄金交易研究站
          </div>
          <div className="navMeta">HMM + XGBoost · {latest.asOf}</div>
        </nav>

        <div className="heroGrid">
          <div className={`decision decision-${signalTone}`}>
            <p className="eyebrow">今日操作</p>
            <h1 className={signalTone}>{guide}</h1>
            <p className="decisionCopy">
              当前 HMM 状态为{latest.marketState}，
              {modelContext}的 XGBoost 止盈优先评分为 {pct(modelProbability, 1)}。
              {modelGateText}
            </p>
            <div className="decisionMeta">
              <span>入场阈值 {pct(latest.thresholds.buyAbove, 0)}</span>
              <span>{horizonDays} 日标签窗口</span>
              <span>建议仓位 {pct(latest.position, 1)}</span>
            </div>
            <div className="decisionGauges">
              <Gauge label="建议仓位" value={latest.position * 100} tone={latest.position > 0 ? "buy" : signalTone} />
              <Gauge
                label="XGBoost 评分"
                value={modelScore}
                threshold={latest.thresholds.buyAbove * 100}
                tone={modelProbability >= latest.thresholds.buyAbove ? "buy" : modelProbability <= latest.thresholds.sellBelow ? "sell" : "watch"}
              />
            </div>
          </div>

          <div className="snapshot">
            <MetricCard label="黄金价格" value={num(latest.price, 2)} detail={`${latest.asset} · 日变化 ${pct(oneDay, 2)}`} />
            <MetricCard label="ATR 止损线" value={latest.atrStop ? num(latest.atrStop, 2) : "无"} detail={`止盈 ${latest.risk.profit_atr_multiple.toFixed(0)} ATR · 止损 ${latest.risk.stop_atr_multiple.toFixed(0)} ATR`} />
            <MetricCard label="样本外 Sharpe" value={num(latest.backtestMetrics.sharpe, 2)} detail={`5bps 净收益 ${pct(latest.backtestMetrics.net_total_return_5bps, 1)}`} />
            <MetricCard label="Raw AUC" value={num(rawAuc, 2)} detail={`测试期交易动作 ${latest.backtestMetrics.test_trades}`} />
          </div>
        </div>
      </section>

      <section className="contentGrid">
        <PriceChart />
        <StateTape />
        <EquityChart />
        <AblationPanel />

        <section className="panel">
          <div className="sectionHead">
            <div>
              <p className="eyebrow">交易框架</p>
              <h2>信号规则</h2>
            </div>
          </div>
          <div className="rules">
            <p><strong>趋势事件</strong><span>当 HMM quality 趋势成立时，用 CUSUM 波动阈值触发候选交易事件，最小间隔 {latest.risk.meta_event_gap_days} 个交易日。</span></p>
            <p><strong>Meta P &gt; {pct(latest.thresholds.buyAbove, 0)}</strong><span>XGBoost 判断候选交易质量足够高时，以最高 {pct(latest.risk.max_position, 0)} 仓位入场。</span></p>
            <p><strong>退出规则</strong><span>买入后不设置强制持仓到期；退出由 {latest.risk.profit_atr_multiple.toFixed(0)} ATR 止盈、{latest.risk.stop_atr_multiple.toFixed(0)} ATR 止损、HMM 熊市/恐慌跌破 60 日均线连续确认 {latest.risk.hmm_exit_confirmation_days} 天，或新 CUSUM 事件下 XGBoost 分数跌破 {pct(latest.thresholds.sellBelow, 0)} 决定。</span></p>
          </div>
        </section>

        <section className="panel">
          <div className="sectionHead">
            <div>
              <p className="eyebrow">风险约束</p>
              <h2>仓位与止损</h2>
            </div>
          </div>
          <div className="riskList">
            <span>最大仓位 {pct(latest.risk.max_position, 0)}</span>
            <span>最大单笔风险 {pct(latest.risk.max_single_loss, 0)}</span>
            <span>最大杠杆 {latest.risk.max_leverage.toFixed(1)}x</span>
            <span>止盈 {latest.risk.profit_atr_multiple.toFixed(0)} ATR</span>
            <span>止损 {latest.risk.stop_atr_multiple.toFixed(0)} ATR</span>
            <span>训练标签窗口 {horizonDays} 天</span>
            <span>HMM 退出确认 {latest.risk.hmm_exit_confirmation_days} 天</span>
            <span>CUSUM {latest.risk.cusum_threshold_mult.toFixed(1)}x</span>
          </div>
        </section>

        <section className="panel">
          <div className="sectionHead">
            <div>
              <p className="eyebrow">特征贡献</p>
              <h2>XGBoost Top 10</h2>
            </div>
          </div>
          <div className="featureList">
            {latest.topFeatures.map((item) => (
              <div key={item.feature} className="featureRow">
                <span>{item.feature}</span>
                <div>
                  <i style={{ width: `${Math.max((item.importance / featureMax) * 100, 4)}%` }} />
                </div>
                <em>{item.importance.toFixed(4)}</em>
              </div>
            ))}
          </div>
        </section>

        <section className="panel wide disclaimer">
          <div className="sectionHead">
            <div>
              <p className="eyebrow">免责声明</p>
              <h2>仅用于研究复盘，不构成投资建议</h2>
            </div>
          </div>
          <div className="disclaimerGrid">
            <p>
              本页面展示的是历史数据驱动的量化研究结果，模型信号可能失效，
              不应被理解为对黄金、期货、ETF 或任何金融产品的买卖建议。
            </p>
            <p>
              黄金价格采用 {latest.asset}；VIX、实际利率和 ETF 资金流中存在 proxy 因子，
              回测结果受数据源、交易成本、滑点和模型设定影响。
            </p>
            <p>
              任何真实交易都需要结合账户风险承受能力、流动性、保证金规则和独立判断。
              使用者需自行承担投资风险。
            </p>
          </div>
        </section>
      </section>
    </main>
  );
}
