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
    point.sma_120,
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
  const sma120Path = linePath(recent.map((point) => point.sma_120 as number), 960, 360, 24, domain);
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
        <path d={sma120Path} className="maLine ma120Line" />
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
        <span><i className="legendMa120" />120 日均线</span>
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
          <p className="eyebrow">开发历史回测 · 8bps 严格执行</p>
          <h2>成本后策略净值 vs 买入持有</h2>
        </div>
        <div className="backtestSummary" aria-label="历史测试回测摘要">
          <span>{start.date} 至 {end.date}</span>
          <strong className="backtestStrategy" data-series="strategy">
            策略 {pct(latest.liveExecutionMetrics.total_return, 1)}
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
        <span><i className="legendEquity" />策略（双边 8bps 成本）</span>
        <span><i className="legendBench" />买入持有</span>
        <span>该区间已用于研究，仅视为开发样本</span>
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
  const signalTone = actionClass(guide);
  const recommendedPosition = latest.recommendedPosition ?? latest.position;

  return (
    <main>
      <section className="hero">
        <nav className="topbar">
          <div className="brand">
            <span>Au</span>
            黄金交易研究站
          </div>
          <div className="navMeta">趋势 + HMM · {latest.asOf}</div>
        </nav>

        <div className="heroGrid">
          <div className={`decision decision-${signalTone}`}>
            <p className="eyebrow">今日操作</p>
            <h1 className={signalTone}>{guide}</h1>
            <p className="decisionCopy">
              当前 HMM 状态为{latest.marketState}。正式信号由 120 日长期趋势与 CUSUM 事件共同触发，
              HMM 只负责确认趋势破坏退出。真正前瞻观察自 {latest.forwardHoldoutMetrics.start} 起，
              当前累计 {latest.forwardHoldoutMetrics.days} 个交易日。
            </p>
            <div className="decisionMeta">
              <span>常态风险 {pct(latest.risk.normal_trend_risk_budget, 0)}</span>
              <span>强趋势风险 {pct(latest.risk.strong_trend_risk_budget, 0)}</span>
              <span>已成交仓位 {pct(latest.position, 1)}</span>
              <span>下一开盘目标 {pct(recommendedPosition, 1)}</span>
            </div>
            <div className="decisionGauges">
              <Gauge
                label="目标仓位"
                value={recommendedPosition * 100}
                tone={recommendedPosition > 0 ? "buy" : signalTone}
              />
            </div>
          </div>

          <div className="snapshot">
            <MetricCard label="黄金价格" value={num(latest.price, 2)} detail={`${latest.asset} · 日变化 ${pct(oneDay, 2)}`} />
            <MetricCard label="ATR 止损线" value={latest.atrStop ? num(latest.atrStop, 2) : "无"} detail={`止盈 ${latest.risk.profit_atr_multiple.toFixed(0)} ATR · 止损 ${latest.risk.stop_atr_multiple.toFixed(0)} ATR`} />
            <MetricCard label="开发回测 Sharpe" value={num(latest.liveExecutionMetrics.sharpe, 2)} detail={`8bps 净收益 ${pct(latest.liveExecutionMetrics.total_return, 1)}`} />
            <MetricCard label="最大回撤" value={pct(latest.liveExecutionMetrics.max_drawdown, 1)} detail={`正式策略交易动作 ${latest.liveExecutionMetrics.test_trades}`} />
          </div>
        </div>
      </section>

      <section className="contentGrid">
        <PriceChart />
        <StateTape />
        <EquityChart />

        <section className="panel">
          <div className="sectionHead">
            <div>
              <p className="eyebrow">交易框架</p>
              <h2>信号规则</h2>
            </div>
          </div>
          <div className="rules">
            <p><strong>趋势事件</strong><span>当 120 日长期趋势成立时，用 CUSUM 波动阈值触发候选交易事件，最小间隔 {latest.risk.meta_event_gap_days} 个交易日；HMM 主要负责状态监控和退出风控。</span></p>
            <p><strong>正式趋势入场</strong><span>候选事件只由长期趋势 + CUSUM 决定。常态计划止损风险为 {pct(latest.risk.normal_trend_risk_budget, 0)}；价格高于 120 日均线且 120 日收益达到 {pct(latest.risk.strong_trend_ret_120_threshold, 0)} 时，提高至 {pct(latest.risk.strong_trend_risk_budget, 0)}。</span></p>
            <p><strong>退出规则</strong><span>买入后不设置强制持仓到期；退出仅由 {latest.risk.profit_atr_multiple.toFixed(0)} ATR 止盈、{latest.risk.stop_atr_multiple.toFixed(0)} ATR 止损，或 HMM 熊市/恐慌跌破 60 日均线连续确认 {latest.risk.hmm_exit_confirmation_days} 天决定。</span></p>
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
            <span>常态风险预算 {pct(latest.risk.normal_trend_risk_budget, 0)}</span>
            <span>强趋势风险预算 {pct(latest.risk.strong_trend_risk_budget, 0)}</span>
            <span>最大杠杆 {latest.risk.max_leverage.toFixed(1)}x</span>
            <span>现金利率折减 {latest.risk.cash_yield_haircut_bps.toFixed(0)}bps</span>
            <span>止盈 {latest.risk.profit_atr_multiple.toFixed(0)} ATR</span>
            <span>止损 {latest.risk.stop_atr_multiple.toFixed(0)} ATR</span>
            <span>HMM 退出确认 {latest.risk.hmm_exit_confirmation_days} 天</span>
            <span>CUSUM {latest.risk.cusum_threshold_mult.toFixed(1)}x</span>
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
              本页面展示的是历史数据驱动的量化研究结果，策略信号可能失效，
              不应被理解为对黄金、期货、ETF 或任何金融产品的买卖建议。
            </p>
            <p>
              黄金价格采用 {latest.asset}；VIX、实际利率和 ETF 资金流中存在 proxy 因子，
              未投资现金利率也只是三个月美债收益率的保守 proxy。回测结果受数据源、
              交易成本、滑点和参数设定影响。
            </p>
            <p>
              任何真实交易都需要结合账户风险承受能力、流动性、保证金规则和独立判断。
              2023 年后的历史已经被多轮研究查看，不是独立样本外证据；使用者需自行承担投资风险。
            </p>
          </div>
        </section>
      </section>
    </main>
  );
}
