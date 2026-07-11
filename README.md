# 黄金交易研究网站

这是一个中文黄金交易研究项目，包含两部分：

- `research/gold_research_pipeline.py`：趋势 + CUSUM + HMM 风控的黄金交易研究算法。
- Next.js 网站：展示黄金价格走势、HMM 市场状态、今日交易指南和样本外回测。

当前正式策略不依赖分类模型，全部交易均由可解释规则生成。

完整的策略合理性、XGBoost 贡献、回测缺陷修正和收益瓶颈记录在 `research/STRATEGY_AUDIT.md`。

## 当前策略摘要

- 标的：COMEX 迷你黄金连续合约 proxy，东方财富 `101.QO00Y`。
- HMM 状态：牛市、熊市、震荡、恐慌；状态概率使用前向过滤，并按约 252 个交易日 expanding walk-forward 重训，避免长期使用 2021 年的静态状态模型。
- 事件采样：120 日长期技术趋势 + CUSUM 波动阈值；HMM 主要用于状态监控和退出风控。
- 低点入场审计：单独测试价格低于 60/120 日均线、240 日新低、低点反转确认，以及与长期趋势的并集；这些模式均未同时改善冻结验证收益、Sharpe 和测试稳定性，因此正式策略继续要求 120 日趋势成立。
- XGBoost 结论：已完全退出日常流水线、依赖、网站数据和正式交易算法；历史失败原因保留在审计文档中。
- 买入：120 日长期趋势成立后，由 CUSUM 波动事件触发下一交易日开盘入场。
- 卖出：HMM 趋势破坏或 ATR 止盈/止损触发退出。
- 持仓：不设置强制持仓到期。
- 风控：最高 100% 仓位、1.0x 杠杆；普通趋势按 10% 计划止损风险预算缩放，价格高于 120 日均线且 120 日收益达到 12% 时使用 14% 强趋势预算；10 ATR 止盈，6 ATR 止损，HMM 熊市/恐慌跌破 60 日均线连续确认 20 天退出。
- 正式回测：`t` 日收盘信号、`t+1` 日开盘成交，ATR 止盈止损按盘中障碍价格结算；部分仓位使用现金与黄金持仓单位逐日盯市，不假设开盘免费再平衡；禁止同一交易日退出后立即重入，并使用标准日收益 Sharpe。
- 实盘模拟：额外输出 `t` 日收盘信号、`t+1` 日开盘成交、盘中 ATR 障碍、双边 8bps 交易成本和回撤降仓约束下的模拟结果。
- 数据修正：CPI 同比按下一月中旬滞后，COT 按报告日后 3 天滞后；实际利率优先使用 FRED `DFII10`，取不到时回退为 `US10Y - 滞后 CPI YoY`；Cboe VIX、CFTC managed money、CPI/核心 CPI/非农 surprise、GPR 和 FOMC 作为宏观/事件特征接入。
- 行情容灾：东方财富历史主序列不可用时，只追加 AkShare/Sina 的 GC、GLD、VIXY、标普500和美债收益率新日期；DXY 使用 UUP 日收益代理延伸，原历史区间不换源。

## 安装网站依赖

```bash
npm install
```

## 安装研究算法依赖

建议使用独立 Python 虚拟环境：

```bash
python3 -m venv .venv-research
source .venv-research/bin/activate
pip install -r research/requirements.txt
```

## 本地运行交易算法

运行研究流水线并刷新网站数据：

```bash
source .venv-research/bin/activate
python research/gold_research_pipeline.py --json
```

外部数据源不可用或需要严格复现实验时，可只使用本地缓存：

```bash
python research/gold_research_pipeline.py --offline --json
```

也可以直接使用项目脚本：

```bash
npm run update:data
```

脚本会输出：

- `public/data/gold_research_latest.json`
- `public/data/gold_price_series.json`
- `public/data/gold_backtest.json`
- `local_logs/gold_signals.csv`
- `local_logs/gold_ablation.csv`
- `local_logs/gold_live_execution.csv`
- `local_logs/gold_backtest_yearly.csv`
- `local_logs/gold_parameter_stability.csv`
- `local_logs/gold_entry_mode_comparison.csv`
- `local_logs/data_quality_report.json`

`public/data/*.json` 会被网站读取并提交到 GitHub；`local_logs/` 和 `data/raw/` 只保存在本地。
网站同时读取 `gold_research_latest.json` 中的 `ablation` 和 `liveExecutionMetrics` 字段，保证策略归因和实盘模拟结果随数据更新。

## 本地运行网站

```bash
npm run dev
```

默认访问：

```text
http://localhost:3000
```

## 更新网站数据

常规更新流程：

```bash
npm run update:site
git add research/gold_research_pipeline.py research/requirements.txt PLAN.md README.md app public/data .gitignore
git commit -m "Update gold trading strategy data"
git push origin main
```

Vercel 会使用仓库里的 Next.js 项目构建网站。研究脚本本身不会在 Vercel 上自动运行；需要先在本地运行研究流水线，再提交更新后的 `public/data/*.json`。

网站页面中的价格、仓位、HMM 状态、回测收益和 Sharpe 等数字都从 `public/data/*.json` 读取，不在 `app/page.tsx` 手动维护。`npm run update:site` 会先运行研究算法刷新 JSON，再执行数据一致性检查和网站构建。

如需测试外部行情源连通性：

```bash
npm run test:sources
```

测试结果会写入 `local_logs/data_source_probe.json`，用于判断 Eastmoney、Yahoo、Stooq、FRED、Cboe VIX、CFTC COT 和 GPR 等源在当前网络下是否可用。

## 宏观变量说明

当前已接入或尝试接入的高质量宏观变量：

- 实际利率：优先 FRED `DFII10` 10Y TIPS real yield；失败时使用 `US10Y - 滞后 CPI YoY` 代理。
- 美元指数：东方财富 `100.UDI`。
- 避险情绪：Cboe 官方 VIX 历史序列，VIXY ETF 仅作为备用 proxy。
- CFTC managed money：CFTC 官方 disaggregated futures-only COT 黄金 managed money 多头、空头、净头寸和净头寸/持仓量。
- 宏观 surprise：CPI MoM、核心 CPI MoM、非农就业 actual - forecast，按公布日进入模型。
- 地缘政治风险：Caldara-Iacoviello GPR 月度指数，按月末后 7 天滞后近似。
- ETF 资金流：GLD 量价方向签名成交额 proxy，不等同于官方 ETF 净申赎。
- 暂不可用：MOVE index、Fed funds futures implied rate、GLD 官方日度持仓、黄金 ETF 官方净流入，目前没有稳定免费 point-in-time 接口，流水线会在 `sources/dataQuality` 中显式标记。

## 构建验证

```bash
npm run verify
```

单独构建网站：

```bash
npm run build
```

如果本地 Next SWC 原生包损坏，Next.js 可能会回退到 WASM 绑定并打印警告；只要构建最终成功即可。

## 目录说明

```text
app/                         网站页面代码
public/data/                 网站展示用 JSON
research/gold_research_pipeline.py  最新正式交易策略
research/requirements.txt    研究算法 Python 依赖
PLAN.md                      策略设计与回测计划
local_logs/                  本地信号和数据质量日志，不提交
data/raw/                    原始数据缓存，不提交
```

## 免责声明

本项目仅用于量化研究和历史复盘，不构成投资建议。黄金、期货和 ETF 交易存在亏损风险，真实交易需结合账户风险承受能力、流动性、保证金规则和独立判断。
