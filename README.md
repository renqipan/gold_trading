# 黄金交易研究网站

这是一个中文黄金交易研究项目，包含两部分：

- `research/gold_research_pipeline.py`：趋势 + CUSUM + HMM 风控的黄金交易研究算法。
- Next.js 网站：展示黄金价格走势、HMM 市场状态、今日交易指南和成本后的开发历史回测。

当前正式策略不依赖分类模型，全部交易均由可解释规则生成。

完整的策略合理性、XGBoost 贡献、回测缺陷修正和收益瓶颈记录在 `research/STRATEGY_AUDIT.md`。

## 当前策略摘要

- 标的：COMEX 迷你黄金连续合约 proxy，东方财富 `101.QO00Y`。
- 执行含义：当前回测表示连续黄金价格的 0%-100% 合成多头敞口，不等同于真实 GC/MGC 整数合约账本；尚未计入分月换仓价差、合约乘数和保证金。
- HMM 状态：牛市、熊市、震荡、恐慌；固定使用 `ret_20 / vol_20 / sma_gap_60 / drawdown_120` 四个黄金核心特征，状态概率使用前向过滤，并按约 252 个交易日 expanding walk-forward 重训。后续数据源变长不会再改变模型维度或截短训练历史。
- 事件采样：120 日长期技术趋势 + CUSUM 波动阈值；HMM 主要用于状态监控和退出风控。
- 低点入场审计：单独测试价格低于 60/120 日均线、240 日新低、低点反转确认，以及与长期趋势的并集；这些模式均未同时改善冻结验证收益、Sharpe 和测试稳定性，因此正式策略继续要求 120 日趋势成立。
- XGBoost 结论：已完全退出日常流水线、依赖、网站数据和正式交易算法；历史失败原因保留在审计文档中。
- 买入：120 日长期趋势成立后，由 CUSUM 波动事件触发下一交易日开盘入场。
- 卖出：HMM 趋势破坏或 ATR 止盈/止损触发退出。
- 持仓：不设置强制持仓到期。
- 风控：只做多或空仓，不做空、不借钱；最高 100% 黄金名义敞口、1.0x 杠杆硬上限。普通趋势按 10% 计划止损风险预算缩放，价格高于 120 日均线且 120 日收益达到 12% 时使用 14% 强趋势预算；10 ATR 止盈，6 ATR 止损，HMM 熊市/恐慌跌破 60 日均线连续确认 20 天退出。
- 回撤约束：风险净值较峰值回撤达到 18% 时，下一开盘把已有仓位降至原目标的一半；达到 30% 时清仓并进入 63 个交易日冷却，防止空仓状态永久锁死。
- 现金管理：未投资现金按上一可得日 FRED `DGS3MO` 三个月美债收益率逐日计息，并保守扣减 50bps。现金利息单独披露，不计作黄金择时 alpha；这要求真实账户具备对应的现金管理工具。
- 正式回测：`t` 日收盘信号、`t+1` 日开盘成交，ATR 止盈止损按盘中障碍价格结算；现金与黄金持仓单位逐日盯市，不假设开盘免费再平衡。网站仓位、待执行动作、止损和止盈与正式回测统一使用同一个执行账本。
- 实盘模拟：额外输出 `t` 日收盘信号、`t+1` 日开盘成交、盘中 ATR 障碍、双边 8bps 交易成本和回撤降仓约束下的模拟结果。
- 数据修正：CPI 同比按下一月中旬滞后，COT 按报告日后 3 天滞后；实际利率优先使用 FRED `DFII10`，取不到时回退为 `US10Y - 滞后 CPI YoY`。低频宏观、COT、GPR 和 GLD 资金流不再进入正式日频 HMM；质量报告会按原始观测日暴露数据陈旧，而不是被前向填充掩盖。
- 成交量修正：volume/amount 不再跨日填充；GLD 缺失 amount 时逐行回退到 `close × volume`，避免替代行情接入后沿用旧成交额。
- 时点对齐：低频水平数据先在联合日历上按最近已知值对齐，再裁到黄金交易日；FOMC 等事件映射到公布后的首个黄金交易日。黄金 open/high/low 禁止跨日填充，缺失开盘价会使严格回测直接失败。
- 行情容灾：东方财富历史主序列不可用时，只追加 AkShare/Sina 的 GC、GLD、VIXY、标普500和美债收益率新日期；DXY 使用 UUP 日收益代理延伸，原历史区间不换源。

## 最新结果口径

数据截至 2026-07-10。2023-02-28 之后的区间已经被多轮研究查看，只能称为开发样本；真正前瞻记录从 2026-07-13 开始，目前为 0 个交易日，并由 `gold_forward_ledger.json` 以配置指纹做 append-only 校验。以后修改正式策略或执行器时必须同时提升声明的策略/执行器版本。

- 毛收益 198.65%；5bps 成本后 195.56%；更保守的 8bps 网站主口径为 193.72%。
- 8bps Sharpe 1.71，最大回撤 -10.88%，11 次买入、11 次卖出。Sharpe 使用同一滞后现金收益率计算超额收益，而不是把现金利息当成无风险的策略 alpha。
- 不计现金收益时，5bps 净收益为 192.06%；保守现金收益贡献约 3.50 个百分点。
- 15bps 与 25bps 成本压力下净收益分别为 189.48% 和 183.53%。
- 冻结验证只有 3 次入场、2 次退出（仅 2 笔完整交易），不能据此继续细调参数。

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
- `public/data/gold_forward_ledger.json`
- `local_logs/gold_signals.csv`
- `local_logs/gold_ablation.csv`
- `local_logs/gold_live_execution.csv`
- `local_logs/gold_backtest_yearly.csv`
- `local_logs/gold_backtest_robustness.json`
- `local_logs/gold_parameter_stability.csv`
- `local_logs/gold_entry_mode_comparison.csv`
- `local_logs/gold_hmm_stability.csv`
- `local_logs/data_quality_report.json`

`public/data/*.json` 会被网站读取并提交到 GitHub；`local_logs/` 和 `data/raw/` 只保存在本地。扩展消融结果仅保留在 `local_logs/gold_ablation.csv`，不再写入网站 JSON，也不在网页展示。网站主净值、收益和 Sharpe 统一使用双边 8bps 成本账本。

候选策略的预注册多窗口检查可单独运行：

```bash
npm run research:candidates
```

该脚本严格排除已经查看过的 2023-02-28 之后开发区间，并且只有达到预设收益、Sharpe、回撤和换手门槛的候选才可能晋级。

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
- 现金收益率：FRED `DGS3MO` 三个月美债收益率，只使用上一可得日数值并扣减 50bps。
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
