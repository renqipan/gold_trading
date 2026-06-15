# 黄金交易研究网站

这是一个中文黄金交易研究项目，包含两部分：

- `research/gold_research_pipeline.py`：HMM + XGBoost 的黄金交易研究算法。
- Next.js 网站：展示黄金价格走势、HMM 市场状态、XGBoost meta-label 分数、今日交易指南和样本外回测。

当前策略使用 triple-barrier/meta-labeling。模型预测的不是固定 30 日后涨跌，而是候选趋势交易是否会先触发止盈，而不是先触发止损。

## 当前策略摘要

- 标的：COMEX 迷你黄金连续合约 proxy，东方财富 `101.QO00Y`。
- HMM 状态：牛市、熊市、震荡、恐慌；状态概率使用前向过滤，显式利用 HMM 转移矩阵。
- 事件采样：HMM quality 趋势过滤 + CUSUM 波动阈值。
- XGBoost 目标：`P(profit first)`。
- 模型闸门：只有验证段 raw AUC >= 0.52 时，正式交易才使用 XGBoost 入场/退出信号。
- 买入：模型闸门通过时要求 `P(profit first) > 60%`；闸门未通过时回退为 HMM + CUSUM + ATR。
- 卖出：模型闸门通过时，新 CUSUM 事件下 `P(profit first) < 36%` 可触发退出；无论模型是否启用，HMM 趋势破坏或 ATR 止盈/止损都可触发退出。
- 持仓：不设置强制持仓到期。60 日只用于训练标签窗口和防止标签泄漏。
- 风控：最高 100% 仓位、1.0x 杠杆、10 ATR 止盈、4 ATR 止损。

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
- `local_logs/data_quality_report.json`

`public/data/*.json` 会被网站读取并提交到 GitHub；`local_logs/` 和 `data/raw/` 只保存在本地。
网站同时读取 `gold_research_latest.json` 中的 `ablation` 字段，展示买入持有、HMM 趋势过滤、HMM + CUSUM、XGBoost 和 ATR 风控的 A-E 消融实验。

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

网站页面中的价格、概率、仓位、阈值、HMM 状态、回测收益、Sharpe、特征重要性等数字都从 `public/data/*.json` 读取，不在 `app/page.tsx` 手动维护。`npm run update:site` 会先运行研究算法刷新 JSON，再执行数据一致性检查和网站构建。

如需测试外部行情源连通性：

```bash
npm run test:sources
```

测试结果会写入 `local_logs/data_source_probe.json`，用于判断 Eastmoney、Yahoo、Stooq、FRED 等源在当前网络下是否可用。

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
