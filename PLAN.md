# 黄金 HMM + XGBoost 交易研究计划

## 当前正式策略

本项目当前只保留一套正式研究策略：`research/gold_research_pipeline.py`。

策略不再预测固定 30 日后是否上涨，而是使用 triple-barrier/meta-labeling：

1. HMM 将黄金市场划分为牛市、熊市、震荡、恐慌四类状态。
2. HMM quality 趋势过滤器确认是否具备做多候选环境。
3. CUSUM 波动阈值触发候选交易事件，减少固定频率采样带来的噪声。
4. Triple barrier 在训练标签窗口内判断候选交易是否先触发止盈，而不是先触发止损。
5. XGBoost 预测候选交易 `P(profit first)`，即这笔趋势交易先触发止盈的概率/评分。
6. 若验证段 raw AUC 未达到 0.52，则正式交易不使用 XGBoost 信号，回退为 HMM + CUSUM + ATR。
7. 交易执行只在候选事件和模型闸门满足规则时入场；买入后不设置强制持仓到期。

## 数据

主交易标的：

- 黄金价格：东方财富国际期货 `101.QO00Y`，COMEX 迷你黄金连续合约 proxy。

辅助因子：

- 美元指数：东方财富 `100.UDI`。
- 美债利率：东方财富 `171.US10Y`。
- 实际利率 proxy：US10Y - 美国 CPI 同比。
- VIX proxy：`107.VIXY`。
- ETF 资金流 proxy：`107.GLD` 成交额按价格方向加权。
- COT 持仓：AkShare CFTC 黄金多空与净仓位。
- 技术指标：收益率、动量、波动率、均线、ATR、ADX、RSI、Donchian、趋势质量。

原始数据缓存保存在 `data/raw/`，本地日志保存在 `local_logs/`，这些目录不提交到 GitHub。

## 训练与回测流程

1. 下载或读取缓存数据，并检查 OHLC、缺失值和数据日期。
2. 构建技术、宏观、资金流、COT 与 HMM 状态特征。
3. 使用前 55% 数据训练 HMM，55%-72% 数据作为验证段，72% 之后作为样本外回测段。
4. 对 CUSUM 候选事件生成 triple-barrier 标签。
5. 用 walk-forward 方式训练 XGBoost，并做标签泄漏 purge。
6. 禁止自动概率反转；若验证段 raw AUC < 0.52，则 XGBoost 只作为研究观察，不参与正式交易。
7. 根据最新分数、HMM 状态、ATR 止盈止损和风险约束生成交易信号。
8. 运行 A-E 消融实验，拆分买入持有、HMM、CUSUM、XGBoost 和 ATR 的贡献。
9. 输出网站 JSON 和本地 CSV 日志。

## 当前交易规则

- XGBoost 启用条件：验证段 raw AUC >= 0.52。
- 买入阈值：模型闸门通过时，`P(profit first) > 60%`。
- 卖出阈值：模型闸门通过时，新 CUSUM 事件下 `P(profit first) < 36%`。
- 模型闸门未通过时：不使用 XGBoost 入场/退出信号，回退为 HMM + CUSUM + ATR。
- 单次仓位：最高 100%，杠杆上限 1.0x。
- 训练标签窗口：60 个交易日，仅用于训练标签和 purge，不作为真实持仓到期日。
- 止盈：10 ATR。
- 止损：4 ATR。
- HMM 退出：熊市/恐慌且价格跌破 60 日均线连续确认 10 天。
- 实际退出：ATR 止盈、ATR 止损、XGBoost 低分事件或 HMM 趋势破坏确认。

## 最新回测摘要

最近一次运行日期：2026-06-15，数据截至 2026-06-12。

- 样本外策略总收益：138.82%。
- 样本外买入持有收益：130.88%。
- 5bps 成本后策略净收益：137.16%。
- Sharpe：1.66。
- 最大回撤：-21.17%。
- 测试期交易动作：20 次。
- XGBoost 验证段 raw AUC：0.34，低于 0.52 闸门，本轮正式策略未启用 XGBoost 信号。

消融实验显示，本轮 XGBoost 直接参与入场会显著压低交易频率和收益；当前样本外收益主要来自 HMM/CUSUM 趋势过滤和 ATR 风控，而不是 XGBoost 预测能力。后续优化应继续改善标签、特征和验证稳定性，只有当验证段表现达标时才允许模型进入正式信号。

上述结果为研究回测，不构成投资建议。

## 输出文件

- `public/data/gold_research_latest.json`：网站摘要、今日信号、模型和回测指标。
- `public/data/gold_price_series.json`：网站价格、均线、状态和信号序列。
- `public/data/gold_backtest.json`：网站样本外净值曲线。
- `local_logs/gold_signals.csv`：本地信号日志，包含概率、状态、仓位、止损线和交易指南。
- `local_logs/gold_ablation.csv`：A-E 消融实验结果，用于策略归因。
- `local_logs/data_quality_report.json`：本地数据质量报告。

网站展示数字应始终由上述 `public/data/*.json` 驱动。下次更新时运行 `npm run update:site`，不要在 `app/page.tsx` 中手动改价格、收益率、概率、仓位或阈值。
