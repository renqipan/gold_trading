# 黄金 HMM + XGBoost 交易研究计划

## 当前正式策略

本项目当前只保留一套正式研究策略：`research/gold_research_pipeline.py`。

策略不再预测固定 30 日后是否上涨，而是使用 triple-barrier/meta-labeling：

1. HMM 将黄金市场划分为牛市、熊市、震荡、恐慌四类状态，并按约 252 个交易日 expanding walk-forward 重训。
2. 120 日长期技术趋势确认是否具备做多候选环境；HMM 不再作为入场硬过滤，主要负责市场状态监控和退出。
3. CUSUM 波动阈值触发候选交易事件，减少固定频率采样带来的噪声。
4. Triple barrier 在训练标签窗口内判断候选交易是否先触发止盈，而不是先触发止损。
5. XGBoost 使用 34 个 `regime_core` 特征预测候选交易 `P(profit first)`。输入重点覆盖 120/252 日收益、长期均线距离、波动率、回撤、趋势年龄、黄金/美元比率和 HMM 状态，降低短周期振荡变量导致的趋势踏空。
6. XGBoost 训练器使用强正则 `max_depth=1` stump，不使用类别权重，避免不同 walk-forward 折的概率尺度被各自的正负样本比例扭曲。
7. 若验证段 raw AUC、买入阈值下的验证信号数、候选覆盖率、precision、recall 或 precision lift 任一不达标，则正式交易不使用 XGBoost 信号。
8. 即使模型闸门通过，XGBoost 硬过滤还必须在验证段相对 HMM/CUSUM/ATR fallback 带来收益和 Sharpe 增益；否则 XGBoost 只作为研究评分，不参与正式交易。
9. 交易执行只在候选事件和闸门满足规则时入场；买入后不设置强制持仓到期。

## 数据

主交易标的：

- 黄金价格：东方财富国际期货 `101.QO00Y`，COMEX 迷你黄金连续合约 proxy。

辅助因子：

- 美元指数：东方财富 `100.UDI`。
- 美债利率：东方财富 `171.US10Y`。
- 实际利率：优先使用 FRED `DFII10` 10Y TIPS real yield；当前网络取不到时回退为 US10Y - 滞后美国 CPI 同比。
- VIX：优先使用 Cboe 官方 VIX 历史序列；`107.VIXY` 仅作为备用风险 proxy。
- ETF 资金流 proxy：`107.GLD` 量价方向签名成交额，按 60 日成交额中位数归一化。
- COT 持仓：AkShare CFTC 黄金多空与净仓位，报告日后 3 天才进入模型。
- CFTC managed money：CFTC 官方 disaggregated futures-only COT 黄金 managed money 多头、空头、净头寸和净头寸/总持仓量，报告日后 3 天才进入模型。
- CPI：AkShare 美国 CPI 同比，按下一月中旬才进入模型，降低隐性未来函数。
- 宏观 surprise：AkShare 美国 CPI MoM、核心 CPI MoM、非农就业 actual - forecast，按公布日进入模型。
- GPR：Caldara-Iacoviello 月度地缘政治风险指数，月末后 7 天才进入模型。
- FOMC：美联储官网 FOMC 会议日历，生成事件日和事件 proximity 特征。
- 暂不可用或仅作记录：MOVE index、Fed funds futures implied rate、GLD 官方日度持仓、黄金 ETF 官方净流入。流水线不会用不稳定来源伪造这些字段。
- 技术指标：收益率、动量、波动率、均线、ATR、ADX、RSI、Donchian、MACD、布林带、随机指标、收益分布、跨市场短滞后收益和趋势质量。

原始数据缓存保存在 `data/raw/`，本地日志保存在 `local_logs/`，这些目录不提交到 GitHub。

## 训练与回测流程

1. 下载或读取缓存数据，并检查 OHLC、缺失值和数据日期。
2. 构建技术、宏观、资金流、COT 与 HMM 状态特征。
3. 冻结训练截止日为 2021-02-23、验证截止日为 2023-02-27，之后为历史测试段；2026-06-20 起作为新增 forward holdout，避免比例切分随数据增长迁移。
4. 对 CUSUM 候选事件生成 triple-barrier 标签。
5. 用 walk-forward 方式训练强正则 stump XGBoost，并按 60 日标签窗口做保守 purge；标签入场价格与真实执行一致，使用信号日后的下一交易日开盘价。
6. 禁止自动概率反转；在预先给定的 0.05-0.25 有限阈值网格中检查覆盖率、precision、recall 和 precision lift。XGBoost 定位为低分尾部否决器，而不是高阈值择时器。
7. 若模型闸门通过，再比较验证段 XGBoost 硬过滤策略与 HMM/CUSUM/ATR fallback 的收益和 Sharpe；没有策略增益则继续禁用 XGBoost 交易接管。
8. 输出总体、分年份、分 HMM 状态的 AUC、Brier、precision 和 recall 验证报告。
9. XGBoost 只用于过滤候选入场，不再用后续事件的低分强制退出已有趋势仓位。
10. 运行扩展消融实验，拆分买入持有、纯趋势、HMM、CUSUM、ATR、XGBoost 和正式闸门策略的贡献。
11. 正式历史回测与实盘模拟统一使用 `t` 日收盘信号、`t+1` 日开盘成交、盘中 ATR 障碍、双边交易成本和回撤降仓约束；收盘成交版本只保留为研究对照。
12. 输出网站 JSON 和本地 CSV 日志。

## 当前交易规则

- XGBoost 模型闸门：验证段 raw AUC >= 0.52，候选阈值下验证买入信号数 >= 15、候选覆盖率 >= 65%、precision >= 0.20、recall >= 0.05、precision lift >= 1.05。
- XGBoost 策略闸门：验证段 XGBoost 过滤策略相对趋势/CUSUM/ATR fallback 的收益增益 >= 0.2 个百分点，且 Sharpe 增益 >= 0。
- 买入阈值：从预先给定的有限阈值网格中仅用验证段选择；当前 7.5% 阈值通过两层闸门，保留约 71% 的候选事件，只否决模型评分最低的一段候选。
- 卖出：XGBoost 默认不负责退出，避免错配“新候选事件质量”与“已有持仓是否应该退出”。
- 模型闸门未通过时：不使用 XGBoost 入场/退出信号，回退为 120 日趋势 + CUSUM + ATR。
- 单次仓位：最高 100%，杠杆上限 1.0x；按 `max_single_loss / (6 ATR / 入场价)` 缩放，当前增长型计划单笔止损风险上限为组合净值的 12%。隔夜跳空可能使实际损失超过计划值。
- 训练标签窗口：60 个交易日，仅用于训练标签和 purge，不作为真实持仓到期日。
- 候选事件间隔：最小 5 个交易日，减少重复入场噪声。
- 止盈：10 ATR。
- 止损：6 ATR。
- HMM 退出：熊市/恐慌且价格跌破 60 日均线连续确认 20 天。
- 实际退出：ATR 止盈、ATR 止损或 HMM 趋势破坏确认。

## 最新回测摘要

最近一次优化回测日期：2026-07-11，数据截至最近交易日 2026-07-10。

- 正式次日开盘、盘中障碍历史测试毛收益：218.17%。
- 同期买入持有收益：125.13%。
- 5bps 成本后策略净收益：214.73%。
- 标准日收益 Sharpe：2.33；5bps 后 2.31；Sortino 4.21。
- 最大回撤：-10.88%，同期买入持有最大回撤为 -25.73%。
- 测试期交易动作：24 次，仓位受 12% 计划单笔止损风险上限和 100% 总仓位上限约束。
- 8bps 严格执行模拟收益：212.69%，标准日收益 Sharpe 2.30，最大回撤 -10.88%。
- XGBoost `regime_core` 34 特征模型的验证段 raw AUC 为 0.78，测试段 AUC 为 0.70；7.5% 阈值在验证段产生 39 个信号、保留约 71% 候选，precision 为 20.51%，相对 fallback 的验证收益增加约 0.27 个百分点、Sharpe 增加约 0.01，因此通过正式交易闸门。

正式策略选择风险约束后的 120 日趋势 + CUSUM + XGBoost 低分否决 + ATR，并用 walk-forward HMM 负责退出。严格执行口径下，5bps 净收益高于同期买入持有，同时最大回撤明显更低。XGBoost 的验证策略增益较小，且 2026-06-20 之后 forward holdout 只有 15 个交易日、期间没有新交易，因此仍需继续积累真正前瞻样本，不能将本次历史提升理解为未来收益保证。

邻近 ATR、HMM 确认期、CUSUM 阈值和事件间隔网格只产生很少的验证交易，不足以证明参数增益，因此保留 10 ATR 止盈、6 ATR 止损、20 日 HMM 确认和 5 日事件间隔，避免根据历史测试结果追参。

## 与 GitHub 基线的收益差异

远端 `origin/main` 与本地 `HEAD` 在本轮开始时均为 `c756158`。GitHub 页面曾公布截至 2026-06-19 的 5bps 净收益 205.66%，但该数字来自旧的收盘持仓回测；旧策略的 `max_single_loss=6%` 并未真正进入仓位计算，实际接近满仓，HMM 也是固定训练模型。把 GitHub 逻辑放入当前严格次日开盘、盘中障碍和跳空成交引擎，并把数据统一延长到 2026-07-10 后，5bps 净收益约为 203.06%、Sharpe 约 1.86、最大回撤约 -18.12%。

前一版当前策略的 5bps 净收益降至 89.66%，主要有三项原因：6% 计划止损风险上限把多数趋势仓位缩小；expanding walk-forward HMM quality 被同时用于入场，验证段没有增益却在历史测试段过滤掉了有效趋势；XGBoost 使用短周期特征和较高硬阈值，验证段失效后完全退出正式策略。本轮保留更严格的执行引擎和 walk-forward HMM，但将 HMM 移到退出端、风险预算提高到验证过的 12%，并让 XGBoost 使用长期状态输入做低分尾部否决。最终 5bps 净收益为 214.73%，不依赖恢复旧回测的乐观成交假设。

上述结果为研究回测，不构成投资建议。

## 输出文件

- `public/data/gold_research_latest.json`：网站摘要、今日信号、模型和回测指标。
- `public/data/gold_price_series.json`：网站价格、均线、状态和信号序列。
- `public/data/gold_backtest.json`：网站样本外净值曲线。
- `local_logs/gold_signals.csv`：本地信号日志，包含概率、状态、仓位、止损线和交易指南。
- `local_logs/gold_ablation.csv`：研究收盘口径的扩展消融实验结果，只用于相同口径下的模块归因；正式结果以严格次日开盘回测为准。
- `local_logs/gold_model_validation.csv`：总体、分年份、分 HMM 状态的模型验证结果。
- `local_logs/gold_live_execution.csv`：`t+1` 开盘成交实盘模拟日志。
- `local_logs/gold_backtest_yearly.csv`：正式执行口径的分年度收益、Sharpe、回撤、仓位和换手。
- `local_logs/gold_parameter_stability.csv`：fallback 参考参数及相邻止盈、止损、HMM 确认期和事件间隔的冻结验证/历史测试稳定性报告。
- `local_logs/data_quality_report.json`：本地数据质量报告。

网站展示数字应始终由上述 `public/data/*.json` 驱动。下次更新时运行 `npm run update:site`，不要在 `app/page.tsx` 中手动改价格、收益率、概率、仓位或阈值。
