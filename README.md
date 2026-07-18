# 黄金交易研究网站

这是一个中文黄金量化研究与网站项目。正式策略使用可解释的趋势、CUSUM、ATR 风控和 walk-forward HMM，只做多或空仓，不做空、不加杠杆。

项目目前只保留一套正式策略。XGBoost、旧收盘成交回测器以及未被采纳的候选策略实现均已从生产代码删除。

## 当前策略

- 标的：COMEX 黄金连续合约 proxy；历史主序列为东方财富 `101.QO00Y` 缓存，日常更新使用 AkShare/Sina `GC` 同资产族序列延伸。
- 入场环境：黄金高于 120 日均线，或 20/60/120 日均线形成多头排列。
- 事件触发：绝对 CUSUM，最小事件间隔 5 个交易日。
- 仓位：按 6 ATR 计划止损距离缩放；普通趋势风险预算 10%，强趋势 14%。
- 强趋势：价格高于 120 日均线且 120 日收益不低于 12%。
- 退出：10 ATR 止盈、6 ATR 止损，或 HMM 熊市/恐慌且跌破 60 日均线连续 20 天。
- 回撤控制：18% 软回撤减半，30% 硬回撤清仓并冷却 63 个交易日。
- 现金：使用上一可得日 FRED DGS3MO，扣减 50bps 后计息。
- 约束：现金、黄金单位和名义仓位均非负；最大仓位与杠杆均为 1.0x。

HMM 固定使用四个黄金特征：

```text
ret_20
vol_20
sma_gap_60
drawdown_120
```

状态按 expanding walk-forward 方式约每 252 个交易日重训，不允许随着辅助数据变多而自动改变特征维度。

## 正式数据源

每日生产流水线只保留两个实际参与策略或回测的输入：

- 黄金 OHLC：`101.QO00Y` 历史缓存 + AkShare/Sina COMEX `GC` 日常延伸。
- 现金收益率：FRED `DGS3MO`；缓存超过 7 天才请求刷新，超过 10 天则发布失败。

早期研究使用的 DXY、US10Y、GLD、VIX/VIXY、CPI、COT、CFTC managed money、非农、GPR、FOMC、MOVE 和 Fed funds futures 均未进入当前四特征 HMM 或正式信号，已从每日生产抓取、质量报告和源探测中删除。

## 回测口径

- 冻结训练截止日：2021-02-23。
- 冻结验证截止日：2023-02-27。
- 2023-02-28 之后属于已查看的开发历史，不是独立样本外测试。
- 真正前瞻记录从 2026-07-13 开始。
- 信号在 `t` 日收盘后生成，交易在 `t+1` 日开盘执行。
- ATR 障碍按盘中价格处理，隔夜跳空穿越障碍按开盘价处理。
- 逐日记录现金和黄金单位，不假设免费再平衡。
- 网站主曲线在买入和卖出时各计 8bps 成本。

数据截至 2026-07-10：

- 毛收益：198.65%。
- 5bps 净收益：195.56%。
- 8bps 网站净收益：193.72%。
- 8bps Sharpe：1.71。
- 最大回撤：-10.88%。
- 11 次买入、11 次卖出。
- 同期买入持有收益：125.13%，最大回撤 -25.73%。

完整审计见 [research/STRATEGY_AUDIT.md](research/STRATEGY_AUDIT.md)，设计与发布计划见 [PLAN.md](PLAN.md)。

## 网站

Next.js 页面展示：

- 最新黄金价格、市场状态和交易指南。
- 最近 360 个交易日收盘价。
- 5、20、60、120 日移动平均线。
- HMM 状态带。
- 8bps 严格执行净值与买入持有对照。
- 数据质量、风险和前瞻样本说明。

页面数字均来自 `public/data/*.json`，不在 `app/page.tsx` 手工维护。

## 安装

网站依赖：

```bash
npm ci
```

Python 研究环境：

```bash
python3 -m venv .venv-research
source .venv-research/bin/activate
python -m pip install -r research/requirements.txt
```

建议使用 Node.js 20.9 或更高版本及 Python 3.11 或更高版本。

## 本地运行

刷新数据：

```bash
npm run update:data
```

严格使用本地缓存复现：

```bash
npm run update:data -- --offline
```

运行网站：

```bash
npm run dev
```

完整验证：

```bash
npm run verify
```

`verify` 包括策略测试、网站数据一致性检查、TypeScript 检查和生产构建。

外部数据源探测：

```bash
npm run test:sources
```

## 每日一键更新网站

确保位于干净的 `main` 分支，然后运行：

```bash
./scripts/update_website_daily.sh
```

也可以运行：

```bash
npm run update:daily
```

脚本会自动：

1. 检查 Git、Node、Python 环境及研究依赖。
2. 防止未完成 Git 操作、脏工作树和并发更新。
3. 快进同步 `origin/main`。
4. 获取最新行情并执行完整验证。
5. 检查资产身份、已确认历史不可改写和前瞻账本 append-only。
6. 只允许以下四个文件被自动提交：
   - `public/data/gold_research_latest.json`
   - `public/data/gold_price_series.json`
   - `public/data/gold_backtest.json`
   - `public/data/gold_forward_ledger.json`
7. 当天未收盘时发布程序运行时取得的最新黄金价格和模型盘中快照；同一天可随价格变化安全更新。
8. 没有新的价格或模型变化时恢复时间戳变化，不创建空提交。
9. 提交前再次检查远端，再直接推送 `main`。

正常输出固定为四个简短步骤；详细命令输出仅在某一步失败时显示：

```text
[1/4] 同步 GitHub main ... 完成
[2/4] 更新行情与模型 ... 完成
[3/4] 验证策略与网站 ... 完成
[4/4] 发布网站 ... 完成（YYYY-MM-DD）
```

最新交易日尚未结束时，页面会分别标注“行情交易日”和完整的北京时间模型生成时间，避免把 COMEX 交易日与程序运行日期混为一谈。当天价格会参与最新趋势、HMM 状态、仓位建议和回测快照；前瞻账本仍只在交易日结束后纳入该日，避免把不断变化的盘中收益固化为历史记录。已发布的前瞻账本快照保持不变；FRED 迟到发布或修订历史利率时，只冻结受影响的现金会计字段，策略版本、操作信号、目标仓位和黄金基准收益仍接受严格一致性校验。

脚本依赖本机已经配置好的 GitHub SSH 凭据，不会保存 PAT、私钥或部署令牌。GitHub 推送成功后，托管平台仍需按其自身状态确认部署完成。

## GitHub Actions 自动更新

`.github/workflows/daily-update.yml` 会在北京时间每天 `02:17`、`08:17`、`14:17`、`20:17`
运行，也可以在 GitHub Actions 页面手动触发。GitHub cron 使用 UTC，因此工作流内对应
`17 0,6,12,18 * * *`。其中 `08:17` 位于 COMEX 日线 `08:00` 收盘确认点之后，可及时把
上一交易日从“盘中快照”更新为“已收盘”。

云端任务会创建隔离的 Node.js/Python 环境，从 `research/seeds/` 恢复确定性的黄金与现金利率
历史种子，然后调用同一个 `scripts/update_website_daily.sh`。工作流拥有最小的
`contents: write` 权限，并使用并发锁避免两个更新任务同时推送 `main`。正常情况下仍只会自动
提交四个 `public/data/*.json`；若行情和模型语义均无变化，则不会创建空提交。

## 数据安全

- 所有 HTTPS 请求验证 TLS 证书。
- 主标的不可静默回退到上金所 Au99.99 等不同资产。
- 数据质量门要求 COMEX 资产身份、OHLC、现金收益率和时效通过。
- 公开 JSON 使用临时文件完成序列化后原子替换。
- 正式配置指纹同时包含正式函数源码 SHA-256。
- 已有前瞻记录后，策略逻辑或配置变化必须声明新版本。

## 目录

```text
app/                              网站页面和样式
public/data/                      提交到 GitHub 的网站 JSON
research/gold_research_pipeline.py  正式策略、回测和数据流水线
research/requirements.txt         Python 依赖
research/STRATEGY_AUDIT.md        当前项目审计
scripts/check-site-data.mjs       网站数据发布门
scripts/test_strategy_engine.py   正式执行器测试
scripts/test_data_sources.py      外部数据源探测
scripts/update_website_daily.sh   每日更新、验证、提交和推送
PLAN.md                           策略与发布计划
```

`data/raw/` 与 `local_logs/` 仅保存在本地，不提交到 GitHub。

## 限制

- 连续价格 proxy 不等同于真实逐合约 GC/MGC 账本。
- 尚未计入真实换月价差、合约乘数、整数张数和经纪商保证金。
- 完整运行缓存不提交；GitHub Actions 使用仓库内压缩的确定性生产种子启动，再联网追加新数据。
- Python 依赖尚未使用带哈希锁文件。
- GitHub Actions 负责定时更新数据，但尚未轮询 Vercel 等托管平台的最终部署结果。

本项目仅用于量化研究和历史复盘，不构成投资建议。
