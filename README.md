# 黄金交易研究网站

这是一个面向 Vercel 部署的中文黄金交易研究看板，展示：

- 黄金价格走势
- HMM 市场状态
- XGBoost 未来 30 日上涨概率
- 今日交易指南
- Kelly/ATR 风控参数
- 样本外回测摘要

量化研究脚本、本地模型、原始数据和 CSV 日志保存在本地目录，并通过 `.gitignore` 排除，不会上传到 GitHub。

## 本地运行

```bash
npm install
npm run dev
```

## 构建

```bash
npm run build
```

