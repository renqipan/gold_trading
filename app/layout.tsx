import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "黄金交易研究站",
  description: "基于 HMM 市场状态与 XGBoost 概率模型的黄金交易研究看板。",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="zh-CN">
      <body>{children}</body>
    </html>
  );
}

