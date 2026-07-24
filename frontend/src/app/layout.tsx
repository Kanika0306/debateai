import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Debate-AI | Real-Time Fact Checker & Fallacy Detector",
  description: "Live dashboard for real-time debate analysis, claim extraction, fact verification, and fallacy detection.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
