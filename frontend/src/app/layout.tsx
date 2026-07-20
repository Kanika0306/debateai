export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <head>
        <title>Debate-AI | Live Fact Checker</title>
        <meta
          name="description"
          content="Real-time debate fact-checking, fallacy detection, and live dashboard."
        />
      </head>
      <body>{children}</body>
    </html>
  );
}
