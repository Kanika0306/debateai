export default function Home() {
  return (
    <main
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        minHeight: "100vh",
        fontFamily: "system-ui, sans-serif",
        background: "#0a0a0a",
        color: "#ededed",
      }}
    >
      <h1 style={{ fontSize: "2.5rem", marginBottom: "0.5rem" }}>
        🎙️ Debate-AI
      </h1>
      <p style={{ color: "#888", fontSize: "1.1rem" }}>
        Live Fact Checker — Frontend coming soon
      </p>
      <p style={{ color: "#555", fontSize: "0.85rem", marginTop: "2rem" }}>
        Backend API:{" "}
        <a
          href="http://localhost:8000/docs"
          style={{ color: "#3b82f6" }}
          target="_blank"
          rel="noopener noreferrer"
        >
          http://localhost:8000/docs
        </a>
      </p>
    </main>
  );
}
