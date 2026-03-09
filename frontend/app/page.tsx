"use client";

import { FormEvent, useState } from "react";

type Message = {
  role: "user" | "assistant";
  content: string;
};

export default function Page() {
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    const trimmed = question.trim();
    if (!trimmed || isLoading) return;

    setError(null);
    setIsLoading(true);

    // Add the user message optimistically
    setMessages((prev) => [...prev, { role: "user", content: trimmed }]);
    setQuestion("");

    try {
      const res = await fetch("http://localhost:8000/api/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ question: trimmed }),
      });

      if (!res.ok) {
        throw new Error(`Request failed with status ${res.status}`);
      }

      const data: { answer: string } = await res.json();

      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: data.answer },
      ]);
    } catch (err) {
      setError("Something went wrong talking to the DreamNest backend.");
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <main
      style={{
        width: "100%",
        maxWidth: "720px",
        backgroundColor: "rgba(15,23,42,0.9)",
        borderRadius: "1.5rem",
        padding: "1.75rem",
        boxShadow:
          "0 20px 60px rgba(15,23,42,0.9), 0 0 0 1px rgba(148,163,184,0.15)",
        border: "1px solid rgba(148,163,184,0.35)",
        backdropFilter: "blur(18px)",
      }}
    >
      <header style={{ marginBottom: "1.5rem" }}>
        <h1
          style={{
            fontSize: "1.75rem",
            fontWeight: 600,
            margin: 0,
            color: "#e5e7eb",
          }}
        >
          DreamNest
        </h1>
        <p
          style={{
            marginTop: "0.35rem",
            fontSize: "0.9rem",
            color: "#9ca3af",
          }}
        >
          Privacy-first reflective retrieval over your dream journal. Your
          dreams never leave your device.
        </p>
      </header>

      <section
        aria-label="Chat history"
        style={{
          display: "flex",
          flexDirection: "column",
          gap: "0.75rem",
          marginBottom: "1.25rem",
          maxHeight: "320px",
          overflowY: "auto",
          paddingRight: "0.25rem",
        }}
      >
        {messages.length === 0 && (
          <p
            style={{
              fontSize: "0.95rem",
              color: "#9ca3af",
            }}
          >
            Ask things like:
            “When have I dreamed about water before?”
          </p>
        )}

        {messages.map((m, idx) => (
          <div
            key={idx}
            style={{
              alignSelf: m.role === "user" ? "flex-end" : "flex-start",
              maxWidth: "80%",
              padding: "0.6rem 0.75rem",
              borderRadius:
                m.role === "user" ? "0.75rem 0.75rem 0.15rem 0.75rem" : "0.75rem 0.75rem 0.75rem 0.15rem",
              backgroundColor:
                m.role === "user" ? "#4f46e5" : "rgba(15,23,42,0.85)",
              color: m.role === "user" ? "#e5e7eb" : "#e5e7eb",
              fontSize: "0.95rem",
              whiteSpace: "pre-wrap",
            }}
          >
            {m.content}
          </div>
        ))}

        {isLoading && (
          <p
            style={{
              fontSize: "0.9rem",
              color: "#9ca3af",
            }}
          >
            Retrieving similar dreams…
          </p>
        )}
      </section>

      <form onSubmit={handleSubmit} style={{ display: "flex", gap: "0.6rem" }}>
        <input
          type="text"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Ask about your dreams..."
          style={{
            flex: 1,
            borderRadius: "999px",
            border: "1px solid rgba(148,163,184,0.6)",
            backgroundColor: "rgba(15,23,42,0.8)",
            padding: "0.6rem 0.9rem",
            color: "#e5e7eb",
            fontSize: "0.95rem",
            outline: "none",
          }}
        />
        <button
          type="submit"
          disabled={isLoading || !question.trim()}
          style={{
            borderRadius: "999px",
            border: "none",
            padding: "0.55rem 1rem",
            background:
              "linear-gradient(to right, #4f46e5, #7c3aed, #ec4899)",
            color: "#f9fafb",
            fontSize: "0.9rem",
            fontWeight: 500,
            cursor: isLoading || !question.trim() ? "not-allowed" : "pointer",
            opacity: isLoading || !question.trim() ? 0.5 : 1,
            transition: "transform 0.1s ease, box-shadow 0.1s ease",
          }}
        >
          {isLoading ? "Thinking…" : "Ask DreamNest"}
        </button>
      </form>

      {error && (
        <p
          style={{
            marginTop: "0.75rem",
            fontSize: "0.85rem",
            color: "#fca5a5",
          }}
        >
          {error}
        </p>
      )}
    </main>
  );
}

