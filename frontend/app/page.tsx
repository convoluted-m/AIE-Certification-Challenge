"use client";

import { FormEvent, useState } from "react";
import Image from "next/image";
import nestImg from "./nest.png";

type Message = {
  role: "user" | "assistant";
  content: string;
};

const SUGGESTIONS: string[] = [
  "Have I dreamt about a house before?",
  "When have I dreamed about water before?",
  "What recurring locations appear in my dreams?",
];

// ─── Design tokens ──────────────────────────────────────────────────────────
const light = {
  card: "#F3EDE5",
  cardBorder: "rgba(120,100,85,0.25)",
  cardShadow: "0 20px 60px rgba(80,60,45,0.18), 0 0 0 1px rgba(200,185,165,0.4), 0 0 40px 6px rgba(243,237,229,0.7)",
  heading: "#3E3A36",
  subtext: "#5E544E",
  footnote: "#7A6E68",
  sectionText: "#5E544E",
  userBubble: "#8a70a8",
  userBubbleText: "#faf8f2",
  asstBubble: "#F4EFE8",
  asstBubbleText: "#3E3A36",
  loading: "#7A6E68",
  input: "#F4EFE8",
  inputBorder: "rgba(120,100,85,0.45)",
  inputText: "#3E3A36",
  inputPlaceholder: "#9A8F89",
  chip: "#eee7d7",
  chipBorder: "rgba(120,100,85,0.4)",
  chipText: "#4B3F3A",
  resetBg: "#EDE6DC",
  resetBorder: "rgba(120,100,85,0.4)",
  resetText: "#4B3F3A",
  toggleBg: "#EDE6DC",
  toggleBorder: "rgba(120,100,85,0.4)",
  toggleText: "#4B3F3A",
  footer: "#9A8F89",
  error: "#9f1239",
};

const dark = {
  card: "#2a2420",
  cardBorder: "rgba(87,83,78,0.55)",
  cardShadow: "0 30px 80px rgba(0,0,0,0.7), 0 0 0 1px rgba(87,83,78,0.4)",
  heading: "#F4EFE8",
  subtext: "#DDD5CA",
  footnote: "#a8a29e",
  sectionText: "#DDD5CA",
  userBubble: "#7c6aa0",
  userBubbleText: "#faf8f2",
  asstBubble: "#38322c",
  asstBubbleText: "#F4EFE8",
  loading: "#a8a29e",
  input: "#38322c",
  inputBorder: "rgba(120,113,108,0.6)",
  inputText: "#F4EFE8",
  inputPlaceholder: "#a8a29e",
  chip: "#38322c",
  chipBorder: "rgba(120,113,108,0.5)",
  chipText: "#DDD5CA",
  resetBg: "#38322c",
  resetBorder: "rgba(120,113,108,0.5)",
  resetText: "#DDD5CA",
  toggleBg: "#38322c",
  toggleBorder: "rgba(120,113,108,0.5)",
  toggleText: "#DDD5CA",
  footer: "#a8a29e",
  error: "#fda4af",
};

const FONT_SERIF = '"Playfair Display", Georgia, "Times New Roman", serif';
const FONT_SANS =
  '"Inter", system-ui, -apple-system, BlinkMacSystemFont, sans-serif';

export default function Page() {
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isDark, setIsDark] = useState(false);

  const c = isDark ? dark : light;

  async function sendQuestion(text: string) {
    const trimmed = text.trim();
    if (!trimmed || isLoading) return;

    setError(null);
    setIsLoading(true);
    setMessages((prev) => [...prev, { role: "user", content: trimmed }]);
    setQuestion("");

    try {
      const res = await fetch("http://localhost:8000/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: trimmed }),
      });

      if (!res.ok) throw new Error(`Request failed with status ${res.status}`);

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

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    await sendQuestion(question);
  }

  function handleResetChat() {
    setMessages([]);
    setQuestion("");
    setError(null);
  }

  function handleSuggestionClick(text: string) {
    setQuestion(text);
    void sendQuestion(text);
  }

  return (
    <div
      style={{
        width: "100%",
        maxWidth: "880px",
        display: "flex",
        flexDirection: "column",
        gap: "1.25rem",
      }}
    >
      {/* ─── Page heading (outside card) ───────────────────────────────── */}
      <header
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "flex-start",
          gap: "1rem",
        }}
      >
        <div>
          <h1
            style={{
              fontSize: "2.25rem",
              fontWeight: 700,
              margin: 0,
              letterSpacing: "0.04em",
              color: c.heading,
              fontFamily: FONT_SERIF,
            }}
          >
            DreamNest
          </h1>
          <p
            style={{
              margin: "0.4rem 0 0",
              fontSize: "0.95rem",
              color: c.footnote,
              fontFamily: FONT_SANS,
            }}
          >
            Privacy-first reflective retrieval over your dream journal.
          </p>
          <p
            style={{
              margin: "0.15rem 0 0",
              fontSize: "0.95rem",
              color: c.footnote,
              fontFamily: FONT_SANS,
            }}
          >
            Your dreams never leave this space.
          </p>
        </div>

        {/* Dark / light mode toggle — top-right of page */}
        <button
          type="button"
          aria-label={isDark ? "Switch to light mode" : "Switch to dark mode"}
          aria-pressed={isDark}
          onClick={() => setIsDark((p) => !p)}
          style={{
            flexShrink: 0,
            borderRadius: "0.4rem",
            border: `1px solid ${c.toggleBorder}`,
            padding: "0.3rem 0.75rem",
            backgroundColor: c.toggleBg,
            color: c.toggleText,
            fontSize: "0.8rem",
            fontFamily: FONT_SANS,
            cursor: "pointer",
            fontWeight: 500,
          }}
        >
          {isDark ? "☀ Light" : "☾ Dark"}
        </button>
      </header>

      {/* ─── Card ───────────────────────────────────────────────────────── */}
      <main
        role="main"
        aria-label="DreamNest — dream archive chat"
        style={{
          width: "100%",
          minHeight: "70vh",
          backgroundColor: c.card,
          borderRadius: "0.25rem",
          padding: "2.25rem",
          boxShadow: c.cardShadow,
          border: "none",
          backdropFilter: "blur(20px)",
          display: "flex",
          flexDirection: "column",
          gap: "1.25rem",
        }}
      >
        {/* ─── Nest illustration ────────────────────────────────────────── */}
        <div style={{ display: "flex", justifyContent: "center" }}>
          <Image
            src={nestImg}
            alt="Nest illustration"
            width={160}
            height={84}
            style={{
              width: "160px",
              height: "auto",
              mixBlendMode: "multiply",
              opacity: 0.85,
              transform: "translateX(-20px)",
            }}
          />
        </div>

        {/* ─── Card controls row ────────────────────────────────────────── */}
        <div style={{ display: "flex", justifyContent: "flex-end" }}>
          <button
            type="button"
            aria-label="Restart the chat and clear all messages"
            onClick={handleResetChat}
            disabled={messages.length === 0}
            style={{
              borderRadius: "0.4rem",
              border: `1px solid ${c.resetBorder}`,
              padding: "0.3rem 0.75rem",
              backgroundColor: c.resetBg,
              color: c.resetText,
              fontSize: "0.8rem",
              fontFamily: FONT_SANS,
              cursor: messages.length === 0 ? "not-allowed" : "pointer",
              opacity: messages.length === 0 ? 0.4 : 1,
              fontWeight: 500,
            }}
          >
            ↺ Restart chat
          </button>
        </div>

        {/* ─── Chat history ─────────────────────────────────────────────── */}
        <section
          aria-label="Chat history"
          aria-live="polite"
          aria-atomic="false"
          style={{
            display: "flex",
            flexDirection: "column",
            gap: "0.75rem",
            flex: "1 1 auto",
            maxHeight: "60vh",
            minHeight: "260px",
            overflowY: "auto",
            paddingRight: "0.25rem",
          }}
        >
          {messages.length === 0 && (
            <div
              style={{
                fontSize: "0.95rem",
                color: c.sectionText,
                fontFamily: FONT_SANS,
                display: "flex",
                flexDirection: "column",
                gap: "0.6rem",
              }}
            >
              <span>Ask a question about your dreams:</span>
              <div
                role="group"
                aria-label="Suggested questions"
                style={{ display: "flex", flexWrap: "wrap", gap: "0.5rem" }}
              >
                {SUGGESTIONS.map((text) => (
                  <button
                    key={text}
                    type="button"
                    aria-label={`Ask: ${text}`}
                    onClick={() => handleSuggestionClick(text)}
                    style={{
                      borderRadius: "0.4rem",
                      border: `1px solid ${c.chipBorder}`,
                      padding: "0.35rem 0.85rem",
                      backgroundColor: c.chip,
                      fontSize: "0.95rem",
                      fontFamily: FONT_SANS,
                      color: c.chipText,
                      cursor: "pointer",
                      fontWeight: 500,
                    }}
                  >
                    {text}
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map((m, idx) => (
            <div
              key={idx}
              role={m.role === "assistant" ? "region" : undefined}
              aria-label={
                m.role === "assistant"
                  ? `DreamNest response ${idx + 1}`
                  : undefined
              }
              style={{
                alignSelf: m.role === "user" ? "flex-end" : "flex-start",
                maxWidth: "80%",
                padding: "0.65rem 0.85rem",
                borderRadius:
                  m.role === "user"
                    ? "0.75rem 0.75rem 0.2rem 0.75rem"
                    : "0.75rem 0.75rem 0.75rem 0.2rem",
                backgroundColor:
                  m.role === "user" ? c.userBubble : c.asstBubble,
                color: m.role === "user" ? c.userBubbleText : c.asstBubbleText,
                fontSize: "0.95rem",
                fontFamily: FONT_SANS,
                lineHeight: 1.6,
                whiteSpace: "pre-wrap",
                boxShadow:
                  m.role === "assistant"
                    ? "0 4px 14px rgba(0,0,0,0.12)"
                    : "0 4px 14px rgba(109,76,125,0.35)",
              }}
            >
              {m.content}
            </div>
          ))}

          {isLoading && (
            <p
              aria-live="polite"
              style={{
                fontSize: "0.9rem",
                color: c.loading,
                fontFamily: FONT_SANS,
                fontStyle: "italic",
              }}
            >
              Retrieving similar dreams…
            </p>
          )}
        </section>

        {/* ─── Input form ───────────────────────────────────────────────── */}
        <form
          onSubmit={handleSubmit}
          style={{ display: "flex", gap: "0.6rem", alignItems: "flex-end" }}
        >
          <label
            htmlFor="dream-question"
            style={{
              position: "absolute",
              width: 1,
              height: 1,
              overflow: "hidden",
              clip: "rect(0,0,0,0)",
              whiteSpace: "nowrap",
            }}
          >
            Type here...
          </label>
          <textarea
            id="dream-question"
            rows={3}
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Type your question here…"
            aria-label="Dream question input"
            style={{
              flex: 1,
              borderRadius: "0.75rem",
              border: `1px solid ${c.inputBorder}`,
              backgroundColor: c.input,
              padding: "0.7rem 0.9rem",
              color: c.inputText,
              fontSize: "0.95rem",
              fontFamily: FONT_SANS,
              outline: "none",
              resize: "vertical",
              minHeight: "3.2rem",
            }}
          />
          <button
            type="submit"
            aria-label={
              isLoading ? "Thinking, please wait" : "Send question to DreamNest"
            }
            disabled={isLoading || !question.trim()}
            style={{
              borderRadius: "0.5rem",
              border:
                isLoading || !question.trim()
                  ? "2px solid #ccc5b8"
                  : "2px solid #b8a898",
              padding: "0.7rem 1.4rem",
              background:
                isLoading || !question.trim()
                  ? isDark
                    ? "#3a3530"
                    : "#ede8df"
                  : "#e5decf",
              color:
                isLoading || !question.trim()
                  ? isDark
                    ? "#8aab87"
                    : "#9a9188"
                  : "#3E3A36",
              fontSize: "1rem",
              fontFamily: FONT_SERIF,
              fontWeight: 700,
              letterSpacing: "0.02em",
              cursor: isLoading || !question.trim() ? "not-allowed" : "pointer",
              boxShadow:
                isLoading || !question.trim()
                  ? "none"
                  : "0 4px 12px rgba(120,100,85,0.25)",
              transition: "background 0.15s ease, box-shadow 0.15s ease",
            }}
          >
            {isLoading ? "Thinking…" : "Ask DreamNest"}
          </button>
        </form>

        {/* ─── Error state ──────────────────────────────────────────────── */}
        {error && (
          <p
            role="alert"
            style={{
              margin: 0,
              fontSize: "0.85rem",
              fontFamily: FONT_SANS,
              color: c.error,
            }}
          >
            {error}
          </p>
        )}
      </main>

      {/* ─── Footer (outside card) ────────────────────────────────────────── */}
      <footer
        style={{
          paddingTop: "0.75rem",
          fontSize: "0.75rem",
          fontFamily: FONT_SANS,
          color: c.footer,
          textAlign: "center",
        }}
      >
        © convoluted-m 2026 — DreamNest Prototype
      </footer>
    </div>
  );
}
