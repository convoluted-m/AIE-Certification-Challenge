import type { ReactNode } from "react";

export const metadata = {
  title: "DreamNest",
  description:
    "Privacy-first reflective dream retrieval, running locally on your machine.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body
        style={{
          margin: 0,
          minHeight: "100vh",
          fontFamily: "system-ui, -apple-system, BlinkMacSystemFont, sans-serif",
          background: "radial-gradient(circle at top, #111827, #020617)",
          color: "#e5e7eb",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          padding: "2rem",
        }}
      >
        {children}
      </body>
    </html>
  );
}

