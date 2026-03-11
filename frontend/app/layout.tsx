import type { ReactNode } from "react";

export const metadata = {
  title: "DreamNest",
  description:
    "Privacy-first reflective dream retrieval, running locally on your machine.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <head>
        <link
          href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,600;0,700;1,400&display=swap"
          rel="stylesheet"
        />
      </head>
      <body
        style={{
          margin: 0,
          minHeight: "100vh",
          fontFamily:
            '"Inter", system-ui, -apple-system, BlinkMacSystemFont, sans-serif',
          background:
            "radial-gradient(circle at center, #F3EDE5 0%, #e5decf 100%)",
          color: "#3E3A36",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          padding: "2rem",
          position: "relative",
        }}
      >
        {/* Paper grain texture overlay */}
        <div
          aria-hidden="true"
          style={{
            position: "fixed",
            inset: 0,
            backgroundImage:
              "url('https://www.transparenttextures.com/patterns/paper-fibers.png')",
            opacity: 0.12,
            mixBlendMode: "multiply",
            pointerEvents: "none",
            zIndex: 0,
          }}
        />
        <div
          style={{
            position: "relative",
            zIndex: 1,
            width: "100%",
            display: "flex",
            justifyContent: "center",
          }}
        >
          {children}
        </div>
      </body>
    </html>
  );
}
