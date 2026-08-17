import type { Metadata } from "next";
import type { ReactNode } from "react";
import { Instrument_Serif, Inter, IBM_Plex_Mono } from "next/font/google";
import { SiteHeader } from "@/components/shell/SiteHeader";
import "./globals.css";

const instrumentSerif = Instrument_Serif({
  variable: "--font-instrument-serif",
  subsets: ["latin"],
  weight: "400",
  style: ["normal", "italic"],
});

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
});

const plexMono = IBM_Plex_Mono({
  variable: "--font-plex-mono",
  subsets: ["latin"],
  weight: ["400", "500"],
});

export const metadata: Metadata = {
  title: {
    default: "Explain The Self",
    template: "%s — Explain The Self",
  },
  description:
    "A microscope for artificial intelligence. Explore what a model sees, what it represents, what it predicts — and what changes its behavior.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html
      lang="en"
      className={`${instrumentSerif.variable} ${inter.variable} ${plexMono.variable}`}
    >
      <body className="min-h-full flex flex-col">
        <SiteHeader />
        <main className="flex-1">{children}</main>
      </body>
    </html>
  );
}
