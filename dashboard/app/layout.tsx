import type { Metadata } from "next";
import { JetBrains_Mono } from "next/font/google";
import { cn } from "@/lib/utils";
import "./globals.css";

const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-mono",
});

export const metadata: Metadata = {
  title: "Nomos",
  description: "Regulation-to-code intelligence",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={cn("theme", jetbrainsMono.variable)}>
      <body className={cn(jetbrainsMono.className, "min-h-screen")}>{children}</body>
    </html>
  );
}
