export const theme = {
  colors: {
    background: "oklch(15% 0.02 240)",
    foreground: "oklch(98% 0.01 200)",
    muted: "oklch(26% 0.03 240)",
    mutedForeground: "oklch(72% 0.025 200)",
    border: "oklch(98% 0.01 200 / 0.22)",
    surface: "oklch(18% 0.022 238)",
    surfaceRaised: "oklch(22% 0.024 236)",
    card: "oklch(18% 0.022 238)",
    primary: "oklch(92% 0.03 200)",
    compliance: {
      emergent: "oklch(60% 0.15 20)",
      caution: "oklch(80% 0.12 70)",
      verified: "oklch(70% 0.12 140)",
    },
    chart: {
      1: "oklch(70% 0.12 140)",
      2: "oklch(80% 0.12 70)",
      3: "oklch(60% 0.15 20)",
      4: "oklch(55% 0.08 240)",
      5: "oklch(85% 0.06 200)",
    },
  },
  radii: { panel: "2px" },
  borders: { hairline: "0.5px" },
  noise: { opacity: 0.03 },
  typography: { monoFontVariable: "var(--font-mono)" },
} as const;

export type NomosTheme = typeof theme;
