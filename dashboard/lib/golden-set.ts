export type TrafficState = "verified" | "caution" | "emergent";

export type GoldenSetRecord = {
  id: string;
  citation: string;
  instrument: string;
  section_ref: string;
  text: string;
  repo_path: string;
  expected_probability: number;
  state: TrafficState;
  reasoning: string;
};

export type TraceabilityRow = {
  id: string;
  euArticle: string;
  instrument: string;
  annexRef: string;
  obligation: string;
  repoPath: string;
  posteriorMean: number;
  traffic: TrafficState;
};

export type PosteriorSeries = {
  id: string;
  label: string;
  mu: number;
  sigma: number;
  strokeVar: "--chart-1" | "--chart-2" | "--chart-3";
};

const STROKE_VARS: PosteriorSeries["strokeVar"][] = [
  "--chart-1",
  "--chart-2",
  "--chart-3",
];

export function recordsToTraceRows(records: GoldenSetRecord[]): TraceabilityRow[] {
  return records.map((r) => ({
    id: r.id,
    euArticle: r.citation,
    instrument: r.instrument,
    annexRef: r.section_ref,
    obligation: r.text,
    repoPath: r.repo_path,
    posteriorMean: r.expected_probability,
    traffic: r.state,
  }));
}

export function recordsToPosteriorSeries(records: GoldenSetRecord[]): PosteriorSeries[] {
  return records.map((r, i) => ({
    id: r.id,
    label: `${r.citation} · ${r.repo_path}`,
    mu: r.expected_probability,
    sigma: Math.max(0.018, 0.045 * (1 - r.expected_probability) + 0.022),
    strokeVar: STROKE_VARS[i % STROKE_VARS.length]!,
  }));
}

export function normalDensity(x: number, mu: number, sigma: number): number {
  if (sigma <= 0) return 0;
  const z = (x - mu) / sigma;
  return Math.exp(-0.5 * z * z) / (sigma * Math.sqrt(2 * Math.PI));
}

export type GoldenSetBundle = {
  sourcePath: string;
  traceRows: TraceabilityRow[];
  posteriors: PosteriorSeries[];
  cotLines: string[];
};

export function buildPosteriorChartData(posteriors: PosteriorSeries[], points = 96) {
  const lo = 0.35;
  const hi = 0.995;
  const step = (hi - lo) / (points - 1);
  const xs = Array.from({ length: points }, (_, i) => lo + step * i);
  const raw = posteriors.map((p) => xs.map((x) => normalDensity(x, p.mu, p.sigma)));
  let globalMax = 0;
  raw.forEach((series) => series.forEach((v) => (globalMax = Math.max(globalMax, v))));
  const scale = globalMax > 0 ? globalMax : 1;
  return xs.map((x, i) => {
    const row: Record<string, number> = { x: Math.round(x * 1000) / 1000 };
    posteriors.forEach((p, pi) => {
      const v = raw[pi]?.[i] ?? 0;
      row[p.id] = v / scale;
    });
    return row;
  });
}
