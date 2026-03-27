export type TrafficState = "verified" | "caution" | "emergent";

export type TraceabilityRow = {
  id: string;
  euArticle: string;
  annexRef: string;
  obligation: string;
  repoPath: string;
  posteriorMean: number;
  traffic: TrafficState;
};

export const mockTraceabilityRows: TraceabilityRow[] = [
  {
    id: "r1",
    euArticle: "Art. 9(2)",
    annexRef: "Annex IV(a)",
    obligation: "Risk management system · iterative identification/mitigation",
    repoPath: "backend/app/main.py",
    posteriorMean: 0.91,
    traffic: "verified",
  },
  {
    id: "r2",
    euArticle: "Art. 10(1)",
    annexRef: "Annex IV(c)",
    obligation: "Logging · automatic recording of events (limited to model scope)",
    repoPath: "backend/app/core/logging.py",
    posteriorMean: 0.76,
    traffic: "caution",
  },
  {
    id: "r3",
    euArticle: "Art. 11(1)",
    annexRef: "Annex IV(d)",
    obligation: "Technical documentation · design / training / monitoring",
    repoPath: "backend/app/api/v1/router.py",
    posteriorMean: 0.88,
    traffic: "verified",
  },
  {
    id: "r4",
    euArticle: "Art. 12(1)",
    annexRef: "—",
    obligation: "Record-keeping · operation logs retrievable on request",
    repoPath: "backend/app/middleware/request_id.py",
    posteriorMean: 0.71,
    traffic: "caution",
  },
  {
    id: "r5",
    euArticle: "Art. 13(1)(b)",
    annexRef: "Annex IV(f)",
    obligation: "Transparency · instructions for deployer / downstream config",
    repoPath: "backend/app/schemas/deployment.py",
    posteriorMean: 0.84,
    traffic: "verified",
  },
  {
    id: "r6",
    euArticle: "Art. 14(4)",
    annexRef: "Annex IV(h)",
    obligation: "Human oversight · intervention / stop / override affordances",
    repoPath: "backend/app/routers/admin.py",
    posteriorMean: 0.64,
    traffic: "emergent",
  },
  {
    id: "r7",
    euArticle: "Art. 15(1)",
    annexRef: "Annex IV(i)",
    obligation: "Accuracy / robustness · measurable service levels",
    repoPath: "backend/app/deps.py",
    posteriorMean: 0.79,
    traffic: "caution",
  },
  {
    id: "r8",
    euArticle: "Art. 16",
    annexRef: "Annex IV(j)",
    obligation: "Post-market monitoring · incident hooks / reporting surface",
    repoPath: "backend/tests/test_health.py",
    posteriorMean: 0.87,
    traffic: "verified",
  },
];

export type PosteriorMock = {
  id: string;
  label: string;
  mu: number;
  sigma: number;
  strokeVar: "--chart-1" | "--chart-2" | "--chart-3";
};

export const mockPosteriors: PosteriorMock[] = [
  { id: "p1", label: "Art. 9 · risk management", mu: 0.91, sigma: 0.028, strokeVar: "--chart-1" },
  { id: "p2", label: "Art. 13 · transparency", mu: 0.84, sigma: 0.041, strokeVar: "--chart-2" },
  { id: "p3", label: "Art. 15 · accuracy", mu: 0.73, sigma: 0.054, strokeVar: "--chart-3" },
];

export const mockCotLines: string[] = [
  "[09:14:22] tool:vector_search corpus=EU_AI_ACT_ANNEX_IV k=12",
  "[09:14:23] ast:package_resolve root=backend/app framework=fastapi",
  "[09:14:24] map:Art.13(1)(b) ↔ backend/app/schemas/deployment.py conf=0.84",
  "[09:14:25] map:Art.14(4) ↔ backend/app/routers/admin.py conf=0.64 → DIAGNOSTIC",
  "[09:14:26] static:semgrep_rule=eidas_high_risk_trace hits=0",
  "[09:14:27] rag:rerank neuralect=eu-ai-act-v0 latency=118ms",
];

export function normalDensity(x: number, mu: number, sigma: number): number {
  if (sigma <= 0) return 0;
  const z = (x - mu) / sigma;
  return Math.exp(-0.5 * z * z) / (sigma * Math.sqrt(2 * Math.PI));
}

export function buildPosteriorChartData(posteriors: PosteriorMock[], points = 96) {
  const lo = 0.42;
  const hi = 0.99;
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
