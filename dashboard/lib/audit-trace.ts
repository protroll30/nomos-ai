import type { PosteriorSeries, TraceabilityRow, TrafficState } from "@/lib/golden-set";

function trafficFromSubtlety(raw: unknown): TrafficState {
  const s = typeof raw === "string" ? raw.trim().toLowerCase() : "";
  if (s === "overt") return "verified";
  if (s === "borderline") return "emergent";
  return "caution";
}

function trafficFromComplianceStatus(status: string): TrafficState {
  const s = status.toLowerCase();
  if (s.includes("non-compliant") || s.includes("non_compliant")) return "emergent";
  if (s.includes("compliant") && !s.includes("non")) return "verified";
  return "caution";
}

function normalizeProb(raw: unknown): number {
  if (typeof raw === "number" && !Number.isNaN(raw)) {
    return Math.min(1, Math.max(0, raw));
  }
  return 0.72;
}

const STROKES: PosteriorSeries["strokeVar"][] = ["--chart-1", "--chart-2", "--chart-3"];

function rowFromParsed(
  parsed: Record<string, unknown>,
  id: string,
  repoPath: string,
): TraceabilityRow {
  const clause = String(parsed.clause ?? "");
  const just = String(parsed.compliance_justification ?? "");
  const obligation = [clause, just].filter(Boolean).join(" — ") || "(no obligation text)";
  const truncated =
    obligation.length > 620 ? `${obligation.slice(0, 620)}…` : obligation;
  return {
    id,
    euArticle: String(parsed.legal_anchor ?? "(unknown)"),
    instrument: String(parsed.instrument ?? "EU AI Act (2024/1689)"),
    annexRef: String(parsed.violation_subtlety ?? "—"),
    obligation: truncated,
    repoPath,
    posteriorMean: normalizeProb(parsed.uncertainty_factor),
    traffic: trafficFromSubtlety(parsed.violation_subtlety),
  };
}

function posteriorFromParsed(
  parsed: Record<string, unknown>,
  id: string,
  strokeVar: PosteriorSeries["strokeVar"],
): PosteriorSeries {
  const mu = normalizeProb(parsed.uncertainty_factor);
  const anchor = String(parsed.legal_anchor ?? "audit");
  return {
    id,
    label: `${anchor.length > 36 ? `${anchor.slice(0, 36)}…` : anchor} · live`,
    mu,
    sigma: Math.max(0.018, 0.045 * (1 - mu) + 0.022),
    strokeVar,
  };
}

function flattenParsedObjects(parsed: Record<string, unknown>): Record<string, unknown>[] {
  const findings = parsed.findings;
  if (
    Array.isArray(findings) &&
    findings.length > 0 &&
    findings.every((x) => x !== null && typeof x === "object" && !Array.isArray(x))
  ) {
    return findings as Record<string, unknown>[];
  }
  return [parsed];
}

function issueLabel(issue: Record<string, unknown>, index: number): string {
  const t = String(issue.type ?? issue.category ?? "").replace(/_/g, " ").trim();
  return t || `issue ${index + 1}`;
}

function rowFromComplianceIssue(
  issue: Record<string, unknown>,
  id: string,
  repoPath: string,
  status: string,
  index: number,
): TraceabilityRow {
  const desc = String(issue.description ?? issue.summary ?? "").trim();
  const typ = issueLabel(issue, index);
  const obligation =
    desc || `(no description for ${typ})`;
  const truncated =
    obligation.length > 620 ? `${obligation.slice(0, 620)}…` : obligation;
  const mu = Math.max(0.12, Math.min(0.88, 0.58 - index * 0.035));
  return {
    id,
    euArticle: status ? `Audit · ${status}` : "Compliance audit",
    instrument: "EU AI Act (2024/1689)",
    annexRef: typ,
    obligation: truncated,
    repoPath,
    posteriorMean: mu,
    traffic: trafficFromComplianceStatus(status),
  };
}

function posteriorFromComplianceIssue(
  issue: Record<string, unknown>,
  id: string,
  strokeVar: PosteriorSeries["strokeVar"],
  mu: number,
  index: number,
): PosteriorSeries {
  const typ = issueLabel(issue, index);
  return {
    id,
    label: `${typ.length > 28 ? `${typ.slice(0, 28)}…` : typ} · live`,
    mu,
    sigma: Math.max(0.018, 0.045 * (1 - mu) + 0.022),
    strokeVar,
  };
}

function fromComplianceAuditWrapper(
  parsed: Record<string, unknown>,
  opts: { repoPath: string },
): { traceRows: TraceabilityRow[]; posteriors: PosteriorSeries[] } | null {
  const ca = parsed.compliance_audit;
  if (ca === null || typeof ca !== "object" || Array.isArray(ca)) {
    return null;
  }
  const audit = ca as Record<string, unknown>;
  const status = String(audit.status ?? "");
  const issues = audit.issues;
  const base = Date.now();

  if (Array.isArray(issues) && issues.length > 0) {
    const traceRows: TraceabilityRow[] = [];
    const posteriors: PosteriorSeries[] = [];
    issues.forEach((issue, i) => {
      if (issue === null || typeof issue !== "object" || Array.isArray(issue)) {
        return;
      }
      const obj = issue as Record<string, unknown>;
      const id = `live-${base}-${i}`;
      traceRows.push(rowFromComplianceIssue(obj, id, opts.repoPath, status, i));
      const mu = traceRows[traceRows.length - 1]!.posteriorMean;
      posteriors.push(
        posteriorFromComplianceIssue(
          obj,
          id,
          STROKES[i % STROKES.length]!,
          mu,
          i,
        ),
      );
    });
    return traceRows.length ? { traceRows, posteriors } : null;
  }

  const recs = audit.recommendations;
  if (
    Array.isArray(recs) &&
    recs.length > 0 &&
    recs.every((r) => r !== null && typeof r === "object" && !Array.isArray(r))
  ) {
    const first = recs[0] as Record<string, unknown>;
    const action = String(first.action ?? "").trim();
    const details = String(first.details ?? "").trim();
    const obligation = [action, details].filter(Boolean).join(" — ") || "(no recommendation text)";
    const id = `live-${base}-0`;
    const mu = 0.55;
    return {
      traceRows: [
        {
          id,
          euArticle: status ? `Audit · ${status}` : "Compliance audit",
          instrument: "EU AI Act (2024/1689)",
          annexRef: "recommendations",
          obligation:
            obligation.length > 620 ? `${obligation.slice(0, 620)}…` : obligation,
          repoPath: opts.repoPath,
          posteriorMean: mu,
          traffic: trafficFromComplianceStatus(status),
        },
      ],
      posteriors: [
        {
          id,
          label: "Recommendations · live",
          mu,
          sigma: Math.max(0.018, 0.045 * (1 - mu) + 0.022),
          strokeVar: "--chart-1",
        },
      ],
    };
  }

  if (status) {
    const id = `live-${base}-0`;
    const mu = 0.5;
    return {
      traceRows: [
        {
          id,
          euArticle: `Audit · ${status}`,
          instrument: "EU AI Act (2024/1689)",
          annexRef: "summary",
          obligation: "(no issues or recommendations in response)",
          repoPath: opts.repoPath,
          posteriorMean: mu,
          traffic: trafficFromComplianceStatus(status),
        },
      ],
      posteriors: [
        {
          id,
          label: `${status} · live`,
          mu,
          sigma: Math.max(0.018, 0.045 * (1 - mu) + 0.022),
          strokeVar: "--chart-1",
        },
      ],
    };
  }

  return null;
}

export function liveAuditToTraceAndPosteriors(
  parsed: Record<string, unknown> | null,
  opts: { repoPath: string },
): { traceRows: TraceabilityRow[]; posteriors: PosteriorSeries[] } | null {
  if (!parsed) return null;

  const compliance = fromComplianceAuditWrapper(parsed, opts);
  if (compliance) return compliance;

  const items = flattenParsedObjects(parsed);
  const base = Date.now();
  const traceRows: TraceabilityRow[] = [];
  const posteriors: PosteriorSeries[] = [];
  items.forEach((obj, i) => {
    const id = `live-${base}-${i}`;
    traceRows.push(rowFromParsed(obj, id, opts.repoPath));
    posteriors.push(
      posteriorFromParsed(obj, id, STROKES[i % STROKES.length]!),
    );
  });
  return { traceRows, posteriors };
}
