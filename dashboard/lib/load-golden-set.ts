import { readFile } from "fs/promises";
import path from "path";

import {
  type GoldenSetBundle,
  type GoldenSetRecord,
  recordsToPosteriorSeries,
  recordsToTraceRows,
} from "@/lib/golden-set";

const REQUIRED_KEYS: (keyof GoldenSetRecord)[] = [
  "id",
  "citation",
  "instrument",
  "section_ref",
  "text",
  "repo_path",
  "expected_probability",
  "state",
  "reasoning",
];

const TRAFFIC: GoldenSetRecord["state"][] = ["verified", "caution", "emergent"];

function iso() {
  return new Date().toISOString();
}

function isGoldenSetRecord(x: unknown): x is GoldenSetRecord {
  if (x === null || typeof x !== "object") return false;
  const o = x as Record<string, unknown>;
  for (const k of REQUIRED_KEYS) {
    if (!(k in o)) return false;
  }
  if (typeof o.expected_probability !== "number") return false;
  if (!TRAFFIC.includes(o.state as GoldenSetRecord["state"])) return false;
  return true;
}

function stateFromViolationSubtlety(raw: unknown): GoldenSetRecord["state"] | null {
  const s = typeof raw === "string" ? raw.trim().toLowerCase() : "";
  if (s === "overt") return "verified";
  if (s === "omission") return "caution";
  if (s === "borderline") return "emergent";
  return null;
}

function coerceGoldenSetRecord(x: unknown): GoldenSetRecord | null {
  if (isGoldenSetRecord(x)) return x;
  if (x === null || typeof x !== "object") return null;
  const o = x as Record<string, unknown>;
  const baseKeys = ["id", "citation", "instrument", "section_ref", "text"] as const;
  for (const k of baseKeys) {
    if (typeof o[k] !== "string" || !String(o[k]).trim()) return null;
  }
  if (typeof o.compliance_justification !== "string") return null;
  if (typeof o.uncertainty_factor !== "number" || Number.isNaN(o.uncertainty_factor)) return null;
  const state = stateFromViolationSubtlety(o.violation_subtlety);
  if (!state) return null;
  return {
    id: String(o.id),
    citation: String(o.citation),
    instrument: String(o.instrument),
    section_ref: String(o.section_ref),
    text: String(o.text),
    repo_path: `synthetic/${String(o.id)}`,
    expected_probability: o.uncertainty_factor,
    state,
    reasoning: String(o.compliance_justification),
  };
}

function buildCotLines(
  sourcePath: string,
  byteLength: number,
  records: GoldenSetRecord[],
  parseMs: number
): string[] {
  const lines: string[] = [];
  lines.push(`[${iso()}] io:read path=${sourcePath} bytes=${byteLength}`);
  lines.push(
    `[${iso()}] parse:json.decode records=${records.length} wall_ms=${parseMs.toFixed(2)}`
  );
  records.forEach((r, i) => {
    const o = r as unknown as Record<string, unknown>;
    const keys = Object.keys(o);
    const allowed = new Set(REQUIRED_KEYS as string[]);
    const unknownKeys = keys.filter((k) => !allowed.has(k));
    lines.push(
      `[${iso()}] validate:row[${i}] id=${r.id} fields=${keys.length} stray=${unknownKeys.length ? unknownKeys.join(",") : "∅"}`
    );
    lines.push(
      `[${iso()}] hydrate:row[${i}] citation=${r.citation} repo_path=${r.repo_path} E[p]=${r.expected_probability} state=${r.state}`
    );
    lines.push(
      `[${iso()}] map:matrix row[${i}] obligation_chars=${r.text.length} reasoning_chars=${r.reasoning.length}`
    );
  });
  lines.push(
    `[${iso()}] done:bundle trace_rows=${records.length} posterior_series=${records.length}`
  );
  return lines;
}

function resolveGoldenSetPath(): string {
  const rel = path.join(process.cwd(), "..", "data", "golden_set.json");
  return path.normalize(rel);
}

export async function loadGoldenSetBundle(): Promise<GoldenSetBundle> {
  const sourcePath = resolveGoldenSetPath();
  const t0 = performance.now();
  const raw = await readFile(sourcePath, "utf8");
  const byteLength = Buffer.byteLength(raw, "utf8");
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    throw new Error(`golden_set.json JSON error at ${sourcePath}: ${msg}`);
  }
  const parseMs = performance.now() - t0;
  if (!Array.isArray(parsed)) {
    throw new Error(`golden_set.json must be an array at ${sourcePath}`);
  }
  const records = parsed.map(coerceGoldenSetRecord).filter((r): r is GoldenSetRecord => r !== null);
  if (records.length !== parsed.length) {
    throw new Error(
      `golden_set.json: ${parsed.length - records.length} record(s) failed schema at ${sourcePath}`
    );
  }
  const traceRows = recordsToTraceRows(records);
  const posteriors = recordsToPosteriorSeries(records);
  const cotLines = buildCotLines(sourcePath, byteLength, records, parseMs);
  return { sourcePath, traceRows, posteriors, cotLines };
}
