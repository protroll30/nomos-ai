"use client";

import { useCallback, useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";

const DEFAULT_SAMPLE = `@app.post("/predict")
async def predict(features: list[float]):
    return {"score": model.predict(features)}`;

type AuditApiResponse = {
  raw_text: string;
  parsed: Record<string, unknown> | null;
  parse_error: string | null;
};

type StatusResponse = {
  adapter_dir: string;
  model_name: string;
  inference_ready: boolean;
  load_error: string | null;
};

function apiBase(): string {
  return (process.env.NEXT_PUBLIC_NOMOS_API_URL ?? "").replace(/\/$/, "");
}

export function AuditPlayground() {
  const base = apiBase();
  const [code, setCode] = useState(DEFAULT_SAMPLE);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AuditApiResponse | null>(null);
  const [status, setStatus] = useState<StatusResponse | null>(null);

  const refreshStatus = useCallback(async () => {
    if (!base) return;
    try {
      const r = await fetch(`${base}/v1/audit/status`);
      if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
      setStatus((await r.json()) as StatusResponse);
    } catch {
      setStatus(null);
    }
  }, [base]);

  useEffect(() => {
    void refreshStatus();
  }, [refreshStatus]);

  async function runAudit() {
    if (!base) {
      setError("Set NEXT_PUBLIC_NOMOS_API_URL to your FastAPI base (e.g. http://127.0.0.1:8000).");
      return;
    }
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const r = await fetch(`${base}/v1/audit`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ code }),
      });
      const data = (await r.json().catch(() => ({}))) as
        | AuditApiResponse
        | { detail?: string };
      if (!r.ok) {
        const detail =
          typeof data === "object" && data && "detail" in data
            ? String((data as { detail: unknown }).detail)
            : r.statusText;
        throw new Error(detail || `HTTP ${r.status}`);
      }
      setResult(data as AuditApiResponse);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
      void refreshStatus();
    }
  }

  return (
    <Card className="flex flex-col gap-0 rounded-panel border-hairline border-border bg-surface py-2 shadow-none ring-0">
      <CardHeader className="border-b border-border px-3 py-2 [.border-b]:pb-2">
        <CardTitle className="font-mono text-[11px] font-semibold tracking-wide text-muted-foreground uppercase">
          Live audit (LoRA)
        </CardTitle>
        <p className="mt-1 font-mono text-[11px] leading-relaxed text-muted-foreground">
          Requires the Nomos API on a GPU machine. PoC only — not legal advice.
        </p>
      </CardHeader>
      <CardContent className="flex flex-col gap-3 px-3 py-3">
        {!base ? (
          <p className="font-mono text-xs text-amber-600 dark:text-amber-400">
            Configure{" "}
            <span className="text-foreground">NEXT_PUBLIC_NOMOS_API_URL</span>{" "}
            and restart <span className="text-foreground">npm run dev</span>.
          </p>
        ) : (
          <p className="font-mono text-[11px] text-muted-foreground">
            API: <span className="text-foreground">{base}</span>
            {status && (
              <>
                {" "}
                · ready:{" "}
                <span className="text-foreground">
                  {status.inference_ready ? "yes" : "no"}
                </span>
                {status.load_error ? (
                  <span className="block pt-1 text-amber-600 dark:text-amber-400">
                    {status.load_error}
                  </span>
                ) : null}
              </>
            )}
          </p>
        )}

        <textarea
          value={code}
          onChange={(e) => setCode(e.target.value)}
          spellCheck={false}
          className="min-h-[200px] w-full resize-y rounded-lg border border-border bg-background px-3 py-2 font-mono text-xs leading-relaxed text-foreground outline-none focus-visible:border-ring focus-visible:ring-2 focus-visible:ring-ring/40"
          aria-label="Python code to audit"
        />

        <div className="flex flex-wrap gap-2">
          <Button
            type="button"
            variant="default"
            size="sm"
            disabled={loading || !base}
            onClick={() => void runAudit()}
          >
            {loading ? "Running…" : "Run audit"}
          </Button>
          <Button
            type="button"
            variant="outline"
            size="sm"
            disabled={!base}
            onClick={() => void refreshStatus()}
          >
            Refresh status
          </Button>
        </div>

        {error ? (
          <p className="font-mono text-xs text-destructive">{error}</p>
        ) : null}

        {result ? (
          <div className="flex flex-col gap-2">
            {result.parse_error ? (
              <p className="font-mono text-[11px] text-amber-600 dark:text-amber-400">
                JSON parse: {result.parse_error}
              </p>
            ) : null}
            <div className="font-mono text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">
              Parsed JSON
            </div>
            <ScrollArea className="h-[min(24rem,50vh)] w-full rounded-lg border border-border bg-muted/30 p-2">
              <pre className="whitespace-pre-wrap break-words font-mono text-[11px] text-foreground">
                {result.parsed
                  ? JSON.stringify(result.parsed, null, 2)
                  : "(not available)"}
              </pre>
            </ScrollArea>
            <details className="font-mono text-[11px] text-muted-foreground">
              <summary className="cursor-pointer text-foreground">
                Raw model output
              </summary>
              <pre className="mt-2 max-h-48 overflow-auto whitespace-pre-wrap break-words text-[11px]">
                {result.raw_text}
              </pre>
            </details>
          </div>
        ) : null}
      </CardContent>
    </Card>
  );
}
