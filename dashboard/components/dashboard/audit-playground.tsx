"use client";

import { useCallback, useEffect, useRef, useState } from "react";
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

const MAX_FILES = 64;
const MAX_TOTAL_CHARS = 250_000;

type AuditApiResponse = {
  raw_text: string;
  parsed: Record<string, unknown> | null;
  parse_error: string | null;
};

type StatusResponse = {
  backend?: string;
  adapter_dir: string;
  model_name: string;
  use_lora: boolean;
  inference_ready: boolean;
  load_error: string | null;
};

function apiBase(): string {
  return (process.env.NEXT_PUBLIC_NOMOS_API_URL ?? "").replace(/\/$/, "");
}

async function fileListToFilesMap(files: FileList): Promise<Record<string, string>> {
  const out: Record<string, string> = {};
  const list = Array.from(files);
  if (list.length > MAX_FILES) {
    throw new Error(`Too many files (max ${MAX_FILES}).`);
  }
  let total = 0;
  for (const f of list) {
    const rel = (f as File & { webkitRelativePath?: string }).webkitRelativePath || f.name;
    const path = rel.replace(/\\/g, "/").trim();
    if (!path) continue;
    const text = await f.text();
    total += path.length + text.length;
    if (total > MAX_TOTAL_CHARS) {
      throw new Error(`Total size exceeds ${MAX_TOTAL_CHARS} characters.`);
    }
    out[path] = text;
  }
  if (!Object.keys(out).length) {
    throw new Error("No files selected.");
  }
  if (!Object.values(out).some((s) => s.trim())) {
    throw new Error("All selected files are empty.");
  }
  return out;
}

export function AuditPlayground() {
  const base = apiBase();
  const snippetInputRef = useRef<HTMLInputElement>(null);
  const folderInputRef = useRef<HTMLInputElement>(null);
  const [sourceMode, setSourceMode] = useState<"snippet" | "files">("snippet");
  const [code, setCode] = useState(DEFAULT_SAMPLE);
  const [filesMap, setFilesMap] = useState<Record<string, string> | null>(null);
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

  async function onPickFiles(e: React.ChangeEvent<HTMLInputElement>) {
    const fl = e.target.files;
    e.target.value = "";
    if (!fl?.length) return;
    try {
      setFilesMap(await fileListToFilesMap(fl));
      setError(null);
    } catch (err) {
      setFilesMap(null);
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function runAudit() {
    if (!base) {
      setError("Set NEXT_PUBLIC_NOMOS_API_URL to your FastAPI base (e.g. http://127.0.0.1:8000).");
      return;
    }
    setLoading(true);
    setError(null);
    setResult(null);
    const body =
      sourceMode === "files" && filesMap
        ? JSON.stringify({ files: filesMap })
        : JSON.stringify({ code });
    try {
      const r = await fetch(`${base}/v1/audit`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body,
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

  const canRun =
    sourceMode === "snippet"
      ? Boolean(code.trim())
      : Boolean(filesMap && Object.keys(filesMap).length);

  return (
    <Card className="flex flex-col gap-0 rounded-panel border-hairline border-border bg-surface py-2 shadow-none ring-0">
      <CardHeader className="border-b border-border px-3 py-2 [.border-b]:pb-2">
        <CardTitle className="font-mono text-[11px] font-semibold tracking-wide text-muted-foreground uppercase">
          Live audit
        </CardTitle>
        <p className="mt-1 font-mono text-[11px] leading-relaxed text-muted-foreground">
          Calls your Nomos API (<span className="text-foreground">/v1/audit</span>). PoC only — not
          legal advice.
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
                {status.backend ? (
                  <>
                    {" "}
                    · backend:{" "}
                    <span className="text-foreground">{status.backend}</span>
                  </>
                ) : null}
                {" "}
                · LoRA:{" "}
                <span className="text-foreground">
                  {status.use_lora ? "on" : "off"}
                </span>
                {" "}
                · model:{" "}
                <span className="text-foreground break-all">{status.model_name}</span>
                {status.load_error ? (
                  <span className="block pt-1 text-amber-600 dark:text-amber-400">
                    {status.load_error}
                  </span>
                ) : null}
              </>
            )}
          </p>
        )}

        <div className="flex flex-wrap gap-2 font-mono text-[11px]">
          <Button
            type="button"
            variant={sourceMode === "snippet" ? "default" : "outline"}
            size="sm"
            onClick={() => setSourceMode("snippet")}
          >
            Snippet
          </Button>
          <Button
            type="button"
            variant={sourceMode === "files" ? "default" : "outline"}
            size="sm"
            onClick={() => setSourceMode("files")}
          >
            Upload files
          </Button>
        </div>

        {sourceMode === "snippet" ? (
          <textarea
            value={code}
            onChange={(e) => setCode(e.target.value)}
            spellCheck={false}
            className="min-h-[200px] w-full resize-y rounded-lg border border-border bg-background px-3 py-2 font-mono text-xs leading-relaxed text-foreground outline-none focus-visible:border-ring focus-visible:ring-2 focus-visible:ring-ring/40"
            aria-label="Python code to audit"
          />
        ) : (
          <div className="flex flex-col gap-2 rounded-lg border border-border bg-muted/20 p-3">
            <input
              ref={snippetInputRef}
              type="file"
              multiple
              className="hidden"
              onChange={(e) => void onPickFiles(e)}
            />
            <input
              ref={folderInputRef}
              type="file"
              multiple
              className="hidden"
              {...({ webkitdirectory: "" } as Record<string, string>)}
              onChange={(e) => void onPickFiles(e)}
            />
            <div className="flex flex-wrap gap-2">
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={() => snippetInputRef.current?.click()}
              >
                Choose files
              </Button>
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={() => folderInputRef.current?.click()}
              >
                Choose folder
              </Button>
              {filesMap ? (
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  onClick={() => setFilesMap(null)}
                >
                  Clear
                </Button>
              ) : null}
            </div>
            <p className="font-mono text-[11px] text-muted-foreground">
              Up to {MAX_FILES} paths, {MAX_TOTAL_CHARS.toLocaleString()} characters total. Relative
              paths are preserved when you pick a folder.
            </p>
            {filesMap ? (
              <ScrollArea className="h-24 w-full rounded border border-border bg-background p-2">
                <ul className="font-mono text-[10px] text-foreground">
                  {Object.keys(filesMap)
                    .sort()
                    .map((p) => (
                      <li key={p} className="truncate" title={p}>
                        {p}
                      </li>
                    ))}
                </ul>
              </ScrollArea>
            ) : (
              <p className="font-mono text-[11px] text-muted-foreground">
                No files selected.
              </p>
            )}
          </div>
        )}

        <div className="flex flex-wrap gap-2">
          <Button
            type="button"
            variant="default"
            size="sm"
            disabled={loading || !base || !canRun}
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
