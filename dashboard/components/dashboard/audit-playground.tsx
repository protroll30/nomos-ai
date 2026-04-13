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
import { cn } from "@/lib/utils";

const DEFAULT_SAMPLE = `@app.post("/predict")
async def predict(features: list[float]):
    return {"score": model.predict(features)}`;

const MAX_FILES = 200;
const MAX_TOTAL_CHARS = 750_000;

function formatApiDetail(data: unknown): string {
  if (!data || typeof data !== "object" || !("detail" in data)) {
    return "";
  }
  const d = (data as { detail: unknown }).detail;
  if (typeof d === "string") return d;
  try {
    return JSON.stringify(d);
  } catch {
    return String(d);
  }
}

type AuditApiResponse = {
  raw_text: string;
  parsed: Record<string, unknown> | null;
  parse_error: string | null;
};

type StatusResponse = {
  backend?: string;
  client_backend_choice_enabled?: boolean;
  adapter_dir: string;
  model_name: string;
  use_lora: boolean;
  inference_ready: boolean;
  load_error: string | null;
};

type AuditBackendMode = "openai" | "hf";

export type AuditCompletePayload = {
  parsed: Record<string, unknown>;
  kind: "snippet" | "files";
  fileCount: number;
};

function apiBase(): string {
  return (process.env.NEXT_PUBLIC_NOMOS_API_URL ?? "").replace(/\/$/, "");
}

function defaultAuditBackend(): AuditBackendMode {
  const v = (process.env.NEXT_PUBLIC_DEFAULT_AUDIT_BACKEND ?? "").trim().toLowerCase();
  if (v === "hf" || v === "local") return "hf";
  if (v === "openai" || v === "oai") return "openai";
  return "openai";
}

async function filesToAuditMap(files: File[]): Promise<Record<string, string>> {
  const list = files;
  if (list.length === 0) {
    throw new Error("No files selected.");
  }
  if (list.length > MAX_FILES) {
    throw new Error(
      `Too many entries (${list.length}). Max ${MAX_FILES} per audit — pick a subfolder (e.g. src) or fewer files.`
    );
  }

  const pathCounts = new Map<string, number>();
  const uniquePath = (basePath: string): string => {
    const n = pathCounts.get(basePath) ?? 0;
    pathCounts.set(basePath, n + 1);
    if (n === 0) return basePath;
    const dot = basePath.lastIndexOf(".");
    if (dot > 0) {
      return `${basePath.slice(0, dot)}__${n + 1}${basePath.slice(dot)}`;
    }
    return `${basePath}__${n + 1}`;
  };

  const rows = await Promise.all(
    list.map(async (f) => {
      const rel = (f as File & { webkitRelativePath?: string }).webkitRelativePath || f.name;
      const raw = rel.replace(/\\/g, "/").trim();
      if (!raw) return null;
      const path = uniquePath(raw);
      const text = await f.text();
      return { path, text };
    })
  );

  const out: Record<string, string> = {};
  let total = 0;
  for (const row of rows) {
    if (!row) continue;
    total += row.path.length + row.text.length;
    if (total > MAX_TOTAL_CHARS) {
      throw new Error(
        `Total size exceeds ${MAX_TOTAL_CHARS.toLocaleString()} characters — choose a smaller folder.`
      );
    }
    out[row.path] = row.text;
  }
  if (!Object.keys(out).length) {
    throw new Error("No usable file paths (empty names).");
  }
  if (!Object.values(out).some((s) => s.trim())) {
    throw new Error("All selected files are empty.");
  }
  return out;
}

type AuditPlaygroundProps = {
  onAuditComplete?: (payload: AuditCompletePayload) => void;
};

export function AuditPlayground({ onAuditComplete }: AuditPlaygroundProps) {
  const base = apiBase();
  const snippetInputRef = useRef<HTMLInputElement>(null);
  const folderInputRef = useRef<HTMLInputElement | null>(null);
  const setFolderInputEl = useCallback((el: HTMLInputElement | null) => {
    folderInputRef.current = el;
    if (el) {
      el.setAttribute("webkitdirectory", "");
      el.setAttribute("mozdirectory", "");
    }
  }, []);
  const [sourceMode, setSourceMode] = useState<"snippet" | "files">("snippet");
  const [auditBackend, setAuditBackend] = useState<AuditBackendMode>(defaultAuditBackend);
  const [code, setCode] = useState(DEFAULT_SAMPLE);
  const [filesMap, setFilesMap] = useState<Record<string, string> | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AuditApiResponse | null>(null);
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [readingFiles, setReadingFiles] = useState(false);

  const fetchStatusFor = useCallback(async (mode: AuditBackendMode) => {
    if (!base) return;
    try {
      const q = new URLSearchParams({ audit_backend: mode });
      const r = await fetch(`${base}/v1/audit/status?${q}`, {
        cache: "no-store",
      });
      if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
      setStatus((await r.json()) as StatusResponse);
    } catch {
      setStatus(null);
    }
  }, [base]);

  useEffect(() => {
    void fetchStatusFor(auditBackend);
  }, [auditBackend, base, fetchStatusFor]);

  async function onPickFiles(e: React.ChangeEvent<HTMLInputElement>) {
    const input = e.target;
    const picked = Array.from(input.files ?? []);
    if (!picked.length) return;
    setReadingFiles(true);
    setError(null);
    try {
      setFilesMap(await filesToAuditMap(picked));
    } catch (err) {
      setFilesMap(null);
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setReadingFiles(false);
      input.value = "";
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
        ? JSON.stringify({ files: filesMap, audit_backend: auditBackend })
        : JSON.stringify({ code, audit_backend: auditBackend });
    try {
      const r = await fetch(`${base}/v1/audit`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body,
      });
      const data = (await r.json().catch(() => ({}))) as
        | AuditApiResponse
        | { detail?: string };
      if (!r.ok) {
        const detail = formatApiDetail(data) || r.statusText;
        throw new Error(detail || `HTTP ${r.status}`);
      }
      const res = data as AuditApiResponse;
      setResult(res);
      if (onAuditComplete && res.parsed) {
        onAuditComplete({
          parsed: res.parsed as Record<string, unknown>,
          kind: sourceMode === "files" ? "files" : "snippet",
          fileCount: filesMap ? Object.keys(filesMap).length : 0,
        });
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
      void fetchStatusFor(auditBackend);
    }
  }

  const canRun =
    sourceMode === "snippet"
      ? Boolean(code.trim())
      : Boolean(filesMap && Object.keys(filesMap).length);

  return (
    <Card className="flex flex-col gap-0 rounded-panel border-hairline border-border bg-surface py-2 shadow-none ring-0">
      <CardHeader className="border-b border-border px-3 py-2 [.border-b]:pb-2">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
          <div className="min-w-0 flex-1">
            <CardTitle className="font-mono text-[11px] font-semibold tracking-wide text-muted-foreground uppercase">
              Live audit
            </CardTitle>
            <p className="mt-1 font-mono text-[11px] leading-relaxed text-muted-foreground">
              Calls your Nomos API (<span className="text-foreground">/v1/audit</span>). PoC only —
              not legal advice.
            </p>
          </div>
          {base ? (
            status?.client_backend_choice_enabled === false ? (
              <div className="shrink-0 rounded-full border border-border bg-muted/30 px-3 py-1.5 font-mono text-[10px] text-muted-foreground">
                Backend:{" "}
                <span className="font-semibold text-foreground">
                  {status.backend ?? "server default"}
                </span>
              </div>
            ) : (
              <div
                className="flex shrink-0 rounded-full border border-border bg-muted/40 p-0.5 shadow-[inset_0_1px_2px_rgba(0,0,0,0.06)] dark:shadow-[inset_0_1px_2px_rgba(0,0,0,0.2)]"
                role="group"
                aria-label="Inference backend"
              >
                <button
                  type="button"
                  onClick={() => setAuditBackend("openai")}
                  className={cn(
                    "rounded-full px-3 py-1.5 font-mono text-[10px] font-semibold uppercase tracking-wide transition-colors",
                    "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background",
                    auditBackend === "openai"
                      ? "bg-primary text-primary-foreground shadow-sm"
                      : "text-muted-foreground hover:text-foreground"
                  )}
                >
                  OpenAI
                </button>
                <button
                  type="button"
                  onClick={() => setAuditBackend("hf")}
                  className={cn(
                    "rounded-full px-3 py-1.5 font-mono text-[10px] font-semibold uppercase tracking-wide transition-colors",
                    "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background",
                    auditBackend === "hf"
                      ? "bg-primary text-primary-foreground shadow-sm"
                      : "text-muted-foreground hover:text-foreground"
                  )}
                >
                  Local GPU
                </button>
              </div>
            )
          ) : null}
        </div>
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
                    {status.backend === "hf" &&
                    /cuda/i.test(status.load_error) ? (
                      <span className="mt-1 block text-muted-foreground">
                        <span className="text-foreground">Local GPU</span> needs CUDA. Pick{" "}
                        <span className="text-foreground">OpenAI</span> above, put{" "}
                        <span className="text-foreground">OPENAI_API_KEY</span> in the repo-root{" "}
                        <span className="text-foreground">.env</span> (loaded when uvicorn starts), then
                        restart the API.
                      </span>
                    ) : null}
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
              ref={setFolderInputEl}
              type="file"
              multiple
              className="hidden"
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
                onClick={() => {
                  folderInputRef.current?.setAttribute("webkitdirectory", "");
                  folderInputRef.current?.setAttribute("mozdirectory", "");
                  folderInputRef.current?.click();
                }}
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
              Up to {MAX_FILES} files and {MAX_TOTAL_CHARS.toLocaleString()} characters total (whole
              repos with <span className="text-foreground">node_modules</span> usually exceed that — pick{" "}
              <span className="text-foreground">src</span> or similar). Folder picks can take a few
              seconds before the list appears.
            </p>
            {readingFiles ? (
              <p className="font-mono text-[11px] text-amber-600 dark:text-amber-400">
                Reading files from disk…
              </p>
            ) : null}
            {filesMap && !readingFiles ? (
              <div className="max-h-40 min-h-[6rem] w-full overflow-y-auto rounded border border-border bg-background p-2">
                <p className="mb-1 font-mono text-[10px] font-semibold text-muted-foreground">
                  {Object.keys(filesMap).length} file(s)
                </p>
                <ul className="font-mono text-[10px] text-foreground">
                  {Object.keys(filesMap)
                    .sort()
                    .map((p) => (
                      <li key={p} className="break-all py-0.5" title={p}>
                        {p}
                      </li>
                    ))}
                </ul>
              </div>
            ) : null}
            {!filesMap && !readingFiles ? (
              <p className="font-mono text-[11px] text-muted-foreground">No files selected.</p>
            ) : null}
          </div>
        )}

        <div className="flex flex-wrap gap-2">
          <Button
            type="button"
            variant="default"
            size="sm"
            disabled={loading || readingFiles || !base || !canRun}
            onClick={() => void runAudit()}
          >
            {loading ? "Running…" : "Run audit"}
          </Button>
          <Button
            type="button"
            variant="outline"
            size="sm"
            disabled={!base}
            onClick={() => void fetchStatusFor(auditBackend)}
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
