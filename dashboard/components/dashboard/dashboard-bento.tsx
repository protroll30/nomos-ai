"use client";

import { useCallback, useState } from "react";

import { AuditPlayground } from "@/components/dashboard/audit-playground";
import { BayesianChartPanel } from "@/components/dashboard/bayesian-chart-panel";
import { CotFeedPanel } from "@/components/dashboard/cot-feed-panel";
import { DatasetHealthCard } from "@/components/dashboard/dataset-health-card";
import { TraceabilityMatrix } from "@/components/dashboard/traceability-matrix";
import { liveAuditToTraceAndPosteriors } from "@/lib/audit-trace";
import type { DatasetHealth } from "@/lib/dataset-health";
import type { GoldenSetBundle, PosteriorSeries, TraceabilityRow } from "@/lib/golden-set";

export function DashboardBento({
  sourcePath,
  traceRows,
  posteriors,
  cotLines,
  datasetHealth,
}: GoldenSetBundle & { datasetHealth: DatasetHealth }) {
  const [liveOverlay, setLiveOverlay] = useState<{
    traceRows: TraceabilityRow[];
    posteriors: PosteriorSeries[];
    sourceLabel: string;
  } | null>(null);

  const displayTraceRows = liveOverlay?.traceRows ?? traceRows;
  const displayPosteriors = liveOverlay?.posteriors ?? posteriors;
  const displayMatrixSource = liveOverlay?.sourceLabel ?? sourcePath;
  const chartLabel = liveOverlay ? "last /v1/audit" : "golden_set.json";

  const handleAuditComplete = useCallback(
    (payload: {
      parsed: Record<string, unknown> | null;
      kind: "snippet" | "files";
      fileCount: number;
    }) => {
      const repoPath =
        payload.kind === "files"
          ? `live/upload (${payload.fileCount} files)`
          : "live/snippet";
      const mapped = liveAuditToTraceAndPosteriors(payload.parsed, {
        repoPath,
      });
      if (!mapped) return;
      setLiveOverlay({
        ...mapped,
        sourceLabel: `${repoPath} · last /v1/audit`,
      });
    },
    [],
  );

  const clearLiveOverlay = useCallback(() => setLiveOverlay(null), []);

  return (
    <div
      data-dashboard-root
      className="nomos-bento-grid min-h-screen p-2 md:min-h-0 md:p-3"
    >
      <div className="col-span-12">
        <AuditPlayground onAuditComplete={handleAuditComplete} />
      </div>
      <div className="col-span-12">
        <DatasetHealthCard legalChunksTotal={datasetHealth.legalChunksTotal} />
      </div>
      <div className="col-span-12 flex min-h-[20rem] flex-col lg:col-span-7 lg:row-span-2 lg:min-h-[calc(100vh-1.5rem)]">
        <TraceabilityMatrix
          traceRows={displayTraceRows}
          sourcePath={displayMatrixSource}
          variant={liveOverlay ? "live" : "golden"}
          onResetLive={liveOverlay ? clearLiveOverlay : undefined}
        />
      </div>
      <div className="col-span-12 flex min-h-[13rem] flex-col lg:col-span-5">
        <BayesianChartPanel
          posteriors={displayPosteriors}
          dataSourceLabel={chartLabel}
        />
      </div>
      <div className="col-span-12 flex min-h-[13rem] flex-col lg:col-span-5">
        <CotFeedPanel cotLines={cotLines} />
      </div>
    </div>
  );
}
