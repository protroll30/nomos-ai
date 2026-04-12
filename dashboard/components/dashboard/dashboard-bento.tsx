"use client";

import { AuditPlayground } from "@/components/dashboard/audit-playground";
import { BayesianChartPanel } from "@/components/dashboard/bayesian-chart-panel";
import { CotFeedPanel } from "@/components/dashboard/cot-feed-panel";
import { DatasetHealthCard } from "@/components/dashboard/dataset-health-card";
import { TraceabilityMatrix } from "@/components/dashboard/traceability-matrix";
import type { DatasetHealth } from "@/lib/dataset-health";
import type { GoldenSetBundle } from "@/lib/golden-set";

export function DashboardBento({
  sourcePath,
  traceRows,
  posteriors,
  cotLines,
  datasetHealth,
}: GoldenSetBundle & { datasetHealth: DatasetHealth }) {
  return (
    <div
      data-dashboard-root
      className="nomos-bento-grid min-h-screen p-2 md:min-h-0 md:p-3"
    >
      <div className="col-span-12">
        <AuditPlayground />
      </div>
      <div className="col-span-12">
        <DatasetHealthCard legalChunksTotal={datasetHealth.legalChunksTotal} />
      </div>
      <div className="col-span-12 flex min-h-[20rem] flex-col lg:col-span-7 lg:row-span-2 lg:min-h-[calc(100vh-1.5rem)]">
        <TraceabilityMatrix traceRows={traceRows} sourcePath={sourcePath} />
      </div>
      <div className="col-span-12 flex min-h-[13rem] flex-col lg:col-span-5">
        <BayesianChartPanel posteriors={posteriors} />
      </div>
      <div className="col-span-12 flex min-h-[13rem] flex-col lg:col-span-5">
        <CotFeedPanel cotLines={cotLines} />
      </div>
    </div>
  );
}
