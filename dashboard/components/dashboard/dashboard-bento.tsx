"use client";

import { BayesianChartPanel } from "@/components/dashboard/bayesian-chart-panel";
import { CotFeedPanel } from "@/components/dashboard/cot-feed-panel";
import { TraceabilityMatrix } from "@/components/dashboard/traceability-matrix";

export function DashboardBento() {
  return (
    <div
      data-dashboard-root
      className="nomos-bento-grid min-h-screen p-2 md:min-h-0 md:p-3"
    >
      <div className="col-span-12 flex min-h-[20rem] flex-col lg:col-span-7 lg:row-span-2 lg:min-h-[calc(100vh-1.5rem)]">
        <TraceabilityMatrix />
      </div>
      <div className="col-span-12 flex min-h-[13rem] flex-col lg:col-span-5">
        <BayesianChartPanel />
      </div>
      <div className="col-span-12 flex min-h-[13rem] flex-col lg:col-span-5">
        <CotFeedPanel />
      </div>
    </div>
  );
}
