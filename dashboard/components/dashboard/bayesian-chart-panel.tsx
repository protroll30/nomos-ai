"use client";

import { useEffect, useMemo, useState } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  type PosteriorSeries,
  buildPosteriorChartData,
} from "@/lib/golden-set";

export function BayesianChartPanel({
  posteriors,
}: {
  posteriors: PosteriorSeries[];
}) {
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);
  const data = useMemo(
    () => buildPosteriorChartData(posteriors),
    [posteriors]
  );

  const xDomain = useMemo((): [number, number] => {
    if (posteriors.length === 0) return [0.35, 0.995];
    let lo = 1;
    let hi = 0;
    posteriors.forEach((p) => {
      lo = Math.min(lo, p.mu - 4 * p.sigma);
      hi = Math.max(hi, p.mu + 4 * p.sigma);
    });
    return [
      Math.max(0.2, Math.round(lo * 1000) / 1000),
      Math.min(0.999, Math.round(hi * 1000) / 1000),
    ];
  }, [posteriors]);

  return (
    <Card className="flex h-full min-h-0 flex-col gap-0 rounded-panel border-hairline border-border bg-surface py-2 shadow-none ring-0">
      <CardHeader className="space-y-0 border-b border-border px-3 py-2 [.border-b]:pb-2">
        <CardTitle className="font-mono text-[11px] font-semibold tracking-wide text-muted-foreground uppercase">
          Posterior density · E[p] from golden_set.json
        </CardTitle>
      </CardHeader>
      <CardContent className="min-h-0 flex-1 px-2 pb-2 pt-1">
        <div className="h-[11rem] w-full min-w-[12rem]">
          {mounted && posteriors.length > 0 ? (
            <ResponsiveContainer width="100%" height="100%">
              <LineChart
                data={data}
                margin={{ top: 4, right: 8, left: 0, bottom: 0 }}
              >
                <CartesianGrid
                  stroke="oklch(98% 0.01 200 / 0.08)"
                  vertical={false}
                />
                <XAxis
                  dataKey="x"
                  type="number"
                  domain={xDomain}
                  tick={{ fill: "oklch(72% 0.025 200)", fontSize: 9 }}
                  tickLine={false}
                  axisLine={{ stroke: "oklch(98% 0.01 200 / 0.18)" }}
                  tickFormatter={(v) => `${(Number(v) * 100).toFixed(0)}¢`}
                />
                <YAxis
                  width={28}
                  tick={{ fill: "oklch(72% 0.025 200)", fontSize: 9 }}
                  tickLine={false}
                  axisLine={false}
                  domain={[0, 1]}
                  tickFormatter={(v) => `${(Number(v) * 100).toFixed(0)}%`}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "oklch(22% 0.024 236)",
                    border: "0.5px solid oklch(98% 0.01 200 / 0.22)",
                    borderRadius: 2,
                    fontFamily: "var(--font-mono), ui-monospace, monospace",
                    fontSize: 11,
                  }}
                  labelFormatter={(v) => `θ ≈ ${(Number(v) * 100).toFixed(1)}%`}
                />
                <Legend
                  wrapperStyle={{
                    fontSize: 10,
                    fontFamily: "var(--font-mono), monospace",
                  }}
                />
                {posteriors.map((p) => (
                  <Line
                    key={p.id}
                    type="monotone"
                    dataKey={p.id}
                    name={p.label}
                    stroke={`var(${p.strokeVar})`}
                    dot={false}
                    strokeWidth={1.25}
                    isAnimationActive={false}
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div className="flex h-full w-full items-center justify-center bg-[color:var(--surface-raised)] font-mono text-[11px] text-muted-foreground">
              {posteriors.length === 0 ? "No expected_probability series" : ""}
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
