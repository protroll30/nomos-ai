"use client";

import {
  type ColumnDef,
  flexRender,
  getCoreRowModel,
  useReactTable,
} from "@tanstack/react-table";
import { useMemo } from "react";

import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  type TraceabilityRow,
  type TrafficState,
} from "@/lib/golden-set";
import { cn } from "@/lib/utils";

function trafficBadgeClass(s: TrafficState) {
  if (s === "verified")
    return "rounded-panel border border-[color:var(--compliance-verified)]/35 bg-[color:var(--compliance-verified)]/12 text-[color:var(--compliance-verified)]";
  if (s === "caution")
    return "rounded-panel border border-[color:var(--compliance-caution)]/35 bg-[color:var(--compliance-caution)]/12 text-[color:var(--compliance-caution)]";
  return "rounded-panel border border-[color:var(--compliance-emergent)]/35 bg-[color:var(--compliance-emergent)]/12 text-[color:var(--compliance-emergent)]";
}

const columns: ColumnDef<TraceabilityRow>[] = [
  {
    accessorKey: "euArticle",
    header: "Citation",
    cell: ({ row }) => (
      <span className="font-mono text-[11px] tabular-nums">
        {row.original.euArticle}
      </span>
    ),
  },
  {
    accessorKey: "instrument",
    header: "Instrument",
    cell: ({ row }) => (
      <span className="font-mono text-[10px] text-muted-foreground">
        {row.original.instrument}
      </span>
    ),
  },
  {
    accessorKey: "annexRef",
    header: "Annex",
    cell: ({ row }) => (
      <span className="font-mono text-[11px] text-muted-foreground">
        {row.original.annexRef}
      </span>
    ),
  },
  {
    accessorKey: "obligation",
    header: "Obligation (excerpt)",
    cell: ({ row }) => (
      <span className="max-w-[18rem] whitespace-normal text-[11px] leading-snug">
        {row.original.obligation}
      </span>
    ),
  },
  {
    accessorKey: "repoPath",
    header: "Repo path",
    cell: ({ row }) => (
      <span className="max-w-[12rem] whitespace-normal font-mono text-[10px] leading-tight text-foreground">
        {row.original.repoPath}
      </span>
    ),
  },
  {
    accessorKey: "posteriorMean",
    header: "E[p]",
    cell: ({ row }) => (
      <span className="font-mono text-[11px] tabular-nums">
        {(row.original.posteriorMean * 100).toFixed(1)}%
      </span>
    ),
  },
  {
    accessorKey: "traffic",
    header: "State",
    cell: ({ row }) => (
      <Badge
        variant="outline"
        className={cn(
          "h-4 px-1.5 text-[10px] font-semibold",
          trafficBadgeClass(row.original.traffic)
        )}
      >
        {row.original.traffic}
      </Badge>
    ),
  },
];

export function TraceabilityMatrix({
  traceRows,
  sourcePath,
}: {
  traceRows: TraceabilityRow[];
  sourcePath: string;
}) {
  const data = useMemo(() => traceRows, [traceRows]);
  const table = useReactTable({
    data,
    columns,
    getCoreRowModel: getCoreRowModel(),
  });

  return (
    <Card className="flex h-full min-h-0 flex-col gap-0 rounded-panel border-hairline border-border bg-surface py-2 shadow-none ring-0">
      <CardHeader className="space-y-0 border-b border-border px-3 py-2 [.border-b]:pb-2">
        <CardTitle className="font-mono text-[11px] font-semibold tracking-wide text-muted-foreground uppercase">
          Traceability matrix · ground truth
        </CardTitle>
        <p className="pt-0.5 font-mono text-[10px] leading-tight text-muted-foreground">
          {sourcePath}
        </p>
      </CardHeader>
      <CardContent className="min-h-0 flex-1 px-0 pb-2">
        <Table>
          <TableHeader>
            {table.getHeaderGroups().map((hg) => (
              <TableRow key={hg.id} className="border-border hover:bg-transparent">
                {hg.headers.map((h) => (
                  <TableHead
                    key={h.id}
                    className="h-7 px-2 font-mono text-[10px] tracking-wide text-muted-foreground uppercase"
                  >
                    {h.isPlaceholder
                      ? null
                      : flexRender(h.column.columnDef.header, h.getContext())}
                  </TableHead>
                ))}
              </TableRow>
            ))}
          </TableHeader>
          <TableBody>
            {table.getRowModel().rows.map((row) => (
              <TableRow
                key={row.id}
                className="border-border hover:bg-muted/30"
              >
                {row.getVisibleCells().map((cell) => (
                  <TableCell
                    key={cell.id}
                    className={cn(
                      "px-2 py-1",
                      (cell.column.id === "obligation" ||
                        cell.column.id === "repoPath") &&
                        "whitespace-normal align-top"
                    )}
                  >
                    {flexRender(cell.column.columnDef.cell, cell.getContext())}
                  </TableCell>
                ))}
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  );
}
