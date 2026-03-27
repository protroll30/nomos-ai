import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

export function DatasetHealthCard({
  legalChunksTotal,
}: {
  legalChunksTotal: number;
}) {
  return (
    <Card className="flex flex-col gap-0 rounded-panel border-hairline border-border bg-surface py-2 shadow-none ring-0">
      <CardHeader className="border-b border-border px-3 py-2 [.border-b]:pb-2">
        <CardTitle className="font-mono text-[11px] font-semibold tracking-wide text-muted-foreground uppercase">
          Dataset health
        </CardTitle>
      </CardHeader>
      <CardContent className="flex flex-wrap items-baseline gap-x-6 gap-y-1 px-3 py-2">
        <div className="font-mono text-[11px] text-muted-foreground uppercase">
          Table
        </div>
        <div className="font-mono text-xs text-foreground">legal_chunks</div>
        <div className="font-mono text-[11px] text-muted-foreground uppercase">
          Total rows
        </div>
        <div className="font-mono text-sm font-semibold tabular-nums text-foreground">
          {legalChunksTotal}
        </div>
      </CardContent>
    </Card>
  );
}
