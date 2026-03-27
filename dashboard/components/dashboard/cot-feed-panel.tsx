"use client";

import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { mockCotLines } from "@/lib/golden-set";

export function CotFeedPanel() {
  return (
    <Card className="flex h-full min-h-0 flex-col gap-0 rounded-panel border-hairline border-border bg-surface py-2 shadow-none ring-0">
      <CardHeader className="space-y-0 border-b border-border px-3 py-2 [.border-b]:pb-2">
        <CardTitle className="font-mono text-[11px] font-semibold tracking-wide text-muted-foreground uppercase">
          CoT stream · agent steps (mock)
        </CardTitle>
      </CardHeader>
      <CardContent className="min-h-0 flex-1 px-2 pb-2 pt-0">
        <ScrollArea className="h-[11rem] rounded-panel border-hairline border-border bg-[color:var(--surface-raised)]">
          <div className="nomos-cot-feed border-0 bg-transparent p-2">
            <ul className="space-y-1.5">
              {mockCotLines.map((line) => (
                <li
                  key={line}
                  className="border-l-2 border-[color:var(--compliance-caution)]/45 pl-2 font-mono text-[11px] leading-snug text-foreground"
                >
                  {line}
                </li>
              ))}
            </ul>
          </div>
        </ScrollArea>
      </CardContent>
    </Card>
  );
}
