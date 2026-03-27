import { readFile } from "fs/promises";
import path from "path";

export type DatasetHealth = {
  legalChunksTotal: number;
};

export async function loadDatasetHealth(): Promise<DatasetHealth> {
  const filePath = path.join(
    process.cwd(),
    "..",
    "data",
    "dataset_health.json"
  );
  try {
    const raw = await readFile(filePath, "utf8");
    const parsed = JSON.parse(raw) as unknown;
    if (
      parsed &&
      typeof parsed === "object" &&
      "legal_chunks_total" in parsed &&
      typeof (parsed as { legal_chunks_total: unknown }).legal_chunks_total ===
        "number"
    ) {
      return {
        legalChunksTotal: (parsed as { legal_chunks_total: number })
          .legal_chunks_total,
      };
    }
  } catch {
    return { legalChunksTotal: 0 };
  }
  return { legalChunksTotal: 0 };
}
