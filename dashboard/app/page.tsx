import { DashboardBento } from "@/components/dashboard/dashboard-bento";
import { loadDatasetHealth } from "@/lib/dataset-health";
import { loadGoldenSetBundle } from "@/lib/load-golden-set";

export default async function Page() {
  const [bundle, datasetHealth] = await Promise.all([
    loadGoldenSetBundle(),
    loadDatasetHealth(),
  ]);
  return <DashboardBento {...bundle} datasetHealth={datasetHealth} />;
}
