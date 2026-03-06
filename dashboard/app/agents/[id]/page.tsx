/**
 * §6.1 Agent Overview — at-a-glance health of a single agent.
 *
 * Surfaces: total invariants by status, violation rate sparkline,
 * last mining run, quick actions (mine again, browse invariants, run check).
 */

"use client";

import Link from "next/link";
import useSWR from "swr";
import { LineChart, Line, ResponsiveContainer, YAxis } from "recharts";
import type { Agent, Invariant } from "@/lib/api";

const fetcher = (url: string) =>
  fetch(url).then((r) => {
    if (!r.ok) throw new Error(String(r.status));
    return r.json();
  });

function StatusCount({
  label,
  count,
  className,
}: {
  label: string;
  count: number;
  className?: string;
}) {
  return (
    <div className="glass p-4 flex flex-col">
      <span className="text-xs uppercase tracking-wide text-zinc-500">
        {label}
      </span>
      <span className={`text-3xl font-semibold mt-1 ${className || ""}`}>
        {count}
      </span>
    </div>
  );
}

export default function AgentOverviewPage({
  params,
}: {
  params: { id: string };
}) {
  const agentId = params.id;

  const { data: agent, error: aErr } = useSWR<Agent>(
    `/api/agents/${agentId}`,
    fetcher
  );
  const { data: invs, error: iErr, mutate: refetchInvs } = useSWR<Invariant[]>(
    `/api/invariants?agent_id=${agentId}`,
    fetcher
  );

  if (aErr) {
    return (
      <div className="glass p-6 text-center">
        Could not load agent <code className="font-mono">{agentId}</code>. Is
        the API running?
      </div>
    );
  }

  const counts = {
    active: invs?.filter((i) => i.status === "active").length ?? 0,
    pending: invs?.filter((i) => i.status === "pending").length ?? 0,
    rejected: invs?.filter((i) => i.status === "rejected").length ?? 0,
  };

  // synthetic sparkline from violation_rate per invariant — until we wire a
  // real time-series query, the spread of current violation rates makes a
  // useful at-a-glance distribution view.
  const sparkData = (invs ?? [])
    .slice(0, 30)
    .map((i, idx) => ({ idx, rate: i.violation_rate }));
