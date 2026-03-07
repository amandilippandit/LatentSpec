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

  const triggerMining = async () => {
    await fetch(`/api/agents/${agentId}/mining-runs`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ limit: 500 }),
    });
    await refetchInvs();
  };

  return (
    <div className="space-y-6">
      <div className="flex items-end justify-between">
        <div>
          <p className="text-sm text-zinc-500">
            <Link href="/" className="hover:text-zinc-900">
              Agents
            </Link>{" "}
            · {agent?.name ?? "loading…"}
          </p>
          <h1 className="text-2xl font-semibold tracking-tight">
            {agent?.name ?? "…"}
          </h1>
        </div>
        <div className="flex items-center gap-2">
          <Link
            href={`/agents/${agentId}/invariants`}
            className="px-3 py-1.5 text-sm bg-white border border-zinc-200 rounded-md hover:border-zinc-300"
          >
            Browse rules
          </Link>
          <button
            type="button"
            onClick={triggerMining}
            className="px-3 py-1.5 text-sm bg-accent-500 hover:bg-accent-600 text-white rounded-md"
          >
            Run mining
          </button>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-4">
        <StatusCount label="Active" count={counts.active} className="text-emerald-600" />
        <StatusCount label="Pending review" count={counts.pending} className="text-amber-600" />
        <StatusCount label="Rejected" count={counts.rejected} className="text-zinc-400" />
      </div>

      <div className="glass p-5">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-sm font-medium text-zinc-700">Violation rate per rule</h2>
          <span className="text-xs text-zinc-400 font-mono">
            {invs?.length ?? 0} rules
          </span>
        </div>
        <div className="h-24">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={sparkData}>
              <YAxis hide domain={[0, 1]} />
              <Line
                type="monotone"
                dataKey="rate"
                stroke="#7c3aed"
                strokeWidth={2}
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {iErr && (
        <div className="glass p-4 text-sm text-red-700 bg-red-50">
          Could not load invariants: {String(iErr)}
        </div>
      )}
    </div>
  );
}
