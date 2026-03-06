/**
 * §6.1 Invariant Explorer — browse and curate discovered behavioral rules.
 *
 * Filterable list by type / status / severity, with one-click confirm and
 * reject actions on pending rules. Per §6.2: the UI says "behavioral rule",
 * not "invariant" — but in code we keep the term internally.
 */

"use client";

import Link from "next/link";
import useSWR from "swr";
import { useState } from "react";
import type { Invariant, InvariantStatus, InvariantType, Severity } from "@/lib/api";

const fetcher = (url: string) =>
  fetch(url).then((r) => {
    if (!r.ok) throw new Error(String(r.status));
    return r.json();
  });

const TYPE_LABELS: Record<InvariantType, string> = {
  ordering: "Sequence",
  conditional: "Conditional",
  negative: "Forbidden action",
  statistical: "Performance",
  output_format: "Output style",
  tool_selection: "Tool routing",
  state: "State machine",
  composition: "Multi-agent",
};

export default function InvariantExplorerPage({
  params,
}: {
  params: { id: string };
}) {
  const agentId = params.id;
  const [status, setStatus] = useState<InvariantStatus | "all">("all");
  const [minConf, setMinConf] = useState(0);

  const url = `/api/invariants?agent_id=${agentId}` +
    (status !== "all" ? `&status=${status}` : "") +
    (minConf > 0 ? `&min_confidence=${minConf}` : "");
  const { data: invs, mutate } = useSWR<Invariant[]>(url, fetcher);

  const patch = async (
    id: string,
    body: { action: "confirm" | "reject" }
  ) => {
    await fetch(`/api/invariants/${id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    await mutate();
  };

  return (
    <div className="space-y-5">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm text-zinc-500">
            <Link href={`/agents/${agentId}`} className="hover:text-zinc-900">
              ← Agent
            </Link>
          </p>
          <h1 className="text-2xl font-semibold tracking-tight">
            Behavioral rules
          </h1>
          <p className="text-sm text-zinc-500">
            {invs?.length ?? 0} rules · review pending ones to activate them
          </p>
        </div>
      </div>

      <div className="glass p-4 flex flex-wrap items-center gap-4">
        <Filter
          label="Status"
          value={status}
          onChange={(v) => setStatus(v as InvariantStatus | "all")}
          options={[
            { v: "all", label: "All" },
            { v: "active", label: "Active" },
            { v: "pending", label: "Pending review" },
            { v: "rejected", label: "Rejected" },
          ]}
        />
        <label className="flex items-center gap-2 text-sm text-zinc-600">
          <span>Min confidence</span>
          <input
            type="range"
            min={0}
            max={1}
            step={0.05}
            value={minConf}
            onChange={(e) => setMinConf(Number(e.target.value))}
            className="w-32"
          />
          <span className="font-mono text-xs">{minConf.toFixed(2)}</span>
        </label>
      </div>

