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

      <div className="space-y-2">
        {(invs ?? []).map((inv) => (
          <RuleRow
            key={inv.id}
            inv={inv}
            onConfirm={() => patch(inv.id, { action: "confirm" })}
            onReject={() => patch(inv.id, { action: "reject" })}
          />
        ))}
        {invs && invs.length === 0 && (
          <div className="glass p-8 text-center text-zinc-500">
            No rules match this filter. Try widening the confidence range or
            triggering a fresh mining run.
          </div>
        )}
      </div>
    </div>
  );
}

function Filter<T extends string>({
  label,
  value,
  onChange,
  options,
}: {
  label: string;
  value: T;
  onChange: (v: T) => void;
  options: { v: T; label: string }[];
}) {
  return (
    <label className="flex items-center gap-2 text-sm text-zinc-600">
      <span>{label}</span>
      <select
        className="text-sm bg-white border border-zinc-200 rounded-md px-2 py-1"
        value={value}
        onChange={(e) => onChange(e.target.value as T)}
      >
        {options.map((o) => (
          <option key={o.v} value={o.v}>
            {o.label}
          </option>
        ))}
      </select>
    </label>
  );
}

function RuleRow({
  inv,
  onConfirm,
  onReject,
}: {
  inv: Invariant;
  onConfirm: () => void;
  onReject: () => void;
}) {
  return (
    <div className="glass p-4">
      <div className="flex items-start justify-between gap-4">
        <div className="flex-1 min-w-0">
          <div className="flex flex-wrap items-center gap-2 text-xs">
            <span className={`pill pill-${inv.severity}`}>{inv.severity}</span>
            <span className="text-zinc-400">·</span>
            <span className="text-zinc-500">{TYPE_LABELS[inv.type]}</span>
            <span className="text-zinc-400">·</span>
            <span className={`pill status-${inv.status}`}>{inv.status}</span>
            <span className="text-zinc-400">·</span>
            <span className="font-mono text-zinc-500">
              conf {inv.confidence.toFixed(2)}
            </span>
            <span className="text-zinc-400">·</span>
            <span className="text-zinc-500">
              {inv.discovered_by === "both" ? "ai + stats" : inv.discovered_by}
            </span>
          </div>
          <p className="mt-2 text-zinc-900">{inv.description}</p>
          <p className="mt-1 text-xs text-zinc-500">
            evidence: {inv.evidence_count} traces · violations: {inv.violation_count}
          </p>
        </div>
        {inv.status === "pending" && (
          <div className="flex flex-col gap-1">
            <button
              type="button"
              onClick={onConfirm}
              className="px-3 py-1 text-xs bg-accent-500 hover:bg-accent-600 text-white rounded-md"
            >
              Confirm
            </button>
            <button
              type="button"
              onClick={onReject}
              className="px-3 py-1 text-xs bg-white border border-zinc-200 rounded-md hover:border-zinc-300"
            >
              Reject
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
