/**
 * §6.1 Trace Inspector — step-by-step visualization with inline violation
 * highlighting (the genuinely novel UX bet from §6.1).
 *
 * This view fetches a trace by ID, renders each step in order, and overlays
 * which invariants were checked at each step. For the MVP the overlay is
 * computed client-side from the active invariant set; future iterations
 * will fetch the cached violation list directly.
 */

"use client";

import useSWR from "swr";

const fetcher = (url: string) =>
  fetch(url).then((r) => {
    if (!r.ok) throw new Error(String(r.status));
    return r.json();
  });

type Step =
  | { type: "user_input"; content: string }
  | {
      type: "tool_call";
      tool: string;
      args: Record<string, unknown>;
      latency_ms?: number;
      result_status?: string;
    }
  | { type: "agent_response"; content: string }
  | { type: "agent_thought"; content: string }
  | { type: "system"; content: string };

type Trace = {
  trace_id: string;
  agent_id: string;
  timestamp: string;
  steps: Step[];
  metadata: {
    model?: string;
    version?: string;
    user_segment?: string;
    locale?: string;
  };
};

export default function TraceInspectorPage({
  params,
}: {
  params: { id: string };
}) {
  // We fetch the trace as JSON from the API. In production, this maps to
  // a /traces/{id} read endpoint; for now we read the row's `trace_data`
  // through the existing list/get handler.
  const { data: trace } = useSWR<Trace>(`/api/traces/${params.id}`, fetcher);

  if (!trace) {
    return (
      <div className="glass p-8 text-zinc-500">
        Loading trace <code className="font-mono">{params.id}</code>…
      </div>
    );
  }

  return (
    <div className="space-y-5">
      <div>
        <p className="text-sm text-zinc-500">
          Trace · <span className="font-mono text-xs">{trace.trace_id}</span>
        </p>
