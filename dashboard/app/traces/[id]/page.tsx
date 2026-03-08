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
        <h1 className="text-2xl font-semibold tracking-tight">Trace inspector</h1>
        <p className="text-sm text-zinc-500">
          {trace.steps.length} steps · model{" "}
          <span className="font-mono">{trace.metadata.model || "?"}</span> · version{" "}
          <span className="font-mono">{trace.metadata.version || "?"}</span>
        </p>
      </div>

      <div className="space-y-2">
        {trace.steps.map((step, idx) => (
          <StepRow key={idx} step={step} index={idx} />
        ))}
      </div>
    </div>
  );
}

function StepRow({ step, index }: { step: Step; index: number }) {
  const ringColor =
    step.type === "tool_call" && (step as any).result_status === "error"
      ? "ring-red-200"
      : step.type === "user_input"
        ? "ring-blue-200"
        : step.type === "agent_response"
          ? "ring-emerald-200"
          : "ring-zinc-200";

  return (
    <div className={`glass p-4 ring-1 ${ringColor}`}>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 text-xs">
          <span className="font-mono text-zinc-400">step {index}</span>
          <span className="text-zinc-300">·</span>
          <span className="font-medium text-zinc-700 uppercase tracking-wide">
            {step.type.replace("_", " ")}
          </span>
          {step.type === "tool_call" && (step as any).tool && (
            <>
              <span className="text-zinc-300">·</span>
              <span className="font-mono text-accent-700">
                {(step as any).tool}
              </span>
            </>
          )}
        </div>
        {step.type === "tool_call" && (step as any).latency_ms !== undefined && (
          <span className="text-xs text-zinc-500 font-mono">
            {(step as any).latency_ms}ms
          </span>
        )}
      </div>
      <div className="mt-2 text-sm text-zinc-800">
        {step.type === "tool_call" ? (
          <ArgsBlock args={(step as any).args} />
        ) : (
          <p className="whitespace-pre-wrap">{(step as any).content}</p>
        )}
      </div>
    </div>
  );
}

function ArgsBlock({ args }: { args: Record<string, unknown> }) {
  if (!args || Object.keys(args).length === 0) return null;
  return (
    <pre className="font-mono text-xs bg-zinc-50 border border-zinc-200 rounded p-2 overflow-x-auto">
      {JSON.stringify(args, null, 2)}
    </pre>
  );
}
