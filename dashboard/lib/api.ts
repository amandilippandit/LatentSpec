/**
 * Thin client for the LatentSpec FastAPI backend.
 *
 * Routed through `/api/...` (see next.config.js rewrites) so the dashboard
 * doesn't need to know the absolute API URL — it talks to its own origin.
 */

export type Severity = "low" | "medium" | "high" | "critical";

export type InvariantStatus = "active" | "pending" | "rejected";

export type InvariantType =
  | "ordering"
  | "conditional"
  | "negative"
  | "statistical"
  | "output_format"
  | "tool_selection"
  | "state"
  | "composition";

export type Agent = {
  id: string;
  org_id: string;
  name: string;
  description: string | null;
  framework: string | null;
  created_at: string;
};

export type Invariant = {
  id: string;
  agent_id: string;
  type: InvariantType;
  description: string;
  formal_rule: string;
  confidence: number;
  severity: Severity;
  status: InvariantStatus;
  evidence_count: number;
  violation_count: number;
  violation_rate: number;
  discovered_at: string;
  last_checked_at: string | null;
  discovered_by: "statistical" | "llm" | "both";
  params: Record<string, unknown>;
};

export type TraceStep =
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

export type Trace = {
  trace_id: string;
  agent_id: string;
  timestamp: string;
  ended_at?: string | null;
  steps: TraceStep[];
  metadata: { model?: string; version?: string; user_segment?: string; locale?: string };
};

const BASE = "/api";

async function request<T>(
  path: string,
  init?: RequestInit
): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers || {}) },
    cache: "no-store",
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${res.status} ${res.statusText}: ${text}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  agents: {
    list: () => request<Agent[]>("/agents"),
    get: (id: string) => request<Agent>(`/agents/${id}`),
    create: (payload: { org_id: string; name: string; description?: string; framework?: string }) =>
      request<Agent>("/agents", { method: "POST", body: JSON.stringify(payload) }),
    triggerMining: (id: string, body: { limit?: number; version_tag?: string }) =>
      request(`/agents/${id}/mining-runs`, {
        method: "POST",
        body: JSON.stringify({ limit: 500, ...body }),
      }),
    runCheck: (
      id: string,
      body: {
        baseline_traces?: Trace[];
        candidate_traces?: Trace[];
        baseline_version_tag?: string;
        candidate_version_tag?: string;
        fail_on?: string;
        agent_name?: string;
      }
    ) =>
      request(`/agents/${id}/check`, {
        method: "POST",
        body: JSON.stringify(body),
      }),
  },
  invariants: {
    list: (agentId: string, opts?: { status?: InvariantStatus; type?: InvariantType; min_confidence?: number }) => {
      const params = new URLSearchParams({ agent_id: agentId });
      if (opts?.status) params.set("status", opts.status);
      if (opts?.type) params.set("type", opts.type);
      if (typeof opts?.min_confidence === "number") params.set("min_confidence", String(opts.min_confidence));
      return request<Invariant[]>(`/invariants?${params.toString()}`);
    },
    patch: (id: string, body: { action?: "confirm" | "reject" | "edit"; description?: string; severity?: Severity }) =>
      request<Invariant>(`/invariants/${id}`, { method: "PATCH", body: JSON.stringify(body) }),
  },
};
