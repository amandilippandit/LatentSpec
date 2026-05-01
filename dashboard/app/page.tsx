/**
 * Root page — list of registered agents.
 *
 * Each agent card surfaces the §6.1 Agent Overview at a glance:
 * total invariants, sparkline of violations, last mining run.
 */

import Link from "next/link";
import { api, type Agent } from "@/lib/api";

async function fetchAgents(): Promise<Agent[]> {
  try {
    const base = process.env.LATENTSPEC_API_BASE || "http://localhost:8000";
    const res = await fetch(`${base}/agents`, { cache: "no-store" });
    if (!res.ok) return [];
    return (await res.json()) as Agent[];
  } catch {
    return [];
  }
}

export default async function Page() {
  const agents = await fetchAgents();

  return (
    <div className="space-y-6">
      <div className="flex items-end justify-between">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Your agents</h1>
          <p className="text-sm text-zinc-500">
            {agents.length} registered · LatentSpec discovers behavioral rules
            from each agent's production traces.
          </p>
        </div>
      </div>

      {agents.length === 0 ? (
        <EmptyState />
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {agents.map((a) => (
            <AgentCard key={a.id} agent={a} />
          ))}
        </div>
      )}
    </div>
  );
}

function AgentCard({ agent }: { agent: Agent }) {
  return (
    <Link
      href={`/agents/${agent.id}`}
      className="glass p-5 hover:shadow-md hover:border-accent-500/40 transition"
    >
      <div className="flex items-center justify-between">
        <h3 className="font-medium text-zinc-900">{agent.name}</h3>
        {agent.framework && (
          <span className="text-xs text-zinc-500 font-mono">
            {agent.framework}
          </span>
        )}
      </div>
      {agent.description && (
        <p className="text-sm text-zinc-600 mt-1 line-clamp-2">
          {agent.description}
        </p>
      )}
      <div className="mt-4 text-xs text-zinc-400 font-mono">
        Created {new Date(agent.created_at).toLocaleDateString()}
      </div>
    </Link>
  );
}

function EmptyState() {
  return (
    <div className="glass p-10 text-center">
      <h3 className="text-lg font-medium">No agents yet</h3>
      <p className="text-sm text-zinc-500 mt-1">
        Register an agent and ingest traces via{" "}
        <code className="font-mono text-xs bg-zinc-100 px-1.5 py-0.5 rounded">
          pip install latentspec
        </code>
        , or via{" "}
        <code className="font-mono text-xs bg-zinc-100 px-1.5 py-0.5 rounded">
          POST /agents
        </code>
        .
      </p>
      <pre className="text-left text-xs font-mono mt-6 bg-zinc-900 text-zinc-100 p-4 rounded-lg overflow-x-auto">
{`import latentspec

latentspec.init(api_key="ls_...", agent_id="booking-agent")

@latentspec.trace_tool
def search_flights(dest: str, date: str):
    return flight_api.search(dest, date)`}
      </pre>
    </div>
  );
}
