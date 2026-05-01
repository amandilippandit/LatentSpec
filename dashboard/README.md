# LatentSpec Dashboard

§6.1 Core Views (Next.js 14 + React + Tailwind + Recharts).

## Views

- **`/`** — Agents list (Agent Overview cards)
- **`/agents/[id]`** — single Agent Overview (status counts + violation sparkline + quick actions)
- **`/agents/[id]/invariants`** — Invariant Explorer (filterable list with one-click confirm/reject)
- **`/traces/[id]`** — Trace Inspector (step-by-step visualization, ring-color coded by step type)

The §6.1 fourth view (Regression Report) is intentionally deferred — the
GitHub Action's PR comment is its v0 surface (see `action/`).

## Run locally

```bash
cd dashboard
npm install
npm run dev
```

The dashboard talks to the LatentSpec FastAPI through a Next.js rewrite
(`/api/* → http://localhost:8000/*`). Override with:

```bash
LATENTSPEC_API_BASE=https://api.latentspec.dev npm run dev
```

## Stack notes

- **Next.js 14 (App Router)** — server components for the agent list page,
  client components for interactive views.
- **Tailwind 3** — utility-first; severity / status pills are defined in
  `globals.css` so any view can use the same vocabulary.
- **Recharts** — sparkline on the Agent Overview. D3 will join when the
  Trace Inspector overlays the active invariant set on the step timeline.
- **SWR** — cached fetch + revalidation for the interactive views.
