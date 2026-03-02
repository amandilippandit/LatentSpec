# latentspec/action

GitHub Action that wraps the `latentspec check` CLI to run a behavioral
regression check on every PR (§4.2).

## Usage

```yaml
# .github/workflows/latentspec.yml
name: LatentSpec Behavioral Regression Check
on: [push, pull_request]
jobs:
  behavioral-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Run LatentSpec regression check
        uses: latentspec/action@v1
        with:
          api-key:    ${{ secrets.LATENTSPEC_API_KEY }}
          agent-id:   booking-agent
          baseline:   tests/fixtures/baseline_traces.json
          candidate:  tests/fixtures/candidate_traces.json
          fail-on:    critical    # block merge on critical violations
```

### Inputs

| Name        | Default                       | Purpose                                                       |
|-------------|-------------------------------|---------------------------------------------------------------|
| `api-key`   | (none)                        | LatentSpec API key (when fetching invariants from the API).   |
| `agent-id`  | (none)                        | Agent UUID. Required when `invariants` is not provided.       |
| `agent-name`| `agent`                       | Header label in the rendered PR comment.                      |
| `invariants`| (none)                        | Path to a JSON file with the invariant set (offline mode).    |
| `baseline`  | (required)                    | Path to a JSON file with baseline traces.                     |
| `candidate` | (required)                    | Path to a JSON file with candidate traces.                    |
| `fail-on`   | `critical`                    | `never` / `warn` / `any` / `high` / `critical`.               |
| `api-base`  | `https://api.latentspec.dev`  | Override for self-hosted deployments.                         |

### Outputs

| Name       | Description                                  |
|------------|----------------------------------------------|
| `report`   | The rendered §4.2 PR-comment body.           |
| `exit-code`| The severity-gated exit code (0 = pass).     |

The action also writes the report into `GITHUB_STEP_SUMMARY` so it appears
on the workflow's run page.
