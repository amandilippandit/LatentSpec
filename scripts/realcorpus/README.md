# Real-corpus harness

This directory contains shape-faithful simulators for three publicly
documented agent architectures so we can run the LatentSpec pipeline
against trace populations whose structure was decided by someone other
than us.

The simulators mirror each agent's published architecture:

| Module | Models | Why |
|---|---|---|
| `autogpt_corpus.py` | AutoGPT-style autonomous loop with `think → act → observe` cycles, occasional `add_to_memory` and `evaluate_progress` calls. Variable-depth task plans. | Tests the pipeline on long, looping, branching traces — the hardest case for PrefixSpan. |
| `opendevin_corpus.py` | OpenDevin software-engineering agent with `read_file`, `write_file`, `run_tests`, `git_commit`, `web_search` tool surface. Tasks branch by error path. | Tests structured engineering workflows with high tool-name diversity (>20). |
| `babyagi_corpus.py` | BabyAGI's `task_creation → prioritization → execution → result_storage` chain. Strong recurring sub-sequences. | Tests the easy case — strong patterns the miner SHOULD recover. |

Each module exposes:

```python
def generate(n_traces: int, *, seed: int = 0) -> list[NormalizedTrace]:
    """Return n traces matching this architecture's published shape."""
```

These are NOT real production traces from those projects — none of those
projects publish trace dumps. They are shape-faithful simulators based on
the published architecture (AutoGPT v0.5 documentation, OpenDevin paper,
BabyAGI README) so the pipeline meets distributions it didn't make.

The end-to-end driver (`run_realcorpus_pipeline.py`) pushes each corpus
through `mine → calibrate → check` and prints what mining recovered vs
what the architecture says SHOULD be there.
