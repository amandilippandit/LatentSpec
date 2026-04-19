"""Multi-version trace handling.

Agents ship new versions all the time; tools get renamed, added, removed.
Without versioning, mining starts from scratch on every change because:
  - the closed-world repertoire fires on every new tool
  - cluster centroids stop matching
  - statistical baselines reset

The versioning module tracks `agent_versions` rows automatically as
new `version_tag` values arrive at ingest. Each version snapshots its
observed tool repertoire so:
  - downstream mining can be scoped per-version
  - the runtime checker only applies version-specific invariants to
    matching traces
  - the dashboard shows a per-version diff of behavioural change
"""

from latentspec.versioning.tracker import (
    VersionDelta,
    diff_versions,
    register_or_update_version,
    resolve_active_version,
)

__all__ = [
    "VersionDelta",
    "diff_versions",
    "register_or_update_version",
    "resolve_active_version",
]
