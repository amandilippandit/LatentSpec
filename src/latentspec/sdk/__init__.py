"""Python SDK — `pip install latentspec` (§5.1).

Three integration patterns, each escalating in coverage:

    import latentspec
    latentspec.init(api_key="ls_...", agent_id="booking-agent")  # 1. bootstrap

    @latentspec.trace_tool                                       # 2. function decorator
    def search_flights(dest: str, date: str): ...

    from latentspec.integrations import langchain                # 3. framework wrap
    traced_agent = langchain.wrap(my_agent)

`agent_id` threads through CLI / dashboard / GitHub Action so traces ingested
via the SDK key cleanly into the same DB rows.
"""

from latentspec.sdk.client import (
    LatentSpecClient,
    SDKConfig,
    flush,
    get_client,
    init,
    is_initialized,
    record_trace,
    shutdown,
)
from latentspec.sdk.decorators import (
    StepCollector,
    current_collector,
    trace,
    trace_tool,
)

__all__ = [
    "LatentSpecClient",
    "SDKConfig",
    "StepCollector",
    "current_collector",
    "flush",
    "get_client",
    "init",
    "is_initialized",
    "record_trace",
    "shutdown",
    "trace",
    "trace_tool",
]
