"""Framework integrations (§5.2).

Day-1: LangChain. Other frameworks (CrewAI, OpenAI Agents SDK, AutoGen,
Anthropic SDK, MCP, OTel) plug in here following the same pattern.
"""

from latentspec.sdk.integrations import langchain  # noqa: F401

__all__ = ["langchain"]
