"""§2.2 Alert Manager — route violations to Slack / generic webhook / PagerDuty.

Each violation event is fanned out to every registered sink in parallel.
Sinks are independently rate-limited and retried with exponential backoff;
a failing sink does not block its peers. The dispatcher is the bridge from
the streaming detector to the outside world.
"""

from latentspec.alerts.dispatcher import (
    AlertDispatcher,
    AlertEvent,
    AlertSink,
    get_dispatcher,
)
from latentspec.alerts.pagerduty import PagerDutySink
from latentspec.alerts.slack import SlackWebhookSink
from latentspec.alerts.webhook import GenericWebhookSink

__all__ = [
    "AlertDispatcher",
    "AlertEvent",
    "AlertSink",
    "GenericWebhookSink",
    "PagerDutySink",
    "SlackWebhookSink",
    "get_dispatcher",
]
