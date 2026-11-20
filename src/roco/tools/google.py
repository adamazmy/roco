"""Google Calendar + Gmail (read-only) tools.

Owner: workstream D. To implement:
- register nothing unless ``google.enabled`` is true AND a valid token exists in ~/.roco.
  When nothing is registered, the system prompt tells ROCO to say calendar/email
  isn't connected (see ``roco.brain.persona``).
- tools: ``get_events``, ``create_event``, ``get_unread_emails``, ``search_emails``,
  ``read_email``. No sending email, ever.
- ``roco auth google`` lives in :mod:`roco.integrations.google_auth`.

Keep ``googleapiclient`` imports inside functions (or let ImportError propagate from
module import) so text mode works without the ``google`` extra.
"""

from __future__ import annotations

from roco.tools import ToolContext
from roco.tools.registry import ToolRegistry


def register(registry: ToolRegistry, ctx: ToolContext) -> None:
    """Not implemented yet: registers nothing (workstream D)."""
