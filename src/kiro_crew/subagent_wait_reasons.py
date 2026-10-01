"""The kinds of WAIT the subagent admission gate labels a queued spawn with.

Deliberately a leaf module -- it imports nothing from ``kiro_crew`` -- so a
surface that only needs to ask "is this wait a deferral?" can import it without
acquiring an edge to :mod:`kiro_crew.subagent`. The channel command layer
(:mod:`kiro_crew.messaging.commands`) is the case in point: it keeps
``kiro_crew.subagent`` duck-typed on purpose, because that module reaches
``kiro_crew.slack`` transitively. :mod:`kiro_crew.subagent` re-exports these names
so its own callers keep reading them from there.

The kinds themselves are a report of a verdict the gate already made; no gate
reads them back. ``concurrency_limit`` is the ordinary wave shape -- a slot is
taken, or the stagger tick has not elapsed -- and clears on its own within
seconds. The other four are DEFERRALS: the row is re-checked on the pump's next
eligible pass and can wait for as long as the host stays below the bar, which is
why the UI and every tool answer must not describe them as a capacity queue.
"""

from __future__ import annotations

QUEUED_REASON_CONCURRENCY_LIMIT = "concurrency_limit"
QUEUED_REASON_LOW_MEMORY = "low_memory"
QUEUED_REASON_POSTURE_CRITICAL = "posture_critical"
QUEUED_REASON_ADAPTIVE_CAP_ZERO = "adaptive_cap_zero"
#: The macOS kernel reports memory pressure (WARN or worse) while a dedicated
#: child of this gateway is running or warming. Carries no GB figures: the
#: reclaimable figure cleared the floor, so any pair of numbers would contradict
#: the verdict.
QUEUED_REASON_MEMORY_PRESSURE = "memory_pressure"

#: The kinds a caller is told ``queued`` for (rather than ``spawned``): the row
#: is accepted but may not run for a long time.
DEFERRED_QUEUED_REASONS: frozenset[str] = frozenset(
    {
        QUEUED_REASON_LOW_MEMORY,
        QUEUED_REASON_POSTURE_CRITICAL,
        QUEUED_REASON_ADAPTIVE_CAP_ZERO,
        QUEUED_REASON_MEMORY_PRESSURE,
    }
)

__all__ = [
    "DEFERRED_QUEUED_REASONS",
    "QUEUED_REASON_ADAPTIVE_CAP_ZERO",
    "QUEUED_REASON_CONCURRENCY_LIMIT",
    "QUEUED_REASON_LOW_MEMORY",
    "QUEUED_REASON_MEMORY_PRESSURE",
    "QUEUED_REASON_POSTURE_CRITICAL",
]
