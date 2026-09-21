"""``AcpClient.shutdown()`` must reset its state even when the kill fails.

``shutdown`` awaited ``_kill_process(force=True)`` and then called
``_reset_state()`` sequentially, so any exception out of the kill skipped the
reset entirely.

``_kill_process`` has several exits: four ``run_in_executor`` awaits (child
scan, record capture, escaped-child sweep) that are not individually guarded,
``subprocess_executor()`` refusing new work once the loop is tearing down, and
``asyncio.CancelledError`` -- a ``BaseException`` -- arriving mid-await, which
is precisely what a shutdown produces.

Nothing retries. Every caller treats ``shutdown`` as terminal and drops the
client right after: ``AcpWorker`` (``knowledge/llm_pool.py``) and
``_shutdown_quietly`` (``connections/mint.py``) both ``except Exception``, log,
and set their reference to ``None``. So a skipped reset is permanent.

The effect asserted here is the one with a security shape: for the claude
backend ``_reset_state`` undoes the session's ``.claude/settings.local.json``
seed, which is what carries ``bypassPermissions`` for the live session. These
tests use a real work directory and a real file rather than asserting a mock was
called. The seed is written through ``_write_claude_local_settings`` because the
cleanup is scoped to what the session itself wrote -- a project file Crew never
touched is the user's and is left in place.
"""

from __future__ import annotations

import asyncio
import contextlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from kiro_crew.acp.client import ACP_BACKEND_CLAUDE, AcpClient


def _claude_client(tmp_path):
    """A client whose reset has one plainly observable effect on disk."""
    client = AcpClient(work_dir=tmp_path, permission_mode="bypassPermissions")
    # `_is_claude` is a read-only property over the backend seam.
    client._acp_backend = ACP_BACKEND_CLAUDE
    client._write_claude_local_settings()
    settings = tmp_path / ".claude" / "settings.local.json"
    assert settings.exists()
    # No live child: the reset's PID bookkeeping is not what these pin.
    client._process = None
    client._pid = None
    client._child_pids = {}
    return client, settings


@pytest.mark.asyncio
async def test_a_cancelled_kill_still_resets_the_client(tmp_path):
    """Cancellation is the shutdown case, and it must not skip the reset."""
    client, settings = _claude_client(tmp_path)
    client._kill_process = AsyncMock(side_effect=asyncio.CancelledError())

    with pytest.raises(asyncio.CancelledError):
        await client.shutdown()

    assert not settings.exists(), (
        "settings.local.json outlived the session it granted bypassPermissions "
        "for; no caller retries shutdown, so nothing else removes it"
    )
    assert client._session_id is None


@pytest.mark.asyncio
async def test_a_failing_kill_still_resets_the_client(tmp_path):
    """The executor refusing work during teardown looks like this."""
    client, settings = _claude_client(tmp_path)
    client._kill_process = AsyncMock(
        side_effect=RuntimeError("cannot schedule new futures after shutdown")
    )

    with pytest.raises(RuntimeError):
        await client.shutdown()

    assert not settings.exists()
    assert client._session_id is None


@pytest.mark.asyncio
async def test_a_clean_shutdown_still_resets(tmp_path):
    """Control: the path that already worked must keep working."""
    client, settings = _claude_client(tmp_path)
    client._kill_process = AsyncMock()

    await client.shutdown()

    assert not settings.exists()
    assert client._session_id is None


@pytest.mark.asyncio
async def test_ensure_ready_recycles_a_process_when_activation_drifts_on(tmp_path, monkeypatch):
    """A live process spawned NON-activated is recycled once gating activates (codex F1).

    The client's credential mask is fixed at spawn; activation is a manual keystone write with
    no watcher. So a process spawned while gating was OFF keeps full git credentials after an
    operator activates it -- an opaque subprocess could publish an unjudged commit.
    ``ensure_ready`` detects the OFF->ON drift on the warm path and kills+resets the process so
    the cold-start below respawns it under the mask. We stub the spawn work to a no-op and only
    assert the recycle fired.
    """
    from unittest.mock import AsyncMock

    client = AcpClient(work_dir=tmp_path)
    client._work_dir_ready = True
    client._process = SimpleNamespace(returncode=None)  # type: ignore[assignment]
    client._session_id = "sess-1"
    client._spawn_push_verdict_activation = False  # spawned before activation

    monkeypatch.setattr("kiro_crew.acp.client._push_verdict_masks_ssh", lambda: True)
    killed = AsyncMock()
    client._kill_process = killed  # type: ignore[method-assign]
    client._discard_claude_settings_seed = AsyncMock()  # type: ignore[method-assign]

    def _reset():
        client._process = None
        client._session_id = None

    client._reset_state = MagicMock(side_effect=_reset)  # type: ignore[method-assign]
    # After the recycle the warm-path early return is not taken; stop cold-start at the spawn
    # so the test exercises only the drift recycle, not a real process launch.
    client._spawn = AsyncMock(side_effect=RuntimeError("stop before spawn"))  # type: ignore[method-assign]

    with contextlib.suppress(Exception):
        await client.ensure_ready()

    killed.assert_awaited()  # the drifted process was recycled
    client._reset_state.assert_called()


@pytest.mark.asyncio
async def test_ensure_ready_keeps_a_process_spawned_already_activated(tmp_path, monkeypatch):
    """A process spawned WHILE activated has no OFF->ON drift; the warm path is not disturbed.

    It must not consult the activation keystone (only a non-activated spawn can drift on), and
    a live, session-bound process is reused unchanged.
    """
    from unittest.mock import AsyncMock

    client = AcpClient(work_dir=tmp_path)
    client._work_dir_ready = True
    client._process = SimpleNamespace(returncode=None)  # type: ignore[assignment]
    client._session_id = "sess-1"
    client._spawn_push_verdict_activation = True  # spawned already activated

    def _must_not_read():
        pytest.fail("activation keystone read for a process spawned already-activated")

    monkeypatch.setattr("kiro_crew.acp.client._push_verdict_masks_ssh", _must_not_read)
    killed = AsyncMock()
    client._kill_process = killed  # type: ignore[method-assign]

    await client.ensure_ready()  # warm-path early return

    killed.assert_not_awaited()


@pytest.mark.asyncio
async def test_activation_drift_refusal_fires_even_when_not_judging(tmp_path, monkeypatch):
    """The push-verdict activation-drift refusal runs for EVERY permission request.

    A shared-runtime session handle has ``_judges_permission_requests`` False, so the refusal
    must run OUTSIDE the judging branch to cover it -- a shared-runtime client's in-flight
    ``git push`` spawned before gating activated otherwise reaches a floor that never fires. The
    refusal is a security floor, not a judging-policy concern: this stands up a non-judging
    client whose process was spawned non-activated with gating now ON, feeds it a permission
    request, and asserts the call is rejected and the stale child retired.
    """
    from kiro_crew.acp.types import JsonRpcMessage

    client = AcpClient(work_dir=tmp_path)
    # A default client (empty spec deny set, backend not in the meta-identity set) does not
    # judge permission requests -- the shared-runtime shape this fix is about.
    assert client._judges_permission_requests is False
    client._spawn_push_verdict_activation = False  # spawned before activation
    client._session_id = "sess-shared"
    monkeypatch.setattr("kiro_crew.acp.client._push_verdict_masks_ssh", lambda: True)

    reject = AsyncMock()
    client.reject_tool = reject  # type: ignore[method-assign]
    client._kill_process = AsyncMock()  # type: ignore[method-assign]
    client._audit_spec_restriction = MagicMock()  # type: ignore[method-assign]

    msg = JsonRpcMessage(
        id=7,
        method="session/requestPermission",
        params={"toolCall": {"title": "shell", "toolCallId": "tc-1"}, "options": []},
    )
    await client._handle_permission(msg)

    reject.assert_awaited_once()  # the drifted call was refused, not approved
    client._kill_process.assert_awaited()  # and the stale child retired for the next-turn respawn


@pytest.mark.asyncio
async def test_activation_drift_no_refusal_when_spawned_activated_and_not_judging(
    tmp_path, monkeypatch
):
    """A non-judging client spawned WHILE activated has no drift: the floor must not fire.

    Guards the hoist from over-refusing -- the self-gate (spawn snapshot is not ``False``) must
    still short-circuit before any keystone read, so an ordinary shared-runtime permission
    request is untouched.
    """
    client = AcpClient(work_dir=tmp_path)
    assert client._judges_permission_requests is False
    client._spawn_push_verdict_activation = True  # spawned already activated -> no drift

    def _must_not_read():
        raise AssertionError("a process spawned activated must not read the activation keystone")

    monkeypatch.setattr("kiro_crew.acp.client._push_verdict_masks_ssh", _must_not_read)
    reject = AsyncMock()
    client.reject_tool = reject  # type: ignore[method-assign]
    client._kill_process = AsyncMock()  # type: ignore[method-assign]

    from kiro_crew.acp.types import AcpEvent

    event = AcpEvent(kind="permission", request_id="rid-2", tool_name="shell")
    # The floor self-gates on the spawn snapshot BEFORE any keystone read, so it returns False
    # (no refusal) and never calls ``_push_verdict_masks_ssh`` (which would raise above).
    refused = await client._refuse_push_verdict_activation_drift(event)

    assert refused is False
    reject.assert_not_awaited()
    client._kill_process.assert_not_awaited()
