"""Account limits reach the current canonical RPC without adding network to stats polls."""
from contextlib import contextmanager

import pytest


@pytest.mark.asyncio
async def test_explicit_usage_reaches_canonical_contract_after_authorization(tmp_path, monkeypatch):
    from agent import account_usage, multi_account_usage
    from gateway.session_controls import AuthorityConnection
    from tests.gateway.test_local_viewer_usage import _agent
    from tests.gateway.test_surface_admission import _authority
    from tui_gateway.contracts.registry import CANONICAL_METHODS

    async def answer(event):
        return "ok"

    authority = await _authority(tmp_path, monkeypatch, answer)
    agent = _agent()
    authority.runner._cached_agent_for = lambda route: agent
    connection = AuthorityConnection(authority, object(), {"user_id": "owner"})
    scope_active = []
    reads = []

    @contextmanager
    def scope(owner, *, hydrate_secrets=False):
        assert owner is authority
        if not hydrate_secrets:
            yield
            return
        scope_active.append(True)
        try:
            yield
        finally:
            scope_active.pop()

    def collect(provider, **kwargs):
        assert scope_active == [True]
        reads.append(provider)
        return [account_usage._snapshot("openai-codex", "test", [
            account_usage.AccountUsageWindow("Week", used_percent=25)], [], title="Account limits — Example")]

    monkeypatch.setattr("gateway.session_authorities.owner_scope", scope)
    monkeypatch.setattr(multi_account_usage, "fetch_usage_accounts", collect)
    try:
        await connection.dispatch({"id": 1, "method": "session.resume", "params": {"session_id": "s"}})
        poll = await connection.dispatch({"id": 2, "method": "session.stats", "params": {"session_id": "s"}})
        assert "result" in poll and "account_lines" not in poll["result"]
        assert reads == []
        reply = await connection.dispatch({"id": 3, "method": "session.stats", "params": {"session_id": "s", "include_accounts": True}})
        assert "result" in reply, reply
        assert any("Account limits — Example" in line for line in reply["result"]["account_lines"])
        assert reads == ["custom"]
        CANONICAL_METHODS["session.stats"].result.model_validate(reply["result"])
        denied = await connection.dispatch({"id": 4, "method": "session.stats", "params": {"session_id": "unknown", "include_accounts": True}})
        assert "error" in denied
        assert reads == ["custom"]
    finally:
        await connection.close()
