"""Profile-local account lines survive CLI and the actual sidecar /usage worker."""
import contextlib
import io
import json
from datetime import datetime
from types import SimpleNamespace


def test_cli_sidecar_and_slash_share_profile_local_accounts(tmp_path, monkeypatch):
    from agent import account_usage, multi_account_usage, secret_scope
    from cli import HermesCLI
    from tui_gateway import launch_profile_policy, server, slash_worker

    a, b = tmp_path / "a", tmp_path / "b"
    for home, label in ((a, "Alpha"), (b, "Beta")):
        home.mkdir()
        (home / "auth.json").write_text(json.dumps({"credential_pool": {"openai-codex": [
            {"label": label, "access_token": "example-" + label}]}}))
        (home / ".env").write_text("OPENCODE_GO_API_KEY=example-go\n" if home == a else "")
    monkeypatch.setenv("HERMES_HOME", str(a))
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "launch-only-must-not-leak")
    monkeypatch.setattr(server, "_hermes_home", a)
    monkeypatch.setattr(launch_profile_policy, "_snapshot", None)
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", True)
    monkeypatch.setattr(account_usage, "nous_credits_lines", lambda **kw: [])
    monkeypatch.setattr(account_usage, "_get_json", lambda *a, **kw: {
        "rate_limit": {"primary_window": {"used_percent": 25}}})
    real_fetch = account_usage.fetch_account_usage

    def fetch(provider, **kwargs):
        if provider == "opencode-go":
            assert kwargs["api_key"] == "example-go"
            return account_usage._snapshot(provider, "test", [], ["Go allowance available"])
        return real_fetch(provider, **kwargs)

    monkeypatch.setattr(multi_account_usage, "fetch_account_usage", fetch)
    monkeypatch.setattr("hermes_cli.plugins.fire_pre_command_hook", lambda **kwargs: None)
    cli = HermesCLI.__new__(HermesCLI)
    cli.agent = None
    cli.provider, cli.base_url, cli.api_key = "openai-codex", None, None
    cli.model, cli.session_start = "example", datetime.now()
    cli._print_nous_credits_block = lambda: False
    outputs = []
    for home in (a, b, a):
        session = {"agent": SimpleNamespace(provider="openai-codex", base_url=None, api_key=None),
                   "profile_home": str(home), "session_key": "example", "history": [], "running": False}
        lines = server._usage_account_blocks(session)
        with server._session_profile_runtime_scope(session):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                cli._print_account_limits()
            worker_output = slash_worker._run(cli, "/usage")
        for line in lines:
            assert line in output.getvalue()
            assert line in worker_output
        outputs.append(lines)
    assert outputs[0] == outputs[2]
    assert "Alpha" in " ".join(outputs[0]) and "Beta" not in " ".join(outputs[0])
    assert "Beta" in " ".join(outputs[1]) and "Alpha" not in " ".join(outputs[1])
    assert "Go allowance" in " ".join(outputs[0]) and "Go allowance" not in " ".join(outputs[1])
