import base64
import json
import threading
import time

import httpx
import pytest

from agent import account_usage, multi_account_usage


@pytest.fixture
def profile(tmp_path, monkeypatch):
    home = tmp_path / "own-profile"
    home.mkdir()
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: home)
    monkeypatch.setattr("agent.secret_scope.get_secret_str", lambda *args: "")

    def write(store):
        (home / "auth.json").write_text(json.dumps(store))
    return write


def response(percent=25):
    return {"rate_limit": {"primary_window": {"used_percent": percent}}}


def test_distinct_accounts_use_explicit_credentials_read_only(profile, monkeypatch):
    profile({"credential_pool": {"openai-codex": [
        {"label": "First", "access_token": "example-first"},
        {"label": "Second", "access_token": "example-second"}]}})
    seen = []

    def get(url, headers, **kw):
        seen.append((url, headers["Authorization"]))
        return response(25 if headers["Authorization"].endswith("first") else 75)

    monkeypatch.setattr(account_usage, "_get_json", get)
    monkeypatch.setattr(account_usage, "resolve_codex_runtime_credentials", lambda **kw: pytest.fail("must not resolve/rotate"))
    result = multi_account_usage.fetch_usage_accounts("openai-codex")
    assert [s.title for s in result] == ["Account limits — First", "Account limits — Second"]
    assert [s.windows[0].used_percent for s in result] == [25, 75]
    assert {token for _, token in seen} == {"Bearer example-first", "Bearer example-second"}
    assert all(url.endswith("/usage") for url, _ in seen)  # No reset/consume requests.


def test_identity_dedup_preserves_workspace_members(profile, monkeypatch):
    def token(subject, suffix):
        claims = {"sub": subject, "https://api.openai.com/auth": {"chatgpt_account_id": "workspace"}}
        encoded = base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=")
        return f"header.{encoded}.{suffix}"
    first, rotated, other = token("one", "old"), token("one", "new"), token("two", "other")
    profile({"credential_pool": {"openai-codex": [
        {"label": "One", "access_token": first}, {"label": "Two", "access_token": other}]},
        "providers": {"openai-codex": {"tokens": {"access_token": rotated}}}})
    seen = []

    def get(url, headers, **kw):
        seen.append(headers["Authorization"].removeprefix("Bearer "))
        return response()
    monkeypatch.setattr(account_usage, "_get_json", get)
    result = multi_account_usage.fetch_usage_accounts("openai-codex", api_key=rotated)
    assert [s.title for s in result] == ["Account limits — One", "Account limits — Two"]
    assert set(seen) == {rotated, other}


def test_deadline_keeps_fast_result(profile, monkeypatch):
    profile({"credential_pool": {"openai-codex": [
        {"label": "Slow", "access_token": "slow"}, {"label": "Fast", "access_token": "fast"}]}})
    release = threading.Event()
    fast_seen = threading.Event()

    def get(url, headers, **kw):
        if headers["Authorization"].endswith("slow"):
            release.wait(8)
        else:
            fast_seen.set()
        return response()
    monkeypatch.setattr(account_usage, "_get_json", get)
    start = time.monotonic()
    try:
        result = multi_account_usage.fetch_usage_accounts(None, timeout=2)
        assert fast_seen.is_set()
        assert "timed out" in result[0].unavailable_reason.lower()
        assert result[1].windows[0].used_percent == 25
        assert time.monotonic() - start < 6
    finally:
        release.set()


def test_errors_and_empty_are_explicit(profile, monkeypatch):
    profile({"credential_pool": {"openai-codex": [
        {"label": "Rejected", "access_token": "rejected"}, {"label": "Empty", "access_token": "empty"}]}})

    def get(url, headers, **kw):
        if headers["Authorization"].endswith("rejected"):
            request = httpx.Request("GET", url)
            raise httpx.HTTPStatusError("do not echo secret", request=request, response=httpx.Response(429, request=request))
        return {}
    monkeypatch.setattr(account_usage, "_get_json", get)
    result = multi_account_usage.fetch_usage_accounts(None)
    assert "HTTP 429" in result[0].unavailable_reason
    assert "no usage" in result[1].unavailable_reason
    assert "secret" not in str(result)


def test_go_hook_and_zen_no_network(profile, monkeypatch):
    profile({"credential_pool": {"opencode-go": [{"access_token": "go-example"}],
                                 "opencode-zen": [{"access_token": "zen-example"}]}})

    class Client:
        def __init__(self, **kw):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def get(self, url, headers):
            assert url == "https://opencode.ai/zen/go/v1/usage"
            assert headers["Authorization"] == "Bearer go-example"
            return httpx.Response(200, json={"usage": {"weekly": {"percent": 30}}}, request=httpx.Request("GET", url))
    monkeypatch.setattr(httpx, "Client", Client)
    account_usage.fetch_account_usage("opencode-go", api_key="go-example", read_only=True, raise_errors=True)
    result = multi_account_usage.fetch_usage_accounts(None)
    assert [s.provider for s in result] == ["opencode-go", "opencode-zen"]
    assert result[0].available, result[0]
    assert result[0].windows[0].used_percent == 30
    assert "deferred" in result[1].unavailable_reason


def test_context_is_copied_a_b_a(tmp_path, monkeypatch):
    from contextvars import ContextVar
    home = ContextVar("usage-test-home")
    secret = ContextVar("usage-test-secret")
    monkeypatch.setattr("hermes_constants.get_hermes_home", home.get)
    monkeypatch.setattr("agent.secret_scope.get_secret_str", lambda name, default: secret.get() if name == "OPENCODE_GO_API_KEY" else "")
    observed = []

    def fetch(provider, **kwargs):
        observed.append((home.get().name, secret.get(), kwargs["api_key"]))
        return account_usage._snapshot(provider, "test", [account_usage.AccountUsageWindow("Week", used_percent=30)], [])
    monkeypatch.setattr(multi_account_usage, "fetch_account_usage", fetch)
    for name in ("a", "b", "a"):
        path = tmp_path / name
        path.mkdir(exist_ok=True)
        ht, st = home.set(path), secret.set("key-" + name)
        try:
            assert multi_account_usage.fetch_usage_accounts(None)[0].available
        finally:
            home.reset(ht)
            secret.reset(st)
    assert observed == [("a", "key-a", "key-a"), ("b", "key-b", "key-b"), ("a", "key-a", "key-a")]


def test_corrupt_store_is_not_rewritten(profile, tmp_path, monkeypatch):
    home = tmp_path / "corrupt"
    home.mkdir()
    original = "{not JSON"
    (home / "auth.json").write_text(original)
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: home)
    result = multi_account_usage.fetch_usage_accounts("openai-codex")
    assert result[0].unavailable_reason
    assert (home / "auth.json").read_text() == original
    assert [p.name for p in home.iterdir()] == ["auth.json"]
