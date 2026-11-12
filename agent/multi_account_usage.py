"""Read-only, profile-local allowance collection shared by explicit /usage requests."""
from __future__ import annotations

import concurrent.futures
import contextvars
import json
import math
import time
from dataclasses import dataclass, replace
from typing import Callable, Optional

import httpx

from agent.account_usage import AccountUsageSnapshot, _snapshot, fetch_account_usage


@dataclass(frozen=True)
class _Job:
    provider: str
    title: str
    fetch: Callable[[], Optional[AccountUsageSnapshot]]


def _unavailable(provider, title, reason):
    return _snapshot(provider, "usage_api", [], [], title=title, unavailable_reason=reason)


def _discover(home):
    from agent.secret_scope import get_secret_str
    # The ordinary auth loader can preserve a corrupt backup. /usage must be a
    # pure credential read even for malformed JSON, with no global-home fallback.
    try:
        with (home / "auth.json").open(encoding="utf-8-sig") as handle:
            raw = handle.read(2 * 1024 * 1024 + 1)
    except FileNotFoundError:
        raw = "{}"
    if len(raw) > 2 * 1024 * 1024:
        raise ValueError("Auth store exceeds read limit")
    store = json.loads(raw)
    if not isinstance(store, dict):
        raise ValueError("Invalid auth store")
    pool, providers = store.get("credential_pool") or {}, store.get("providers") or {}
    if not isinstance(pool, dict) or not isinstance(providers, dict):
        raise ValueError("Invalid auth sections")
    singleton = (providers.get("openai-codex") or {}).get("tokens") or {}
    if not isinstance(singleton, dict):
        raise ValueError("Invalid Codex tokens")
    return pool, singleton, get_secret_str("OPENCODE_GO_API_KEY", "").strip(), get_secret_str("OPENCODE_ZEN_API_KEY", "").strip()


def _codex_job(token, label, endpoint, identity):
    title = "Account limits — " + " ".join(str(label).split())[:80]
    identity_id = f"codex:{identity[0]}:{identity[1]}" if identity else None
    return _Job("openai-codex", title, lambda: fetch_account_usage(
        "openai-codex", base_url=endpoint, api_key=token, read_only=True,
        identity_id=identity_id, raise_errors=True))


def _codex_jobs(pool, singleton, current, api_key, base_url):
    from agent.credential_pool import _codex_principal_identity
    jobs, seen_tokens, seen_identities = [], set(), set()

    def add(token, label, endpoint=""):
        token = str(token or "").strip()
        if not token:
            return
        identity = _codex_principal_identity(token)
        if token in seen_tokens or (identity is not None and identity in seen_identities):
            return
        seen_tokens.add(token)
        if identity is not None:
            seen_identities.add(identity)
        jobs.append(_codex_job(token, label, endpoint, identity))

    rows = pool.get("openai-codex") or []
    if not isinstance(rows, list):
        raise ValueError("Invalid profile credential pool")
    singleton_token = str(singleton.get("access_token") or "").strip()
    singleton_identity = _codex_principal_identity(singleton_token)
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            continue
        token = str(row.get("access_token") or "").strip()
        identity = _codex_principal_identity(token)
        # Prefer a persisted refreshed singleton over its matching stale pool alias.
        same = singleton_token and (token == singleton_token or
                                   (identity is not None and identity == singleton_identity))
        add(singleton_token if same else token, row.get("label") or f"ChatGPT account {index}", str(row.get("base_url") or ""))
    add(singleton_token, "ChatGPT account")
    if current == "openai-codex":
        add(api_key, "Current ChatGPT account", base_url or "")
        if not jobs:
            jobs.append(_Job(current, "Account limits — ChatGPT", lambda: None))
    return jobs


def _pool_key(pool, name):
    entries = pool.get(name) or []
    if not isinstance(entries, list):
        return ""
    return next((str(row.get("access_token") or "").strip() for row in entries
                 if isinstance(row, dict) and row.get("access_token")), "")


def _other_jobs(pool, go_key, zen_key, current, api_key, base_url):
    jobs = []
    go_key = (api_key if current == "opencode-go" and api_key else _pool_key(pool, "opencode-go")) or go_key
    if go_key or current == "opencode-go":
        jobs.append(_Job("opencode-go", "Account limits — OpenCode Go", lambda: fetch_account_usage(
            "opencode-go", api_key=go_key, base_url=base_url if current == "opencode-go" else None,
            read_only=True, raise_errors=True) if go_key else None))
    if zen_key or _pool_key(pool, "opencode-zen") or current == "opencode-zen":
        jobs.append(_Job("opencode-zen", "Account limits — OpenCode Zen", lambda: _snapshot(
            "opencode-zen", "deferred", [], [], unavailable_reason="Zen credits are deferred")))
    if current and current not in {"openai-codex", "opencode-go", "opencode-zen", "nous"}:
        jobs.append(_Job(current, "Account limits", lambda: fetch_account_usage(
            current, api_key=api_key, base_url=base_url, read_only=True, raise_errors=True)))
    return jobs


def _collect(job):
    try:
        snapshot = job.fetch()
        if snapshot and (snapshot.available or snapshot.unavailable_reason):
            return replace(snapshot, title=job.title)
        reason = "Provider returned no usage windows or balance"
    except httpx.HTTPStatusError as exc:
        reason = f"Provider returned HTTP {exc.response.status_code}"
    except (httpx.HTTPError, OSError, RuntimeError, ValueError, TypeError, AttributeError, ImportError):
        # Never echo exception text: it can carry URLs, tokens or auth paths.
        reason = "Could not retrieve provider usage"
    return _unavailable(job.provider, job.title, reason)


def _run_jobs(jobs, deadline):
    from tools.daemon_pool import DaemonThreadPoolExecutor
    if not jobs:
        return []
    executor = DaemonThreadPoolExecutor(max_workers=min(8, len(jobs)))
    futures = [executor.submit(contextvars.copy_context().run, _collect, job) for job in jobs]
    try:
        concurrent.futures.wait(futures, timeout=max(0.0, deadline - time.monotonic()))
        return [future.result() if future.done() else _unavailable(job.provider, job.title, "Usage collection timed out")
                for future, job in zip(futures, jobs)]
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def fetch_usage_accounts(
    provider: Optional[str], *, base_url: Optional[str] = None,
    api_key: Optional[str] = None, timeout: float = 12.0,
) -> list[AccountUsageSnapshot]:
    """Inspect this profile's stored credentials, never select/refresh a pool entry.

    Workers inherit separate copies of the caller's home/secret context. A single
    deadline bounds discovery and collection; completed providers survive slower
    ones. Timed-out daemon work may finish its bounded HTTP read/cache write, but
    never rotates credentials or holds process shutdown open.
    """
    from agent.deadline import run_bounded_sync
    from hermes_constants import get_hermes_home
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be finite and positive")
    deadline = time.monotonic() + timeout
    current, home = str(provider or "").strip().lower(), get_hermes_home()
    try:
        discovery = run_bounded_sync(lambda: _discover(home), timeout, label="usage-discovery")
        if discovery.timed_out:
            return [_unavailable(current or "accounts", "Account limits", "Usage credential discovery timed out")]
        pool, singleton, go_key, zen_key = discovery.value
        jobs = _codex_jobs(pool, singleton, current, api_key, base_url)
        jobs += _other_jobs(pool, go_key, zen_key, current, api_key, base_url)
    except (OSError, ValueError, TypeError, AttributeError):
        return [_unavailable(current or "accounts", "Account limits", "Could not read profile credentials")]
    return _run_jobs(jobs, deadline)
