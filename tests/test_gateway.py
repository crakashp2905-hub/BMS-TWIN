"""Tests: the authenticated, read-only telemetry gateway and its reconnecting client.

Layers, cheapest first: token and hub units, the SSE parser, the HTTP service in-process, then
a real uvicorn server on a loopback socket with a TCP proxy that cuts, stalls and redirects live
connections, to show that a client resumes without losing or repeating a record.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import re
import threading
import time
from dataclasses import replace

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")
httpx = pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402
from hypothesis import HealthCheck, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402
from netfault import ChaosProxy, collect, wait_until  # noqa: E402
from starlette.websockets import WebSocketDisconnect  # noqa: E402

from bms.gateway import (  # noqa: E402
    SCOPE_READ,
    AuthError,
    ClientStats,
    DriveFeeder,
    FailureThrottle,
    GatewayAuthError,
    GatewayConfig,
    GatewayConnectionError,
    GatewayProtocolError,
    GatewayServer,
    TelemetryClient,
    TelemetryHub,
    TokenAuthority,
    create_gateway_app,
    generate_secret,
    main,
    parse_event_id,
    parse_sse,
)
from bms.telemetry import (  # noqa: E402
    DEFAULT_EPOCH_UNIX_S,
    TelemetryRecord,
    check_sequence,
    format_timestamp,
    json_schema,
)


def _rec(i: int, **kw) -> TelemetryRecord:
    base = dict(seq=i, t_s=float(i), timestamp=format_timestamp(DEFAULT_EPOCH_UNIX_S + i), speed_mps=10.0 + i % 7,
                soc=0.9, voltage_V=350.0, current_A=12.0, temperature_C=25.0, power_W=4200.0)
    base.update(kw)
    return TelemetryRecord(**base)


def _bearer(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture()
def authority():
    return TokenAuthority(generate_secret())


@pytest.fixture()
def hub():
    return TelemetryHub(capacity=50)


@pytest.fixture()
def app(hub, authority):
    return create_gateway_app(hub, authority, GatewayConfig(heartbeat_s=0.2, max_history=20, max_auth_failures=500,
                                                          auth_failure_window_s=0.6))


@pytest.fixture()
def http(app):
    return TestClient(app)


@pytest.fixture()
def token(authority):
    return authority.issue("dashboard", ttl_s=600)


def _events(response) -> list:
    return list(parse_sse(response.text.splitlines()))


# ── Tokens ─────────────────────────────────────────────────────────────
class TestTokens:
    def test_round_trip(self, authority):
        t = authority.issue("alice", ttl_s=120, now=1000.0)
        c = authority.verify(t, now=1060.0)
        assert c.subject == "alice" and c.scopes == (SCOPE_READ,)
        assert (c.issued_at, c.expires_at) == (1000.0, 1120.0) and c.key_id == "k1"

    def test_tokens_are_unique(self, authority):
        assert len({authority.issue("a") for _ in range(50)}) == 50

    def test_expiry_with_a_small_leeway_for_clock_skew(self, authority):
        t = authority.issue("alice", ttl_s=100, now=1000.0)
        authority.verify(t, now=1100.0 + 4.0)                                    # inside the 5 s leeway
        with pytest.raises(AuthError) as err:
            authority.verify(t, now=1100.0 + 6.0)
        assert err.value.code == "expired"

    def test_a_token_from_the_future_is_refused(self, authority):
        t = authority.issue("alice", now=5000.0)
        with pytest.raises(AuthError) as err:
            authority.verify(t, now=1000.0)
        assert err.value.code == "not_yet_valid"

    def test_scope_is_enforced(self, authority):
        t = authority.issue("alice", scopes=("something:else",))
        with pytest.raises(AuthError) as err:
            authority.verify(t)
        assert err.value.code == "scope"
        assert authority.verify(t, required_scope="something:else").subject == "alice"

    def test_a_token_signed_with_another_secret_is_refused(self, authority):
        other = TokenAuthority(generate_secret())
        with pytest.raises(AuthError) as err:
            authority.verify(other.issue("mallory"))
        assert err.value.code == "bad_signature"

    def test_revocation_by_token_and_by_id(self, authority):
        a, b = authority.issue("a"), authority.issue("b")
        authority.revoke(a)
        authority.revoke(json.loads(__import__("base64").urlsafe_b64decode(b.split(".")[2] + "==="))["jti"])
        for t in (a, b):
            with pytest.raises(AuthError) as err:
                authority.verify(t)
            assert err.value.code == "revoked"
        assert authority.verify(authority.issue("c")).subject == "c"
        with pytest.raises(ValueError):
            authority.revoke("a.b.c.d")

    def test_key_rotation_keeps_old_tokens_valid_until_their_key_is_removed(self):
        s_old, s_new = generate_secret(), generate_secret()
        before = TokenAuthority({"old": s_old})
        old_token = before.issue("alice")
        rotating = TokenAuthority({"old": s_old, "new": s_new}, active_key_id="new")
        assert rotating.verify(old_token).key_id == "old"
        fresh = rotating.issue("alice")
        assert rotating.verify(fresh).key_id == "new"
        retired = TokenAuthority({"new": s_new})
        assert retired.verify(fresh).subject == "alice"
        with pytest.raises(AuthError) as err:
            retired.verify(old_token)
        assert err.value.code == "unknown_key"

    @pytest.mark.parametrize("bad", [b"", b"short", b"x" * 31])
    def test_a_short_secret_is_refused(self, bad):
        with pytest.raises(ValueError, match="at least 32 bytes"):
            TokenAuthority(bad)

    def test_key_ids_and_active_key_are_validated(self):
        with pytest.raises(ValueError):
            TokenAuthority({"bad id!": generate_secret()})
        with pytest.raises(ValueError):
            TokenAuthority({"k1": generate_secret()}, active_key_id="nope")
        with pytest.raises(ValueError):
            TokenAuthority({})

    @pytest.mark.parametrize("kw", [dict(subject="bad subject"), dict(subject="a" * 65), dict(subject=""),
                                    dict(subject="x\ny"), dict(subject="ok", ttl_s=0.0),
                                    dict(subject="ok", ttl_s=-5.0), dict(subject="ok", ttl_s=10 * 365 * 86400.0)])
    def test_issue_validates_subject_and_lifetime(self, authority, kw):
        with pytest.raises(ValueError):
            authority.issue(**kw)

    @pytest.mark.parametrize("junk", [None, 123, "", "x", "a.b.c", "a.b.c.d.e", "v2.k1.AAAA.BBBB", "v1.k1..",
                                      "v1.k1.%%%.%%%", "v1." + "k" * 3000, b"bytes", "v1.k1.e30.e30"])
    def test_garbage_is_refused_with_an_auth_error_only(self, authority, junk):
        with pytest.raises(AuthError):
            authority.verify(junk)

    def test_the_signature_is_compared_in_constant_time(self, authority, monkeypatch):
        calls = []
        real = hmac.compare_digest
        monkeypatch.setattr(hmac, "compare_digest", lambda a, b: calls.append((len(a), len(b))) or real(a, b))
        authority.verify(authority.issue("alice"))
        assert calls == [(32, 32)]

    def test_the_payload_is_not_parsed_before_the_signature_checks_out(self, authority, monkeypatch):
        import bms.gateway as gw
        parsed = []
        real = gw.json.loads
        monkeypatch.setattr(gw.json, "loads", lambda *a, **k: parsed.append(1) or real(*a, **k))
        t = authority.issue("alice")
        head, kid, payload, sig = t.split(".")
        with pytest.raises(AuthError):
            authority.verify(f"{head}.{kid}.{payload}.{'A' * len(sig)}")
        assert parsed == []                                      # a forged token never reaches the JSON parser

    @settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(st.text(max_size=300))
    def test_arbitrary_text_never_verifies_and_never_crashes(self, text):
        auth = TokenAuthority(b"k" * 32)
        with pytest.raises(AuthError):
            auth.verify(text)

    def test_changing_any_single_character_of_a_valid_token_invalidates_it(self, authority):
        t = authority.issue("alice", ttl_s=600)
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
        for i in range(len(t)):
            if t[i] == ".":
                continue
            other = alphabet[(alphabet.index(t[i]) + 1) % 64] if t[i] in alphabet else "A"
            forged = t[:i] + other + t[i + 1:]
            with pytest.raises(AuthError):
                authority.verify(forged)

    def test_failure_throttle_blocks_after_the_limit_and_recovers(self):
        now = [0.0]
        th = FailureThrottle(max_failures=3, window_s=10.0, clock=lambda: now[0])
        for _ in range(3):
            assert th.blocked_for("ip") == 0.0
            th.record_failure("ip")
        assert th.blocked_for("ip") == pytest.approx(10.0)
        assert th.blocked_for("other") == 0.0
        now[0] = 4.0
        assert th.blocked_for("ip") == pytest.approx(6.0)
        now[0] = 11.0
        assert th.blocked_for("ip") == 0.0
        th.record_failure("ip")
        th.record_success("ip")
        assert th.blocked_for("ip") == 0.0

    def test_failure_throttle_memory_is_bounded(self):
        th = FailureThrottle(max_failures=3, window_s=100.0, max_clients=50)
        for i in range(500):
            th.record_failure(f"10.0.0.{i}")
        assert len(th._events) <= 50


# ── Hub ────────────────────────────────────────────────────────────────
class TestHub:
    def test_ids_are_contiguous_and_the_epoch_is_hex(self, hub):
        assert re.fullmatch(r"[0-9a-f]{8}", hub.epoch)
        assert [hub.publish(_rec(i)) for i in range(5)] == [0, 1, 2, 3, 4]
        assert hub.latest_id == 4 and hub.oldest_id == 0

    def test_the_ring_evicts_the_oldest_and_reports_a_gap(self):
        h = TelemetryHub(capacity=10)
        for i in range(25):
            h.publish(_rec(i))
        assert (h.oldest_id, h.latest_id) == (15, 24)
        items, gap = h.read_after(3, 100)
        assert gap and [i for i, _ in items] == list(range(15, 25))
        items, gap = h.read_after(14, 100)                       # 15 is the next one wanted: nothing lost
        assert not gap and items[0][0] == 15
        items, gap = h.read_after(20, 2)
        assert not gap and [i for i, _ in items] == [21, 22]

    def test_read_after_the_end_and_on_an_empty_hub(self, hub):
        assert hub.read_after(-1, 10) == ([], False)
        hub.publish(_rec(0))
        assert hub.read_after(0, 10) == ([], False)

    def test_snapshot_returns_the_latest_in_order(self, hub):
        for i in range(8):
            hub.publish(_rec(i))
        assert [i for i, _ in hub.snapshot(3)] == [5, 6, 7] and hub.snapshot(0) == []

    def test_a_bad_record_is_refused_before_it_can_reach_a_client(self, hub):
        with pytest.raises(Exception):
            hub.publish(replace(_rec(0), soc=7.0))
        assert hub.latest_id == -1

    def test_publishing_to_a_closed_hub_fails(self, hub):
        hub.close()
        with pytest.raises(RuntimeError):
            hub.publish(_rec(0))

    def test_capacity_and_epoch_are_validated(self):
        with pytest.raises(ValueError):
            TelemetryHub(capacity=0)
        with pytest.raises(ValueError):
            TelemetryHub(epoch="XYZ")

    def test_wait_wakes_when_another_thread_publishes(self, hub):
        async def go():
            t0 = time.monotonic()
            threading.Timer(0.1, lambda: hub.publish(_rec(0))).start()
            await hub.wait(-1, timeout=5.0)
            return time.monotonic() - t0
        assert 0.05 < asyncio.run(go()) < 2.0

    def test_wait_returns_at_once_if_there_is_already_something_newer(self, hub):
        hub.publish(_rec(0))
        assert asyncio.run(asyncio.wait_for(hub.wait(-1, timeout=5.0), 1.0)) is None

    def test_wait_times_out_and_close_wakes_waiters(self, hub):
        async def go():
            t0 = time.monotonic()
            await hub.wait(-1, timeout=0.1)
            timed_out = time.monotonic() - t0
            threading.Timer(0.1, hub.close).start()
            t1 = time.monotonic()
            await hub.wait(-1, timeout=5.0)
            return timed_out, time.monotonic() - t1
        timed_out, woken = asyncio.run(go())
        assert 0.08 < timed_out < 1.0 and woken < 2.0

    def test_event_ids_parse_strictly(self):
        assert parse_event_id("0a1b2c3d-17") == ("0a1b2c3d", 17)
        for bad in (None, "", "17", "0a1b2c3d-", "0A1B2C3D-1", "0a1b2c3d-1x", "0a1b2c3d--1", "0a1b2c3d-" + "9" * 20,
                    "zzzzzzzz-1", "0a1b2c3d-1\nid: 2"):
            assert parse_event_id(bad) is None


# ── SSE parser ─────────────────────────────────────────────────────────
class TestSSEParser:
    def test_fields_comments_and_multiline_data(self):
        text = ": a comment\nretry: 1500\nid: e-1\nevent: telemetry\ndata: line one\ndata: line two\n\n"
        (ev,) = list(parse_sse(text.splitlines()))
        assert (ev.event, ev.id, ev.retry, ev.data) == ("telemetry", "e-1", 1500, "line one\nline two")

    def test_keepalives_and_incomplete_events_produce_nothing(self):
        assert list(parse_sse([": keepalive", "", ": keepalive", ""])) == []
        assert list(parse_sse(["event: telemetry", "data: half an event"])) == []      # no terminating blank line

    def test_default_event_name_and_missing_space_after_colon(self):
        (ev,) = list(parse_sse(["data:no-space", ""]))
        assert ev.event == "message" and ev.data == "no-space"

    def test_unknown_fields_and_a_non_numeric_retry_are_ignored(self):
        (ev,) = list(parse_sse(["bogus: 1", "retry: soon", "data: x", ""]))
        assert ev.retry is None and ev.data == "x"

    def test_consecutive_events_are_separated(self):
        evs = list(parse_sse(["id: a", "data: 1", "", "id: b", "data: 2", ""]))
        assert [(e.id, e.data) for e in evs] == [("a", "1"), ("b", "2")]

    def test_an_oversized_event_is_a_protocol_error_not_an_out_of_memory(self):
        flood = (f"data: {'x' * 100_000}" for _ in range(30))
        with pytest.raises(GatewayProtocolError):
            list(parse_sse(flood))


# ── The service, in-process ────────────────────────────────────────────
class TestServiceAuthentication:
    def test_health_is_open_and_says_nothing_else(self, http):
        r = http.get("/healthz")
        assert r.status_code == 200 and r.json() == {"status": "ok"}

    @pytest.mark.parametrize("path", ["/v1/whoami", "/v1/telemetry/schema", "/v1/telemetry/latest",
                                      "/v1/telemetry/history", "/v1/telemetry/stream"])
    def test_every_data_endpoint_demands_a_token(self, http, path):
        r = http.get(path)
        assert r.status_code == 401 and r.json() == {"error": "unauthorized"}
        assert r.headers["www-authenticate"].startswith("Bearer")

    @pytest.mark.parametrize("header", ["Basic dXNlcjpwYXNz", "Bearer", "Bearer ", "bearer", "Token abc", "abc", ""])
    def test_other_authorization_forms_are_refused(self, http, header):
        assert http.get("/v1/whoami", headers={"Authorization": header}).status_code == 401

    def test_a_valid_token_is_accepted_and_whoami_leaks_nothing_secret(self, http, token):
        r = http.get("/v1/whoami", headers=_bearer(token))
        assert r.status_code == 200
        body = r.json()
        assert body["subject"] == "dashboard" and body["scopes"] == [SCOPE_READ]
        assert set(body) == {"subject", "scopes", "expires_at"}
        assert token not in r.text

    def test_every_kind_of_bad_token_gets_the_same_answer(self, http, authority):
        other = TokenAuthority(generate_secret())
        revoked = authority.issue("r")
        authority.revoke(revoked)
        bad = [other.issue("x"), authority.issue("e", ttl_s=1, now=1.0), revoked, "v1.k1.e30.e30", "nonsense",
               "v1.zz.e30.e30"]
        answers = {(r.status_code, r.text, r.headers.get("www-authenticate"))
                   for r in (http.get("/v1/whoami", headers=_bearer(t)) for t in bad)}
        assert len(answers) == 1 and next(iter(answers))[0] == 401         # no oracle for guessing why

    def test_a_token_without_the_scope_is_forbidden_not_unauthorized(self, http, authority):
        t = authority.issue("limited", scopes=("other",))
        assert http.get("/v1/whoami", headers=_bearer(t)).status_code == 403

    def test_the_token_is_never_echoed_in_an_error(self, http):
        secret_looking = "v1.k1.c2VjcmV0.c2VjcmV0"
        r = http.get("/v1/whoami", headers=_bearer(secret_looking))
        assert secret_looking not in r.text and secret_looking not in str(r.headers)

    def test_repeated_failures_are_throttled_then_forgiven(self, hub, authority, token):
        strict = TestClient(create_gateway_app(hub, authority, GatewayConfig(max_auth_failures=5,
                                                                            auth_failure_window_s=0.6)))
        for _ in range(5):
            assert strict.get("/v1/whoami", headers=_bearer("garbage")).status_code == 401
        blocked = strict.get("/v1/whoami", headers=_bearer(token))
        assert blocked.status_code == 429 and int(blocked.headers["retry-after"]) >= 1
        time.sleep(0.7)                                                      # the window is 0.6 s
        assert strict.get("/v1/whoami", headers=_bearer(token)).status_code == 200

    def test_docs_and_the_schema_endpoint_that_would_describe_write_routes_are_off(self, http):
        for path in ("/docs", "/redoc", "/openapi.json"):
            assert http.get(path).status_code == 404

    def test_the_schema_endpoint_serves_the_telemetry_json_schema(self, http, token):
        assert http.get("/v1/telemetry/schema", headers=_bearer(token)).json() == json_schema()


class TestServiceIsReadOnly:
    METHODS = ("POST", "PUT", "PATCH", "DELETE")
    PATHS = ("/healthz", "/v1/whoami", "/v1/telemetry/latest", "/v1/telemetry/history", "/v1/telemetry/stream",
             "/v1/telemetry/schema", "/v1/anything/else", "/")

    def test_no_write_method_is_accepted_on_any_path_even_with_a_valid_token(self, http, token, hub):
        for method in self.METHODS:
            for path in self.PATHS:
                r = http.request(method, path, headers=_bearer(token), json=_rec(0).to_dict())
                assert r.status_code == 405, (method, path)
                assert r.headers["allow"] == "GET, HEAD, OPTIONS" and r.json() == {"error": "read_only"}
        assert hub.latest_id == -1                                           # nothing reached the hub

    def test_a_write_is_refused_before_authentication_so_it_reveals_nothing(self, http):
        assert http.post("/v1/whoami").status_code == 405

    def test_publishing_over_http_is_impossible_not_merely_forbidden(self, app):
        methods = {m for r in app.app.routes for m in (getattr(r, "methods", None) or ())}
        assert methods <= {"GET", "HEAD"}

    def test_websockets_are_refused(self, http):
        with pytest.raises(WebSocketDisconnect) as err:
            with http.websocket_connect("/v1/telemetry/stream"):
                pass
        assert err.value.code == 1008

    def test_security_headers_are_on_every_response(self, http, token):
        for r in (http.get("/healthz"), http.get("/v1/whoami"), http.get("/v1/whoami", headers=_bearer(token)),
                  http.get("/nope"), http.post("/v1/whoami"),
                  http.get("/v1/telemetry/stream?follow=false", headers=_bearer(token))):
            assert r.headers["cache-control"] == "no-store"
            assert r.headers["x-content-type-options"] == "nosniff" and r.headers["referrer-policy"] == "no-referrer"

    def test_no_cors_headers_are_ever_added(self, http, token):
        r = http.get("/v1/whoami", headers={**_bearer(token), "Origin": "https://evil.example"})
        assert not any(k.lower().startswith("access-control") for k in r.headers)


class TestServiceData:
    def test_latest_is_404_until_something_is_published(self, http, token, hub):
        assert http.get("/v1/telemetry/latest", headers=_bearer(token)).status_code == 404
        hub.publish(_rec(7))
        body = http.get("/v1/telemetry/latest", headers=_bearer(token)).json()
        assert body["id"] == 0 and body["epoch"] == hub.epoch
        assert TelemetryRecord.from_dict(body["record"]) == _rec(7)

    def test_history_pages_and_clamps(self, http, token, hub):
        for i in range(30):
            hub.publish(_rec(i))
        r = http.get("/v1/telemetry/history", params={"after": 9, "limit": 5}, headers=_bearer(token)).json()
        assert [e["id"] for e in r["events"]] == [10, 11, 12, 13, 14] and r["latest"] == 29 and not r["gap"]
        big = http.get("/v1/telemetry/history", params={"limit": 1000}, headers=_bearer(token)).json()
        assert len(big["events"]) == 20                                          # max_history in this fixture
        for bad in ({"limit": 0}, {"after": -5}):
            assert http.get("/v1/telemetry/history", params=bad, headers=_bearer(token)).status_code == 422

    def test_history_flags_a_gap_when_the_ring_has_moved_on(self, http, token, hub):
        for i in range(120):
            hub.publish(_rec(i))
        r = http.get("/v1/telemetry/history", params={"after": 5, "limit": 5}, headers=_bearer(token)).json()
        assert r["gap"] and r["events"][0]["id"] == 70


class TestServiceStream:
    def _get(self, http, token, **kw):
        headers = {**_bearer(token), **kw.pop("headers", {})}
        return http.get("/v1/telemetry/stream", params={"follow": "false", **kw}, headers=headers)

    def test_a_new_client_gets_hello_then_the_whole_backlog_in_order(self, http, token, hub):
        for i in range(6):
            hub.publish(_rec(i))
        r = self._get(http, token)
        assert r.headers["content-type"].startswith("text/event-stream")
        evs = _events(r)
        assert evs[0].event == "hello" and evs[0].retry == 2000
        hello = json.loads(evs[0].data)
        assert hello["epoch"] == hub.epoch and hello["schema_version"] == 1
        assert (hello["oldest"], hello["latest"]) == (0, 5)
        data = [e for e in evs if e.event == "telemetry"]
        assert [e.id for e in data] == [f"{hub.epoch}-{i}" for i in range(6)]
        assert [TelemetryRecord.from_json(e.data).seq for e in data] == list(range(6))

    def test_last_event_id_resumes_with_exactly_the_missed_events(self, http, token, hub):
        for i in range(10):
            hub.publish(_rec(i))
        evs = _events(self._get(http, token, headers={"Last-Event-ID": f"{hub.epoch}-6"}))
        assert [e.id for e in evs if e.event == "telemetry"] == [f"{hub.epoch}-{i}" for i in (7, 8, 9)]
        assert not [e for e in evs if e.event == "reset"]

    def test_since_is_the_query_form_of_last_event_id(self, http, token, hub):
        for i in range(5):
            hub.publish(_rec(i))
        evs = _events(self._get(http, token, since=f"{hub.epoch}-3"))
        assert [e.id for e in evs if e.event == "telemetry"] == [f"{hub.epoch}-4"]

    def test_resuming_at_the_head_returns_nothing_new(self, http, token, hub):
        hub.publish(_rec(0))
        assert [e for e in _events(self._get(http, token, headers={"Last-Event-ID": f"{hub.epoch}-0"}))
                if e.event == "telemetry"] == []

    def test_replay_none_and_replay_n(self, http, token, hub):
        for i in range(10):
            hub.publish(_rec(i))
        none = [e for e in _events(self._get(http, token, replay="none")) if e.event == "telemetry"]
        last3 = [e.id for e in _events(self._get(http, token, replay="3")) if e.event == "telemetry"]
        assert none == [] and last3 == [f"{hub.epoch}-{i}" for i in (7, 8, 9)]

    @pytest.mark.parametrize("replay", ["x", "-1", "1.5", "9999999999", "all;drop", "", " 3"])
    def test_an_invalid_replay_is_refused(self, http, token, replay):
        assert self._get(http, token, replay=replay).status_code == 422

    def test_replay_zero_means_the_last_zero_records_which_is_none(self, http, token, hub):
        hub.publish(_rec(0))
        assert [e for e in _events(self._get(http, token, replay="0")) if e.event == "telemetry"] == []

    def test_a_resume_from_another_epoch_is_a_reset_then_a_full_replay(self, http, token, hub):
        for i in range(4):
            hub.publish(_rec(i))
        other = "deadbeef" if hub.epoch != "deadbeef" else "feedface"
        evs = _events(self._get(http, token, headers={"Last-Event-ID": f"{other}-99"}))
        resets = [json.loads(e.data) for e in evs if e.event == "reset"]
        assert resets == [{"epoch": hub.epoch, "reason": "epoch_changed"}]
        assert len([e for e in evs if e.event == "telemetry"]) == 4

    def test_a_resume_ahead_of_the_server_is_a_reset_too(self, http, token, hub):
        hub.publish(_rec(0))
        evs = _events(self._get(http, token, headers={"Last-Event-ID": f"{hub.epoch}-500"}))
        assert json.loads([e for e in evs if e.event == "reset"][0].data)["reason"] == "ahead_of_server"

    def test_falling_behind_the_ring_is_reported_as_a_reset_never_hidden(self, http, token, hub):
        for i in range(120):                                                 # capacity 50: ids 70..119 remain
            hub.publish(_rec(i))
        evs = _events(self._get(http, token, headers={"Last-Event-ID": f"{hub.epoch}-10"}))
        reset = json.loads([e for e in evs if e.event == "reset"][0].data)
        assert reset["reason"] == "buffer_overrun" and reset["oldest"] == 70
        ids = [int(e.id.split("-")[1]) for e in evs if e.event == "telemetry"]
        assert ids == list(range(70, 120))

    def test_a_malformed_last_event_id_is_ignored_like_a_fresh_connection(self, http, token, hub):
        for i in range(3):
            hub.publish(_rec(i))
        evs = _events(self._get(http, token, headers={"Last-Event-ID": "x'; DROP TABLE--"}))
        assert len([e for e in evs if e.event == "telemetry"]) == 3 and not [e for e in evs if e.event == "reset"]

    def test_a_closed_hub_ends_the_stream_with_a_closed_event(self, authority, token):
        h = TelemetryHub(capacity=10)
        h.publish(_rec(0))
        h.close()
        c = TestClient(create_gateway_app(h, authority, GatewayConfig(heartbeat_s=0.2)))
        r = c.get("/v1/telemetry/stream", headers=_bearer(token))             # follow=true, but the hub is closed
        assert [e.event for e in _events(r)] == ["hello", "telemetry", "closed"]


# ── Client against scripted servers (no network) ───────────────────────
def _sse_body(*events: str) -> bytes:
    return ("".join(events)).encode()


def _telemetry_event(epoch: str, n: int, record: TelemetryRecord | None = None) -> str:
    return f"id: {epoch}-{n}\nevent: telemetry\ndata: {(record or _rec(n)).to_json()}\n\n"


def _hello(epoch: str) -> str:
    return f'retry: 1000\nevent: hello\ndata: {{"epoch":"{epoch}","latest":0,"oldest":0,"schema_version":1}}\n\n'


def _mock_client(handler, **kw):
    kw.setdefault("sleep", lambda s: None)
    kw.setdefault("jitter", 0.0)
    return TelemetryClient("http://gw.test", "tok", transport=httpx.MockTransport(handler), **kw)


class TestClientProtocol:
    EPOCH = "0a0b0c0d"

    def test_backoff_is_exponential_and_capped(self):
        delays, calls = [], {"n": 0}

        def handler(request):
            calls["n"] += 1
            if calls["n"] <= 6:
                return httpx.Response(503)
            return httpx.Response(200, content=_sse_body(_hello(self.EPOCH), _telemetry_event(self.EPOCH, 0),
                                                         "event: closed\ndata: {}\n\n"))
        c = _mock_client(handler, sleep=delays.append, backoff_initial_s=0.5, backoff_max_s=4.0)
        assert [r.seq for r in c.stream()] == [0]
        assert delays == [0.5, 1.0, 2.0, 4.0, 4.0, 4.0] and c.stats.reconnects == 6

    def test_backoff_restarts_from_the_beginning_after_progress(self):
        # Two failures push the delay up; a connection that delivers data and then drops counts as progress,
        # so the next delay starts over instead of carrying on growing.
        plain_hello = f'event: hello\ndata: {{"epoch":"{self.EPOCH}","latest":0,"oldest":0,"schema_version":1}}\n\n'
        delays, calls = [], {"n": 0}

        def handler(request):
            calls["n"] += 1
            if calls["n"] <= 2:
                return httpx.Response(503)
            if calls["n"] == 3:
                return httpx.Response(200, content=_sse_body(plain_hello, _telemetry_event(self.EPOCH, 0)))
            return httpx.Response(200, content=_sse_body(plain_hello, "event: closed\ndata: {}\n\n"))
        c = _mock_client(handler, sleep=delays.append, backoff_initial_s=0.5, backoff_max_s=15.0)
        assert [r.seq for r in c.stream()] == [0]
        assert delays == [0.5, 1.0, 0.5]

    def test_a_new_epoch_resets_the_duplicate_filter_even_without_a_reset_event(self):
        # A restarted server should say so with a reset event, but if it does not, the client must not
        # discard the new server's low-numbered events as duplicates of the old one's.
        other = "0f0e0d0c"
        calls = {"n": 0}

        def handler(request):
            calls["n"] += 1
            if calls["n"] == 1:
                return httpx.Response(200, content=_sse_body(
                    _hello(self.EPOCH), *[_telemetry_event(self.EPOCH, n) for n in range(3)]))
            return httpx.Response(200, content=_sse_body(
                _hello(other), _telemetry_event(other, 0, _rec(100)), _telemetry_event(other, 1, _rec(101)),
                "event: closed\ndata: {}\n\n"))
        c = _mock_client(handler)
        assert [r.seq for r in c.stream()] == [0, 1, 2, 100, 101]
        assert c.stats.duplicates_dropped == 0 and c.stats.resets == 0

    def test_a_reset_means_forget_what_you_have_seen(self):
        # Within one epoch, a reset says the stream restarts here: events numbered lower than ones already
        # seen are new, not duplicates.
        calls = {"n": 0}

        def handler(request):
            calls["n"] += 1
            if calls["n"] == 1:
                return httpx.Response(200, content=_sse_body(
                    _hello(self.EPOCH), *[_telemetry_event(self.EPOCH, n) for n in range(5)]))
            return httpx.Response(200, content=_sse_body(
                _hello(self.EPOCH), 'event: reset\ndata: {"reason":"buffer_overrun"}\n\n',
                _telemetry_event(self.EPOCH, 1, _rec(201)), "event: closed\ndata: {}\n\n"))
        c = _mock_client(handler)
        assert [r.seq for r in c.stream()] == [0, 1, 2, 3, 4, 201]
        assert c.stats.resets == 1 and c.stats.duplicates_dropped == 0

    def test_a_server_retry_hint_is_honoured(self):
        delays, calls = [], {"n": 0}

        def handler(request):
            calls["n"] += 1
            if calls["n"] == 1:
                return httpx.Response(200, content=_sse_body(_hello(self.EPOCH)))     # hint: 1000 ms, then EOF
            return httpx.Response(200, content=_sse_body(_hello(self.EPOCH), "event: closed\ndata: {}\n\n"))
        c = _mock_client(handler, sleep=delays.append, backoff_initial_s=0.5)
        list(c.stream())
        assert delays and delays[0] == pytest.approx(1.0)

    def test_jitter_is_deterministic_for_a_seed_and_bounded(self):
        def run(seed):
            delays = []
            c = _mock_client(lambda req: httpx.Response(503), sleep=delays.append, jitter=0.25, seed=seed,
                             backoff_initial_s=1.0, backoff_max_s=1.0, max_consecutive_failures=20)
            with pytest.raises(GatewayConnectionError):
                list(c.stream())
            return delays
        a, b, other = run(5), run(5), run(6)
        assert a == b and a != other and all(0.75 <= d <= 1.25 for d in a)

    def test_the_attempt_limit_raises_a_connection_error(self):
        c = _mock_client(lambda req: (_ for _ in ()).throw(httpx.ConnectError("refused")), max_consecutive_failures=3)
        with pytest.raises(GatewayConnectionError, match="3 attempts"):
            list(c.stream())
        assert c.stats.reconnects == 3

    def test_an_auth_failure_is_raised_at_once_and_never_retried(self):
        seen = []

        def handler(request):
            seen.append(request)
            return httpx.Response(401)
        c = _mock_client(handler)
        with pytest.raises(GatewayAuthError):
            list(c.stream())
        assert len(seen) == 1 and c.stats.reconnects == 0
        c2 = _mock_client(lambda req: httpx.Response(403))
        with pytest.raises(GatewayAuthError):
            c2.latest()

    def test_429_with_retry_after_waits_at_least_that_long(self):
        delays, calls = [], {"n": 0}

        def handler(request):
            calls["n"] += 1
            if calls["n"] == 1:
                return httpx.Response(429, headers={"retry-after": "7"})
            return httpx.Response(200, content=_sse_body(_hello(self.EPOCH), "event: closed\ndata: {}\n\n"))
        c = _mock_client(handler, sleep=delays.append, backoff_initial_s=0.5)
        list(c.stream())
        assert delays[0] >= 7.0

    def test_resume_header_carries_the_last_event_id(self):
        sent, calls = [], {"n": 0}

        def handler(request):
            calls["n"] += 1
            sent.append(request.headers.get("last-event-id"))
            if calls["n"] == 1:
                return httpx.Response(200, content=_sse_body(_hello(self.EPOCH), _telemetry_event(self.EPOCH, 0),
                                                             _telemetry_event(self.EPOCH, 1)))
            return httpx.Response(200, content=_sse_body(_hello(self.EPOCH), _telemetry_event(self.EPOCH, 2),
                                                         "event: closed\ndata: {}\n\n"))
        c = _mock_client(handler)
        assert [r.seq for r in c.stream()] == [0, 1, 2]
        assert sent == [None, f"{self.EPOCH}-1"] and c.stats.last_event_id == f"{self.EPOCH}-2"

    def test_duplicates_and_foreign_or_invalid_events_are_counted_not_fatal(self):
        bad_record = json.dumps({**json.loads(_rec(9).to_json()), "soc": 9.0})
        body = _sse_body(
            _hello(self.EPOCH), _telemetry_event(self.EPOCH, 0), _telemetry_event(self.EPOCH, 0),    # duplicate
            _telemetry_event("ffffffff", 1),                                                        # wrong epoch
            f"id: {self.EPOCH}-2\nevent: telemetry\ndata: {bad_record}\n\n",                         # invalid record
            f"id: {self.EPOCH}-3\nevent: telemetry\ndata: not json\n\n",
            "id: junk\nevent: telemetry\ndata: {}\n\n", _telemetry_event(self.EPOCH, 4),
            "event: closed\ndata: {}\n\n")
        c = _mock_client(lambda req: httpx.Response(200, content=body))
        assert [r.seq for r in c.stream()] == [0, 4]
        s = c.stats
        assert (s.duplicates_dropped, s.invalid_dropped, s.records) == (1, 4, 2)

    def test_a_reset_is_surfaced_to_the_caller(self):
        resets = []
        body = _sse_body(_hello(self.EPOCH), 'event: reset\ndata: {"reason":"buffer_overrun"}\n\n',
                         _telemetry_event(self.EPOCH, 5), "event: closed\ndata: {}\n\n")
        c = _mock_client(lambda req: httpx.Response(200, content=body), on_reset=resets.append)
        list(c.stream())
        assert c.stats.resets == 1 and resets == [{"reason": "buffer_overrun"}]

    def test_an_expired_event_triggers_a_reconnect_with_a_fresh_token(self):
        tokens, calls = [], {"n": 0}

        def handler(request):
            calls["n"] += 1
            tokens.append(request.headers["authorization"])
            if calls["n"] == 1:
                return httpx.Response(200, content=_sse_body(_hello(self.EPOCH), "event: expired\ndata: {}\n\n"))
            return httpx.Response(200, content=_sse_body(_hello(self.EPOCH), _telemetry_event(self.EPOCH, 0),
                                                         "event: closed\ndata: {}\n\n"))
        issued = iter(["t-one", "t-two", "t-three"])
        c = TelemetryClient("http://gw.test", lambda: next(issued), transport=httpx.MockTransport(handler),
                            sleep=lambda s: None, jitter=0.0)
        assert [r.seq for r in c.stream()] == [0]
        assert tokens == ["Bearer t-one", "Bearer t-two"]

    def test_follow_false_stops_at_the_end_of_the_backlog(self):
        body = _sse_body(_hello(self.EPOCH), _telemetry_event(self.EPOCH, 0))
        c = _mock_client(lambda req: httpx.Response(200, content=body))
        assert [r.seq for r in c.stream(follow=False)] == [0] and c.stats.reconnects == 0

    def test_a_hostile_oversized_event_raises_a_protocol_error(self):
        body = ("event: telemetry\n" + "data: " + "x" * 1_200_000 + "\n\n").encode()
        c = _mock_client(lambda req: httpx.Response(200, content=body))
        with pytest.raises(GatewayProtocolError):
            list(c.stream())

    def test_the_token_never_appears_in_repr_or_exceptions(self):
        c = TelemetryClient("http://gw.test", "super-secret-token-123", transport=httpx.MockTransport(
            lambda req: httpx.Response(401)), sleep=lambda s: None)
        assert "super-secret" not in repr(c)
        with pytest.raises(GatewayAuthError) as err:
            list(c.stream())
        assert "super-secret" not in str(err.value)


# ── Real server, real sockets ──────────────────────────────────────────
@pytest.fixture()
def live(authority):
    hub = TelemetryHub(capacity=5000)
    app = create_gateway_app(hub, authority, GatewayConfig(heartbeat_s=0.15, retry_ms=10, max_streams_per_subject=3))
    server = GatewayServer(app).start()
    try:
        yield server, hub, authority
    finally:
        hub.close()
        server.stop()


class TestLiveGateway:
    def test_a_client_receives_live_records_in_order(self, live):
        server, hub, authority = live
        client = TelemetryClient(server.url, authority.issue("dash"), sleep=lambda s: None)

        def produce():
            for i in range(40):
                hub.publish(_rec(i))
                time.sleep(0.005)
            hub.close()
        threading.Thread(target=produce, daemon=True).start()
        got = collect(client)
        assert [r.seq for r in got] == list(range(40)) and client.stats.resets == 0

    def test_backlog_then_live_continues_seamlessly(self, live):
        server, hub, authority = live
        for i in range(5):
            hub.publish(_rec(i))
        client = TelemetryClient(server.url, authority.issue("dash"))
        threading.Timer(0.3, lambda: [hub.publish(_rec(i)) for i in range(5, 10)] and hub.close()).start()
        assert [r.seq for r in collect(client)] == list(range(10))

    def test_http_helpers_work_against_the_real_server(self, live):
        server, hub, authority = live
        hub.publish(_rec(3))
        client = TelemetryClient(server.url, authority.issue("dash"))
        assert client.whoami()["subject"] == "dash" and client.latest()["id"] == 0
        assert client.history()["events"][0]["record"]["seq"] == 3

    def test_a_wrong_token_is_refused_once_and_not_retried(self, live):
        server, hub, authority = live
        client = TelemetryClient(server.url, TokenAuthority(generate_secret()).issue("mallory"),
                                 sleep=lambda s: pytest.fail("must not retry"))
        with pytest.raises(GatewayAuthError):
            collect(client)
        assert client.stats.connects == 0 and client.stats.reconnects == 0

    def test_idle_streams_send_keepalives(self, live):
        server, hub, authority = live
        with httpx.Client(timeout=5) as c, c.stream("GET", server.url + "/v1/telemetry/stream",
                                                    headers=_bearer(authority.issue("dash"))) as r:
            seen = b""
            for chunk in r.iter_raw():
                seen += chunk
                if b": keepalive" in seen:
                    break
        assert b"event: hello" in seen and b": keepalive" in seen

    def test_the_stream_cap_is_enforced_and_slots_are_returned_when_clients_vanish(self, live):
        server, hub, authority = live
        tok = authority.issue("capped")

        def open_stream():
            c = httpx.Client(timeout=5)
            cm = c.stream("GET", server.url + "/v1/telemetry/stream", headers=_bearer(tok))
            return c, cm, cm.__enter__()
        opened = [open_stream() for _ in range(3)]
        assert all(r.status_code == 200 for _, _, r in opened)
        extra = httpx.get(server.url + "/v1/telemetry/stream", headers=_bearer(tok), timeout=5)
        assert extra.status_code == 429 and extra.json() == {"error": "too_many_streams"}
        assert httpx.get(server.url + "/v1/telemetry/stream", params={"follow": "false"},
                         headers=_bearer(authority.issue("someone_else")), timeout=5).status_code == 200
        for c, cm, _ in opened:                                               # vanish abruptly
            c.close()
        deadline, status = time.monotonic() + 10, None
        while time.monotonic() < deadline:
            status = httpx.get(server.url + "/v1/telemetry/stream", params={"follow": "false"},
                               headers=_bearer(tok), timeout=5).status_code
            if status == 200:
                break
            time.sleep(0.1)
        assert status == 200                                                  # the slots were freed

    def test_a_stream_ends_when_its_token_expires(self, authority):
        clock = [1000.0]
        hub = TelemetryHub(capacity=100)
        app = create_gateway_app(hub, authority, GatewayConfig(heartbeat_s=0.1), clock=lambda: clock[0])
        tok = authority.issue("short", ttl_s=30, now=1000.0)
        with GatewayServer(app) as server:
            events = []

            def read():
                with httpx.Client(timeout=10) as c, c.stream("GET", server.url + "/v1/telemetry/stream",
                                                            headers=_bearer(tok)) as r:
                    events.extend(parse_sse(r.iter_lines()))
            th = threading.Thread(target=read, daemon=True)
            th.start()
            time.sleep(0.4)
            assert th.is_alive()                                               # token still good: stream open
            clock[0] = 1100.0                                                  # well past expiry + leeway
            th.join(10)
            assert not th.is_alive() and events[-1].event == "expired"
            hub.close()

    def test_a_client_resumes_across_repeated_connection_cuts_without_loss_or_repeats(self, live):
        server, hub, authority = live
        proxy = ChaosProxy(server.port)
        proxy.cut_after_bytes = 3500                                           # ~6 events per connection
        try:
            client = TelemetryClient(f"http://127.0.0.1:{proxy.port}", authority.issue("dash"),
                                     backoff_initial_s=0.01, backoff_max_s=0.05, jitter=0.0, timeout_s=10)

            def produce():
                for i in range(150):
                    hub.publish(_rec(i))
                    time.sleep(0.004)
                hub.close()
            threading.Thread(target=produce, daemon=True).start()
            got = collect(client)
        finally:
            proxy.stop()
        assert [r.seq for r in got] == list(range(150))                        # complete, ordered, no duplicates
        assert client.stats.reconnects >= 5 and proxy.connections >= 6
        assert client.stats.resets == 0 and check_sequence(got).ok

    def test_severing_the_link_mid_stream_is_survived(self, live):
        server, hub, authority = live
        proxy = ChaosProxy(server.port)
        try:
            client = TelemetryClient(f"http://127.0.0.1:{proxy.port}", authority.issue("dash"),
                                     backoff_initial_s=0.01, jitter=0.0, timeout_s=10)

            def produce():
                wait_until(lambda: client.stats.connects >= 1)                  # never race the client's start-up
                for i in range(60):
                    hub.publish(_rec(i))
                    if i in (15, 35):
                        wait_until(lambda i=i: client.stats.records >= i + 1)   # it has everything so far
                        connects = client.stats.connects
                        proxy.sever_all()
                        wait_until(lambda: client.stats.connects > connects)    # and has reconnected
                    time.sleep(0.005)
                hub.close()
            threading.Thread(target=produce, daemon=True).start()
            got = collect(client)
        finally:
            proxy.stop()
        assert [r.seq for r in got] == list(range(60))
        assert client.stats.reconnects == 2 and client.stats.connects == 3 and client.stats.duplicates_dropped == 0

    def test_a_silent_dead_link_is_detected_by_the_read_timeout(self, live):
        server, hub, authority = live
        proxy = ChaosProxy(server.port)
        try:
            client = TelemetryClient(f"http://127.0.0.1:{proxy.port}", authority.issue("dash"),
                                     backoff_initial_s=0.01, jitter=0.0, timeout_s=0.8)      # heartbeat is 0.15 s
            hub.publish(_rec(0))

            def produce():
                wait_until(lambda: client.stats.records >= 1)                # connected and streaming
                proxy.blackhole = True                                        # link stays up but goes silent
                for i in range(1, 6):
                    hub.publish(_rec(i))
                wait_until(lambda: client.stats.reconnects >= 1)             # the read timeout noticed
                proxy.blackhole = False                                       # the network heals
                hub.publish(_rec(6))
                wait_until(lambda: client.stats.records >= 7)
                hub.close()
            threading.Thread(target=produce, daemon=True).start()
            got = collect(client)
        finally:
            proxy.stop()
        assert [r.seq for r in got] == list(range(7)) and client.stats.reconnects >= 1

    def test_a_server_restart_is_reported_as_a_reset_and_the_new_data_still_arrives(self, authority):
        hub_a = TelemetryHub(capacity=100)
        server_a = GatewayServer(create_gateway_app(hub_a, authority, GatewayConfig(heartbeat_s=0.15))).start()
        proxy = ChaosProxy(server_a.port)
        resets: list[dict] = []
        try:
            client = TelemetryClient(f"http://127.0.0.1:{proxy.port}", authority.issue("dash"),
                                     backoff_initial_s=0.01, jitter=0.0, timeout_s=10, on_reset=resets.append)
            for i in range(10):
                hub_a.publish(_rec(i))
            hub_b = TelemetryHub(capacity=100)
            server_b = GatewayServer(create_gateway_app(hub_b, authority, GatewayConfig(heartbeat_s=0.15))).start()
            for i in range(100, 108):
                hub_b.publish(_rec(i))

            def restart():
                time.sleep(0.5)
                server_a.stop()                                               # the old gateway goes away…
                proxy.upstream_port = server_b.port                           # …and a new one takes its place
                proxy.sever_all()
                time.sleep(0.6)
                hub_b.close()
            threading.Thread(target=restart, daemon=True).start()
            got = collect(client)
        finally:
            proxy.stop()
            hub_a.close()
            server_b.stop()
        assert [r.seq for r in got] == list(range(10)) + list(range(100, 108))
        assert client.stats.resets == 1 and resets[0]["reason"] == "epoch_changed"

    def test_a_client_that_connects_late_to_a_busy_hub_sees_only_what_the_ring_kept(self, authority):
        hub = TelemetryHub(capacity=20)
        for i in range(200):
            hub.publish(_rec(i))
        hub.close()
        with GatewayServer(create_gateway_app(hub, authority)) as server:
            got = collect(TelemetryClient(server.url, authority.issue("late")))
        assert [r.seq for r in got] == list(range(180, 200))

    def test_garbage_from_a_compromised_server_cannot_crash_the_client(self):
        body = _sse_body("id: x\nevent: telemetry\ndata: \x00\x01\n\n", "event: nonsense\ndata: {}\n\n",
                         ": comment\n\n", "retry: -5\n\n", "event: closed\ndata: {}\n\n")
        c = _mock_client(lambda req: httpx.Response(200, content=body))
        assert list(c.stream()) == []


# ── Feeder, server wrapper and command line ────────────────────────────
class TestFeederAndCli:
    def test_the_feeder_publishes_a_continuous_stream_and_closes_the_hub(self):
        from bms.drive_cycles import get_cycle
        hub = TelemetryHub(capacity=1000)
        feeder = DriveFeeder(hub, get_cycle("ece15"), speed=1e9, sleep=lambda s: None)
        feeder.run()
        recs = [TelemetryRecord.from_json(t) for _, t in hub.snapshot(1000)]
        assert hub.closed and feeder.published == len(recs) == 196
        assert check_sequence(recs).ok

    def test_repeating_a_cycle_renumbers_and_retimes_the_records(self):
        from bms.drive_cycles import get_cycle
        hub = TelemetryHub(capacity=1000)
        DriveFeeder(hub, get_cycle("ece15"), speed=1e9, repeat=2, sleep=lambda s: None).run()
        recs = [TelemetryRecord.from_json(t) for _, t in hub.snapshot(1000)]
        rep = check_sequence(recs)
        assert len(recs) == 392 and rep.ok and rep.max_dt_s == pytest.approx(1.0)
        assert recs[196].seq == 196 and recs[196].t_s == pytest.approx(196.0)
        assert recs[196].timestamp > recs[195].timestamp and recs[-1].t_s == pytest.approx(391.0)

    def test_the_feeder_paces_itself_against_wall_time(self):
        from bms.drive_cycles import constant_speed
        hub = TelemetryHub(capacity=100)
        waits: list[float] = []
        DriveFeeder(hub, constant_speed(50.0, 10.0), speed=10.0, sleep=waits.append).run()
        assert waits and waits[-1] == pytest.approx(10.0 / 10.0, rel=0.2) and waits == sorted(waits)

    def test_the_feeder_can_be_stopped(self):
        from bms.drive_cycles import constant_speed
        hub = TelemetryHub(capacity=100)
        feeder = DriveFeeder(hub, constant_speed(50.0, 100.0), speed=1.0, repeat=0).start()
        time.sleep(0.3)
        feeder.stop()
        assert hub.closed

    def test_cli_creates_secrets_and_tokens(self, monkeypatch, capsys):
        assert main(["secret"]) == 0
        secret = capsys.readouterr().out.strip()
        assert re.fullmatch(r"[0-9a-f]{64}", secret)
        monkeypatch.setenv("BMS_GATEWAY_SECRET", secret)
        assert main(["token", "--subject", "ops", "--ttl", "90"]) == 0
        token = capsys.readouterr().out.strip()
        claims = TokenAuthority(secret.encode()).verify(token)
        assert claims.subject == "ops" and claims.expires_at - claims.issued_at == pytest.approx(90.0)

    def test_cli_refuses_to_run_without_a_strong_secret(self, monkeypatch):
        monkeypatch.delenv("BMS_GATEWAY_SECRET", raising=False)
        with pytest.raises(SystemExit, match="BMS_GATEWAY_SECRET"):
            main(["token", "--subject", "ops"])
        monkeypatch.setenv("BMS_GATEWAY_SECRET", "too-short")
        with pytest.raises(SystemExit, match="at least 32"):
            main(["serve"])

    def test_cli_refuses_plaintext_on_a_public_address(self, monkeypatch):
        monkeypatch.setenv("BMS_GATEWAY_SECRET", "s" * 40)
        with pytest.raises(SystemExit, match="plaintext"):
            main(["serve", "--host", "0.0.0.0"])

    def test_the_gateway_server_binds_a_free_loopback_port_and_stops_cleanly(self, authority, hub):
        server = GatewayServer(create_gateway_app(hub, authority)).start()
        try:
            assert server.host == "127.0.0.1" and server.port > 0
            assert httpx.get(server.url + "/healthz", timeout=5).json() == {"status": "ok"}
        finally:
            server.stop()
        with pytest.raises(httpx.TransportError):
            httpx.get(server.url + "/healthz", timeout=1)

    def test_stats_object_defaults(self):
        assert ClientStats() == ClientStats(0, 0, 0, 0, 0, 0, None)
