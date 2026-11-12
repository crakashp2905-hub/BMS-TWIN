"""Authenticated, read-only telemetry gateway with resumable streaming.

Streams :class:`bms.telemetry.TelemetryRecord` s to network clients over HTTP, as
Server-Sent Events.  Three properties are the point:

* **Authenticated.**  Every data endpoint needs ``Authorization: Bearer <token>``.  Tokens are
  HMAC-SHA256 signed, carry a subject, scope and expiry, can be revoked, and verify against a
  rotating key set.  Signatures are compared in constant time; a secret shorter than 32 bytes
  is refused.  Repeated failures from one client are throttled.
* **Read-only.**  The service defines GET routes only, and a pure-ASGI layer rejects every
  other method (and every WebSocket) before routing, so there is no write path to find.
  Records enter through :meth:`TelemetryHub.publish` in the producing process, never over HTTP.
* **Resumable.**  Each event has an id ``<epoch>-<n>``.  A client that drops and reconnects
  sends ``Last-Event-ID`` and receives exactly the events it missed.  If the server restarted
  (new epoch) or the client fell further behind than the ring buffer holds, the server says so
  with a ``reset`` event instead of silently skipping data.

A reference :class:`TelemetryClient` implements the client half: reconnect with jittered
exponential backoff, resume, de-duplication, and refusal to retry on an authentication failure.

Not provided, and not claimed: TLS (terminate it at a reverse proxy, or give uvicorn a key and
certificate; ``serve`` refuses a non-loopback bind without it), per-token rate limits beyond the
stream cap, or protection against a compromised host.  Token revocation is in memory and does not
survive a restart; short lifetimes are the real defence.

    hub = TelemetryHub()
    authority = TokenAuthority(secrets.token_bytes(32))
    app = create_gateway_app(hub, authority)          # run with uvicorn
    token = authority.issue("dashboard", ttl_s=3600)
    for record in TelemetryClient("http://127.0.0.1:8765", token).stream():
        ...

The service is not exposed from ``bms/__init__`` so the core library never requires FastAPI;
install the extras with ``pip install '.[api]'``.
"""

# NOTE: no ``from __future__ import annotations`` — FastAPI must see real annotations.

import asyncio
import base64
import binascii
import hashlib
import hmac
import itertools
import json
import logging
import re
import secrets
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, replace
from typing import Callable, Iterable, Iterator, Optional, Union

from .telemetry import (
    SCHEMA_VERSION,
    TelemetryRecord,
    TelemetryValidationError,
    format_timestamp,
    json_schema,
    parse_timestamp,
)

log = logging.getLogger("bms.gateway")

SCOPE_READ = "telemetry:read"
_TOKEN_VERSION = "v1"
_MAX_TOKEN_CHARS = 2048
_MIN_SECRET_BYTES = 32
_MAX_TTL_S = 7 * 24 * 3600.0
_KEY_ID = re.compile(r"^[A-Za-z0-9_-]{1,16}$")
_SUBJECT = re.compile(r"^[A-Za-z0-9_.@:-]{1,64}$")
_EVENT_ID = re.compile(r"^([0-9a-f]{8})-(\d{1,15})$")
_SECURITY_HEADERS = ((b"cache-control", b"no-store"), (b"x-content-type-options", b"nosniff"),
                     (b"referrer-policy", b"no-referrer"))


# ── Tokens ─────────────────────────────────────────────────────────────
class AuthError(Exception):
    """A token was refused.  ``code`` says why internally; callers never see it over the wire."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class Claims:
    subject: str
    scopes: tuple
    issued_at: float
    expires_at: float
    token_id: str
    key_id: str


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _unb64(text: str) -> bytes:
    """Strict base64url: only the URL alphabet, and only the canonical encoding.

    A lenient decoder accepts several strings for the same bytes (the unused low bits of the last
    character are ignored), so a token could be altered without invalidating its signature.  Requiring
    the canonical form closes that: every character of a valid token matters.
    """
    if not re.fullmatch(r"[A-Za-z0-9_-]*", text):
        raise ValueError("not base64url")
    raw = base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))
    if _b64(raw) != text:
        raise ValueError("not canonical base64url")
    return raw


def generate_secret() -> bytes:
    """A fresh 32-byte signing secret."""
    return secrets.token_bytes(_MIN_SECRET_BYTES)


class TokenAuthority:
    """Issues and verifies HMAC-signed bearer tokens.

    Token format: ``v1.<key id>.<base64url payload>.<base64url HMAC-SHA256>``.  The signature covers
    the version, key id and payload, and is checked *before* the payload is parsed.  Pass a dict of
    ``{key id: secret}`` to rotate: new tokens use ``active_key_id``, old ones keep verifying until
    their key is removed.
    """

    def __init__(self, keys: Union[bytes, dict], *, active_key_id: Optional[str] = None,
                 clock: Callable[[], float] = time.time, leeway_s: float = 5.0) -> None:
        if isinstance(keys, (bytes, bytearray)):
            keys = {"k1": bytes(keys)}
        if not keys:
            raise ValueError("at least one signing key is required")
        for kid, secret in keys.items():
            if not _KEY_ID.match(kid):
                raise ValueError("key ids must be 1-16 characters of letters, digits, '_' or '-'")
            if not isinstance(secret, (bytes, bytearray)) or len(secret) < _MIN_SECRET_BYTES:
                raise ValueError(f"signing keys must be at least {_MIN_SECRET_BYTES} bytes")
        self._keys = {k: bytes(v) for k, v in keys.items()}
        self._active = active_key_id if active_key_id is not None else next(iter(self._keys))
        if self._active not in self._keys:
            raise ValueError("active_key_id is not among the keys")
        self._clock, self._leeway = clock, float(leeway_s)
        self._revoked: set = set()
        self._lock = threading.Lock()

    def _sign(self, key_id: str, payload_b64: str) -> str:
        msg = f"{_TOKEN_VERSION}.{key_id}.{payload_b64}".encode("ascii")
        return _b64(hmac.new(self._keys[key_id], msg, hashlib.sha256).digest())

    def issue(self, subject: str, *, ttl_s: float = 3600.0, scopes: Iterable[str] = (SCOPE_READ,),
              now: Optional[float] = None) -> str:
        if not _SUBJECT.match(subject):
            raise ValueError("subject must be 1-64 characters of letters, digits and _ . @ : -")
        if not 0.0 < ttl_s <= _MAX_TTL_S:
            raise ValueError(f"ttl_s must be in (0, {_MAX_TTL_S:.0f}] seconds")
        t = self._clock() if now is None else float(now)
        payload = {"sub": subject, "scp": sorted(set(scopes)), "iat": t, "exp": t + float(ttl_s),
                   "jti": secrets.token_hex(8)}
        payload_b64 = _b64(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())
        return f"{_TOKEN_VERSION}.{self._active}.{payload_b64}.{self._sign(self._active, payload_b64)}"

    def verify(self, token: str, *, required_scope: str = SCOPE_READ, now: Optional[float] = None) -> Claims:
        """Return the token's claims, or raise :class:`AuthError` with the reason."""
        if not isinstance(token, str) or not token or len(token) > _MAX_TOKEN_CHARS:
            raise AuthError("malformed")
        parts = token.split(".")
        if len(parts) != 4 or parts[0] != _TOKEN_VERSION:
            raise AuthError("malformed")
        _, key_id, payload_b64, sig = parts
        if key_id not in self._keys:
            raise AuthError("unknown_key")
        try:
            given = _unb64(sig)
        except (ValueError, binascii.Error):
            raise AuthError("malformed") from None
        expected = hmac.new(self._keys[key_id], f"{_TOKEN_VERSION}.{key_id}.{payload_b64}".encode("ascii"),
                            hashlib.sha256).digest()
        if not hmac.compare_digest(expected, given):
            raise AuthError("bad_signature")
        try:                                                  # only now is the payload trusted enough to parse
            p = json.loads(_unb64(payload_b64))
            claims = Claims(subject=str(p["sub"]), scopes=tuple(p["scp"]), issued_at=float(p["iat"]),
                            expires_at=float(p["exp"]), token_id=str(p["jti"]), key_id=key_id)
        except (ValueError, KeyError, TypeError, binascii.Error):
            raise AuthError("malformed") from None
        t = self._clock() if now is None else float(now)
        if t > claims.expires_at + self._leeway:
            raise AuthError("expired")
        if claims.issued_at > t + self._leeway:
            raise AuthError("not_yet_valid")
        with self._lock:
            if claims.token_id in self._revoked:
                raise AuthError("revoked")
        if required_scope not in claims.scopes:
            raise AuthError("scope")
        return claims

    def revoke(self, token_or_id: str) -> None:
        """Revoke a token (or its id) until the process restarts."""
        tid = token_or_id
        if token_or_id.count(".") == 3:
            try:
                tid = str(json.loads(_unb64(token_or_id.split(".")[2]))["jti"])
            except (ValueError, KeyError, TypeError, binascii.Error):
                raise ValueError("not a token") from None
        with self._lock:
            self._revoked.add(tid)


class FailureThrottle:
    """Counts authentication failures per client in a sliding window and blocks repeat offenders."""

    def __init__(self, max_failures: int = 10, window_s: float = 60.0,
                 clock: Callable[[], float] = time.monotonic, max_clients: int = 10_000) -> None:
        self.max_failures, self.window_s, self._clock, self._max_clients = max_failures, window_s, clock, max_clients
        self._events: dict = defaultdict(deque)

    def _trim(self, key: str, now: float) -> deque:
        q = self._events[key]
        while q and now - q[0] > self.window_s:
            q.popleft()
        return q

    def blocked_for(self, key: str) -> float:
        """Seconds until ``key`` may try again (0 if it may now)."""
        now = self._clock()
        q = self._trim(key, now)
        if len(q) >= self.max_failures:
            return max(0.0, self.window_s - (now - q[0]))
        return 0.0

    def record_failure(self, key: str) -> None:
        now = self._clock()
        if key not in self._events and len(self._events) >= self._max_clients:
            self._events.pop(next(iter(self._events)))        # bounded memory under a spray of clients
        self._trim(key, now).append(now)

    def record_success(self, key: str) -> None:
        self._events.pop(key, None)


# ── The hub ────────────────────────────────────────────────────────────
class TelemetryHub:
    """A thread-safe ring buffer of recent events with wake-up for async readers.

    The hub assigns each published record an event id (0, 1, 2, …) of its own, so a producer
    that restarts its record numbering cannot confuse a resuming client.  ``epoch`` identifies this
    hub instance: ids are only comparable within one epoch.
    """

    def __init__(self, capacity: int = 10_000, *, epoch: Optional[str] = None) -> None:
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self.capacity = int(capacity)
        self.epoch = epoch or secrets.token_hex(4)
        if not re.fullmatch(r"[0-9a-f]{8}", self.epoch):
            raise ValueError("epoch must be 8 lowercase hex characters")
        self._buf: deque = deque(maxlen=self.capacity)      # (event id, json text)
        self._next = 0
        self._lock = threading.Lock()
        self._waiters: list = []
        self.closed = False

    def publish(self, record: TelemetryRecord) -> int:
        """Add a record; returns its event id.  Validates the record (raises on a bad one)."""
        text = record.to_json()
        with self._lock:
            if self.closed:
                raise RuntimeError("hub is closed")
            event_id = self._next
            self._next += 1
            self._buf.append((event_id, text))
            waiters = list(self._waiters)
        self._wake(waiters)
        return event_id

    def close(self) -> None:
        with self._lock:
            self.closed = True
            waiters = list(self._waiters)
        self._wake(waiters)

    @staticmethod
    def _wake(waiters) -> None:
        for loop, ev in waiters:
            try:
                loop.call_soon_threadsafe(ev.set)
            except RuntimeError:                              # that loop is already gone
                pass

    @property
    def latest_id(self) -> int:
        with self._lock:
            return self._next - 1

    @property
    def oldest_id(self) -> Optional[int]:
        with self._lock:
            return self._buf[0][0] if self._buf else None

    def read_after(self, last_id: int, limit: int) -> tuple:
        """Events with id > ``last_id`` (up to ``limit``) and whether any were lost to the ring.

        ``gap`` is True when ``last_id + 1`` has already been evicted, so the caller knows the
        returned events do not follow on from the last one it saw.
        """
        with self._lock:
            if not self._buf:
                return [], False
            oldest = self._buf[0][0]
            gap = last_id + 1 < oldest
            first = max(last_id + 1, oldest) - oldest            # ids are contiguous, so id − oldest is the index
            return list(itertools.islice(self._buf, first, first + limit)), gap

    def snapshot(self, n: int) -> list:
        """The last ``n`` events, oldest first."""
        with self._lock:
            return list(itertools.islice(reversed(self._buf), n))[::-1] if n > 0 else []

    async def wait(self, last_id: int, timeout: float) -> None:
        """Return when an event newer than ``last_id`` exists, the hub closes, or ``timeout`` passes."""
        loop = asyncio.get_running_loop()
        ev = asyncio.Event()
        with self._lock:
            if self._next - 1 > last_id or self.closed:
                return
            waiter = (loop, ev)
            self._waiters.append(waiter)
        try:
            await asyncio.wait_for(ev.wait(), timeout)
        except asyncio.TimeoutError:
            pass
        finally:
            with self._lock:
                if waiter in self._waiters:
                    self._waiters.remove(waiter)


def parse_event_id(text: Optional[str]) -> Optional[tuple]:
    """``"<epoch>-<n>"`` → ``(epoch, n)``, or None if it is not a well-formed event id."""
    m = _EVENT_ID.match(text or "")
    return (m.group(1), int(m.group(2))) if m else None


# ── The service ────────────────────────────────────────────────────────
@dataclass(frozen=True)
class GatewayConfig:
    heartbeat_s: float = 10.0               # keep-alive comment when idle; clients treat silence as a dead link
    retry_ms: int = 2000                    # reconnect delay suggested to clients
    max_streams_per_subject: int = 4
    max_history: int = 1000
    batch: int = 256                        # events sent per read of the buffer
    max_auth_failures: int = 10
    auth_failure_window_s: float = 60.0


def _sse(event: str, data: str, event_id: Optional[str] = None, retry_ms: Optional[int] = None) -> bytes:
    lines = []
    if retry_ms is not None:
        lines.append(f"retry: {retry_ms}")
    if event_id is not None:
        lines.append(f"id: {event_id}")
    lines.append(f"event: {event}")
    lines.extend(f"data: {part}" for part in data.split("\n"))
    return ("\n".join(lines) + "\n\n").encode("utf-8")


class _ReadOnlyASGI:
    """Rejects every method but GET/HEAD/OPTIONS and every WebSocket, and adds security headers."""

    _ALLOWED = {"GET", "HEAD", "OPTIONS"}

    def __init__(self, app) -> None:
        self.app = app

    @property
    def state(self):
        """The wrapped application's state (hub, authority, config, throttle)."""
        return self.app.state

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 1008})
            return
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        if scope["method"] not in self._ALLOWED:
            body = b'{"error":"read_only"}'
            await send({"type": "http.response.start", "status": 405, "headers": [
                (b"content-type", b"application/json"), (b"content-length", str(len(body)).encode()),
                (b"allow", b"GET, HEAD, OPTIONS"), *_SECURITY_HEADERS]})
            await send({"type": "http.response.body", "body": body})
            return

        async def send_with_headers(message) -> None:
            if message["type"] == "http.response.start":
                have = {k.lower() for k, _ in message.get("headers", [])}
                message = {**message, "headers": [*message.get("headers", []),
                                                  *[h for h in _SECURITY_HEADERS if h[0] not in have]]}
            await send(message)
        await self.app(scope, receive, send_with_headers)


def create_gateway_app(hub: TelemetryHub, authority: TokenAuthority, config: Optional[GatewayConfig] = None,
                       *, clock: Callable[[], float] = time.time):
    """Build the read-only ASGI application serving ``hub`` to authenticated clients."""
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, StreamingResponse

    class _SlotResponse(StreamingResponse):
        """A stream response that gives its subject's slot back however it ends, even if it never started."""

        def __init__(self, *args, release: Callable[[], None], **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self._release = release

        async def __call__(self, scope, receive, send) -> None:
            try:
                await super().__call__(scope, receive, send)
            finally:
                self._release()

    cfg = config or GatewayConfig()
    throttle = FailureThrottle(cfg.max_auth_failures, cfg.auth_failure_window_s)
    streams: dict = defaultdict(int)
    app = FastAPI(title="BMS telemetry gateway", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.hub, app.state.authority, app.state.config, app.state.throttle = hub, authority, cfg, throttle

    class _Refuse(Exception):
        def __init__(self, status: int, error: str, headers: Optional[dict] = None) -> None:
            self.status, self.error, self.headers = status, error, headers or {}

    @app.exception_handler(_Refuse)
    async def _refuse(request: Request, exc: _Refuse):
        return JSONResponse({"error": exc.error}, status_code=exc.status, headers=exc.headers)

    def authenticate(request: Request) -> Claims:
        who = request.client.host if request.client else "unknown"
        wait = throttle.blocked_for(who)
        if wait > 0.0:
            raise _Refuse(429, "too_many_failures", {"Retry-After": str(int(wait) + 1)})
        scheme, _, token = request.headers.get("authorization", "").partition(" ")
        challenge = {"WWW-Authenticate": 'Bearer realm="bms-telemetry"'}
        if scheme.lower() != "bearer" or not token.strip():
            throttle.record_failure(who)
            raise _Refuse(401, "unauthorized", challenge)
        try:
            claims = authority.verify(token.strip(), now=clock())
        except AuthError as exc:
            throttle.record_failure(who)
            log.warning("auth refused: %s (client %s)", exc.code, who)          # never the token itself
            if exc.code == "scope":
                raise _Refuse(403, "forbidden") from None
            raise _Refuse(401, "unauthorized", {"WWW-Authenticate": 'Bearer realm="bms-telemetry", '
                                                                    'error="invalid_token"'}) from None
        throttle.record_success(who)
        return claims

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}                                  # liveness only: no version, no hub details

    @app.get("/v1/whoami")
    async def whoami(request: Request):
        c = authenticate(request)
        return {"subject": c.subject, "scopes": list(c.scopes),
                "expires_at": format_timestamp(c.expires_at)}

    @app.get("/v1/telemetry/schema")
    async def schema(request: Request):
        authenticate(request)
        return json_schema()

    @app.get("/v1/telemetry/latest")
    async def latest(request: Request):
        authenticate(request)
        snap = hub.snapshot(1)
        if not snap:
            raise _Refuse(404, "no_data")
        return {"epoch": hub.epoch, "id": snap[0][0], "record": json.loads(snap[0][1])}

    @app.get("/v1/telemetry/history")
    async def history(request: Request, after: int = -1, limit: int = 100):
        authenticate(request)
        if limit < 1 or after < -1:
            raise _Refuse(422, "invalid_parameters")
        events, gap = hub.read_after(after, min(limit, cfg.max_history))
        return {"epoch": hub.epoch, "gap": gap, "latest": hub.latest_id,
                "events": [{"id": i, "record": json.loads(t)} for i, t in events]}

    @app.get("/v1/telemetry/stream")
    async def stream(request: Request, follow: bool = True, replay: str = "all", since: Optional[str] = None):
        claims = authenticate(request)
        if streams[claims.subject] >= cfg.max_streams_per_subject:
            raise _Refuse(429, "too_many_streams", {"Retry-After": "1"})
        resume = parse_event_id(request.headers.get("last-event-id") or since)
        if replay not in ("all", "none") and not (replay.isdigit() and 0 < len(replay) <= 9):
            raise _Refuse(422, "invalid_parameters")
        streams[claims.subject] += 1

        def release(subject: str = claims.subject) -> None:
            streams[subject] -= 1

        async def events():
            reset_reason = None
            latest_id, oldest_id = hub.latest_id, hub.oldest_id
            floor = (oldest_id if oldest_id is not None else 0) - 1          # "before the oldest event"
            if resume is not None and resume[0] == hub.epoch and resume[1] <= latest_id:
                last = resume[1]
            elif resume is not None:
                reset_reason = "epoch_changed" if resume[0] != hub.epoch else "ahead_of_server"
                last = floor
            elif replay == "none":
                last = latest_id
            elif replay == "all":
                last = floor
            else:
                last = max(latest_id - int(replay), floor)
            yield _sse("hello", json.dumps({"epoch": hub.epoch, "schema_version": SCHEMA_VERSION,
                                            "oldest": oldest_id, "latest": latest_id}, sort_keys=True),
                       retry_ms=cfg.retry_ms)
            if reset_reason:
                yield _sse("reset", json.dumps({"reason": reset_reason, "epoch": hub.epoch}))
            last_sent = time.monotonic()
            while True:
                items, gap = hub.read_after(last, cfg.batch)
                if gap:
                    yield _sse("reset", json.dumps({"reason": "buffer_overrun", "epoch": hub.epoch,
                                                    "oldest": hub.oldest_id}))
                for event_id, text in items:
                    yield _sse("telemetry", text, event_id=f"{hub.epoch}-{event_id}")
                    last = event_id
                if items:
                    last_sent = time.monotonic()
                    continue
                if not follow:
                    return
                if hub.closed:
                    yield _sse("closed", "{}")
                    return
                if clock() > claims.expires_at + authority._leeway:          # a stream cannot outlive its token
                    yield _sse("expired", "{}")
                    return
                await hub.wait(last, cfg.heartbeat_s)
                if time.monotonic() - last_sent >= cfg.heartbeat_s:
                    yield b": keepalive\n\n"
                    last_sent = time.monotonic()

        return _SlotResponse(events(), release=release, media_type="text/event-stream",
                             headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})

    return _ReadOnlyASGI(app)


# ── Client ─────────────────────────────────────────────────────────────
class GatewayAuthError(Exception):
    """The gateway refused the token (401 or 403).  Retrying cannot help."""


class GatewayConnectionError(Exception):
    """The gateway stayed unreachable past the configured number of attempts."""


class GatewayProtocolError(Exception):
    """The server sent something that is not valid SSE (or an oversized event)."""


@dataclass
class SSEEvent:
    event: str = "message"
    data: str = ""
    id: Optional[str] = None
    retry: Optional[int] = None


_MAX_EVENT_BYTES = 1_048_576


def parse_sse(lines: Iterable[str]) -> Iterator[SSEEvent]:
    """Parse an SSE stream (one decoded line at a time) into events, per the HTML specification."""
    ev, data, size = SSEEvent(), [], 0
    for raw in lines:
        line = raw.rstrip("\r\n")
        size += len(line)
        if size > _MAX_EVENT_BYTES:
            raise GatewayProtocolError("event exceeds the size limit")
        if line == "":
            if data or ev.event != "message" or ev.id is not None:
                ev.data = "\n".join(data)
                yield ev
            ev, data, size = SSEEvent(), [], 0
            continue
        if line.startswith(":"):
            continue
        name, _, value = line.partition(":")
        value = value[1:] if value.startswith(" ") else value
        if name == "event":
            ev.event = value
        elif name == "data":
            data.append(value)
        elif name == "id":
            ev.id = value
        elif name == "retry" and value.isdigit():
            ev.retry = int(value)


@dataclass
class ClientStats:
    connects: int = 0
    reconnects: int = 0
    records: int = 0
    resets: int = 0
    duplicates_dropped: int = 0
    invalid_dropped: int = 0
    last_event_id: Optional[str] = None


class _Retry(Exception):
    def __init__(self, delay: float = 0.0) -> None:
        self.delay = delay


class TelemetryClient:
    """Reference client: streams records, reconnecting and resuming across drops.

    ``token`` may be a string or a zero-argument callable returning a fresh one (called on every
    connect), so a long-lived consumer can renew expiring tokens.  An authentication failure raises
    :class:`GatewayAuthError` at once.  Anything else (network error, 5xx, 429, a silent link past the
    read timeout, a clean close while following) is retried with jittered exponential backoff,
    resuming from the last event seen.  Events at or before the last one seen are dropped as duplicates;
    a ``reset`` from the server is counted and reported, never hidden.
    """

    def __init__(self, base_url: str, token: Union[str, Callable[[], str]], *, timeout_s: float = 30.0,
                 backoff_initial_s: float = 0.5, backoff_max_s: float = 15.0, backoff_factor: float = 2.0,
                 jitter: float = 0.25, max_consecutive_failures: Optional[int] = None, seed: Optional[int] = None,
                 sleep: Callable[[float], None] = time.sleep, on_reset: Optional[Callable[[dict], None]] = None,
                 verify: Union[bool, str] = True, transport=None) -> None:
        import random

        import httpx
        self._httpx = httpx
        self._base = base_url.rstrip("/")
        self._token = token
        self._timeout = timeout_s
        self._b0, self._bmax, self._bf, self._jitter = backoff_initial_s, backoff_max_s, backoff_factor, jitter
        self._max_failures = max_consecutive_failures
        self._rng = random.Random(seed)
        self._sleep = sleep
        self._on_reset = on_reset
        self._verify, self._transport = verify, transport
        self.stats = ClientStats()
        self._epoch: Optional[str] = None
        self._last_n = -1
        self._retry_hint_s: Optional[float] = None

    def __repr__(self) -> str:                                    # the token must never reach a log line
        return f"TelemetryClient({self._base!r}, token=<hidden>)"

    def _auth_header(self) -> dict:
        token = self._token() if callable(self._token) else self._token
        return {"Authorization": f"Bearer {token}"}

    def _delay(self, attempt: int) -> float:
        base = self._retry_hint_s if self._retry_hint_s is not None else self._b0
        raw = min(self._bmax, base * self._bf ** (attempt - 1))
        return max(0.0, raw * (1.0 + self._jitter * (2.0 * self._rng.random() - 1.0)))

    def _client(self):
        return self._httpx.Client(timeout=self._timeout, verify=self._verify, transport=self._transport)

    def _get_json(self, path: str, **params) -> dict:
        with self._client() as c:
            r = c.get(self._base + path, headers=self._auth_header(), params=params or None)
        if r.status_code in (401, 403):
            raise GatewayAuthError(f"gateway refused the token (HTTP {r.status_code})")
        r.raise_for_status()
        return r.json()

    def latest(self) -> dict:
        return self._get_json("/v1/telemetry/latest")

    def history(self, after: int = -1, limit: int = 100) -> dict:
        return self._get_json("/v1/telemetry/history", after=after, limit=limit)

    def whoami(self) -> dict:
        return self._get_json("/v1/whoami")

    def stream(self, *, follow: bool = True, replay: str = "all") -> Iterator[TelemetryRecord]:
        """Yield records until the server closes the stream (or, with ``follow=False``, the backlog ends)."""
        attempt = 0
        while True:
            try:
                got_event = False
                headers = {**self._auth_header(), "Accept": "text/event-stream"}
                if self.stats.last_event_id:
                    headers["Last-Event-ID"] = self.stats.last_event_id
                with self._client() as c, c.stream("GET", self._base + "/v1/telemetry/stream", headers=headers,
                                                  params={"follow": str(follow).lower(), "replay": replay}) as r:
                    if r.status_code in (401, 403):
                        raise GatewayAuthError(f"gateway refused the token (HTTP {r.status_code})")
                    if r.status_code in (429, 503):
                        raise _Retry(float(r.headers.get("retry-after", "0") or 0))
                    if r.status_code != 200:
                        raise _Retry()
                    self.stats.connects += 1
                    for event in parse_sse(r.iter_lines()):
                        got_event = True
                        if event.retry is not None:
                            self._retry_hint_s = min(60.0, max(0.1, event.retry / 1000.0))
                        done = yield from self._handle(event)
                        if done:
                            return
                    if not follow:
                        return
                    raise _Retry()                                  # the server closed a live stream: reconnect
            except GatewayAuthError:
                raise
            except (self._httpx.TransportError, _Retry) as exc:
                attempt = 1 if got_event else attempt + 1           # progress resets the backoff
                if self._max_failures is not None and attempt > self._max_failures:
                    raise GatewayConnectionError(f"gateway unreachable after {self._max_failures} attempts") from exc
                self.stats.reconnects += 1
                delay = max(self._delay(attempt), exc.delay if isinstance(exc, _Retry) else 0.0)
                self._sleep(delay)

    def _handle(self, event: SSEEvent):
        """Process one event; returns True when the stream is finished.  A generator: yields records."""
        if event.event == "hello":
            try:
                epoch = json.loads(event.data).get("epoch")
            except (ValueError, AttributeError):
                raise GatewayProtocolError("malformed hello") from None
            if self._epoch is not None and epoch != self._epoch:
                self._epoch, self._last_n = epoch, -1
            self._epoch = epoch
            return False
        if event.event == "reset":
            self.stats.resets += 1
            self._last_n = -1                                       # what follows is a fresh start of the buffer
            payload = {}
            try:
                payload = json.loads(event.data)
            except ValueError:
                pass
            if self._on_reset:
                self._on_reset(payload)
            return False
        if event.event in ("closed",):
            return True
        if event.event == "expired":
            raise _Retry()
        if event.event != "telemetry":
            return False
        parsed = parse_event_id(event.id)
        if parsed is None or parsed[0] != self._epoch:
            self.stats.invalid_dropped += 1
            return False
        if parsed[1] <= self._last_n:
            self.stats.duplicates_dropped += 1
            return False
        try:
            record = TelemetryRecord.from_json(event.data)
        except TelemetryValidationError:
            self.stats.invalid_dropped += 1
            return False
        self._last_n = parsed[1]
        self.stats.last_event_id = event.id
        self.stats.records += 1
        yield record
        return False


# ── Producing and serving ──────────────────────────────────────────────
class DriveFeeder:
    """Feeds a hub from a drive-cycle simulation in a background thread, paced against wall time.

    ``speed`` is simulated seconds per wall second (60 = a minute of driving per second).  With
    ``repeat`` > 1 the cycle is re-driven (``0`` = until stopped); each pass restarts from the
    configured initial SOC, and records are re-numbered and re-timed so the stream is continuous.
    """

    def __init__(self, hub: TelemetryHub, cycle, config=None, *, speed: float = 1.0, repeat: int = 1,
                 telemetry_dt_s: float = 1.0, sleep: Callable[[float], None] = time.sleep) -> None:
        self.hub, self.cycle, self.config = hub, cycle, config
        self.speed, self.repeat, self.dt = float(speed), int(repeat), float(telemetry_dt_s)
        self._sleep = sleep
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.published = 0

    def _records(self) -> Iterator[TelemetryRecord]:
        from .ev_drive import EVConfig, EVSimulator
        seq_off, t_off, n = 0, 0.0, 0
        while not self._stop.is_set() and (self.repeat == 0 or n < self.repeat):
            last = None
            for rec in EVSimulator(self.config or EVConfig()).stream(self.cycle, telemetry_dt_s=self.dt):
                last = replace(rec, seq=rec.seq + seq_off, t_s=rec.t_s + t_off,
                               timestamp=format_timestamp(parse_timestamp(rec.timestamp) + t_off))
                yield last
            if last is not None:
                seq_off, t_off = last.seq + 1, last.t_s + self.dt
            n += 1

    def run(self) -> None:
        start = time.monotonic()
        t0 = None
        for rec in self._records():
            if self._stop.is_set():
                break
            t0 = rec.t_s if t0 is None else t0
            wait = (rec.t_s - t0) / self.speed - (time.monotonic() - start)
            if wait > 0:
                self._sleep(wait)
            try:
                self.hub.publish(rec)
            except RuntimeError:                                  # the hub was closed under us
                break
            self.published += 1
        self.hub.close()

    def start(self) -> "DriveFeeder":
        self._thread = threading.Thread(target=self.run, name="drive-feeder", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)


class GatewayServer:
    """Runs the gateway app under uvicorn in a background thread (for tests, demos and embedding)."""

    def __init__(self, app, host: str = "127.0.0.1", port: int = 0, **uvicorn_kwargs) -> None:
        import uvicorn
        uvicorn_kwargs.setdefault("timeout_graceful_shutdown", 2)       # open event streams must not stall stop()
        self._server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level="warning",
                                                     **uvicorn_kwargs))
        self._thread = threading.Thread(target=self._server.run, name="gateway-server", daemon=True)
        self.host = host
        self._port: Optional[int] = None

    def start(self, timeout_s: float = 10.0) -> "GatewayServer":
        self._thread.start()
        deadline = time.monotonic() + timeout_s
        while not self._server.started:
            if time.monotonic() > deadline or not self._thread.is_alive():
                raise RuntimeError("gateway server failed to start")
            time.sleep(0.01)
        self._port = self._server.servers[0].sockets[0].getsockname()[1]     # kept: uvicorn drops its sockets on stop
        return self

    @property
    def port(self) -> int:
        if self._port is None:
            raise RuntimeError("the server has not been started")
        return self._port

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def stop(self) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=10.0)

    def __enter__(self) -> "GatewayServer":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()


# ── Command line ───────────────────────────────────────────────────────
def _secret_from_env() -> bytes:
    import os
    value = os.environ.get("BMS_GATEWAY_SECRET", "")
    if len(value.encode()) < _MIN_SECRET_BYTES:
        raise SystemExit(f"BMS_GATEWAY_SECRET must be set to at least {_MIN_SECRET_BYTES} bytes "
                         "(create one with: python -m bms.gateway secret)")
    return value.encode()


def main(argv: Optional[list] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="python -m bms.gateway", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("secret", help="print a fresh signing secret")
    p_tok = sub.add_parser("token", help="issue a read-only token (needs BMS_GATEWAY_SECRET)")
    p_tok.add_argument("--subject", required=True)
    p_tok.add_argument("--ttl", type=float, default=3600.0, help="lifetime in seconds")
    p_srv = sub.add_parser("serve", help="serve a drive-cycle feed (needs BMS_GATEWAY_SECRET)")
    p_srv.add_argument("--cycle", default="udds")
    p_srv.add_argument("--speed", type=float, default=10.0, help="simulated seconds per wall second")
    p_srv.add_argument("--repeat", type=int, default=0, help="passes of the cycle (0 = forever)")
    p_srv.add_argument("--host", default="127.0.0.1")
    p_srv.add_argument("--port", type=int, default=8765)
    p_srv.add_argument("--tls-keyfile")
    p_srv.add_argument("--tls-certfile")
    p_srv.add_argument("--allow-insecure", action="store_true", help="allow a non-loopback bind without TLS")
    args = ap.parse_args(argv)

    if args.cmd == "secret":
        print(secrets.token_hex(32))
        return 0
    authority = TokenAuthority(_secret_from_env())
    if args.cmd == "token":
        print(authority.issue(args.subject, ttl_s=args.ttl))
        return 0

    loopback = args.host in ("127.0.0.1", "localhost", "::1")
    tls = bool(args.tls_keyfile and args.tls_certfile)
    if not loopback and not tls and not args.allow_insecure:
        raise SystemExit("refusing to serve bearer tokens over plaintext on a non-loopback address; "
                         "give --tls-keyfile and --tls-certfile, or --allow-insecure behind a TLS proxy")
    import uvicorn

    from .drive_cycles import get_cycle
    hub = TelemetryHub()
    feeder = DriveFeeder(hub, get_cycle(args.cycle), speed=args.speed, repeat=args.repeat).start()
    try:
        uvicorn.run(create_gateway_app(hub, authority), host=args.host, port=args.port, log_level="info",
                    ssl_keyfile=args.tls_keyfile, ssl_certfile=args.tls_certfile)
    finally:
        feeder.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
