# Virtual CAN output and the telemetry gateway

Two ways to get a virtual EV's telemetry out of the process: as **CAN frames** on a virtual
bus, and as an **authenticated, read-only network stream**. Both carry the same drive; a test
checks that they agree. Measured decode fidelity, throughput and latency are in
[Virtual EV benchmark results](ev_benchmark.md).

!!! warning "Simulation only"
    The traffic here is simulated, on a virtual interface or a loopback socket. It says nothing
    about timing on a real vehicle network, and the gateway has not had a security review by anyone
    but its tests. Read [what it does not cover](#what-the-gateway-does-not-cover) before exposing it.

```
DriveRun ──► EVCanPublisher ──► transport (virtual | SocketCAN vcan0/can0) ──► receiver
                                                                                  │
                                       cantools + shipped bms.dbc  ◄─────────────┘ independent decode

DriveRun ──► TelemetryHub (ring buffer) ──► gateway (HTTP, SSE) ──► TelemetryClient
                                              ▲ bearer token, GET only          (reconnect, resume)
```

## CAN: publishing a drive

```python
import bms
from bms.can_transport import VirtualCanTransport
from bms.ev_can import EVCanPublisher, load_dbc, verify_against_telemetry

run = bms.simulate_cycle("udds")
with VirtualCanTransport("pack1") as tx, VirtualCanTransport("pack1") as rx:
    pub = EVCanPublisher(tx)                       # or SocketCanTransport("vcan0") on Linux
    frames = []
    pub.publish_run(run, after_sample=lambda i, sent: frames.extend(rx.recv(1.0) for _ in sent))
report = verify_against_telemetry(run, frames, db=load_dbc("bms.dbc"))
assert report.ok
```

No new message layouts: every frame comes from the existing `BMSCanBus` encoder, whose output is
the shipped `bms.dbc` (a test asserts the file and the encoder agree). Each telemetry sample
produces one broadcast cycle of seven classic 8-byte frames:

| ID | DBC message | Carries |
|---|---|---|
| 0x180 | `BMS_Status` | pack voltage, current, mean SOC, state, fault |
| 0x181 | `BMS_CellExtrema` | lowest and highest group voltage and SOC, group count |
| 0x182 | `BMS_Thermal` | coldest, hottest, spread, cooling duty |
| 0x183 / 0x184 / 0x185 | `BMS_SOP_2s` / `_10s` / `_30s` | traction and regen power and current limits |
| 0x186 | `BMS_Health` | alive counter, status checksum, transmit count, bus-off |

The decode side is independent of the encoder: frames are read with `cantools` and the shipped
DBC file. On full UDDS, HWFET and US06 cycles (2,737 samples, 19,159 frames, none missing) the
largest error of every signal sits at half its quantization step, which is the floor the
encoding itself allows:

| Signal | Resolution | Largest error observed |
|---|---|---|
| Pack voltage | 0.01 V | 0.0050 V |
| Pack current | 0.1 A | 0.050 A |
| SOC | 0.1 % | 0.050 % |
| Cell voltage min / max | 1 mV | 0.0005 V |
| Temperature min / max | 0.1 °C | 0.050 °C |

!!! note "The SOP power field saturates for a pack this size"
    The shipped DBC carries SOP power as 16-bit watts (65,535 W maximum). The default 65 kWh pack
    can deliver far more, so its SOP power reads 65,535 W on the wire. The publisher counts these
    samples (`PublishStats.sop_saturated_samples`) rather than hiding it; the current fields are
    unaffected. Widening the field means changing the DBC, which this work deliberately did not do.

### Virtual SocketCAN on Linux

```bash
sudo modprobe vcan
sudo ip link add dev vcan0 type vcan
sudo ip link set up vcan0
```

The `socketcan` CI job creates `vcan0` and runs a full UDDS drive over it with
`BMS_REQUIRE_VCAN=1`, so a missing interface is a failure there, not a skip. Elsewhere,
`VirtualCanTransport` runs the same path in process.

### A bad bus

`bms.ImpairedTransport` wraps any transport and, with seeded probabilities, drops, duplicates or
bit-flips frames, or goes silent. What a receiver sees:

| Fault | Detected by |
|---|---|
| Dropped frames | alive-counter gaps; missing samples in `verify_against_telemetry`; a stale message |
| Corrupted payload | the status checksum in the health frame; the CAN error counter, then bus-off at 256 |
| Duplicated frames | alive counter repeating |
| Dead transmitter | `CanBusMonitor.is_stale` once the timeout passes |

A lossy bus is never reported as faithful: `verify_against_telemetry(...).ok` is false and
`missing` counts what is gone.

## Gateway

```bash
export BMS_GATEWAY_SECRET=$(python -m bms.gateway secret)
python -m bms.gateway serve --cycle udds --speed 20           # 127.0.0.1:8765
TOKEN=$(python -m bms.gateway token --subject dashboard --ttl 3600)
```

```python
from bms.gateway import TelemetryClient
for record in TelemetryClient("http://127.0.0.1:8765", TOKEN).stream():
    print(record.t_s, record.speed_mps, record.soc)
```

Install with `pip install '.[api]'`. The gateway is not imported from `bms/__init__`, so the core
library never needs FastAPI.

### Endpoints

All are `GET`. Everything except `/healthz` needs `Authorization: Bearer <token>`.

| Path | Returns |
|---|---|
| `/healthz` | `{"status": "ok"}`, nothing else, no authentication |
| `/v1/whoami` | the token's subject, scopes and expiry |
| `/v1/telemetry/schema` | the record JSON Schema |
| `/v1/telemetry/latest` | the newest event |
| `/v1/telemetry/history?after=&limit=` | a bounded page of events, with a `gap` flag |
| `/v1/telemetry/stream` | Server-Sent Events |

### The event stream

Each record is an event with id `<epoch>-<n>`. The epoch identifies one hub instance; `n` counts
from 0. The server numbers events itself, so a producer that restarts its own record numbering
cannot confuse a resuming client.

| Event | Meaning |
|---|---|
| `hello` | first event: epoch, schema version, oldest and latest id, and a reconnect-delay hint |
| `telemetry` | one record; the `id` is the resume point |
| `reset` | the stream does not continue from where the client was; `reason` says why |
| `closed` | the producer finished |
| `expired` | the token's lifetime ended; reconnect with a fresh one |
| *(comment)* | `: keepalive` every few seconds when idle, so a dead link is noticed |

A client that reconnects sends `Last-Event-ID` and receives exactly the events it missed. Nothing is
skipped silently:

| Situation | The server sends |
|---|---|
| Client is current | the missed events only |
| Client fell further behind than the ring buffer holds | `reset` (`buffer_overrun`), then everything still held |
| The gateway restarted (new epoch) | `reset` (`epoch_changed`), then the new hub's buffer |
| Client claims an id the server never issued | `reset` (`ahead_of_server`), then the buffer |

New clients get the whole buffer by default (`replay=all`); `replay=none` starts at the next event
and `replay=N` replays the last *N*.

### Authentication

- **Tokens** are `v1.<key id>.<payload>.<HMAC-SHA256>`. The payload carries subject, scopes,
  issue and expiry times and a random id. The signature is checked, in constant time
  (`hmac.compare_digest`), *before* the payload is parsed. Base64 is accepted only in its
  canonical form, so no character of a valid token can be changed without invalidating it (a test
  tries every position).
- **Lifetime** is at most 7 days; a live stream ends with `expired` when its token does.
- **Secrets** must be at least 32 bytes or the authority refuses to start. The CLI reads
  `BMS_GATEWAY_SECRET` and never prints it.
- **Rotation**: give the authority several keys; new tokens use the active one and older tokens keep
  verifying until their key is removed.
- **Revocation** is by token id, in memory.
- **No oracle**: a malformed, forged, expired, revoked or unknown-key token all get the same 401, and
  the token is never echoed. A valid token without the scope gets 403.
- **Throttling**: repeated failures from one peer address are blocked with `429` and `Retry-After`.
- **Stream cap**: at most four concurrent streams per subject (configurable), released when a client
  disappears, however abruptly.

### Read-only

The service defines `GET` routes only, and a pure-ASGI layer in front of it answers every other
method with 405 and every WebSocket with a close, *before routing and before authentication*, so a
write attempt reveals nothing and reaches nothing. The interactive docs and OpenAPI schema are off.
Records enter through `TelemetryHub.publish` in the process that owns the hub, never over HTTP.
Responses carry `Cache-Control: no-store`, `X-Content-Type-Options: nosniff` and
`Referrer-Policy: no-referrer`, and no CORS headers are ever added.

### The client

`TelemetryClient` reconnects with jittered exponential backoff (honouring the server's hint and any
`Retry-After`), resumes from the last event, drops duplicates, and counts `reset` events instead of
hiding them. An authentication failure raises `GatewayAuthError` at once and is never retried. A
silent link is detected by the read timeout, because the server sends keep-alives. `token` may be a
callable, called on every connect, so a long-lived consumer can renew expiring tokens. The token never
appears in `repr` or in an exception.

### What was tested

On real sockets, with a TCP proxy that cuts, stalls and redirects live connections:

| Scenario | Result |
|---|---|
| Connection cut every ~3.5 kB, 150 records | all 150 received, in order, none repeated, 5 or more reconnects |
| Link severed twice mid-stream | exactly 2 reconnects, all 60 records |
| Link stays up but goes silent | detected by the read timeout, reconnected, nothing lost |
| Gateway replaced by a new one | one `reset` (`epoch_changed`); the new data arrives |
| Client connects late to a busy hub | only what the ring kept |
| Network cut through a 400-record drive | the received records equal the simulation's, exactly |
| Full UDDS through the gateway | 1,370 records, equal to `run.telemetry()` |
| Bad token, forged token, write attempt | 401, 401, 405 |

A mutation check applied 23 deliberate bugs to the gateway one at a time, and the tests now catch all 23.
The first run caught 18, let 3 survive (the client's backoff not restarting after progress, and two
redundant resets of its de-duplication state) and could not apply one because its target text had changed.
Three client tests were added for those gaps (backoff restarts after progress, a new epoch resets the
duplicate filter, a reset means forget what you have seen), the mutation target was corrected, and the
second run caught the five that were left. The 23 cover the signature check, expiry, not-yet-valid tokens,
scope, revocation, the read-only guard, WebSocket refusal, canonical encoding, gap detection, resume,
de-duplication, no retry on an authentication failure, slot release, keep-alives, throttling, the expiry
event, 403 versus 401, the security headers, epoch handling, the minimum secret length, backoff reset and
the disabled docs routes.

### What the gateway does not cover

- **TLS.** Not in process. Bearer tokens over plaintext are credentials on the wire. `serve` refuses a
  non-loopback bind without `--tls-keyfile` and `--tls-certfile`, unless you pass `--allow-insecure`
  to say you are terminating TLS in front of it.
- **Throttling behind a proxy.** It keys on the socket peer address, so behind a reverse proxy every
  client looks like the proxy. Throttle at the proxy as well.
- **Durable revocation.** The revoked set is in memory and is lost on restart; keep lifetimes short.
- **Multiple processes.** One hub per process; there is no shared buffer between workers.
- **Per-token rate limits** beyond the stream cap, request size policy beyond the server's defaults,
  or audit logging beyond a warning line per refused token (which never contains the token).
- **A compromised host.** Anyone who can read the process's memory has the signing key.
