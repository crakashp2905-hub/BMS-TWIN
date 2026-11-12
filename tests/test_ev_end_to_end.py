"""Sprint 1 acceptance, end to end: one virtual drive through every output path.

  simulation ──► CAN frames (virtual bus) ──► cantools + shipped bms.dbc ──► decoded signals
      │
      └────────► telemetry records ──► authenticated, read-only gateway ──► reconnecting client

These tests tie the pieces together; each piece has its own detailed tests.  Nothing here is
validation against a physical vehicle.
"""

from __future__ import annotations

import threading
import time
import uuid
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")
cantools = pytest.importorskip("cantools")
httpx = pytest.importorskip("httpx")
from netfault import ChaosProxy, collect, wait_until  # noqa: E402

import bms  # noqa: E402
from bms.can import BMSCanBus  # noqa: E402
from bms.can_transport import VirtualCanTransport  # noqa: E402
from bms.ev_can import EVCanPublisher, decode_frames, load_dbc, verify_against_telemetry  # noqa: E402
from bms.ev_drive import EVConfig, EVSimulator  # noqa: E402
from bms.gateway import (  # noqa: E402
    GatewayConfig,
    GatewayServer,
    TelemetryClient,
    TelemetryHub,
    TokenAuthority,
    create_gateway_app,
    generate_secret,
)
from bms.telemetry import check_sequence  # noqa: E402

_DBC = Path(bms.__file__).resolve().parents[1] / "bms.dbc"


def _run(cycle="ece15", **kw):
    return EVSimulator(EVConfig(dt_sim_s=0.5, **kw)).run(cycle)


def _can_frames(run):
    ch = f"e2e-{uuid.uuid4().hex[:8]}"
    sent, received = [], []
    with VirtualCanTransport(ch) as tx, VirtualCanTransport(ch) as rx:
        def drain(i, frames):
            sent.extend(frames)
            received.extend(rx.recv(timeout=1.0) for _ in frames)
        EVCanPublisher(tx).publish_run(run, after_sample=drain)
    return sent, received


def _serve(records, *, capacity=5000):
    hub = TelemetryHub(capacity=capacity)
    authority = TokenAuthority(generate_secret())
    for r in records:
        hub.publish(r)
    app = create_gateway_app(hub, authority, GatewayConfig(heartbeat_s=0.15, retry_ms=10))
    return hub, authority, GatewayServer(app)


class TestAcceptance:
    def test_reproducible_virtual_drive_cycle(self):
        """Same configuration → bit-identical simulation, CAN bytes and gateway records."""
        a, b = _run("udds"), _run("udds")
        assert a.digest() == b.digest() and a.manifest() == b.manifest()
        assert [f.data for f in _can_frames(a)[0]] == [f.data for f in _can_frames(b)[0]]
        assert a.telemetry(1.0) == b.telemetry(1.0)

    def test_physically_consistent_energy_accounting(self):
        for cycle in ("udds", "hwfet", "us06"):
            res = _run(cycle).ledger.relative_residuals()
            assert res["vehicle"] < 1e-8 and res["pack"] < 1e-3 and res["spline"] < 1e-3, (cycle, res)

    def test_dbc_compatible_virtual_can_output(self):
        run = _run("udds")
        sent, received = _can_frames(run)
        assert received == sent
        report = verify_against_telemetry(run, received, db=load_dbc(_DBC))
        assert report.ok and report.missing == 0
        assert _DBC.read_text().replace("\r\n", "\n") == BMSCanBus().to_dbc()      # the DBC is the shipped one

    def test_can_and_gateway_agree_about_the_same_drive(self):
        run = _run()
        records = run.telemetry(1.0)
        _, received = _can_frames(run)
        status = [d for d in decode_frames(received, load_dbc(_DBC)) if d["message"] == "BMS_Status"]
        assert len(status) == len(records)
        for d, r in zip(status, records):
            assert d["pack_voltage_V"] == pytest.approx(r.voltage_V, abs=0.005)
            assert d["pack_current_A"] == pytest.approx(r.current_A, abs=0.05)
            assert d["soc_pct"] == pytest.approx(r.soc * 100.0, abs=0.05)

    def test_authenticated_read_only_stream_delivers_exactly_what_the_simulation_produced(self):
        records = _run("udds").telemetry(1.0)
        hub, authority, server = _serve(records)
        hub.close()
        with server:
            client = TelemetryClient(server.url, authority.issue("e2e"))
            got = list(client.stream())
            # no token, a forged token and a write are all refused
            assert httpx.get(server.url + "/v1/telemetry/latest", timeout=5).status_code == 401
            forged = TokenAuthority(generate_secret()).issue("mallory")
            assert httpx.get(server.url + "/v1/telemetry/latest", headers={"Authorization": f"Bearer {forged}"},
                             timeout=5).status_code == 401
            assert httpx.post(server.url + "/v1/telemetry/stream",
                              headers={"Authorization": f"Bearer {authority.issue('e2e')}"}, timeout=5
                              ).status_code == 405
        assert got == records and check_sequence(got).ok and len(got) == 1370

    def test_the_stream_survives_a_flaky_network_with_the_same_exact_records(self):
        records = _run("udds").telemetry(1.0)[:400]
        hub, authority, server = _serve([])
        with server:
            proxy = ChaosProxy(server.port)
            proxy.cut_after_bytes = 20_000
            try:
                client = TelemetryClient(f"http://127.0.0.1:{proxy.port}", authority.issue("e2e"),
                                         backoff_initial_s=0.01, backoff_max_s=0.05, jitter=0.0, timeout_s=10)

                def produce():
                    wait_until(lambda: client.stats.connects >= 1)
                    for r in records:
                        hub.publish(r)
                        time.sleep(0.001)
                    hub.close()
                threading.Thread(target=produce, daemon=True).start()
                got = collect(client)
            finally:
                proxy.stop()
        assert got == records and client.stats.reconnects >= 3 and client.stats.resets == 0

    def test_the_simulation_can_feed_the_gateway_live(self):
        from bms.drive_cycles import get_cycle
        from bms.gateway import DriveFeeder
        hub = TelemetryHub(capacity=1000)
        authority = TokenAuthority(generate_secret())
        app = create_gateway_app(hub, authority, GatewayConfig(heartbeat_s=0.15))
        with GatewayServer(app) as server:
            feeder = DriveFeeder(hub, get_cycle("ece15"), EVConfig(dt_sim_s=0.5), speed=2000.0)
            client = TelemetryClient(server.url, authority.issue("e2e"))
            feeder.start()
            got = collect(client)
            feeder.stop()
        assert len(got) == 196 and check_sequence(got).ok
        assert got == _run("ece15").telemetry(1.0)
