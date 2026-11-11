"""Tests: virtual-EV telemetry as DBC-compatible CAN frames over virtual and SocketCAN buses.

The decode side is deliberately independent: frames are decoded with ``cantools`` and the
*shipped* ``bms.dbc`` file, not with the encoder that produced them.
"""

from __future__ import annotations

import os
import platform
import uuid
from pathlib import Path

import numpy as np
import pytest

import bms
from bms.can import BMSCanBus, CanBusMonitor, CANFrame
from bms.can_transport import (
    ImpairedTransport,
    SocketCanTransport,
    VirtualCanTransport,
    consume_telemetry,
)
from bms.ev_can import (
    EVCanBus,
    EVCanPublisher,
    decode_frames,
    load_dbc,
    verify_against_telemetry,
)
from bms.ev_drive import EVConfig, EVSimulator
from bms.sop import SOPLimit

cantools = pytest.importorskip("cantools")
_DBC = Path(bms.__file__).resolve().parents[1] / "bms.dbc"

_N_FRAMES_PER_SAMPLE = 7        # status, cell extrema, thermal, three SOP horizons, health


def _run(cycle="ece15", **kw):
    return EVSimulator(EVConfig(dt_sim_s=0.5, **kw)).run(cycle)


def _channel() -> str:
    return f"ev-{uuid.uuid4().hex[:8]}"


def _drive_over_virtual_bus(run):
    """Publish ``run`` and return (frames sent, frames received, stats), draining as it goes."""
    ch = _channel()
    sent, received = [], []
    with VirtualCanTransport(ch) as tx, VirtualCanTransport(ch) as rx:
        pub = EVCanPublisher(tx)

        def drain(i, frames):
            sent.extend(frames)
            for _ in frames:
                f = rx.recv(timeout=1.0)
                assert f is not None, "a frame was lost on the lossless virtual bus"
                received.append(f)
        stats = pub.publish_run(run, after_sample=drain)
    return sent, received, stats


class TestEncoding:
    def test_summary_frames_are_identical_to_the_existing_encoders_output(self):
        # The same broadcast built the old way (per-group arrays) and the new way (summaries).
        soc = np.array([0.50, 0.62, 0.71, 0.80])
        v = np.array([3.61, 3.72, 3.80, 3.91])
        t = np.array([24.0, 25.5, 26.0, 27.5])
        sop = {h: SOPLimit(h, 40_000.0 - 1000 * h, 20_000.0, 120.0 - h, 60.0, 1.0) for h in (2.0, 10.0, 30.0)}
        old_bus, new_bus = BMSCanBus(), EVCanBus()
        for _ in range(3):                                       # the alive counter must advance identically
            old = old_bus.broadcast({"v_pack": float(v.sum()), "cmd_current": 33.3, "soc": soc, "state": "operating",
                                     "fault_label": "none", "v_cells": v, "T_cells": t, "cooling_duty": 0.4}, sop)
            new = new_bus.broadcast_summary(
                v_pack_V=float(v.sum()), current_A=33.3, soc_mean=float(soc.mean()), soc_min=float(soc.min()),
                soc_max=float(soc.max()), n_groups=4, cell_v_min_V=float(v.min()), cell_v_max_V=float(v.max()),
                temp_min_C=float(t.min()), temp_max_C=float(t.max()), cooling_duty=0.4, state="operating",
                fault="none", sop=sop)
            assert [f.data for f in old] == [f.data for f in new]
            assert [f.arbitration_id for f in old] == [f.arbitration_id for f in new]

    def test_every_frame_is_classic_can_with_a_known_identifier(self):
        sent, _, _ = _drive_over_virtual_bus(_run())
        known = {BMSCanBus.STATUS_ID, BMSCanBus.CELL_EXTREMA_ID, BMSCanBus.THERMAL_ID, BMSCanBus.SOP_2S_ID,
                 BMSCanBus.SOP_10S_ID, BMSCanBus.SOP_30S_ID, BMSCanBus.HEALTH_ID}
        assert all(len(f.data) == 8 and not f.is_extended_id and f.arbitration_id in known for f in sent)

    def test_one_broadcast_cycle_per_telemetry_sample(self):
        run = _run()
        _, _, stats = _drive_over_virtual_bus(run)
        assert stats.samples == len(run.sample_indices())
        assert stats.frames == stats.samples * _N_FRAMES_PER_SAMPLE
        assert set(stats.per_id.values()) == {stats.samples}
        assert stats.frames_per_sample == _N_FRAMES_PER_SAMPLE

    def test_shipped_dbc_is_the_one_the_encoder_emits(self):
        assert _DBC.read_text().replace("\r\n", "\n") == BMSCanBus().to_dbc()


class TestDriveOverTheVirtualBus:
    def test_a_full_udds_decodes_to_the_simulation_within_half_a_quantization_step(self):
        run = _run("udds")
        sent, received, stats = _drive_over_virtual_bus(run)
        assert received == sent and stats.samples == 1370
        report = verify_against_telemetry(run, received, db=load_dbc(_DBC))
        assert report.ok, report
        assert report.missing == 0 and report.samples_compared == 1370
        # the error really is at the quantization floor: not zero (it is quantized), not larger
        assert 0.0 < report.max_abs_error["pack_voltage_V"] <= 0.005 + 1e-9
        assert 0.0 < report.max_abs_error["pack_current_A"] <= 0.05 + 1e-9

    def test_decoding_with_the_shipped_file_and_with_the_generated_text_agree(self):
        _, received, _ = _drive_over_virtual_bus(_run())
        assert decode_frames(received, load_dbc(_DBC)) == decode_frames(received, load_dbc())

    def test_decoded_values_track_the_drive(self):
        run = _run("udds")
        _, received, _ = _drive_over_virtual_bus(run)
        status = [d for d in decode_frames(received, load_dbc(_DBC)) if d["message"] == "BMS_Status"]
        assert status[0]["state_code"] == 0 and any(d["state_code"] == 2 for d in status)    # idle → operating
        assert max(d["pack_current_A"] for d in status) == pytest.approx(run["current_A"].max(), abs=0.1)
        assert min(d["pack_current_A"] for d in status) < 0.0                                  # regeneration
        assert status[-1]["soc_pct"] < status[0]["soc_pct"]
        assert all(d["fault_code"] == 0 for d in status)

    def test_state_of_power_frames_carry_the_pack_limits_up_to_the_16_bit_field(self):
        run = _run()
        _, received, stats = _drive_over_virtual_bus(run)
        sop = {d["message"]: d for d in decode_frames(received[:_N_FRAMES_PER_SAMPLE], load_dbc(_DBC))
               if d["message"].startswith("BMS_SOP")}
        want = run.sop_at(0)
        for h in (2.0, 10.0, 30.0):
            got = sop[f"BMS_SOP_{int(h)}s"]
            assert got["traction_power_W"] == pytest.approx(min(want[h].traction_power_W, 65_535.0), abs=1.0)
            assert got["traction_current_A"] == pytest.approx(want[h].traction_current_A, abs=0.1)
        # a 65 kW-class field cannot carry this pack's limit: the saturation is reported, not hidden
        assert stats.sop_saturated_samples == stats.samples
        assert sop["BMS_SOP_2s"]["traction_power_W"] == 65_535.0

    def test_a_small_pack_fits_the_sop_field_and_reports_no_saturation(self):
        run = _run(pack=bms.PackSpec(n_series=96, n_parallel=6, cell_capacity_Ah=3.0))
        _, _, stats = _drive_over_virtual_bus(run)
        assert stats.sop_saturated_samples == 0

    def test_a_healthy_bus_passes_every_receiver_check(self):
        run = _run("udds")
        _, received, stats = _drive_over_virtual_bus(run)
        bus, monitor = BMSCanBus(), CanBusMonitor(timeout_s=2.5)
        status_data = None
        for f in received:
            msg = bus.parse(f)
            if msg["message"] == "BMS_Status":
                status_data = f.data
            elif msg["message"] == "BMS_Health":
                assert monitor.alive_ok(msg["alive_counter"])
                assert monitor.checksum_ok(status_data, msg["status_checksum"])
        assert monitor.error_count == 0 and not monitor.bus_off
        last = bus.parse(received[-1])
        assert last["tx_count"] == stats.frames % 65_536               # the transmit counter keeps true count

    def test_the_alive_counter_wraps_cleanly_past_255_samples(self):
        run = _run("udds")                                              # 1370 samples: wraps five times
        _, received, _ = _drive_over_virtual_bus(run)
        bus, monitor = BMSCanBus(), CanBusMonitor()
        alive = [bus.parse(f)["alive_counter"] for f in received if f.arbitration_id == BMSCanBus.HEALTH_ID]
        assert max(alive) == 255 and all(monitor.alive_ok(a) for a in alive)

    def test_realtime_pacing_spaces_samples_by_the_telemetry_interval(self):
        run = _run()
        waits: list[float] = []
        ch = _channel()
        with VirtualCanTransport(ch) as tx, VirtualCanTransport(ch):
            EVCanPublisher(tx).publish_run(run, realtime=True, speed=10.0, sleep=waits.append)
        assert len(waits) >= 190 and all(b >= a - 1e-9 for a, b in zip(waits, waits[1:]))
        assert waits[-1] == pytest.approx(run["t_s"][-1] / 10.0, rel=0.05)      # 195 s at 10x → ~19.5 s


class TestFaultInjectionOnTheBus:
    """A receiver must notice a bad bus, never mistake it for a healthy one."""

    @staticmethod
    def _drive(run, **impair):
        ch = _channel()
        monitor = CanBusMonitor(timeout_s=2.5)
        bus = BMSCanBus()
        counts = {"alive_breaks": 0, "checksum_errors": 0, "frames": 0}
        state = {"status": None}
        with VirtualCanTransport(ch) as raw_tx, VirtualCanTransport(ch) as rx:
            tx = ImpairedTransport(raw_tx, **impair)
            pub = EVCanPublisher(tx)

            def drain(i, frames):
                now = float(run["t_s"][i])
                for f in consume_telemetry(rx, monitor, now=now, max_frames=64, timeout=0.005):
                    counts["frames"] += 1
                    msg = bus.parse(f)
                    if msg["message"] == "BMS_Status":
                        state["status"] = f.data
                    elif msg["message"] == "BMS_Health":
                        if not monitor.alive_ok(msg["alive_counter"]):
                            counts["alive_breaks"] += 1
                        if state["status"] is not None and not monitor.checksum_ok(state["status"],
                                                                                    msg["status_checksum"]):
                            counts["checksum_errors"] += 1
            pub.publish_run(run, after_sample=drain)
        return tx, monitor, counts

    def test_dropped_frames_break_the_alive_sequence_and_the_comparison(self):
        run = _run()
        tx, monitor, counts = self._drive(run, drop=0.25, seed=3)
        assert tx.dropped > 0 and counts["frames"] == tx.sent - tx.dropped
        assert counts["alive_breaks"] > 0                                # health frames went missing

    def test_a_lossy_bus_is_never_reported_as_a_faithful_one(self):
        run = _run()
        ch = _channel()
        received = []
        with VirtualCanTransport(ch) as raw_tx, VirtualCanTransport(ch) as rx:
            tx = ImpairedTransport(raw_tx, drop=0.15, seed=11)

            def drain(i, frames):
                while (f := rx.recv(timeout=0.005)) is not None:
                    received.append(f)
            EVCanPublisher(tx).publish_run(run, after_sample=drain)
        report = verify_against_telemetry(run, received, db=load_dbc(_DBC))
        assert tx.dropped > 0 and report.missing > 0 and not report.ok

    def test_corrupted_payloads_are_caught_by_the_status_checksum(self):
        run = _run()
        tx, monitor, counts = self._drive(run, corrupt=0.3, seed=5)
        assert tx.corrupted > 0 and counts["checksum_errors"] > 0 and monitor.error_count > 0

    def test_a_bus_that_corrupts_everything_goes_bus_off_and_can_recover(self):
        run = _run()
        _, monitor, counts = self._drive(run, corrupt=1.0, seed=1)
        assert monitor.bus_off and counts["checksum_errors"] >= 32
        monitor.recover()
        assert not monitor.bus_off and monitor.error_count == 0

    def test_duplicated_frames_are_caught_by_the_alive_counter(self):
        run = _run()
        tx, _, counts = self._drive(run, duplicate=1.0, seed=2)
        assert tx.duplicated == tx.sent and counts["alive_breaks"] >= len(run.sample_indices()) - 1

    def test_a_transmitter_that_dies_goes_stale(self):
        run = _run()
        ch = _channel()
        monitor = CanBusMonitor(timeout_s=2.5)
        idx = list(run.sample_indices())
        with VirtualCanTransport(ch) as raw_tx, VirtualCanTransport(ch) as rx:
            tx = ImpairedTransport(raw_tx)
            pub = EVCanPublisher(tx)
            for k, i in enumerate(idx):
                if k == 50:
                    tx.silent = True
                pub.publish_sample(run, int(i))
                consume_telemetry(rx, monitor, now=float(run["t_s"][i]), max_frames=64, timeout=0.005)
                if k == 49:
                    assert not monitor.is_stale(BMSCanBus.STATUS_ID, now=float(run["t_s"][i]))
        assert monitor.is_stale(BMSCanBus.STATUS_ID, now=float(run["t_s"][idx[-1]]))

    def test_damage_is_seeded_and_reproducible(self):
        run = _run()
        a, b = (self._drive(run, drop=0.2, corrupt=0.2, duplicate=0.2, seed=9)[0] for _ in range(2))
        assert (a.dropped, a.corrupted, a.duplicated) == (b.dropped, b.corrupted, b.duplicated)
        c = self._drive(run, drop=0.2, corrupt=0.2, duplicate=0.2, seed=10)[0]
        assert (a.dropped, a.corrupted, a.duplicated) != (c.dropped, c.corrupted, c.duplicated)

    def test_a_clean_wrapper_changes_nothing(self):
        run = _run()
        tx, monitor, counts = self._drive(run)
        assert (tx.dropped, tx.corrupted, tx.duplicated) == (0, 0, 0)
        assert counts["alive_breaks"] == 0 and counts["checksum_errors"] == 0 and monitor.error_count == 0

    @pytest.mark.parametrize("kw", [dict(drop=-0.1), dict(duplicate=1.5), dict(corrupt=2.0)])
    def test_impairment_probabilities_are_validated(self, kw):
        with VirtualCanTransport(_channel()) as t, pytest.raises(ValueError, match="probability"):
            ImpairedTransport(t, **kw)

    def test_a_frame_with_one_flipped_bit_differs_in_exactly_one_bit(self):
        ch = _channel()
        with VirtualCanTransport(ch) as raw_tx, VirtualCanTransport(ch) as rx:
            tx = ImpairedTransport(raw_tx, corrupt=1.0, seed=4)
            frame = CANFrame(0x180, bytes(range(8)))
            tx.send(frame)
            got = rx.recv(timeout=1.0)
        diff = sum(bin(a ^ b).count("1") for a, b in zip(frame.data, got.data))
        assert diff == 1 and got.arbitration_id == frame.arbitration_id


@pytest.mark.skipif(platform.system() != "Linux", reason="SocketCAN is Linux-only")
class TestSocketCanDrive:
    """The same drive over a real kernel vcan0 interface (the dedicated CI job creates it)."""

    def test_a_drive_over_vcan0_decodes_with_the_shipped_dbc(self):
        try:
            tx = SocketCanTransport("vcan0")
        except Exception:
            if os.environ.get("BMS_REQUIRE_VCAN"):
                raise
            pytest.skip("vcan0 is not configured on this host")
        rx = SocketCanTransport("vcan0")
        run = _run("udds")
        received: list[CANFrame] = []
        try:
            def drain(i, frames):                                       # keep the kernel rx buffer empty
                for _ in frames:
                    f = rx.recv(timeout=1.0)
                    assert f is not None, "a frame was lost on vcan0"
                    received.append(f)
            stats = EVCanPublisher(tx).publish_run(run, after_sample=drain)
        finally:
            tx.close()
            rx.close()
        assert len(received) == stats.frames == 1370 * _N_FRAMES_PER_SAMPLE
        report = verify_against_telemetry(run, received, db=load_dbc(_DBC))
        assert report.ok, report

    def test_vcan0_carries_impaired_traffic_the_same_way_as_the_virtual_bus(self):
        try:
            raw = SocketCanTransport("vcan0")
        except Exception:
            if os.environ.get("BMS_REQUIRE_VCAN"):
                raise
            pytest.skip("vcan0 is not configured on this host")
        rx = SocketCanTransport("vcan0")
        run = _run()
        tx = ImpairedTransport(raw, corrupt=1.0, seed=1)
        received: list[CANFrame] = []
        try:
            def drain(i, frames):
                for _ in frames:
                    received.append(rx.recv(timeout=1.0))
            EVCanPublisher(tx).publish_run(run, after_sample=drain)
        finally:
            tx.close()
            rx.close()
        # What a clean publisher would have sent, frame for frame (same deterministic content).
        clean = [f for i in run.sample_indices() for f in EVCanPublisher(raw).frames_for(run, int(i))]
        assert tx.corrupted == tx.sent == len(received) == len(clean)
        for got, want in zip(received, clean):
            assert got.arbitration_id == want.arbitration_id
            assert sum(bin(a ^ b).count("1") for a, b in zip(got.data, want.data)) == 1   # one bit, nothing else
