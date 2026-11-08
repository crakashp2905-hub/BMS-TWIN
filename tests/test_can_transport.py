"""Tests: CAN transport — telemetry across a virtual bus (and SocketCAN where present)."""

from __future__ import annotations

import platform
import uuid

import numpy as np
import pytest

import bms
from bms.can import CanBusMonitor, CANFrame, can_checksum
from bms.can_transport import (
    SocketCanTransport,
    VirtualCanTransport,
    consume_telemetry,
    publish_telemetry,
)


def _channel(prefix: str = "bms") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def _cycle_result():
    """One real supervisor cycle — the same dict the encoder is designed for."""
    pack = bms.BatteryPack(bms.PackConfig(n_cells=4, seed=0))
    sup = bms.BMSSupervisor(pack, bms.ThermalModel(n_cells=4), bms.HybridFaultDetector())
    result = sup.step(1.0, 1.0, k=0)
    sop = bms.StateOfPower().calculate(pack, np.full(4, 25.0))
    return result, sop


class TestVirtualTransport:
    def test_single_frame_loopback(self):
        ch = _channel()
        with VirtualCanTransport(ch) as tx, VirtualCanTransport(ch) as rx:
            frame = CANFrame(0x123, bytes(range(8)))
            tx.send(frame)
            got = rx.recv(timeout=1.0)
        assert got == frame

    def test_recv_times_out_when_silent(self):
        ch = _channel()
        with VirtualCanTransport(ch) as rx:
            assert rx.recv(timeout=0.05) is None

    def test_full_telemetry_round_trip_decodes_the_same_values(self):
        ch = _channel()
        result, sop = _cycle_result()
        bus = bms.BMSCanBus()
        with VirtualCanTransport(ch) as tx, VirtualCanTransport(ch) as rx:
            n_sent = publish_telemetry(tx, bus, result, sop)
            monitor = CanBusMonitor(timeout_s=0.5)
            frames = consume_telemetry(rx, monitor, now=0.0, max_frames=64)
        assert len(frames) == n_sent
        status = next(f for f in frames if f.arbitration_id == bus.STATUS_ID)
        decoded = bus.parse(status)
        # the encoder stores pack voltage at 0.01 V and SOC at 0.1 % resolution
        assert abs(decoded["pack_voltage_V"] - float(result["v_pack"])) < 0.05
        assert abs(decoded["soc_pct"] - float(np.mean(result["soc"])) * 100.0) < 0.2

    def test_monitor_flags_a_silent_transmitter(self):
        ch = _channel()
        result, sop = _cycle_result()
        bus = bms.BMSCanBus()
        with VirtualCanTransport(ch) as tx, VirtualCanTransport(ch) as rx:
            publish_telemetry(tx, bus, result, sop)
            monitor = CanBusMonitor(timeout_s=0.5)
            consume_telemetry(rx, monitor, now=10.0)
        assert not monitor.is_stale(bus.STATUS_ID, now=10.2)   # fresh
        assert monitor.is_stale(bus.STATUS_ID, now=11.0)       # no traffic for >0.5 s

    def test_malformed_frame_is_dropped_not_crashed_on(self):
        import can
        ch = _channel()
        with VirtualCanTransport(ch) as tx, VirtualCanTransport(ch) as rx:
            # a 5-byte payload is not a valid BMS frame: it must be ignored
            tx._bus.send(can.Message(arbitration_id=0x123, data=b"\x01\x02\x03\x04\x05",
                                     is_extended_id=False))
            tx.send(CANFrame(0x124, bytes(8)))
            monitor = CanBusMonitor()
            frames = consume_telemetry(rx, monitor, now=0.0)
        assert [f.arbitration_id for f in frames] == [0x124]

    def test_corrupted_payload_fails_the_checksum(self):
        result, sop = _cycle_result()
        bus = bms.BMSCanBus()
        status = bus.status_frame(result)
        expected = can_checksum(status.data)
        corrupted = bytearray(status.data)
        corrupted[0] ^= 0xFF
        monitor = CanBusMonitor()
        assert monitor.checksum_ok(status.data, expected)
        assert not monitor.checksum_ok(bytes(corrupted), expected)
        assert monitor.error_count > 0


class TestSocketCanTransport:
    def test_refuses_to_open_off_linux(self):
        if platform.system() == "Linux":
            pytest.skip("this check is for non-Linux hosts")
        with pytest.raises(RuntimeError, match="SocketCAN is a Linux"):
            SocketCanTransport("vcan0")

    @pytest.mark.skipif(platform.system() != "Linux", reason="SocketCAN is Linux-only")
    def test_vcan_loopback_when_interface_exists(self):
        try:
            tx = SocketCanTransport("vcan0")
        except Exception:
            pytest.skip("vcan0 is not configured on this host")
        try:
            frame = CANFrame(0x321, bytes(range(8)))
            tx.send(frame)
            assert tx.recv(timeout=1.0) == frame
        finally:
            tx.close()
