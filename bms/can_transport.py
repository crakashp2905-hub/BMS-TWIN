"""CAN transport — move BMS telemetry frames across a real or virtual bus.

:mod:`bms.can` *encodes* and *validates* frames; this module *moves* them.  One
interface, two backends:

* :class:`VirtualCanTransport` — an in-process bus built on python-can's
  ``virtual`` interface.  Works on Windows, macOS and Linux, so the full
  transmit → receive → health-monitor path runs in CI everywhere.
* :class:`SocketCanTransport` — the Linux kernel's SocketCAN, for a real
  ``can0`` adapter or a ``vcan0`` virtual interface.  Refuses to open off Linux
  with a clear error instead of failing obscurely.

    tx = VirtualCanTransport("pack1")
    rx = VirtualCanTransport("pack1")
    publish_telemetry(tx, BMSCanBus(), supervisor_result, sop_limits)
    frames = consume_telemetry(rx, CanBusMonitor(timeout_s=0.5))

Frames are the same validated :class:`bms.can.CANFrame` objects everywhere, so a
test that passes on the virtual bus exercises the code the SocketCAN path runs.
"""

from __future__ import annotations

import platform
from abc import ABC, abstractmethod

from .can import BMSCanBus, CanBusMonitor, CANFrame


def _to_message(frame: CANFrame):
    import can
    return can.Message(arbitration_id=frame.arbitration_id, data=frame.data,
                       is_extended_id=frame.is_extended_id)


def _to_frame(msg) -> CANFrame | None:
    """python-can message → validated CANFrame; malformed frames are dropped."""
    data = bytes(msg.data)
    if len(data) != 8:
        return None
    try:
        return CANFrame(int(msg.arbitration_id), data, bool(msg.is_extended_id))
    except ValueError:
        return None


class CanTransport(ABC):
    """A bidirectional CAN endpoint."""

    name: str

    @abstractmethod
    def send(self, frame: CANFrame) -> None:
        """Transmit one frame."""

    @abstractmethod
    def recv(self, timeout: float = 0.1) -> CANFrame | None:
        """Receive one frame, or ``None`` if none arrives within ``timeout`` s."""

    @abstractmethod
    def close(self) -> None:
        """Release the underlying bus."""

    def __enter__(self) -> "CanTransport":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class _PythonCanTransport(CanTransport):
    _interface = ""

    def __init__(self, channel: str, **bus_kwargs) -> None:
        import can
        self.name = channel
        self._bus = can.Bus(interface=self._interface, channel=channel,
                            receive_own_messages=False, **bus_kwargs)

    def send(self, frame: CANFrame) -> None:
        self._bus.send(_to_message(frame))

    def recv(self, timeout: float = 0.1) -> CANFrame | None:
        # Malformed frames are skipped *within* the timeout, so one bad frame never
        # hides the valid frames queued behind it.
        import time
        deadline = time.monotonic() + timeout
        while True:
            remaining = max(0.0, deadline - time.monotonic())
            msg = self._bus.recv(timeout=remaining)
            if msg is None:
                return None
            frame = _to_frame(msg)
            if frame is not None:
                return frame
            if remaining <= 0.0:
                return None

    def close(self) -> None:
        self._bus.shutdown()


class VirtualCanTransport(_PythonCanTransport):
    """In-process virtual bus: endpoints sharing a ``channel`` name see each other."""

    _interface = "virtual"


class SocketCanTransport(_PythonCanTransport):
    """Linux SocketCAN (``can0`` hardware or ``vcan0`` virtual interface)."""

    _interface = "socketcan"

    def __init__(self, channel: str = "vcan0", **bus_kwargs) -> None:
        if platform.system() != "Linux":
            raise RuntimeError(
                "SocketCAN is a Linux kernel interface; on this platform use "
                "VirtualCanTransport, or run on Linux with a vcan0 interface.")
        super().__init__(channel, **bus_kwargs)


def publish_telemetry(transport: CanTransport, bus: BMSCanBus, result: dict,
                      sop: dict) -> int:
    """Encode one supervisor cycle with :class:`BMSCanBus` and transmit it.

    Returns the number of frames sent (status, cells, thermal, SOP, health).
    """
    frames = bus.broadcast(result, sop)
    for f in frames:
        transport.send(f)
    return len(frames)


def consume_telemetry(transport: CanTransport, monitor: CanBusMonitor, *,
                      now: float, max_frames: int = 64, timeout: float = 0.05) -> list:
    """Drain up to ``max_frames`` frames into ``monitor`` and return the frames read.

    ``now`` is the receiver's clock, used for freshness/timeout checks, so the
    monitor behaves identically on a virtual bus and on SocketCAN.
    """
    received = []
    for _ in range(max_frames):
        frame = transport.recv(timeout=timeout)
        if frame is None:
            break
        monitor.receive(frame, now)
        received.append(frame)
    return received
