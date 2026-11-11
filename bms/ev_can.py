"""Virtual-EV telemetry as DBC-compatible CAN frames, over a virtual or SocketCAN bus.

:mod:`bms.can` already defines the BMS message set (status, cell extrema, thermal, three
state-of-power horizons, health) and the shipped ``bms.dbc`` describes it.  This module
feeds that encoder from a :class:`bms.ev_drive.DriveRun` and puts the frames on a
:class:`bms.can_transport.CanTransport`:

    DriveRun ──► EVCanBus (existing frame encoders) ──► transport (virtual | SocketCAN)
                                                              │
                  cantools + bms.dbc  ◄── receiver frames ◄───┘   (independent decode)

No new message layouts are invented: every frame is one the existing encoder produces, and
:func:`verify_against_telemetry` decodes the bus traffic with ``cantools`` and the DBC and
compares each signal with the simulation, within half a quantization step.

Scope: this is simulated traffic on a virtual interface.  It says nothing about timing on a
real vehicle network.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .can import BMSCanBus, CANFrame
from .can_transport import CanTransport
from .ev_drive import DriveRun
from .sop import SOPLimit

_U16_MAX_W = 65_535.0     # SOP power fields are 16-bit watts in the shipped DBC


class EVCanBus(BMSCanBus):
    """:class:`BMSCanBus` that builds a broadcast cycle from summary values.

    ``BMSCanBus.broadcast`` takes per-group arrays; a drive run keeps only the extrema and the
    mean per sample (which is all the CAN frames carry).  This builds the same frames, with the
    same alive counter and transmit count behaviour, from those summaries.
    """

    def broadcast_summary(self, *, v_pack_V: float, current_A: float, soc_mean: float, soc_min: float,
                          soc_max: float, n_groups: int, cell_v_min_V: float, cell_v_max_V: float,
                          temp_min_C: float, temp_max_C: float, cooling_duty: float, state: str,
                          fault: str, sop: dict[float, SOPLimit]) -> list[CANFrame]:
        n = max(int(n_groups), 2)
        v = np.full(n, cell_v_min_V)
        v[-1] = cell_v_max_V                       # extrema and group count are all the frame carries
        s = np.full(n, soc_min)
        s[-1] = soc_max
        status = self.status_frame({"v_pack": v_pack_V, "cmd_current": current_A, "soc": [soc_mean],
                                    "state": state, "fault_label": fault})
        frames = [status, self.cell_extrema_frame(v, s),
                  self.thermal_frame(np.array([temp_min_C, temp_max_C]), cooling_duty)]
        frames.extend(self.sop_frame(sop[h]) for h in sorted(sop) if h in self.SOP_IDS)
        self._alive = (self._alive + 1) & 0xFF
        self._tx_count = (self._tx_count + len(frames) + 1) & 0xFFFF
        frames.append(self.health_frame(status.data))
        return frames


@dataclass
class PublishStats:
    """What a publishing run sent."""

    samples: int = 0
    frames: int = 0
    per_id: dict[int, int] = field(default_factory=dict)
    sim_duration_s: float = 0.0
    wall_s: float = 0.0
    # The SOP power fields are 16-bit watts in the shipped DBC (max 65,535 W), so a pack whose
    # limit is higher saturates on the wire; this counts the samples where that happened.
    sop_saturated_samples: int = 0

    @property
    def frames_per_sample(self) -> float:
        return self.frames / self.samples if self.samples else 0.0


class EVCanPublisher:
    """Publishes a drive run's telemetry as BMS CAN frames on a transport."""

    def __init__(self, transport: CanTransport, *, bus: EVCanBus | None = None) -> None:
        self.transport = transport
        self.bus = bus or EVCanBus()

    def frames_for(self, run: DriveRun, i: int) -> list[CANFrame]:
        """The broadcast cycle for channel index ``i`` (advances the alive counter)."""
        c = run.channels
        driving = c["speed_mps"][i] > 0.0 or c["speed_target_mps"][i] > 0.0
        return self.bus.broadcast_summary(
            v_pack_V=float(c["v_pack_V"][i]), current_A=float(c["current_A"][i]),
            soc_mean=float(c["soc_mean"][i]), soc_min=float(c["soc_min"][i]), soc_max=float(c["soc_max"][i]),
            n_groups=run.config.pack.n_series, cell_v_min_V=float(c["cell_v_min_V"][i]),
            cell_v_max_V=float(c["cell_v_max_V"][i]), temp_min_C=float(c["temp_min_C"][i]),
            temp_max_C=float(c["temp_max_C"][i]), cooling_duty=float(c["cooling_duty"][i]),
            state="operating" if driving else "idle", fault="none", sop=run.sop_at(i))

    def publish_sample(self, run: DriveRun, i: int) -> list[CANFrame]:
        frames = self.frames_for(run, i)
        for f in frames:
            self.transport.send(f)
        return frames

    def publish_run(self, run: DriveRun, *, telemetry_dt_s: float | None = None, realtime: bool = False,
                    speed: float = 1.0, sleep=time.sleep, after_sample=None) -> PublishStats:
        """Send one broadcast cycle per telemetry sample.

        With ``realtime`` the publisher waits so that sample ``k`` leaves at ``t_k / speed`` seconds
        of wall time; otherwise it sends as fast as the transport accepts.  ``after_sample(i, frames)``
        runs after each sample, e.g. to drain a receiver so a bounded kernel buffer cannot overflow.
        """
        stats = PublishStats()
        t_start = time.perf_counter()
        for i in run.sample_indices(telemetry_dt_s):
            if realtime:
                due = float(run["t_s"][i]) / speed
                wait = due - (time.perf_counter() - t_start)
                if wait > 0.0:
                    sleep(wait)
            frames = self.publish_sample(run, int(i))
            for f in frames:
                stats.frames += 1
                stats.per_id[f.arbitration_id] = stats.per_id.get(f.arbitration_id, 0) + 1
            if any(lim.traction_power_W > _U16_MAX_W or lim.regen_power_W > _U16_MAX_W
                   for lim in run.sop_at(int(i)).values()):
                stats.sop_saturated_samples += 1
            stats.samples += 1
            if after_sample is not None:
                after_sample(int(i), frames)
        stats.sim_duration_s = float(run["t_s"][-1])
        stats.wall_s = time.perf_counter() - t_start
        return stats


# ── Independent decode and verification ────────────────────────────────
def load_dbc(path: str | Path | None = None):
    """The BMS DBC as a ``cantools`` database: the shipped ``bms.dbc`` if given, else the encoder's own."""
    import cantools
    if path is not None:
        return cantools.database.load_file(str(path))
    return cantools.database.load_string(BMSCanBus().to_dbc(), database_format="dbc")


def decode_frames(frames, db=None) -> list[dict]:
    """Decode frames with ``cantools`` and the DBC: ``{"message": name, **signals}`` per frame."""
    db = db or load_dbc()
    out = []
    for f in frames:
        msg = db.get_message_by_frame_id(f.arbitration_id)
        out.append({"message": msg.name, **db.decode_message(f.arbitration_id, f.data)})
    return out


# Half of each signal's quantization step: the largest error the encoding itself can introduce.
_TOLERANCE = {"pack_voltage_V": 0.005, "pack_current_A": 0.05, "soc_pct": 0.05,
              "cell_voltage_min_V": 0.0005, "cell_voltage_max_V": 0.0005,
              "temperature_min_C": 0.05, "temperature_max_C": 0.05, "cooling_duty_pct": 0.5}


@dataclass(frozen=True)
class CanFidelityReport:
    """How closely the decoded bus traffic matches the simulation."""

    samples_compared: int
    max_abs_error: dict[str, float]
    tolerance: dict[str, float]
    missing: int

    @property
    def ok(self) -> bool:
        return self.missing == 0 and all(self.max_abs_error[k] <= self.tolerance[k] + 1e-9
                                         for k in self.max_abs_error)


def verify_against_telemetry(run: DriveRun, frames: list[CANFrame], *, telemetry_dt_s: float | None = None,
                             db=None) -> CanFidelityReport:
    """Decode ``frames`` (in send order) and compare every signal with the run, sample by sample.

    ``missing`` counts samples whose status, cell, or thermal frame is absent, so a lossy bus
    shows up as missing data rather than as a quietly shorter comparison.
    """
    decoded = decode_frames(frames, db)
    by_msg: dict[str, list[dict]] = {}
    for d in decoded:
        by_msg.setdefault(d["message"], []).append(d)
    c = run.channels
    idx = run.sample_indices(telemetry_dt_s)
    worst = {k: 0.0 for k in _TOLERANCE}
    missing = 0
    sources = {
        "BMS_Status": {"pack_voltage_V": "v_pack_V", "pack_current_A": "current_A"},
        "BMS_CellExtrema": {"cell_voltage_min_V": "cell_v_min_V", "cell_voltage_max_V": "cell_v_max_V"},
        "BMS_Thermal": {"temperature_min_C": "temp_min_C", "temperature_max_C": "temp_max_C",
                        "cooling_duty_pct": None},
    }
    for msg, signals in sources.items():
        got = by_msg.get(msg, [])
        missing += max(0, len(idx) - len(got))
        for d, i in zip(got, idx):
            for sig, chan in signals.items():
                want = float(c[chan][i]) if chan else float(c["cooling_duty"][i]) * 100.0
                worst[sig] = max(worst[sig], abs(float(d[sig]) - want))
    for d, i in zip(by_msg.get("BMS_Status", []), idx):
        worst["soc_pct"] = max(worst["soc_pct"], abs(float(d["soc_pct"]) - float(c["soc_mean"][i]) * 100.0))
    return CanFidelityReport(len(idx), worst, dict(_TOLERANCE), missing)
