# Virtual EV drive simulation

A time-domain electric vehicle that follows a speed profile, draws power from a
battery pack through a drivetrain, and reports what happened as timestamped telemetry.

!!! warning "Not validated against a physical vehicle"
    Everything here is a simulation with generic, illustrative parameters. Its energy
    accounting is checked for **internal consistency**, and its outputs for **physical
    plausibility**. Neither is validation against measured vehicle data, and no result on
    this page should be read as a prediction for a real car. The one external anchor is the
    drive cycles themselves, which are the published schedules.

## What it does

```
target speed v*(t)
   │  driver: feed-forward + speed-error term → wheel force command
   ▼
powertrain limits: motor torque/power, tyre grip, pack discharge/charge power
   │  regenerative braking blended with friction brakes (fades near standstill)
   ▼
wheel force → gear → motor + inverter loss model → DC bus (+ auxiliary loads)
   │  bus power P
   ▼
VehiclePack: solve P = V(I)·I → current, SOC, polarization, temperature → terminal voltage
   ▼
telemetry (timestamped speed, SOC, voltage, current, temperature, power, …)
```

The older `bms.vehicle_dynamics.simulate_drive` solves the *inverse* problem (given the speed,
what power?) and assumes the car always keeps up. This module solves the *forward* problem:
when the pack, the motor or the tyres cannot deliver, the car falls behind the trace, and the
telemetry says which limit was responsible. The two agree where both apply: net motor-side
energy differs by 0.08–0.48 % on UDDS, HWFET, US06, ECE-15 and an urban stop-go cycle.

## Quick start

```python
import bms

run = bms.simulate_cycle("udds")                    # a bundled standard cycle
print(run.metrics()["consumption_Wh_per_km"])       # battery-terminal energy per km
print(run.ledger.relative_residuals())              # how well the energy accounting closes

records = run.telemetry(1.0)                        # 1 Hz TelemetryRecord list
cfg = bms.EVConfig(ambient_C=-10.0, soc0=0.6)       # change the conditions
run = bms.EVSimulator(cfg).run(bms.load_cycle_csv("trip.csv", speed_unit="kmh"))
```

## Vehicle dynamics

Longitudinal force balance, with the rotating inertia folded into an effective mass:

`m_eff·a = F_wheel − F_brake − ½ρ·Cd·A·v² − Crr·m·g·cosθ − m·g·sinθ`

- **Mass, drag, rolling resistance, grade** are parameters of `bms.Vehicle`.
- **Traction** is limited by tyre-road friction, `μ·m·g·cosθ` on the driven axle.
- **The driver** applies the acceleration that reaches the next target sample exactly
  (a deadbeat controller), so a feasible trace is followed exactly and a limited car
  approaches the trace without overshoot. Set `driver_kp_per_s` to model a slower driver.
- **At rest** the car stays put unless the trace is about to move or the command exceeds
  grade plus rolling resistance. A stopping step applies exactly the force that brings the
  car to rest, so the energy balance still closes.

## Electric drivetrain

| Element | Model |
|---|---|
| Gearbox | One ratio, constant efficiency (0.97), applied as a loss in both directions |
| Motor | Constant torque to base speed, constant power beyond it, a speed limit |
| Motor losses | Copper `k_cu·τ²`, iron `k_fe·ω²`, controller `P_idle`; the ledger reports each |
| Inverter | `k_inv·|P_shaft|` |
| Regeneration | A fraction of motor torque, fading to zero below 1.5 m/s, capped by the pack's charge limit; the rest goes to friction brakes. Regeneration that would not recover net energy is skipped. |
| Auxiliaries | Low-voltage base load through a DC-DC converter, HVAC from the ambient-to-setpoint gap, and cooling pump power |

`bms.generic_compact_ev()` is the default vehicle: 1,500 kg, a 300 N·m / 110 kW motor, motor and
inverter efficiency of roughly 95 % at mid load falling to the mid 80s at very light load.
The legacy `Vehicle()` default has much smaller loss coefficients (about 97 % across the map);
it is unchanged.

## Battery coupling

`bms.VehiclePack` is the repository's second-order ECM vectorized over series groups: each group
is one equivalent cell with `R / n_parallel` and `Q × n_parallel`. A test pins it to
`SecondOrderECM` step for step (difference below 1e-9 V with temperature changing) and to
`BatteryPack` for parallel groups, both with the chemistry's static hysteresis on.

`P = V(I)·I` is solved on the model's own terminal voltage, so after the step the terminal power
equals the request to numerical precision. Discharge and charge are each limited by the
tightest of:

| Limit | Default |
|---|---|
| Current | 3 C discharge, 1.5 C charge |
| Cell voltage | the chemistry's min / max, set by the weakest series group |
| Power | optional caps |
| SOC | no discharge at the floor; charge allowance tapers from 90 % to the ceiling |
| Temperature | derates both directions from 45 °C to zero at 60 °C; limits charge below 10 °C, to zero at −10 °C |

The limit reports **which constraint binds**. Telemetry carries it as `limit_flags`:
`pack_discharge_limit`, `motor_limit`, `traction_limit`, `regen_pack_limit`, `regen_motor_limit`,
`soc_empty`, `thermal_derate` (see `bms.describe_flags`).

## Drive cycles

| Name | Source | Notes |
|---|---|---|
| `udds`, `hwfet`, `us06` | EPA dynamometer schedules | bundled unchanged, checksummed |
| `ece15`, `eudc` | UN/ECE Reg 83 parts, as published by the EPA | bundled unchanged |
| `nedc` | four ECE-15 parts then one EUDC part | a convenience composite, not a certified test |
| `urban_stop_go`, `arterial`, `highway`, `cruise_100kmh` | synthetic | simple shapes |

Provenance, checksums and the published statistics each file was checked against are in
`bms/cycles/README.md`. A custom profile comes from a CSV (`time_s`, `speed_kmh`, optional grade)
via `bms.load_cycle_csv`, in km/h, m/s or mph. Every constructor validates: NaN, negative speed,
time going backwards, and a unit mistake that implies a 30 m/s² launch all raise
`DriveCycleError` instead of producing a plausible-looking simulation.

## Telemetry

One `bms.TelemetryRecord` per sample. The field table in `bms/telemetry.py` drives validation,
the JSON Schema (`bms.telemetry.json_schema()`), the CSV header and this table.

| Field | Unit | Required | Meaning |
|---|---|---|---|
| `seq` | 1 | yes | Record number, contiguous from 0 |
| `t_s` | s | yes | Seconds since the start of the run |
| `timestamp` | ISO-8601 UTC | yes | Epoch plus `t_s`, millisecond precision |
| `speed_mps` | m/s | yes | Vehicle speed |
| `soc` | 1 | yes | Pack state of charge (mean), 0..1 |
| `voltage_V` | V | yes | Pack terminal voltage |
| `current_A` | A | yes | Pack current, positive = discharge |
| `temperature_C` | degC | yes | Pack temperature (mean) |
| `power_W` | W | yes | Pack terminal power, positive = discharge |
| `accel_mps2` | m/s^2 | no | Vehicle acceleration |
| `speed_target_mps` | m/s | no | Drive-cycle target speed |
| `odometer_m` | m | no | Distance since the start |
| `energy_Wh` | Wh | no | Net energy drawn from the pack since the start |
| `motor_torque_Nm` | N*m | no | Motor shaft torque, negative = regeneration |
| `cell_v_min_V`, `cell_v_max_V` | V | no | Lowest and highest series-group voltage |
| `soc_min`, `soc_max` | 1 | no | Lowest and highest series-group SOC |
| `temp_min_C`, `temp_max_C` | degC | no | Coldest and hottest series group |
| `limit_flags` | bitmask | no | Active power limits |
| `state` | enum | no | `idle`, `driving` or `fault` |

Non-finite numbers are rejected rather than serialized: JSON has no NaN, and a silently dropped
value would look like a healthy sensor. The default epoch is fixed (2026-01-01T00:00:00Z) so a
run is reproducible; pass `epoch_unix_s` to anchor it to a real clock.

## Energy accounting

Every step computes each power flow from the same force, speed and current. The ledger then sums
them independently:

```
pack terminal energy = Δ kinetic + Δ potential + aerodynamic + rolling + friction brakes
                       + gearbox + motor copper + motor iron + inverter + controller + auxiliaries
```

On the battery side, the chemical energy the pack gave up is computed two independent ways: as
the time integral of `OCV·I`, and as an exact integral of the OCV curve over the SOC drop, which
does not depend on the time stepping at all. Chemical energy equals terminal energy plus ohmic,
polarization and hysteresis losses, less the reversible temperature shift of the EMF.

Measured on the bundled cycles at the default 0.25 s step, as a fraction of the energy that
flowed:

| Residual | Typical value | What it is |
|---|---|---|
| Vehicle side | 1e-11 to 2e-10 | floating-point rounding only |
| Pack side | 1e-6 to 3e-5 | time-stepping error in the RC polarization states; it falls as the step shrinks (tested) |
| Chemical vs exact OCV integral | below 1e-5 | Riemann-sum error |

A closing ledger shows the bookkeeping is consistent: no energy is double-counted or lost
between the force balance, the drivetrain, the auxiliaries and the pack. It does not show that
the *parameters* match any real vehicle.

## Reproducibility

`run.manifest()` pins the package version, NumPy and SciPy versions, the cycle's fingerprint, a
hash of the full configuration, the seed, the step, and a SHA-256 digest of every output channel.
Two runs of the same configuration give bit-identical digests, including from a fresh
interpreter (tested). Cell-to-cell scatter is seeded. The per-cycle results, the convergence with the
internal step and these reproducibility checks are reported, with the numbers measured, in
[Virtual EV benchmark results](ev_benchmark.md).

## Fault injection

`EVSimulator(config, on_step=hook)` calls `hook(k, t_s, pack)` before every step, and the hook
may mutate the pack: raise a series group's resistance, drop its SOC, heat it. Faults enter
through the plant, so the limits, flags and ledger respond as they would to a real degradation.
The tests inject a weak cell mid-drive, a sudden cell imbalance, a thermal event, failed cooling
in a hot climate, a cold soak, an impossible hill, a dead battery and a pack too small to carry
the auxiliaries (which raises `PackPowerError` rather than returning a wrong answer). Sensor
faults on the telemetry stream (stuck, dropout, spike) are caught by the repository's
`SensorMonitor`.

## Known limitations

- Longitudinal only: no cornering, no weight transfer, one driven axle fraction.
- The motor is a lumped loss model with generic coefficients, not a measured efficiency map.
- The pack is one equivalent cell per series group; cells within a parallel group are identical.
- Battery heating has no active heater; there is no battery-to-cabin thermal coupling.
- Self-discharge, aging and calendar effects are not modeled over a drive.
- `BatteryPack` applies the static hysteresis offset to a parallel group by the sign of the group
  current. For identical cells that is the equivalent cell the vehicle pack computes, and the tests
  compare the two with hysteresis on. In a group of mismatched cells that circulate current of both
  signs the offset is not resolved cell by cell; the vehicle pack has no such cells.
