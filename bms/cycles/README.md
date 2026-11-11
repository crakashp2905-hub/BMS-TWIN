# Bundled standard drive cycles

These five speed-versus-time schedules are bundled **unchanged** so the drive-cycle
runner can replay standardized profiles without a download. They are loaded by
`bms.drive_cycles.get_cycle(...)`.

| Cycle | File | Source file on epa.gov | Bytes | SHA-256 |
|---|---|---|---|---|
| UDDS (urban) | `uddscol.txt` | `sites/default/files/2015-10/uddscol.txt` | 12,611 | `cc9f3edf2f0626976a6552d9f5c88867e1805ed3f321df9ec943a3e0353687b6` |
| HWFET (highway) | `hwycol.txt` | `sites/default/files/2015-10/hwycol.txt` | 7,266 | `e48e278a40312dfe84d31b5f448b601e7046c72487a6a20d3db1d4f85fb3e10a` |
| US06 (aggressive) | `us06col.txt` | `system/files/other-files/2025-03/us06col.txt` | 5,700 | `84e9d8cc16fcd4ee3c30c3880f333eb25b61352e56f24e453411a7245c725ad2` |
| ECE-15 (urban, one part) | `ececol.txt` | `sites/default/files/2015-10/ececol.txt` | 1,609 | `4c04507a838e4774694daa7f03811a861ab361d2267acee3c31677f4668a2e8f` |
| EUDC (extra-urban) | `eudccol.txt` | `sites/default/files/2015-10/eudccol.txt` | 4,835 | `dc557fbfcebeb33889822864bd2937377ffea2295270c5485de5729488bec43a` |

Source page: <https://www.epa.gov/vehicle-and-fuel-emissions-testing/dynamometer-drive-schedules>
(retrieved 2026-11-12).

## Provenance and licence

The files are published by the U.S. Environmental Protection Agency. Works of the
U.S. Government are generally not subject to copyright in the United States
(17 U.S.C. § 105). The ECE-15 and EUDC schedules are the UN/ECE Regulation 83
cycles, reproduced here as the EPA publishes them. The files are bundled byte for
byte (`.gitattributes` disables line-ending conversion for them) so the checksums
above stay valid; a test checks them.

## Checked against the published statistics

Duration, distance (sum of speeds × 1 s) and peak speed, computed from the bundled
files:

| Cycle | Duration | Distance | Peak speed |
|---|---|---|---|
| UDDS | 1,369 s | 7.45 mi | 56.7 mph |
| HWFET | 765 s | 10.26 mi | 59.9 mph |
| US06 | 600 s | 8.01 mi | 80.3 mph |
| ECE-15 | 195 s | 0.994 km | 50 km/h |
| EUDC | 400 s | 6.955 km | 120 km/h |

## Format

Two header lines, then tab-separated `time_s  speed` rows at 1 Hz. The second header
line names the speed unit (`mph` or `kph`); the loader reads it from there.

The `nedc` cycle in the registry is a composite built from these files: four ECE-15
parts followed by one EUDC part. It is a convenience composite, not a certified
type-approval test.
