# Aging-model validation against measured cells

The repository's `AgingModel` ships *literature-plausible* coefficients. This
page reports how those coefficients — and two fitted alternatives — perform on
measured capacity-fade data, scored **leave-one-cell-out**: each cell is predicted
by models fitted to the other seven. A model that only reproduces its own training
data is not evidence of anything, so every number here is on a held-out cell.

Reproduce:

```bash
python scripts/validate_aging.py                 # downloads (cached) and prints the tables
python scripts/validate_aging.py --write-csv     # also refreshes the committed derived CSV
```

## Data

Eight cells from the NASA Ames PCoE Li-ion aging dataset (public domain, US
Government). Capacity per cycle is taken from the measured discharges, despiked
with a robust median/MAD filter (`bms.signal.hampel`), and normalised to each cell's
first-cycle capacity. The stress rate is **measured**, from the 90th percentile of
each discharge's current against the 2 Ah rating:

| Cell | C-rate | Cycles | SoH at end | Cycle to 80 % |
|---|---|---|---|---|
| B0005 | 1.0 C | 168 | 0.71 | 100 |
| B0006 | 1.0 C | 168 | 0.58 | 60 |
| B0007 | 1.0 C | 168 | 0.76 | 124 |
| B0018 | 1.0 C | 132 | 0.72 | 74 |
| B0028 | 1.0 C | 28 | 0.96 | not reached |
| B0025 | 2.0 C | 28 | 0.96 | not reached |
| B0026 | 2.0 C | 28 | 0.98 | not reached |
| B0027 | 2.0 C | 28 | 0.98 | not reached |

B0028 sits in the "2 C" batch by name but was measured at 1 C; the labels here
come from the data, not the batch names.

## Models

- **Literature** — the repository's shipped `AgingModel` per-cycle fade, applied unchanged.
- **Linear pooled** — `SoH = 1 − k·n`, one slope for every cell (no stress term).
- **Stress power law** — `SoH = 1 − exp(α + β·ln C)·nᵇ`, fitted by nonlinear least squares.

## Results (leave-one-cell-out)

| Model | Mean SoH RMSE | EOL error, 1 C cells |
|---|---|---|
| Literature (shipped defaults) | **10.1 %** | **≈ +1,900 cycles** (predicts end-of-life far too late) |
| Linear pooled | **3.9 %** | within ±50 cycles |
| Stress power law | **3.9 %** | within ±50 cycles |

What the numbers say:

1. **The shipped literature defaults are wrong for this data**, by roughly an
   order of magnitude in predicted life. They are a reasonable prior for
   well-behaved field cells, not for these accelerated test cells. This is the
   main finding, and it is a correction to the repository's own defaults.
2. **Fitting to measured data removes most of that error** (10 % → 4 %), and
   the 1 C end-of-life predictions land within tens of cycles.
3. **A stress term does not help on these cells.** The fitted C-rate exponent is
   negative (β ≈ −0.56): the 2 C cells fade *more slowly* over their first 28
   cycles. With only 28 cycles and one cell per batch of evidence, C-rate is
   confounded with batch and cell lot. The data cannot support a stress law, and
   we do not claim one.

## Uncertainty is overconfident

A nominal 95 % interval from the residual spread should cover 95 % of held-out
points. It does not:

| Model | 95 % interval coverage | Verdict | Scale to make nominal |
|---|---|---|---|
| Linear pooled | 0.77 | overconfident | ×1.93 |
| Stress power law | 0.74 | overconfident | ×1.94 |

The residual spread underestimates cell-to-cell variation, so the intervals are
about half as wide as they should be. `bms.calibration_scale` returns that
correction. The scale above is computed on the same held-out cells it is
reported on, so it is an *upper bound on the fix*, not a validated correction:
a nested evaluation (calibrate the scale on the training cells only) is the next
step and is not done here.

## Limitations

- **Eight cells, two stress levels, mostly 1 C.** The 2 C evidence is 28 cycles
  from three cells. Stress-dependence claims need a dataset with matched lots and
  a wider range of C-rates and temperatures.
- **Single chemistry (LCO-type 18650) and single laboratory.** Generalisation to
  other chemistries or manufacturers is not tested here.
- **Capacity-only.** Resistance growth and degradation mode are not modelled in
  this validation.
- **No calendar ageing or temperature history.** Every cell was cycled at 24 °C.
