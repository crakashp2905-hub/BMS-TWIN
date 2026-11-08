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

## Uncertainty is overconfident, and the fix holds out of sample

A nominal 95 % interval from the residual spread should cover 95 % of held-out
points. It does not: the uncorrected intervals cover about 0.87 of held-out points
in the nested run below (0.77 when computed pooled across all cells).

The correction is therefore learned **without the held-out cell**. For each cell,
the scale comes from inner leave-one-cell-out predictions on the other cells
(`bms.aging_fit.nested_interval_calibration`), then is applied to the held-out cell:

| Model | Raw coverage | Nested-scaled coverage | Nominal |
|---|---|---|---|
| Linear pooled | 0.87 | **0.93** | 0.95 |
| Stress power law | 0.85 | **0.99** | 0.95 |

Per-cell scales range from about 1.8 to 3.3. B0006 is the hardest cell and drives
most of that spread. With eight cells, coverage estimates are coarse: a difference
of one cell moves them by about 0.06, so the figures are indicative, not precise.

## Second dataset: Oxford Battery Degradation Dataset 1

A different cell (Kokam 740 mAh pouch), lab and stress: eight cells aged at 40 °C
under an urban drive cycle to 5,000–8,200 cycles, with a C/1 characterisation
every 100 cycles. The aging current is not stated as a number, so the models use
C = 1 as an explicit assumption. Reproduce with
`python scripts/validate_oxford.py <path to the .mat file>`.

The data file is **not committed**: the Oxford dataset is licensed under the ODbL,
whose share-alike terms apply to derived databases. The repository is MIT licensed,
so the derived capacity table is not published here either. The tests skip when
the file is absent.

| Test | What it asks | Mean SoH RMSE |
|---|---|---|
| A. Shipped literature defaults at 40 °C | Do the defaults hold on another dataset? | **49 %** (predicts far too much fade) |
| B. Leave-one-cell-out, linear pooled | Do fits generalise across cells within Oxford? | **2.6 %** |
| B. Leave-one-cell-out, stress power law | Same, power law | **1.6 %** |
| C. Fit on NASA, predict Oxford | Does a model transfer across datasets? | **678–838 %** (fails) |

What the numbers say:

1. **The shipped defaults are inconsistent across datasets.** On NASA cells at
   24 °C they predict too little fade (about 10 % error); on Oxford cells at 40 °C
   they predict too much (about 49 % error). Neither dataset supports the defaults
   as a general prior.
2. **Within one dataset, fitting works.** Held-out cells in Oxford are predicted
   to within 2–3 %. Here the power law beats the linear model, the reverse of NASA,
   so the best model depends on the data.
3. **Across datasets, a fit from one dataset does not transfer.** A model fitted
   on the NASA cells cannot predict the Oxford cells, which cycled 50× as far and
   ran 16 °C hotter. Transfer needs covariates (temperature, throughput, C-rate)
   that these two datasets cannot yet estimate reliably. This is a negative result,
   and it is the main reason not to claim cross-chemistry or cross-lab accuracy.

## Limitations

- **Eight cells, two stress levels, mostly 1 C.** The 2 C evidence is 28 cycles
  from three cells. Stress-dependence claims need a dataset with matched lots and
  a wider range of C-rates and temperatures.
- **Single chemistry (LCO-type 18650) and single laboratory.** Generalisation to
  other chemistries or manufacturers is not tested here.
- **Capacity-only.** Resistance growth and degradation mode are not modelled in
  this validation.
- **No calendar ageing or temperature history.** Every cell was cycled at 24 °C.
