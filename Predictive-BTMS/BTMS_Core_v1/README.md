# Predictive-BTMS Core v1.0

Source-independent Python reference architecture. Python 3.10+, standard library only.

Included: canonical signal/state contracts, parameter provenance registry, packet validation, common precursor interface/registry, heat ledger with duplicate-owner protection, placeholder Model 01, smoke test and unit tests.

Not included: validated precursor equations, sensor/CAN drivers, ML/DRL, cooling actuation, or claims of physical validation. Example values are interface examples only.

Run tests: `python -m unittest discover -s tests -v`
Run smoke test: `python main.py`

Missing data is never silently replaced by zero. Signals carry units, timestamps, source type, quality and uncertainty. Model outputs distinguish event status, data sufficiency and identifiability. Heat contributions require explicit physical ownership to reduce double counting.

## Model 01 — Internal Micro-Short

The first physics module is in `btms_core/models/internal_micro_short/`. It requires direct voltage-across-path and current-through-path observables. It deliberately does not infer a micro-short from normal cell-terminal measurements. See that folder's README for equations, input contract, status semantics, and limitations.
