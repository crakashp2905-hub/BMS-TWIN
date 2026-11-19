# Predictive-BTMS

## Physics-Based Battery Precursor Evaluation Framework

**Project status:** Six-model software integration completed; external
battery dataset evaluation is the next milestone.\
**Current integration version:** Unified Data Interface v1.0 + Six-Model
Orchestrator v1.0\
**Implementation language:** Python\
**Scope:** Research/reference implementation --- not safety-certified
firmware.

------------------------------------------------------------------------

## 1. Project Overview

Predictive-BTMS is a source-independent, physics-based battery precursor
evaluation framework. It is being developed to evaluate battery
measurements and independently estimated states against six physical
precursor mechanisms, quantify effects where the supplied evidence and
model assumptions permit, and report uncertainty, identifiability,
evidence sufficiency and associated heat contributions.

The longer-term objective is to develop a predictive battery thermal
management framework that can eventually inform embedded BTMS software.
The current implementation is a Python research/reference pipeline. It
does **not** implement cooling actuation, embedded control, safety
functions or AI/ML.

The initial modelling focus is LFP/Graphite, with the architecture
intended to remain extensible to other lithium-ion chemistries once
appropriate cell-specific parameters and validation evidence are
available.

### Core engineering principles

-   Keep the ingestion contract independent of any one sensor, BMS,
    laboratory or simulation source.
-   Preserve model-specific evidence semantics; do not force
    incompatible model inputs into one generic signal schema.
-   Keep measured signals, estimated states, cell parameters and
    model-specific evidence distinct.
-   Carry units, timestamps, source/provenance and quality information
    with acquired records.
-   Do not silently replace missing values with zero or infer
    unsupported measurements.
-   Distinguish an evaluated condition, a conditional parameterised
    estimate, an evidence-gated result and an identifiable physical
    mechanism.
-   Do not report arbitrary confidence percentages or interpret
    screening scores as probabilities.
-   Keep local contact/path dissipation distinct from cell-level heat.
-   Do not sum heat contributions into total cell heat until physical
    ownership, time-window compatibility and double-counting have been
    established.
-   Treat successful software tests as software verification, not
    experimental validation.

------------------------------------------------------------------------

## 2. Six Precursor Mechanisms

The framework currently contains six standalone reference models and six
corresponding adapters.

  ----------------------------------------------------------------------------------
  ID                Precursor         Main evaluation basis    Important limitation
                    mechanism                                  
  ----------------- ----------------- ------------------------ ---------------------
  01                Internal          Direct voltage/current   Excess loss or path
                    micro-short       across the same          dissipation does not
                                      identified path, and/or  uniquely prove an
                                      independent rest-period  internal micro-short.
                                      charge-loss evidence     Ordinary terminal
                                                               V/I/T/SOC/SOH alone
                                                               cannot localise or
                                                               uniquely identify
                                                               one.

  02                Lithium plating   Negative-electrode       Terminal cell voltage
                                      potential versus Li/Li+  is not a substitute
                                      during charging,         for anode potential.
                                      operating-condition      A condition screen is
                                      screening and optional   not confirmation of
                                      paired stripping traces  plated lithium mass
                                                               or morphology.

  03                SEI growth        Temperature-dependent,   Produces a
                                      parameterised parabolic  conditional estimate
                                      growth model             from supplied
                                                               parameters and
                                                               temperature history;
                                                               it is not independent
                                                               event detection.

  04                Electrolyte       Parameterised reaction   Represents a
                    decomposition     pathway using            specified lumped
                                      temperature and          pathway, not the
                                      calibrated decomposition complete electrolyte
                                      overpotential            reaction network.

  05                Abnormal          Rest-window charge loss  Excess charge loss is
                    self-discharge    relative to a matched    not
                                      healthy baseline, with   mechanism-specific;
                                      independent SOC          it may reflect
                                      endpoints and            parasitic reactions,
                                      external-current         internal leakage,
                                      accounting               redox shuttles or
                                                               other causes.

  06                Abnormal contact  Voltage drop directly    Requires valid
                    resistance        across an identified     measurement geometry
                                      contact and synchronous  and reference
                                      current through the same matching. Incremental
                                      path, compared with a    heat is local to the
                                      matched healthy          contact, not cell
                                      reference                bulk heat.
  ----------------------------------------------------------------------------------

### Model-specific evidence requirements

The models intentionally retain their native input payloads:

-   **Model 01:** Direct-path voltage and current must refer to the same
    identified path. Its indirect rest route requires independent SOC
    endpoints, validated usable capacity, external-current accounting
    and a matched healthy self-discharge baseline.
-   **Model 02:** Anode potential must be measured against Li/Li+ or
    supplied by a validated electrochemical observer. Positive current
    denotes charging in the native model.
-   **Model 03:** Requires calibrated SEI growth, geometry, conductivity
    and capacity-mapping parameters, alongside time-aligned
    temperature/current samples.
-   **Model 04:** Requires a defined decomposition pathway, calibrated
    kinetic parameters, reaction-driving overpotential convention,
    stoichiometry and reaction enthalpy.
-   **Model 05:** SOC endpoints must be independent of the current
    integration being audited. The healthy baseline must be matched to
    the evaluated cell and rest conditions.
-   **Model 06:** Contact voltage must span the identified joint;
    current must be measured in the same path and time-aligned.
    Four-wire/Kelvin sensing is preferred, and the healthy reference
    must be appropriately matched.

The native model documentation and source code remain authoritative for
detailed field definitions, equations, assumptions and limitations.

------------------------------------------------------------------------

## 3. System Architecture

### 3.1 High-level execution flow

``` text
       BATTERY DATA SOURCES
  (BMS / Sensors / Lab / Simulation)
                  |
                  v
       UNIFIED BATTERY DATA PACKET
                  |
                  v
       UNIFIED DATA INTERFACE v1.0
       - Schema validation
       - Units and timestamps
       - Source and provenance
       - Quality checks
       - Measurements / states /
         cell parameters / evidence
                  |
                  v
          MODEL EVIDENCE ROUTER
                  |
        +---------+---------+
        |         |         |
        v         v         v
     Adapter 01 Adapter 02 Adapter 03
        |         |         |
        v         v         v
     Model 01  Model 02  Model 03

        +---------+---------+
        |         |         |
        v         v         v
     Adapter 04 Adapter 05 Adapter 06
        |         |         |
        v         v         v
     Model 04  Model 05  Model 06
        |         |         |
        +---------+---------+
                  |
                  v
        RESULT NORMALISATION
                  |
           +------+------+
           |             |
           v             v
      MODEL REPORTS   HEAT LEDGER
           |             |
           +------+------+
                  |
                  v
       UNIFIED EVALUATION REPORT
                  |
                  v
             JSON OUTPUT
```

### 3.2 Main software components

  -------------------------------------------------------------------------------------------
  Component                                               Responsibility
  ------------------------------------------------------- -----------------------------------
  `Unified_Data_Interface_v1.py`                          Validates the canonical
                                                          source-independent packet and
                                                          exposes enabled model-specific
                                                          evidence payloads.

  `Model01_Adapter_v1.py` ... `Model06_Adapter_v1.py`     Validate each model's evidence
                                                          block and provenance, then pass the
                                                          native payload to its corresponding
                                                          frozen model.

  `Model01_Internal_MicroShort_Final.py` ...              Standalone physics/reference
  `Model06_Abnormal_Contact_Resistance_Final.py`          calculations for the six precursor
                                                          mechanisms.

  `Unified_Battery_Precursor_Evaluation_Engine_v0_2.py`   Existing six-model result
                                                          normalisation, evidence gating,
                                                          execution reporting and heat-ledger
                                                          logic.

  `Unified_Six_Model_Orchestrator_v1.py`                  Integration entry point that
                                                          validates one unified packet,
                                                          invokes the six adapters, and
                                                          produces the consolidated
                                                          evaluation report.
  -------------------------------------------------------------------------------------------

### 3.3 Canonical packet contract

The Unified Data Interface v1.0 packet uses this top-level structure:

``` json
{
  "schema_version": "1.0.0",
  "packet_id": "UNIQUE_PACKET_ID",
  "cell": {
    "cell_id": "CELL_IDENTIFIER",
    "chemistry": "LFP_GRAPHITE"
  },
  "acquisition": {
    "session_id": "ACQUISITION_SESSION",
    "started_at_utc": "2026-11-13T12:00:00Z"
  },
  "measurements": [],
  "estimated_states": [],
  "cell_parameters": [],
  "model_evidence": {
    "model_01": {
      "enabled": true,
      "provenance": {
        "source": "SOURCE_IDENTIFIER",
        "acquisition_method": "METHOD",
        "timestamp_utc": "2026-11-13T12:00:00Z",
        "quality": "GOOD",
        "notes": "Evidence provenance"
      },
      "payload": {}
    }
  }
}
```

The example is a structural illustration, not a complete runnable
six-model packet. Each enabled model requires its own native payload.

The interface currently validates: - Required top-level fields and
schema version. - Cell identity and chemistry. - Acquisition session and
timezone-aware timestamp. - Measurement, estimated-state and
cell-parameter records. - Canonical units, finite numeric values,
timestamps, source identifiers and quality. - Uncertainty values and
uncertainty method when sigma is supplied. - Estimated-state estimator
metadata. - Cell-parameter status metadata. - Model evidence block
structure and enabled flags.

**Important:** Unit conversion is not implicit in interface v1.0.
Unsupported units are rejected rather than silently converted.

### 3.4 Evidence routing and current integration boundary

The orchestrator currently passes explicit model-specific evidence
payloads through the corresponding adapters. Generic measurements are
**not automatically mapped** into model-specific evidence fields.

This is deliberate. A generic terminal voltage, current or temperature
channel must not be substituted for a physically different quantity such
as anode potential versus Li/Li+, voltage across a specific contact, or
independently estimated SOC.

The next development stage will investigate dataset-specific signal
mapping and evidence availability without weakening the model-specific
requirements.

------------------------------------------------------------------------

## 4. Heat Accounting

The system maintains a traceability-oriented heat ledger. It records
only quantities directly returned by the applicable model and preserves
their physical ownership and scale.

Current heat contributors:

  -----------------------------------------------------------------------
  Model                   Heat quantity           Scale / interpretation
                          represented             
  ----------------------- ----------------------- -----------------------
  01                      Directly measured path  Identified conductive
                          power and energy, when  path; not proof of
                          available               micro-short origin

  03                      Incremental SEI ohmic   Conditional
                          heat                    reduced-order estimate;
                                                  not total SEI reaction
                                                  heat

  04                      Reaction enthalpy heat  Conditional estimate
                                                  for the specified
                                                  reaction pathway; sign
                                                  is meaningful

  06                      Incremental contact     Local
                          heat relative to        identified-contact
                          healthy reference       heat; not cell-average
                                                  heat
  -----------------------------------------------------------------------

The heat ledger deliberately reports:

``` json
{
  "aggregate_cell_heat_J": null,
  "aggregation_status": "NOT_AGGREGATED_POTENTIAL_OVERLAP_AND_SCALE_MISMATCH"
}
```

A traceability subtotal is **not** total cell heat. Contributions may
have different spatial scales, overlapping physical ownership or
incompatible time windows. They must not be combined with baseline
electrothermal heat without a formal double-counting and
thermal-boundary audit.

Missing heat is not zero. A lumped or cell-average estimate does not
establish a local hotspot temperature.

------------------------------------------------------------------------

## 5. Work Completed

### 5.1 Physics/reference models

-   Six standalone precursor model scripts established.
-   Native input contracts and model-specific limitations retained.
-   Individual model test suites were run during model development.
-   Models remain research/reference implementations with parameter and
    evidence requirements.

### 5.2 Adapters

-   Six adapters created against Unified Data Interface v1.0.
-   Each adapter checks whether its model evidence block is enabled.
-   Each adapter applies a provenance quality gate.
-   Each adapter validates the expected native payload structure.
-   Each adapter passes the model-specific payload to its native model
    without flattening or guessing field mappings.
-   Individual synthetic adapter integration tests passed for all six
    models.

### 5.3 Unified interface and orchestrator

-   Unified Data Interface v1.0 implemented and validated.
-   Six-model orchestrator implemented.
-   Existing evaluation engine result-normalisation and heat-ledger
    logic reused.
-   One synthetic packet containing evidence for all six models executed
    through the integrated pipeline.
-   Consolidated JSON evaluation report generated.
-   Integration output reported no model execution errors.
-   Existing engine self-tests passed.

### 5.4 Synthetic six-model integration outcome

The completed synthetic integration run reported:

  --------------------------------------------------------------------------------
  Model                               Reported status
  ----------------------------------- --------------------------------------------
  01 -- Internal micro-short          `EVALUATED`

  02 -- Lithium plating               `EVALUATED`

  03 -- SEI growth                    `CONDITIONAL_PARAMETERISED_ESTIMATE`

  04 -- Electrolyte decomposition     `CONDITIONAL_PARAMETERISED_ESTIMATE`

  05 -- Abnormal self-discharge       `EXCESS_SELF_DISCHARGE_CANDIDATE_RESOLVED`

  06 -- Abnormal contact resistance   `ABNORMAL_CONTACT_RESISTANCE_RESOLVED`
  --------------------------------------------------------------------------------

The consolidated report recorded:

``` json
{
  "model_execution_errors": {},
  "heat_ledger": {
    "aggregate_cell_heat_J": null,
    "aggregation_status": "NOT_AGGREGATED_POTENTIAL_OVERLAP_AND_SCALE_MISMATCH"
  }
}
```

These are results from synthetic software fixtures. They demonstrate
that the software components execute together; they do **not**
demonstrate experimental detection accuracy or validate model parameters
for a real target cell.

------------------------------------------------------------------------

## 6. Repository Structure

The relevant project structure is:

``` text
Predictive-BTMS/
|
+-- LFP_Model/
|   +-- electrochemical/
|       +-- electrothermal_baseline.py
|       +-- electrochemical_state_extractor.py
|
+-- Precursor_Event_Models/
|   +-- Model01_Adapter_v1.py
|   +-- Model01_Internal_MicroShort_Final.py
|   +-- Model02_Adapter_v1.py
|   +-- Model02_Lithium_Plating_Final.py
|   +-- Model03_Adapter_v1.py
|   +-- Model03_SEI_Growth_Final.py
|   +-- Model04_Adapter_v1.py
|   +-- Model04_Electrolyte_Decomposition_Final.py
|   +-- Model05_Adapter_v1.py
|   +-- Model05_Abnormal_Self_Discharge_Final.py
|   +-- Model06_Adapter_v1.py
|   +-- Model06_Abnormal_Contact_Resistance_Final.py
|   +-- Unified_Data_Interface_v1.py
|   +-- Unified_Battery_Precursor_Evaluation_Engine_v0_2.py
|   +-- Unified_Six_Model_Orchestrator_v1.py
|   +-- battery_packet/
|   |   +-- six_model_synthetic_integration_packet.json
|   |   +-- [individual model test packets]
|   +-- results/
|       +-- six_model_synthetic_integration_report.json
|
+-- results/
+-- battery_packet/
+-- README.md
```

This is the intended layout for the files relevant to the current
integration. Other project folders and files may exist.

------------------------------------------------------------------------

## 7. Running the Six-Model Orchestrator

### 7.1 Prerequisites

-   Python environment used for the project.
-   Existing six native model scripts.
-   Six adapter scripts.
-   Unified Data Interface v1.0.
-   Existing evaluation engine v0.2.
-   Synthetic integration packet.

Run commands from the `Predictive-BTMS` project root unless otherwise
stated.

### 7.2 Run synthetic integration

``` powershell
python .\Precursor_Event_Models\Unified_Six_Model_Orchestrator_v1.py --input .\Precursor_Event_Models\battery_packet\six_model_synthetic_integration_packet.json --output .\Precursor_Event_Models\results\six_model_synthetic_integration_report.json
```

Expected output file:

``` text
Precursor_Event_Models/results/six_model_synthetic_integration_report.json
```

The successful run should include six model reports and an empty
`model_execution_errors` object. Individual model statuses depend on the
supplied evidence and are not all expected to be identical.

### 7.3 Run the existing engine self-tests

``` powershell
python .\Precursor_Event_Models\Unified_Battery_Precursor_Evaluation_Engine_v0_2.py --self-test
```

### 7.4 Validate a Unified Data Interface packet independently

``` powershell
python .\Precursor_Event_Models\Unified_Data_Interface_v1.py --input .\Precursor_Event_Models\battery_packet\six_model_synthetic_integration_packet.json --output .\Precursor_Event_Models\results\interface_validation_report.json
```

------------------------------------------------------------------------

## 8. Known Limitations and Current Boundaries

1.  **Synthetic evidence:** The completed integrated run uses synthetic
    test fixtures, not a real battery experiment.
2.  **No target-cell validation:** Model parameters and thresholds have
    not been established as validated for a specific production cell.
3.  **Explicit model payloads:** Generic acquired measurements are not
    automatically translated into the six native model evidence schemas.
4.  **Evidence availability:** Ordinary BMS channels will not provide
    every required quantity. Examples include anode potential versus
    Li/Li+, direct contact voltage drop, independent SOC endpoints and
    pathway-specific decomposition overpotential.
5.  **Conditional estimates:** SEI growth and electrolyte decomposition
    outputs depend on supplied parameterisations; they are not
    independent defect detectors.
6.  **Mechanism identifiability:** Excess rest-period loss does not
    uniquely identify micro-short or any other single mechanism.
7.  **Heat aggregation:** The heat ledger does not calculate total cell
    heat or local hotspot temperature.
8.  **No actuation:** No cooling control, thermal-management actuation
    or embedded control loop is implemented.
9.  **No AI/ML yet:** Machine learning and deep reinforcement learning
    are intentionally deferred until physics-model integration, data
    ingestion and validation are established.
10. **Not safety-certified:** The code is not safety-certified firmware
    and must not be used as a safety function.

------------------------------------------------------------------------

## 9. Next Development Milestones

### Stage 1 --- Public battery dataset selection and audit

Investigate suitable public sources, including: - NASA Ames lithium-ion
battery ageing datasets. - CALCE battery research datasets. - Battery
Data Commons.

For each candidate dataset, inspect: - Cell chemistry and cell design. -
Test protocol and operating conditions. - Voltage, current and
temperature availability. - Sampling interval and synchronisation. -
Capacity/SOC/SOH reference methods. - Cell metadata and parameter
availability. - Whether any of the six precursor mechanisms have
independent ground-truth labels or supporting measurements.

Do not assume that a cycling/ageing dataset can validate all six
mechanisms.

### Stage 2 --- Dataset ingestion

Build a dataset-specific ingestion layer that: - Reads the source data
without altering the original files. - Preserves original units,
timestamps and metadata. - Records source and acquisition provenance. -
Checks missing values, duplicates, sampling gaps and quality. - Maps
only semantically equivalent signals into the Unified Data Interface. -
Reports unavailable evidence explicitly.

### Stage 3 --- Evidence mapping and model evaluation

For each dataset, create a signal-availability matrix: - Directly
available. - Derivable with a documented and validated method. -
Available only through an independent observer or additional
instrumentation. - Not available.

Run only models whose evidence requirements are met. Keep
insufficient-data outcomes rather than fabricating inputs.

### Stage 4 --- Physics-model calibration and validation

Identify suitable reference experiments and measurements for each
mechanism. Separate parameter identification, calibration and
independent validation. Quantify uncertainty and systematic measurement
limitations.

### Stage 5 --- AI/ML integration

Only after the physics and data pipeline is stable: - Establish labelled
datasets and train/validation/test separation. - Evaluate suitable
anomaly-detection and predictive models. - Compare ML-only, physics-only
and hybrid approaches. - Consider DRL only when a well-defined control
problem, plant model, constraints and safety strategy exist.

### Stage 6 --- Embedded implementation

After algorithm and model validation: - Identify calculations suitable
for deterministic embedded execution. - Port appropriate algorithms to
C/C++. - Define timing, memory, numerical precision and fault-handling
requirements. - Develop hardware-in-the-loop and target-hardware
verification. - Treat functional-safety engineering as a separate formal
workstream.

------------------------------------------------------------------------

## 10. Development Status Summary

  -----------------------------------------------------------------------
  Workstream                          Status
  ----------------------------------- -----------------------------------
  Six standalone precursor reference  Completed
  models                              

  Six individual adapters             Completed; synthetic integration
                                      tests passed

  Unified Data Interface v1.0         Implemented

  Six-model orchestration             Implemented

  Result normalisation                Integrated

  Heat contribution ledger            Integrated; total cell heat
                                      intentionally not aggregated

  Synthetic end-to-end software       Passed
  integration                         

  External dataset selection          Next

  Dataset ingestion and evidence      Pending
  mapping                             

  Experimental calibration and        Pending
  validation                          

  AI/ML integration                   Deferred

  Embedded C/C++ implementation       Future work

  BTMS cooling actuation              Not implemented
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## 11. Project Philosophy

Predictive-BTMS is being developed around a central engineering
principle:

**A model should only report what its available evidence and physical
assumptions can support.**

The framework is intended to distinguish what is measured, what is
estimated, what is conditionally modelled, what is identifiable and what
remains unknown. Its purpose is not to produce a fault label at any
cost, but to build a traceable, physics-grounded evaluation pipeline
that can be progressively tested against real battery data.
