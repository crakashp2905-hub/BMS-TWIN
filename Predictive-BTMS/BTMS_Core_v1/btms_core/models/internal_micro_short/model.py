"""Model 01: internal micro-short path assessment.

This model intentionally requires direct observables for the suspected path.
It does not infer a micro-short from ordinary terminal-voltage residuals.
"""
from btms_core.contracts.types import (
    DataSufficiency, Identifiability, ModelStatus, PrecursorResult, Quantity,
)
from btms_core.models.base import PrecursorModel
from .requirements import REQUIRED_INPUTS, OPTIONAL_INPUTS
from .electrical import estimate_path
from .heat import estimate_heat


class InternalMicroShortModel(PrecursorModel):
    model_id = "INTERNAL_MICRO_SHORT"
    model_version = "1.0.0"

    @property
    def required_inputs(self):
        return REQUIRED_INPUTS

    @property
    def optional_inputs(self):
        return OPTIONAL_INPUTS

    def evaluate(self, packet, validation):
        if validation.sufficiency != DataSufficiency.SUFFICIENT:
            status = (ModelStatus.INVALID_INPUT if validation.sufficiency == DataSufficiency.INVALID
                      else ModelStatus.INSUFFICIENT_DATA)
            return self._unavailable(packet, status, validation.sufficiency,
                                     tuple(i.message for i in validation.issues))

        v_sig = packet.get_signal("isc_path_voltage_v") or packet.get_state("isc_path_voltage_v")
        i_sig = packet.get_signal("isc_path_current_a") or packet.get_state("isc_path_current_a")
        # A zero/negative reading is not converted into a resistance estimate.
        if v_sig.value <= 0 or i_sig.value <= 0:
            return self._unavailable(
                packet, ModelStatus.NO_CANDIDATE, DataSufficiency.SUFFICIENT,
                ("No positive dissipative path voltage/current pair was observed.",),
                identifiability=Identifiability.NOT_IDENTIFIABLE,
            )

        try:
            estimate = estimate_path(v_sig.value, i_sig.value,
                v_sig.uncertainty.standard, i_sig.uncertainty.standard)
        except ValueError as exc:
            return self._unavailable(packet, ModelStatus.INVALID_INPUT,
                DataSufficiency.INVALID, (str(exc),))

        # Candidate status is evidence of a measured path, not proof that its
        # physical origin is an internal micro-short rather than another shunt.
        current_sigma = i_sig.uncertainty.standard
        if current_sigma is not None and i_sig.value - current_sigma <= 0:
            status = ModelStatus.CANDIDATE
            diagnostic = "Positive path current observed, but its one-standard-uncertainty lower bound includes zero."
        else:
            status = ModelStatus.DETECTED
            diagnostic = "Dissipative current is directly observed in the declared conductive path. Physical origin still requires independent confirmation."

        duration_sig = packet.get_signal("isc_observation_duration_s") or packet.get_state("isc_observation_duration_s")
        volume_sig = packet.get_signal("isc_effective_volume_m3") or packet.get_state("isc_effective_volume_m3")
        duration = duration_sig.value if duration_sig else None
        volume = volume_sig.value if volume_sig else None
        try:
            heat = estimate_heat(estimate.power_w, duration, volume)
        except ValueError as exc:
            return self._unavailable(packet, ModelStatus.INVALID_INPUT,
                DataSufficiency.INVALID, (str(exc),))

        evidence = {
            "voltage_source_id": v_sig.source_id,
            "current_source_id": i_sig.source_id,
            "voltage_source_type": v_sig.source_type.value,
            "current_source_type": i_sig.source_type.value,
            "same_path_measurement_required": True,
            "physical_origin_confirmed": False,
            "uncertainty_assumption": "First-order propagation; voltage/current errors assumed independent.",
            "heat_interpretation": "Path electrical dissipation; not a cell-average temperature-rise prediction.",
        }
        return PrecursorResult(
            model_id=self.model_id, model_version=self.model_version,
            evaluation_timestamp_s=packet.evaluation_timestamp_s,
            status=status, data_sufficiency=DataSufficiency.SUFFICIENT,
            identifiability=Identifiability.PARTIALLY_IDENTIFIABLE,
            evidence=evidence,
            magnitude={
                "path_resistance": Quantity(estimate.resistance_ohm, "ohm"),
                "path_current": Quantity(i_sig.value, "A", i_sig.uncertainty),
                "path_voltage": Quantity(v_sig.value, "V", v_sig.uncertainty),
            },
            heat_power_w=Quantity(heat.power_w, "W"),
            heat_density_w_m3=(Quantity(heat.density_w_m3, "W/m^3") if heat.density_w_m3 is not None
                               else Quantity.unavailable("W/m^3", "Validated effective defect volume not supplied.")),
            cumulative_heat_j=(Quantity(heat.energy_j, "J") if heat.energy_j is not None
                               else Quantity.unavailable("J", "Representative observation duration not supplied.")),
            severity=None,
            diagnostics=(diagnostic,
                "R_path = V_path / I_path; P_path = V_path * I_path.",
                "No temperature-rise or local hotspot estimate is produced by this electrical-path model."),
            validity_domain=("Positive dissipative path voltage and current", "Same physical path for both measurements"),
        )

    def _unavailable(self, packet, status, sufficiency, diagnostics,
                     identifiability=Identifiability.NOT_ASSESSED):
        return PrecursorResult(
            model_id=self.model_id, model_version=self.model_version,
            evaluation_timestamp_s=packet.evaluation_timestamp_s,
            status=status, data_sufficiency=sufficiency,
            identifiability=identifiability,
            heat_power_w=Quantity.unavailable("W", "Required direct path observables unavailable."),
            heat_density_w_m3=Quantity.unavailable("W/m^3", "No valid path heat estimate."),
            cumulative_heat_j=Quantity.unavailable("J", "No valid path heat estimate."),
            diagnostics=diagnostics,
        )
