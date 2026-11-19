"""Run a clearly labelled interface demonstration for Model 01."""
from btms_core.contracts.types import BatteryStatePacket, Signal, SourceType, Uncertainty
from btms_core.models.registry import ModelRegistry
from btms_core.models.internal_micro_short import InternalMicroShortModel
from btms_core.validation.validator import validate_packet


def main():
    # Illustrative arithmetic inputs only. This is not measured cell data,
    # a physical fault simulation, or evidence of an actual micro-short.
    signals = {
        "isc_path_voltage_v": Signal("isc_path_voltage_v", 0.20, "V", 100.0,
            "illustrative_path_voltage_channel", SourceType.MEASURED,
            uncertainty=Uncertainty(standard=0.001)),
        "isc_path_current_a": Signal("isc_path_current_a", 0.010, "A", 100.0,
            "illustrative_path_current_channel", SourceType.MEASURED,
            uncertainty=Uncertainty(standard=0.0001)),
        "isc_observation_duration_s": Signal("isc_observation_duration_s", 10.0, "s", 100.0,
            "illustrative_duration", SourceType.CONFIGURED),
        "isc_effective_volume_m3": Signal("isc_effective_volume_m3", 1e-6, "m^3", 100.0,
            "illustrative_defect_volume", SourceType.CONFIGURED),
    }
    packet = BatteryStatePacket("ILLUSTRATIVE_CELL", "LFP_GRAPHITE", 100.0, signals)
    registry = ModelRegistry()
    registry.register(InternalMicroShortModel())

    for model in registry.all():
        report = validate_packet(packet, (*model.required_inputs, *model.optional_inputs), max_signal_age_s=2.0)
        result = model.evaluate(packet, report)
        print(f"\n{result.model_id} v{result.model_version}")
        print(f"Status: {result.status.value}")
        print(f"Identifiability: {result.identifiability.value}")
        for key, quantity in result.magnitude.items():
            print(f"{key}: {quantity.value:.8g} {quantity.unit}")
        print(f"Path heat: {result.heat_power_w.value:.8g} W" if result.heat_power_w.available else "Path heat: unavailable")
        print(f"Path energy: {result.cumulative_heat_j.value:.8g} J" if result.cumulative_heat_j.available else "Path energy: unavailable")
        print(f"Heat density: {result.heat_density_w_m3.value:.8g} W/m^3" if result.heat_density_w_m3.available else "Heat density: unavailable")
        for item in result.diagnostics:
            print("-", item)


if __name__ == "__main__":
    main()
