import unittest
from btms_core.contracts.types import (
    BatteryStatePacket, Signal, SourceType, Uncertainty, ModelStatus,
)
from btms_core.models.internal_micro_short import InternalMicroShortModel
from btms_core.models.internal_micro_short.electrical import estimate_path
from btms_core.models.internal_micro_short.heat import estimate_heat
from btms_core.validation.validator import validate_packet


def sig(name, value, unit, timestamp=10.0, sigma=None):
    return Signal(name, value, unit, timestamp, f"sensor_{name}", SourceType.MEASURED,
                  uncertainty=Uncertainty(standard=sigma))


class InternalMicroShortTests(unittest.TestCase):
    def setUp(self):
        self.model = InternalMicroShortModel()

    def evaluate(self, signals):
        packet = BatteryStatePacket("CELL_TEST", "LFP_GRAPHITE", 10.0, signals)
        report = validate_packet(packet, (*self.model.required_inputs, *self.model.optional_inputs), 2.0)
        return self.model.evaluate(packet, report)

    def test_electrical_equations_and_uncertainty(self):
        out = estimate_path(0.2, 0.01, 0.001, 0.0001)
        self.assertAlmostEqual(out.resistance_ohm, 20.0)
        self.assertAlmostEqual(out.power_w, 0.002)
        self.assertGreater(out.resistance_std_ohm, 0)
        self.assertGreater(out.power_std_w, 0)

    def test_heat_units(self):
        out = estimate_heat(0.002, 10.0, 1e-6)
        self.assertAlmostEqual(out.energy_j, 0.02)
        self.assertAlmostEqual(out.density_w_m3, 2000.0)

    def test_direct_path_observables_produce_result(self):
        result = self.evaluate({
            "isc_path_voltage_v": sig("isc_path_voltage_v", 0.2, "V"),
            "isc_path_current_a": sig("isc_path_current_a", 0.01, "A"),
            "isc_observation_duration_s": sig("isc_observation_duration_s", 10, "s"),
            "isc_effective_volume_m3": sig("isc_effective_volume_m3", 1e-6, "m^3"),
        })
        self.assertEqual(result.status, ModelStatus.DETECTED)
        self.assertAlmostEqual(result.magnitude["path_resistance"].value, 20.0)
        self.assertAlmostEqual(result.heat_power_w.value, 0.002)
        self.assertAlmostEqual(result.cumulative_heat_j.value, 0.02)
        self.assertAlmostEqual(result.heat_density_w_m3.value, 2000.0)

    def test_missing_path_measurements_do_not_infer_from_terminal_voltage(self):
        result = self.evaluate({"cell_voltage": sig("cell_voltage", 3.3, "V")})
        self.assertEqual(result.status, ModelStatus.INSUFFICIENT_DATA)
        self.assertFalse(result.heat_power_w.available)

    def test_uncertainty_overlaps_zero_gives_candidate(self):
        result = self.evaluate({
            "isc_path_voltage_v": sig("isc_path_voltage_v", 0.2, "V"),
            "isc_path_current_a": sig("isc_path_current_a", 0.001, "A", sigma=0.002),
        })
        self.assertEqual(result.status, ModelStatus.CANDIDATE)

    def test_zero_or_negative_values_rejected_by_equation(self):
        with self.assertRaises(ValueError): estimate_path(0.2, 0.0)
        with self.assertRaises(ValueError): estimate_path(-0.2, 0.01)

    def test_bad_duration_rejected(self):
        with self.assertRaises(ValueError): estimate_heat(0.1, 0.0)


if __name__ == "__main__":
    unittest.main()
