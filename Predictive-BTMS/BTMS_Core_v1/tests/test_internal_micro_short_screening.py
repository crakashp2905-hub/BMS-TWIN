import unittest
from btms_core.contracts.types import (
    BatteryStatePacket, Signal, SourceType, Uncertainty, ModelStatus,
    DataSufficiency, Quality,
)
from btms_core.models.internal_micro_short.screening import (
    InternalMicroShortScreeningModel, _weighted_linear_trend,
)
from btms_core.validation.validator import validate_packet


class ScreeningTests(unittest.TestCase):
    def setUp(self):
        self.model = InternalMicroShortScreeningModel(
            rest_current_limit_a=0.01, max_temperature_span_k=0.5,
            max_sample_skew_s=0.1, minimum_samples=5)

    def packet(self, residuals=None, current=0.0, temp_values=None):
        residuals = residuals if residuals is not None else [0.0] * 8
        temp_values = temp_values if temp_values is not None else [298.15] * len(residuals)
        signals = {}
        for i, residual in enumerate(residuals):
            t = 100.0 + i * 600.0
            healthy = 3.3 - i * 0.00001
            entries = (
                ("cell_voltage_v", healthy + residual, "V", SourceType.MEASURED, "cell_v", 2e-5),
                ("healthy_reference_voltage_v", healthy, "V", SourceType.MODEL_DERIVED, "reference_v1", 2e-5),
                ("cell_temperature_k", temp_values[i], "K", SourceType.MEASURED, "temp", None),
                ("external_current_a", current, "A", SourceType.MEASURED, "current", None),
            )
            for channel, value, unit, source_type, source_id, sigma in entries:
                key = f"isc_screen.{channel}.{i:03d}"
                signals[key] = Signal(key, value, unit, t, source_id, source_type,
                    Quality.VALID, Uncertainty(standard=sigma))
        packet = BatteryStatePacket("CELL_TEST", "LFP_GRAPHITE", 100.0 + (len(residuals)-1)*600.0, signals)
        return packet

    def evaluate(self, packet):
        report = validate_packet(packet, (), None)
        return self.model.evaluate(packet, report)

    def test_negative_residual_trend_is_candidate_not_detection(self):
        packet = self.packet([0.0, -0.0002, -0.0004, -0.0006, -0.0008, -0.0010, -0.0012, -0.0014])
        result = self.evaluate(packet)
        self.assertEqual(result.status, ModelStatus.CANDIDATE)
        self.assertEqual(result.data_sufficiency, DataSufficiency.SUFFICIENT)
        self.assertFalse(result.heat_power_w.available)
        self.assertEqual(result.identifiability.value, "NOT_IDENTIFIABLE")

    def test_flat_residual_has_no_candidate(self):
        result = self.evaluate(self.packet())
        self.assertEqual(result.status, ModelStatus.NO_CANDIDATE)

    def test_missing_sample_channel_is_insufficient(self):
        packet = self.packet()
        del packet.signals["isc_screen.external_current_a.003"]
        result = self.evaluate(packet)
        self.assertEqual(result.status, ModelStatus.INSUFFICIENT_DATA)

    def test_too_few_samples_is_insufficient(self):
        result = self.evaluate(self.packet([0.0, -0.001, -0.002, -0.003]))
        self.assertEqual(result.status, ModelStatus.INSUFFICIENT_DATA)

    def test_non_rest_current_rejected(self):
        result = self.evaluate(self.packet(current=0.02))
        self.assertEqual(result.status, ModelStatus.INVALID_INPUT)

    def test_temperature_span_rejected(self):
        result = self.evaluate(self.packet(temp_values=[298.0, 298.1, 298.2, 298.3, 298.4, 298.5, 298.6, 298.7]))
        self.assertEqual(result.status, ModelStatus.INVALID_INPUT)

    def test_voltage_uncertainty_required(self):
        packet = self.packet()
        old = packet.signals["isc_screen.cell_voltage_v.000"]
        packet.signals["isc_screen.cell_voltage_v.000"] = Signal(
            old.signal_id, old.value, old.unit, old.timestamp_s, old.source_id,
            old.source_type, old.quality, Uncertainty())
        result = self.evaluate(packet)
        self.assertEqual(result.status, ModelStatus.INVALID_INPUT)

    def test_reference_source_must_be_consistent(self):
        packet = self.packet()
        old = packet.signals["isc_screen.healthy_reference_voltage_v.004"]
        packet.signals["isc_screen.healthy_reference_voltage_v.004"] = Signal(
            old.signal_id, old.value, old.unit, old.timestamp_s, "different_reference",
            old.source_type, old.quality, old.uncertainty)
        result = self.evaluate(packet)
        self.assertEqual(result.status, ModelStatus.INVALID_INPUT)

    def test_channel_timestamp_skew_rejected(self):
        packet = self.packet()
        old = packet.signals["isc_screen.cell_temperature_k.002"]
        packet.signals["isc_screen.cell_temperature_k.002"] = Signal(
            old.signal_id, old.value, old.unit, old.timestamp_s + 1.0,
            old.source_id, old.source_type, old.quality, old.uncertainty)
        result = self.evaluate(packet)
        self.assertEqual(result.status, ModelStatus.INVALID_INPUT)

    def test_suspect_quality_rejected(self):
        packet = self.packet()
        old = packet.signals["isc_screen.external_current_a.001"]
        packet.signals["isc_screen.external_current_a.001"] = Signal(
            old.signal_id, old.value, old.unit, old.timestamp_s, old.source_id,
            old.source_type, Quality.SUSPECT, old.uncertainty)
        result = self.evaluate(packet)
        self.assertEqual(result.status, ModelStatus.INVALID_INPUT)

    def test_regression_requires_solvable_time_axis(self):
        with self.assertRaises(ValueError):
            _weighted_linear_trend([1, 1, 1, 1, 1], [1, 2, 3, 4, 5], [0.1]*5)

    def test_invalid_configuration_rejected(self):
        with self.assertRaises(ValueError):
            InternalMicroShortScreeningModel(rest_current_limit_a=-1, max_temperature_span_k=0.5)


if __name__ == "__main__":
    unittest.main()
