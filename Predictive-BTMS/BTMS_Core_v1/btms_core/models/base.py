from abc import ABC, abstractmethod
from btms_core.contracts.types import BatteryStatePacket, PrecursorResult, RequiredInput
from btms_core.validation.validator import ValidationReport

class PrecursorModel(ABC):
    model_id="UNDEFINED"
    model_version="0.0.0"
    @property
    @abstractmethod
    def required_inputs(self): ...
    @property
    def optional_inputs(self): return ()
    @abstractmethod
    def evaluate(self, packet: BatteryStatePacket, validation: ValidationReport) -> PrecursorResult: ...
