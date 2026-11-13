from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional

class SourceType(str, Enum):
    MEASURED="MEASURED"; ESTIMATED="ESTIMATED"; MODEL_DERIVED="MODEL_DERIVED"; CONFIGURED="CONFIGURED"
class Quality(str, Enum):
    VALID="VALID"; SUSPECT="SUSPECT"; INVALID="INVALID"; STALE="STALE"; UNAVAILABLE="UNAVAILABLE"
class ModelStatus(str, Enum):
    NOT_EVALUATED="NOT_EVALUATED"; INSUFFICIENT_DATA="INSUFFICIENT_DATA"; INVALID_INPUT="INVALID_INPUT"
    NO_CANDIDATE="NO_CANDIDATE"; CANDIDATE="CANDIDATE"; DETECTED="DETECTED"; OUT_OF_DOMAIN="OUT_OF_DOMAIN"
class Identifiability(str, Enum):
    IDENTIFIABLE="IDENTIFIABLE"; PARTIALLY_IDENTIFIABLE="PARTIALLY_IDENTIFIABLE"
    NOT_IDENTIFIABLE="NOT_IDENTIFIABLE"; NOT_ASSESSED="NOT_ASSESSED"
class DataSufficiency(str, Enum):
    SUFFICIENT="SUFFICIENT"; INSUFFICIENT="INSUFFICIENT"; INVALID="INVALID"; STALE="STALE"

@dataclass(frozen=True)
class Uncertainty:
    standard: Optional[float]=None
    lower: Optional[float]=None
    upper: Optional[float]=None
    distribution: Optional[str]=None
    def __post_init__(self):
        if self.standard is not None and self.standard < 0: raise ValueError("Uncertainty cannot be negative")
        if self.lower is not None and self.upper is not None and self.lower > self.upper:
            raise ValueError("Uncertainty lower bound exceeds upper bound")

@dataclass(frozen=True)
class Signal:
    signal_id: str
    value: Optional[float]
    unit: str
    timestamp_s: Optional[float]
    source_id: str
    source_type: SourceType
    quality: Quality=Quality.VALID
    uncertainty: Uncertainty=field(default_factory=Uncertainty)
    calibration_id: Optional[str]=None
    def __post_init__(self):
        if not self.signal_id or not self.unit or not self.source_id:
            raise ValueError("signal_id, unit and source_id are required")

@dataclass(frozen=True)
class OperatingContext:
    mode: str="UNKNOWN"
    ambient_temperature_k: Optional[float]=None
    pack_id: Optional[str]=None
    module_id: Optional[str]=None

@dataclass(frozen=True)
class BatteryStatePacket:
    cell_id: str
    chemistry_id: str
    evaluation_timestamp_s: float
    signals: Mapping[str, Signal]
    states: Mapping[str, Signal]=field(default_factory=dict)
    context: OperatingContext=field(default_factory=OperatingContext)
    parameter_set_id: Optional[str]=None
    schema_version: str="1.0"
    def get_signal(self, key): return self.signals.get(key)
    def get_state(self, key): return self.states.get(key)

@dataclass(frozen=True)
class RequiredInput:
    name: str
    unit: str
    description: str=""
    source_types_allowed: tuple=(SourceType.MEASURED, SourceType.ESTIMATED, SourceType.MODEL_DERIVED)
    required: bool=True

@dataclass(frozen=True)
class Quantity:
    value: Optional[float]
    unit: str
    uncertainty: Uncertainty=field(default_factory=Uncertainty)
    available: bool=True
    reason_unavailable: Optional[str]=None
    def __post_init__(self):
        if not self.unit: raise ValueError("Quantity unit is required")
        if self.available and self.value is None: raise ValueError("Available quantity needs a value")
        if not self.available and self.value is not None: raise ValueError("Unavailable quantity cannot have a value")
    @classmethod
    def unavailable(cls, unit, reason): return cls(None, unit, available=False, reason_unavailable=reason)

@dataclass(frozen=True)
class PrecursorResult:
    model_id: str
    model_version: str
    evaluation_timestamp_s: float
    status: ModelStatus
    data_sufficiency: DataSufficiency
    identifiability: Identifiability=Identifiability.NOT_ASSESSED
    evidence: Mapping[str, Any]=field(default_factory=dict)
    magnitude: Mapping[str, Quantity]=field(default_factory=dict)
    heat_power_w: Quantity=field(default_factory=lambda: Quantity.unavailable("W","Not calculated"))
    heat_density_w_m3: Quantity=field(default_factory=lambda: Quantity.unavailable("W/m^3","Not calculated"))
    cumulative_heat_j: Quantity=field(default_factory=lambda: Quantity.unavailable("J","Not calculated"))
    severity: Optional[str]=None
    diagnostics: tuple=()
    validity_domain: tuple=()
