from dataclasses import dataclass
from enum import Enum

class ParameterStatus(str, Enum):
    UNSOURCED="UNSOURCED"; LITERATURE_REPORTED="LITERATURE_REPORTED"
    MANUFACTURER_SPECIFIED="MANUFACTURER_SPECIFIED"; MEASURED="MEASURED"
    FITTED="FITTED"; PROVISIONAL="PROVISIONAL"; VALIDATED_FOR_DEFINED_DOMAIN="VALIDATED_FOR_DEFINED_DOMAIN"

@dataclass(frozen=True)
class ParameterRecord:
    name: str
    value: float
    unit: str
    chemistry_id: str
    source: str
    status: ParameterStatus
    valid_min: float|None=None
    valid_max: float|None=None
    valid_range_unit: str|None=None
    standard_uncertainty: float|None=None
    version: str="1"
    notes: str=""
    def __post_init__(self):
        if not self.name or not self.unit or not self.chemistry_id: raise ValueError("Parameter identity/unit required")
        if self.standard_uncertainty is not None and self.standard_uncertainty < 0: raise ValueError("Negative uncertainty")
        if self.valid_min is not None and self.valid_max is not None and self.valid_min > self.valid_max:
            raise ValueError("Invalid validity range")

class ParameterRegistry:
    def __init__(self): self._records={}
    def add(self, record, replace=False):
        key=(record.chemistry_id,record.name,record.version)
        if key in self._records and not replace: raise KeyError(f"Already registered: {key}")
        self._records[key]=record
    def get(self, chemistry_id, name, version="1"):
        key=(chemistry_id,name,version)
        if key not in self._records: raise KeyError(f"Missing parameter: {key}")
        return self._records[key]
    def get_value(self, chemistry_id, name, unit, version="1"):
        r=self.get(chemistry_id,name,version)
        if r.unit != unit: raise ValueError(f"Unit mismatch: stored {r.unit}, requested {unit}")
        return r.value
    def list_for_chemistry(self, chemistry_id):
        return [r for r in self._records.values() if r.chemistry_id==chemistry_id]
