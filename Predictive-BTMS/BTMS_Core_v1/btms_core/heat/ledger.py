from dataclasses import dataclass
from enum import Enum

class HeatBasis(str, Enum):
    CELL_TOTAL="CELL_TOTAL"; LOCAL_DEFECT="LOCAL_DEFECT"; VOLUMETRIC="VOLUMETRIC"

@dataclass(frozen=True)
class HeatContribution:
    contribution_id: str
    physical_owner: str
    source_model_id: str
    power_w: float
    basis: HeatBasis
    already_in_baseline: bool=False
    incremental_abnormal: bool=False
    notes: str=""
    def __post_init__(self):
        if not self.contribution_id or not self.physical_owner or not self.source_model_id:
            raise ValueError("Contribution ID, physical owner and source model required")
        if self.power_w < 0: raise ValueError("Power must be non-negative")
        if self.already_in_baseline and self.incremental_abnormal:
            raise ValueError("Cannot be baseline-included and incremental")

class HeatLedger:
    def __init__(self): self._entries={}
    def add(self, entry):
        if entry.contribution_id in self._entries: raise KeyError(entry.contribution_id)
        if any(e.physical_owner==entry.physical_owner for e in self._entries.values()):
            raise ValueError(f"Physical owner already attributed: {entry.physical_owner}")
        self._entries[entry.contribution_id]=entry
    def total_incremental_abnormal_power_w(self):
        return sum(e.power_w for e in self._entries.values() if e.incremental_abnormal and not e.already_in_baseline)
    def entries(self): return tuple(self._entries.values())
