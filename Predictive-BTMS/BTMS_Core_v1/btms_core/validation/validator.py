from dataclasses import dataclass
import math
from btms_core.contracts.types import DataSufficiency, Quality

@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str
    signal_id: str|None=None
@dataclass(frozen=True)
class ValidationReport:
    sufficiency: DataSufficiency
    issues: tuple
    @property
    def is_usable(self): return self.sufficiency == DataSufficiency.SUFFICIENT

def validate_packet(packet, requirements, max_signal_age_s=None):
    issues=[]; stale=False; invalid=False
    if not packet.cell_id or not packet.chemistry_id:
        issues.append(ValidationIssue("PACKET_ID_MISSING","cell_id and chemistry_id required")); invalid=True
    if not math.isfinite(packet.evaluation_timestamp_s):
        issues.append(ValidationIssue("EVALUATION_TIME_INVALID","Evaluation time must be finite")); invalid=True
    for req in requirements:
        sig=packet.get_signal(req.name) or packet.get_state(req.name)
        if sig is None:
            if req.required: issues.append(ValidationIssue("REQUIRED_SIGNAL_MISSING",f"{req.name} missing",req.name))
            continue
        if sig.unit != req.unit:
            issues.append(ValidationIssue("UNIT_MISMATCH",f"{req.name}: expected {req.unit}, got {sig.unit}",req.name)); invalid=True; continue
        if sig.quality == Quality.INVALID:
            issues.append(ValidationIssue("SIGNAL_INVALID",f"{req.name} marked invalid",req.name)); invalid=True
        elif sig.quality == Quality.STALE:
            issues.append(ValidationIssue("SIGNAL_STALE",f"{req.name} marked stale",req.name)); stale=True
        elif sig.quality == Quality.UNAVAILABLE:
            if req.required: issues.append(ValidationIssue("SIGNAL_UNAVAILABLE",f"{req.name} unavailable",req.name))
            continue
        elif sig.quality == Quality.SUSPECT:
            issues.append(ValidationIssue("SIGNAL_SUSPECT",f"{req.name} marked suspect",req.name))
        if sig.value is None or not math.isfinite(sig.value):
            issues.append(ValidationIssue("SIGNAL_VALUE_MISSING",f"{req.name} has no finite value",req.name)); invalid=True; continue
        if sig.timestamp_s is None or not math.isfinite(sig.timestamp_s):
            issues.append(ValidationIssue("SIGNAL_TIMESTAMP_MISSING",f"{req.name} has no valid timestamp",req.name)); invalid=True; continue
        if sig.timestamp_s > packet.evaluation_timestamp_s:
            issues.append(ValidationIssue("FUTURE_SIGNAL",f"{req.name} timestamp is in the future",req.name)); invalid=True
        if max_signal_age_s is not None and packet.evaluation_timestamp_s-sig.timestamp_s > max_signal_age_s:
            issues.append(ValidationIssue("SIGNAL_TOO_OLD",f"{req.name} is stale by age",req.name)); stale=True
        if sig.source_type not in req.source_types_allowed:
            issues.append(ValidationIssue("SOURCE_TYPE_NOT_ALLOWED",f"{req.name} source type not allowed",req.name)); invalid=True
    if invalid: state=DataSufficiency.INVALID
    elif stale: state=DataSufficiency.STALE
    elif any(i.code in ("REQUIRED_SIGNAL_MISSING","SIGNAL_UNAVAILABLE") for i in issues): state=DataSufficiency.INSUFFICIENT
    else: state=DataSufficiency.SUFFICIENT
    return ValidationReport(state,tuple(issues))
