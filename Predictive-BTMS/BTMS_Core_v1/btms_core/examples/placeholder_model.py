from btms_core.models.base import PrecursorModel
from btms_core.contracts.types import (
    PrecursorResult, ModelStatus, DataSufficiency, Identifiability, Quantity
)

class InternalMicroShortPlaceholder(PrecursorModel):
    model_id="INTERNAL_MICRO_SHORT"
    model_version="0.1.0-skeleton"
    @property
    def required_inputs(self):
        # Frozen only after Model 01 observability/identifiability review.
        return ()
    def evaluate(self, packet, validation):
        status = ModelStatus.INSUFFICIENT_DATA
        suff = DataSufficiency.INSUFFICIENT
        diagnostics=("Architecture smoke test only; no micro-short diagnosis performed.",
                     "Model 01 equations, required inputs and identifiability criteria are pending.")
        if validation.sufficiency == DataSufficiency.INVALID:
            status=ModelStatus.INVALID_INPUT; suff=DataSufficiency.INVALID
        elif validation.sufficiency == DataSufficiency.STALE:
            status=ModelStatus.INSUFFICIENT_DATA; suff=DataSufficiency.STALE
        return PrecursorResult(
            self.model_id,self.model_version,packet.evaluation_timestamp_s,status,suff,
            Identifiability.NOT_ASSESSED,
            heat_power_w=Quantity.unavailable("W","Model 01 physics not implemented"),
            diagnostics=diagnostics
        )
