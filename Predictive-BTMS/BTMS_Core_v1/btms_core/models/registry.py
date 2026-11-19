from btms_core.models.base import PrecursorModel

class ModelRegistry:
    def __init__(self): self._models={}
    def register(self, model: PrecursorModel, replace=False):
        if model.model_id=="UNDEFINED": raise ValueError("Model must define model_id")
        if model.model_id in self._models and not replace: raise KeyError(model.model_id)
        self._models[model.model_id]=model
    def get(self, model_id): return self._models[model_id]
    def all(self): return tuple(self._models.values())
