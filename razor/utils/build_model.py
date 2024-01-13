from seg.registry import MODELS as SEG_MODELS
from mmpretrain.registry import MODELS as CLS_MODELS


def build_model(model_cfg, task):
    if task == 'seg':
        return SEG_MODELS.build(model_cfg)
    if task == 'cls':
        return CLS_MODELS.build(model_cfg)
    else:
        raise TypeError()
