from mmengine.config import read_base
from seg.models.backbones import ShuffleNetV1_KD

with read_base():
    from .shufflenet_v1_fcn_80k_lits17 import *  # noqa

model.update(dict(backbone=dict(type=ShuffleNetV1_KD)))
