from mmengine.config import read_base

with read_base():
    from .mobilenet_v2_fcn_synapse import *  # noqa

model.update(dict(
    backbone=dict(type='MobileNetV2_KD')))
