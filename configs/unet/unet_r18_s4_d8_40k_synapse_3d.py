from mmengine.config import read_base
from seg.models.data_preprocessor_3d import SegDataPreProcessor3D
with read_base():
    from .._base_.models.unet_r18_s4_d8 import *  # noqa
    from .._base_.datasets.synapse_3d import *  # noqa
    from .._base_.schedules.schedule_40k import *  # noqa
    from .._base_.default_runtime import *  # noqa
# model settings

model['data_preprocessor'] = dict(type=SegDataPreProcessor3D)
model.update(dict(
    backbone=dict(
        conv_cfg=dict(type='Conv3d')),
    neck=dict(
        conv_cfg=dict(type='Conv3d')),
    ))

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='synapse', name='r18-unet-40k'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(type=SegLocalVisualizer,
                  vis_backends=vis_backends,
                  name='visualizer')
