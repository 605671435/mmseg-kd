from mmengine.config import read_base
from seg.models.backbones.hamnet import HamNet

with read_base():
    from ..unet.unet_r50_s4_d8_40k_synapse import *  # noqa

model.update(dict(
    backbone=dict(type=HamNet)
))

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='synapse', name='unet-r50-hamnet'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(
    type=SegLocalVisualizer, vis_backends=vis_backends, name='visualizer')
