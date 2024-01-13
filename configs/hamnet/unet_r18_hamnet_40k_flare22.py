from mmengine.config import read_base
from seg.models.backbones.hamnet import HamNet

with read_base():
    from ..unet.unet_r18_s4_d8_40k_flare22 import *  # noqa

model.update(dict(
    backbone=dict(type=HamNet)
))

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='flare22', name='unet-r18-hamnet'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(
    type=SegLocalVisualizer, vis_backends=vis_backends, name='visualizer')
