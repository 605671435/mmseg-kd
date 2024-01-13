from mmengine.config import read_base
from seg.models.backbones.hamnet import HamNet

with read_base():
    from ..fcn.fcn_r18_d8_40k_lits17 import *  # noqa
    
model.update(dict(
    backbone=dict(type=HamNet)
))

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='lits17', name='fcn-r18-hamnet-40k'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(
    type=SegLocalVisualizer, vis_backends=vis_backends, name='visualizer')
