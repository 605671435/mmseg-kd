from mmengine.config import read_base
from seg.models.utils.hamburger import Ham

with read_base():
    from ..resnet.fcn_r50_d8_40k_synapse import *  # noqa


model.update(dict(
    backbone=dict(
        final_plugins=dict(type=Ham))))

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='synapse', name='fcn-r50-hamnet-40k'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(
    type=SegLocalVisualizer, vis_backends=vis_backends, name='visualizer')
