from mmengine.config import read_base
from mmcv.ops import CrissCrossAttention


with read_base():
    from ..fcn.fcn_r50_d8_40k_flare22 import * # noqa

model.update(dict(
    backbone=dict(
        plugins=[dict(cfg=dict(type=CrissCrossAttention),
                      stages=(False, False, False, True),
                      position='after_res'),
                 ]
    )
))

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='flare22', name='fcn-r50-ccnet-40k'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(
    type=SegLocalVisualizer, vis_backends=vis_backends, name='visualizer')

