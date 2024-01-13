from mmengine.config import read_base
from seg.models.utils.danet import DANet


with read_base():
    from ..fcn.fcn_r18_d8_40k_flare22 import * # noqa

model.update(dict(
    backbone=dict(
        plugins=[dict(cfg=dict(type=DANet),
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
            project='flare22', name='fcn-r18-danet'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(
    type=SegLocalVisualizer, vis_backends=vis_backends, name='visualizer')
