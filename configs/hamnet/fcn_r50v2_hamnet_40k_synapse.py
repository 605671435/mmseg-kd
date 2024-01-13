from mmengine.config import read_base
from seg.models.utils.hamburger import Ham

with read_base():
    from ..resnetv2.fcn_r50v2_40k_synapse import *  # noqa


model.update(dict(
    backbone=dict(
        plugins=[dict(cfg=dict(type=Ham),
                      stages=(False, False, True),
                      position='after_res'),
                 ]
    )
))

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='synapse', name='fcn-r50v2-hamnet-40k'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(
    type=SegLocalVisualizer, vis_backends=vis_backends, name='visualizer')
