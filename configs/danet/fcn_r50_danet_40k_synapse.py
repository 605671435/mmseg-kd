from mmengine.config import read_base
from seg.models.utils.danet import DANet


with read_base():
    from ..fcn.fcn_r50_d8_40k_synapse import * # noqa

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
            project='synapse', name='fcn-r50-danet-40k'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(
    type=SegLocalVisualizer, vis_backends=vis_backends, name='visualizer')

