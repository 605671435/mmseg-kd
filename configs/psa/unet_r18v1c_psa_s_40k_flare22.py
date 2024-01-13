from mmengine.config import read_base
from seg.models.utils.PSA import PSA_s


with read_base():
    from ..unet.unet_r18v1c_d8_40k_flare22 import * # noqa

model.update(dict(
    backbone=dict(
        plugins=[dict(cfg=dict(type=PSA_s),
                      stages=(True, True, True, True),
                      position='after_conv1'),
                 ]
    )
))

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='flare22', name='unet-r18v1c-psa-s-40k'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(
    type=SegLocalVisualizer, vis_backends=vis_backends, name='visualizer')

