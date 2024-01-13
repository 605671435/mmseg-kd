from mmengine.config import read_base
from seg.models.utils import External_Attention

with read_base():
    from ..unet.unet_r18_s4_d8_40k_synapse import *  # noqa

model.update(dict(
    backbone=dict(
        plugins=[dict(cfg=dict(type=External_Attention, dim=512, in_dim=512),
                      stages=(False, False, False, True),
                      position='after_conv2'),
                 ]
    )
))

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='synapse', name='unet-r18-eanet-40k'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(
    type=SegLocalVisualizer, vis_backends=vis_backends, name='visualizer')
