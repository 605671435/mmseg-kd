from mmengine.config import read_base
from seg.models.utils.dsa import DSA_V13
with read_base():
    from .new_unet_resnet50v2_40k_synapse import * # noqa

model.update(dict(
    backbone=dict(
        plugins=[dict(cfg=dict(type=DSA_V13, attn_types=('sp', 'ch'), fusion_type='dsa', bias=False, ratio=4),
                      stages=(True, True, True),
                      position='after_conv1'),
                 ]
    )
))

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='synapse', name='dsnet-new-unet-r50v2-40k'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(type=SegLocalVisualizer,
                  vis_backends=vis_backends,
                  name='visualizer')
