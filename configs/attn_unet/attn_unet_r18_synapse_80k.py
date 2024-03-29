from mmengine import read_base
from seg.models.attention_unet.networks import SingleAttentionBlock2D
with read_base():
    from ..unet.unet_r18_s4_d8_80k_synapse import *  # noqa

model['neck'].update(
    dict(fusion_cfg=dict(type=SingleAttentionBlock2D)))

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='synapse', name='attn-unet-r18-80k'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(type=SegLocalVisualizer,
                  vis_backends=vis_backends,
                  name='visualizer')