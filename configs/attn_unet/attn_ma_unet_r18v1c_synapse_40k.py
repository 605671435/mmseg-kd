from mmengine import read_base
from seg.models.attention_unet.networks import MultiAttentionBlock2D
with read_base():
    from ..unet.unet_r18v1c_d8_40k_synapse import *  # noqa

model['neck'].update(
    dict(fusion_cfg=dict(type=MultiAttentionBlock2D)))

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='synapse', name='attn-ma-unet-r18v1c-40k'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(type=SegLocalVisualizer,
                  vis_backends=vis_backends,
                  name='visualizer')