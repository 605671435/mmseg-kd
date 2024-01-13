from mmengine.config import read_base
from seg.models.attention_unet.networks import SingleAttentionBlock2D

with read_base():
    from .unet_r34_s4_d8_40k_synapse import * # noqa
# model settings
model['neck'].update(
    dict(fusion_cfg=dict(type=SingleAttentionBlock2D)))

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='synapse', name='r34-attn-unet-40k'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(type=SegLocalVisualizer,
                  vis_backends=vis_backends,
                  name='visualizer')
