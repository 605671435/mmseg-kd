from mmengine import read_base
from seg.models.necks.unet import UNet_Neck_legacy
with read_base():
    from ..unet.unet_r50_s4_d8_40k_flare22 import *  # noqa

model['neck'].update(
    dict(
        type=UNet_Neck_legacy,
        fusion_type='attn_unet'))

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='flare22', name='attn-unet-r18-40k'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(type=SegLocalVisualizer,
                  vis_backends=vis_backends,
                  name='visualizer')