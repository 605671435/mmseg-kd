from mmengine import read_base
from seg.models.utils.aam import AAM
with read_base():
    from ..unet.unet_r18_d16_ds_40k_flare22 import *  # noqa

model['neck'].update(
    dict(fusion_cfg=dict(type=AAM)))

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