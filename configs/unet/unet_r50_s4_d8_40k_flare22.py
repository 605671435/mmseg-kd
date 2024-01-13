from mmengine.config import read_base
with read_base():
    from .unet_r18_s4_d8_40k_flare22 import * # noqa
# model settings
model.update(dict(
    backbone=dict(depth=50),
    neck=dict(base_channels=256),
    decode_head=dict(
        in_channels=256,
        channels=256)))

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='synapse', name='r50-unet-40k'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(type=SegLocalVisualizer,
                  vis_backends=vis_backends,
                  name='visualizer')
