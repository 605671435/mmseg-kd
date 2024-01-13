from mmengine.config import read_base
from seg.models.utils.dsa import DSFormer_V14
with read_base():
    from .unet_r50v2_40k_flare22 import * # noqa
# model settings
model['neck'] = [
    dict(
        type=DSFormer_V14,
        in_channels=1024,
        index=3,
        dsa_cfg=dict(fusion_type='dsa', ratio=1)),
    dict(
        type=UNet_Neck,
        in_channels=[64, 256, 512, 1024],
        dec_num_convs=(2, 2, 2),
        norm_cfg=dict(type=InstanceNorm2d, requires_grad=True),
        act_cfg=dict(type=LeakyReLU))
]

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='flare22', name='dsnet-r50v2-40k'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(type=SegLocalVisualizer,
                  vis_backends=vis_backends,
                  name='visualizer')
