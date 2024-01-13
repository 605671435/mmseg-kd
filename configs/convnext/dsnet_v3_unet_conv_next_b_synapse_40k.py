from mmengine.config import read_base
from seg.models.utils.dsa import DSFormer_V14
with read_base():
    from .unet_conv_next_b_synapse_40k import *  # noqa

model['neck'] = [
    dict(
        type=DSFormer_V14,
        in_channels=768,
        index=3),
    dict(
        type=UNet_Neck,
        in_channels=[96, 192, 384, 768],
        downsamples=(True, True, True),
        dec_num_convs=(1, 1, 1),
        norm_cfg=norm_cfg,
        act_cfg=dict(type=LeakyReLU)),
]

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='synapse', name='dsnetV3-convnext-40k'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(type=SegLocalVisualizer,
                  vis_backends=vis_backends,
                  name='visualizer')
