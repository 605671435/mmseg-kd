from mmengine.config import read_base
from seg.models.utils.dsa import DSFormer_V14

with read_base():
    from .unet_swin_base_patch4_window7_40k_synapse import *  # noqa

model['neck'] = [
    dict(
        type=DSFormer_V14,
        in_channels=1024,
        index=3),
    dict(
        type=UNet_Neck,
        in_channels=[128, 256, 512, 1024],
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
            project='synapse', name='dsnet-v3-unet-swin_b-40k'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(type=SegLocalVisualizer,
                  vis_backends=vis_backends,
                  name='visualizer')
