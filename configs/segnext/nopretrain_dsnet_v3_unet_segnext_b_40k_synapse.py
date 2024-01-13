from mmengine.config import read_base
from seg.models.utils.dsa import DSFormer_V14

with read_base():
    from .nopretrain_unet_segnext_b_40k_synapse import *  # noqa

model['neck'] = [
    dict(
        type=DSFormer_V14,
        in_channels=512,
        index=3),
    dict(
        type=UNet_Neck,
        in_channels=[64, 128, 320, 512],
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
            project='synapse', name='unet-segnext_b-40k'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(type=SegLocalVisualizer,
                  vis_backends=vis_backends,
                  name='visualizer')
