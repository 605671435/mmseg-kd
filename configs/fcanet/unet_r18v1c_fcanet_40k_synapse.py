from mmengine.config import read_base
from seg.models.utils.fcanet_layers import MultiSpectralAttentionLayer


with read_base():
    from ..unet.unet_r18v1c_d8_40k_synapse import * # noqa

model.update(dict(
    backbone=dict(
        plugins=[dict(cfg=dict(type=MultiSpectralAttentionLayer,
                               channel=64,
                               dct_h=56,
                               dct_w=56,
                               reduction=16),
                      stages=(True, False, False, False),
                      position='after_conv2'),
                 dict(cfg=dict(type=MultiSpectralAttentionLayer,
                               channel=128,
                               dct_h=28,
                               dct_w=28,
                               reduction=16),
                      stages=(False, True, False, False),
                      position='after_conv2'),
                 dict(cfg=dict(type=MultiSpectralAttentionLayer,
                               channel=256,
                               dct_h=14,
                               dct_w=14,
                               reduction=16),
                      stages=(False, False, True, False),
                      position='after_conv2'),
                 dict(cfg=dict(type=MultiSpectralAttentionLayer,
                               channel=512,
                               dct_h=7,
                               dct_w=7,
                               reduction=16),
                      stages=(False, False, False, True),
                      position='after_conv2'),
                 ]
    )
))

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='synapse', name='unet-r18v1c-fcanet-40k'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(
    type=SegLocalVisualizer, vis_backends=vis_backends, name='visualizer')

