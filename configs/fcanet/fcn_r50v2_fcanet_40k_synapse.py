from mmengine.config import read_base
from pretrain.models.utils.layers import MultiSpectralAttentionLayer

with read_base():
    from ..resnetv2.fcn_r50v2_40k_synapse import * # noqa

model.update(dict(
    backbone=dict(
        plugins=[dict(cfg=dict(type=MultiSpectralAttentionLayer,
                               channel=256,
                               dct_h=56,
                               dct_w=56,
                               reduction=16),
                      stages=(True, False, False),
                      position='after_conv3'),
                 dict(cfg=dict(type=MultiSpectralAttentionLayer,
                               channel=512,
                               dct_h=28,
                               dct_w=28,
                               reduction=16),
                      stages=(False, True, False),
                      position='after_conv3'),
                 dict(cfg=dict(type=MultiSpectralAttentionLayer,
                               channel=1024,
                               dct_h=14,
                               dct_w=14,
                               reduction=16),
                      stages=(False, False, True),
                      position='after_conv3'),
                 ]
    )
))

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='synapse', name='fcn-r50v2-fcanet'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(
    type=SegLocalVisualizer, vis_backends=vis_backends, name='visualizer')

