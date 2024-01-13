from mmengine.config import read_base
# from seg.models.backbones.fcanet_resnet import ResNetV1c as FcaNet_resnet
from pretrain.models.utils.layers import MultiSpectralAttentionLayer

with read_base():
    from ..fcn.fcn_r50_d8_40k_flare22 import * # noqa

# model.update(dict(
#     backbone=dict(
#         type=FcaNet_resnet,
#     )
# ))
model.update(dict(
    backbone=dict(
        plugins=[dict(cfg=dict(type=MultiSpectralAttentionLayer,
                               channel=256,
                               dct_h=56,
                               dct_w=56,
                               reduction=16),
                      stages=(True, False, False, False),
                      position='after_conv3'),
                 dict(cfg=dict(type=MultiSpectralAttentionLayer,
                               channel=512,
                               dct_h=28,
                               dct_w=28,
                               reduction=16),
                      stages=(False, True, False, False),
                      position='after_conv3'),
                 dict(cfg=dict(type=MultiSpectralAttentionLayer,
                               channel=1024,
                               dct_h=14,
                               dct_w=14,
                               reduction=16),
                      stages=(False, False, True, False),
                      position='after_conv3'),
                 dict(cfg=dict(type=MultiSpectralAttentionLayer,
                               channel=2048,
                               dct_h=7,
                               dct_w=7,
                               reduction=16),
                      stages=(False, False, False, True),
                      position='after_conv3'),
                 ]
    )
))
vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='flare22', name='fcn-fcanet50'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(
    type=SegLocalVisualizer, vis_backends=vis_backends, name='visualizer')

