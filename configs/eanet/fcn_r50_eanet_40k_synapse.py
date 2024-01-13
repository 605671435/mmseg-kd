from mmengine.config import read_base
from seg.models.decode_heads import EAHead

with read_base():
    from ..resnet.fcn_r50_d8_40k_synapse import *  # noqa

model.update(dict(
    backbone=dict(
        plugins=[dict(cfg=dict(type=External_Attention, dim=2048, in_dim=2048),
                      stages=(False, False, False, True),
                      position='after_conv3'),
                 ]
    )
))
    
# model['decode_head'] = dict(
#     type=EAHead,
#     in_channels=2048,
#     in_index=3,
#     channels=512,
#     dropout_ratio=0.1,
#     num_classes=9,
#     norm_cfg=norm_cfg,
#     align_corners=False,
#     loss_decode=[
#         dict(
#             type=CrossEntropyLoss, use_sigmoid=False, loss_weight=1.0),
#         dict(
#             type=DiceLoss, naive_dice=True, eps=1e-5, use_sigmoid=False, loss_weight=1.0)])

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='synapse', name='fcn-r50-eanet-40k'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(
    type=SegLocalVisualizer, vis_backends=vis_backends, name='visualizer')
