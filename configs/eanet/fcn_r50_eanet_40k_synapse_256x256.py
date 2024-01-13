from mmengine.config import read_base
from seg.models.decode_heads import EAHead

with read_base():
    from ..fcn.fcn_r50_d8_40k_synapse_256x256 import *  # noqa

model['decode_head'] = dict(
    type=EAHead,
    in_channels=2048,
    in_index=3,
    channels=512,
    dropout_ratio=0.1,
    num_classes=9,
    norm_cfg=norm_cfg,
    align_corners=False,
    loss_decode=[
        dict(
            type=CrossEntropyLoss, use_sigmoid=False, loss_weight=1.0),
        dict(
            type=DiceLoss, naive_dice=True, eps=1e-5, use_sigmoid=False, loss_weight=1.0)])

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='synapse', name='fcn-r50-eanet-40k-256x256'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(
    type=SegLocalVisualizer, vis_backends=vis_backends, name='visualizer')
