from mmengine.config import read_base

from seg.models.decode_heads.dsnet_head import DSNetHead

with read_base():
    from .unet_segnext_b_40k_synapse import *  # noqa

model['decode_head'] = dict(
        type=DSNetHead,
        dsnet_cfg=norm_cfg,
        in_channels=64,
        in_index=3,
        channels=64,
        dropout_ratio=0.1,
        num_classes=9,
        norm_cfg=norm_cfg,
        resize_mode='bilinear',
        align_corners=False,
        loss_decode=[
            dict(
                type=CrossEntropyLoss, use_sigmoid=False, loss_weight=1.0),
            dict(
                type=MemoryEfficientSoftDiceLoss, loss_weight=1.0)])

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
