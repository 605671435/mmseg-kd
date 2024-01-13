from mmengine.config import read_base
from seg.models.decode_heads.dsnet_head import DSNHeadV2
with read_base():
    from .unet_conv_next_b_synapse_40k import *  # noqa

model['decode_head'] = dict(
        type=DSNHeadV2,
        in_channels=96,
        in_index=3,
        channels=96,
        concat_input=True,
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
            project='synapse', name='dsnetV2-convnext-40k'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(type=SegLocalVisualizer,
                  vis_backends=vis_backends,
                  name='visualizer')
