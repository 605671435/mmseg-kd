from mmengine.config import read_base
from mmseg.models.decode_heads import PSAHead
with read_base():
    from ..unet.unet_r18v1c_d8_40k_synapse import * # noqa

model['decode_head'] = dict(
        type=PSAHead,
        in_channels=64,
        in_index=3,
        channels=64,
        mask_size=(512, 512),
        psa_type='bi-direction',
        compact=False,
        shrink_factor=2,
        normalization_factor=1.0,
        psa_softmax=True,
        dropout_ratio=0.1,
        num_classes=9,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=[
            dict(
                type=CrossEntropyLoss, use_sigmoid=False, loss_weight=loss_weights[0]),
            dict(
                type=MemoryEfficientSoftDiceLoss, loss_weight=loss_weights[0])]),
vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='synapse', name='psanet-r18-40k'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer.update(
    dict(vis_backends=vis_backends,
         name='visualizer'))
