from mmengine.config import read_base
from mmseg.models.decode_heads import PSAHead
with read_base():
    from ..fcn.fcn_r18_d8_40k_synapse import * # noqa

model['decode_head'] = dict(
        type=PSAHead,
        in_channels=512,
        in_index=3,
        channels=128,
        mask_size=(64, 64),
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
                type=CrossEntropyLoss, use_sigmoid=False, loss_weight=1.0),
            dict(
                type=DiceLoss, naive_dice=True, eps=1e-5, use_sigmoid=False, loss_weight=1.0)])
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
