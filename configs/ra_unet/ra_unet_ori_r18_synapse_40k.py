from mmengine import read_base
from seg.models.unet.RAUNet import RAUNet
from seg.models.decode_heads.decode_head import LossHead
with read_base():
    from ..unet.unet_r18_d16_ds_40k_synapse import *  # noqa
model = dict(
    type=EncoderDecoder,
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type=RAUNet,
        depth=18,
        in_ch=1,
        num_classes=9,
        pretrained=True),
    decode_head=dict(
        type=LossHead,
        in_channels=64,
        channels=64,
        in_index=0,
        num_classes=9,
        loss_decode=[
            dict(
                type=CrossEntropyLoss, use_sigmoid=False, loss_weight=1),
            dict(
                type=MemoryEfficientSoftDiceLoss, loss_weight=1)]),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='synapse', name='ori-attn-unet-r18-40k'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(type=SegLocalVisualizer,
                  vis_backends=vis_backends,
                  name='visualizer')