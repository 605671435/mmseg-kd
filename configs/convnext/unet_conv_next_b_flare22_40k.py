from mmengine.config import read_base
from mmpretrain.models.backbones import ConvNeXt
from mmengine.optim.optimizer import AmpOptimWrapper
from torch.optim import AdamW
from mmengine.optim.scheduler import LinearLR
with read_base():
    from .._base_.models.unet_r18_d16_ds import *  # noqa
    from .._base_.datasets.flare22 import *  # noqa
    from .._base_.schedules.schedule_40k import *  # noqa
    from .._base_.default_runtime import *  # noqa

# model settings
checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-tiny_3rdparty_32xb128-noema_in1k_20220301-795e9634.pth'  # noqa
norm_cfg = dict(type=SyncBatchNorm, requires_grad=True)

crop_size = (512, 512)
data_preprocessor = dict(
    type=SegDataPreProcessor,
    mean=None,
    std=None,
    # mean=[123.675, 116.28, 103.53],
    # std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size)
model = dict(
    type=EncoderDecoder,
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type=ConvNeXt,
        in_channels=3,
        arch='tiny',
        out_indices=[0, 1, 2, 3],
        drop_path_rate=0.4,
        layer_scale_init_value=1.0,
        gap_before_final_norm=False,
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint_file,
            prefix='backbone.')
    ),
    neck=dict(
        type=UNet_Neck,
        in_channels=[96, 192, 384, 768],
        downsamples=(True, True, True),
        dec_num_convs=(1, 1, 1),
        norm_cfg=norm_cfg,
        act_cfg=dict(type=LeakyReLU)),
    decode_head=dict(
        type=FCNHead,
        in_channels=96,
        in_index=3,
        channels=96,
        num_convs=2,
        concat_input=True,
        dropout_ratio=0.1,
        num_classes=14,
        norm_cfg=norm_cfg,
        resize_mode='bilinear',
        align_corners=False,
        loss_decode=[
            dict(
                type=CrossEntropyLoss, use_sigmoid=False, loss_weight=1.0),
            dict(
                type=MemoryEfficientSoftDiceLoss, loss_weight=1.0)]),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'),
)

train_pipeline = [
    dict(type=LoadImageFromFile),
    dict(type=Normalize),
    dict(type=LoadAnnotations),
    dict(type=Resize, scale=img_scale, keep_ratio=True),
    dict(type=RandomRotFlip, rotate_prob=0.5, flip_prob=0.5, degree=20),
    dict(type=PackSegInputs)
]
test_pipeline = [
    dict(type=LoadImageFromFile),
    dict(type=Normalize),
    dict(type=Resize, scale=img_scale, keep_ratio=True),
    dict(type=LoadAnnotations),
    dict(type=PackSegInputs)
]
train_dataloader['dataset']['pipeline'] = train_pipeline
val_dataloader['dataset']['pipeline'] = test_pipeline
test_dataloader = val_dataloader

optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(
        type=AdamW, lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05),
    paramwise_cfg={
        'custom_keys': {
                    'pos_block': dict(decay_mult=0.),
                    'norm': dict(decay_mult=0.),
                    'head': dict(lr_mult=10.),
                    'neck': dict(lr_mult=10.)},
        'decay_rate': 0.9,
        'decay_type': 'stage_wise',
        'num_layers': 12
    },
    constructor='mmseg.LearningRateDecayOptimizerConstructor',
    loss_scale='dynamic')

param_scheduler = [
    dict(
        type=LinearLR, start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type=PolyLR,
        power=1.0,
        begin=1500,
        end=40000,
        eta_min=0.0,
        by_epoch=False,
    )
]

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='FLARE22', name='convnext-40k'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(type=SegLocalVisualizer,
                  vis_backends=vis_backends,
                  name='visualizer')
