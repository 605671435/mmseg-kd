from mmengine.config import read_base
from torch.nn import BatchNorm2d, GELU
from mmseg.models.backbones import MSCAN
from mmseg.models.decode_heads import LightHamHead
with read_base():
    from .._base_.datasets.synapse import *  # noqa
    from .._base_.models.unet_r18_d16_ds import *  # noqa
    from .._base_.schedules.schedule_40k import *  # noqa
    from .._base_.default_runtime import *  # noqa
# model settings
norm_cfg = dict(type=SyncBatchNorm, requires_grad=True)

crop_size = (512, 512)
data_preprocessor = dict(
    type=SegDataPreProcessor,
    mean=None,
    std=None,
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size)
model = dict(
    type=EncoderDecoder,
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type=MSCAN,
        in_channels=3,
        init_cfg=None,
        embed_dims=[64, 128, 320, 512],
        mlp_ratios=[8, 8, 4, 4],
        drop_rate=0.0,
        drop_path_rate=0.1,
        depths=[3, 3, 12, 3],
        attention_kernel_sizes=[5, [1, 7], [1, 11], [1, 21]],
        attention_kernel_paddings=[2, [0, 3], [0, 5], [0, 10]],
        act_cfg=dict(type=GELU),
        norm_cfg=dict(type=BatchNorm2d, requires_grad=True)),
    neck=None,
    decode_head=dict(
        type=LightHamHead,
        in_channels=[128, 320, 512],
        in_index=[1, 2, 3],
        channels=512,
        ham_channels=512,
        dropout_ratio=0.1,
        num_classes=9,
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        align_corners=False,
        loss_decode=[
            dict(
                type=CrossEntropyLoss, use_sigmoid=False, loss_weight=1.0),
            dict(
                type=MemoryEfficientSoftDiceLoss, loss_weight=1.0)]),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

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
