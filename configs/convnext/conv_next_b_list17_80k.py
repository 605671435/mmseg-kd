from mmengine.config import read_base
from torch.nn import SyncBatchNorm
from mmseg.models import SegDataPreProcessor
from seg.models.segmentors import EncoderDecoder
from mmpretrain.models.backbones import ConvNeXt
from mmseg.models.decode_heads import UPerHead
from mmseg.models.losses import CrossEntropyLoss, DiceLoss
from mmengine.model.weight_init import PretrainedInit
from mmengine.optim.optimizer import AmpOptimWrapper
from mmseg.engine.optimizers import LearningRateDecayOptimizerConstructor
from torch.optim import AdamW
from mmengine.optim.scheduler import LinearLR
from seg.engine.hooks import MyCheckpointHook
with read_base():
    from .._base_.datasets.lits17 import *  # noqa
    from .._base_.schedules.schedule_80k import *  # noqa
    from .._base_.default_runtime import *  # noqa

# model settings
checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-base_3rdparty_32xb128-noema_in1k_20220301-2a0ee547.pth'   # noqa
norm_cfg = dict(type=SyncBatchNorm, requires_grad=True)
data_preprocessor = dict(
    type=SegDataPreProcessor,
    mean=None,
    std=None,
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=(256, 256))
model = dict(
    type=EncoderDecoder,
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type=ConvNeXt,
        arch='base',
        out_indices=[0, 1, 2, 3],
        drop_path_rate=0.4,
        layer_scale_init_value=1.0,
        gap_before_final_norm=False,
        init_cfg=dict(
            type=PretrainedInit, checkpoint=checkpoint_file,
            prefix='backbone.')),
    decode_head=dict(
        type=UPerHead,
        in_channels=[128, 256, 512, 1024],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=3,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=[
            dict(
                type=CrossEntropyLoss, use_sigmoid=False, loss_weight=1.2),
            dict(
                type=DiceLoss, loss_weight=0.8)]),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'),
)

# By default, models are trained on 8 GPUs with 2 images per GPU
train_dataloader.update(dict(batch_size=2))

optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(
        type=AdamW, lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05),
    paramwise_cfg={
        'decay_rate': 0.9,
        'decay_type': 'stage_wise',
        'num_layers': 12
    },
    constructor=LearningRateDecayOptimizerConstructor,
    loss_scale='dynamic')

param_scheduler = [
    dict(
        type=LinearLR, start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type=PolyLR,
        power=1.0,
        begin=1500,
        end=80000,
        eta_min=0.0,
        by_epoch=False,
    )
]

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='lits17', name='convnext_b-80k'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(type=SegLocalVisualizer,
                  vis_backends=vis_backends,
                  name='visualizer')

default_hooks.update(dict(
    checkpoint=dict(type=MyCheckpointHook,
                    by_epoch=False,
                    interval=8000,
                    max_keep_ckpts=1,
                    save_best=['mDice'], rule='greater')))