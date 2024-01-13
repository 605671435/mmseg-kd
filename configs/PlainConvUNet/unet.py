# import modules
from torch.nn import SyncBatchNorm
from mmseg.models import SegDataPreProcessor
from seg.models.segmentors import EncoderDecoder
from mmseg.models.losses import CrossEntropyLoss, DiceLoss
from seg.models.losses.dice import MemoryEfficientSoftDiceLoss
from mmengine.config import read_base
from seg.models.decode_heads import UNet
from torch.nn import Identity
with read_base():
    from .._base_.datasets.synapse import *  # noqa
    from .._base_.schedules.schedule_80k import *  # noqa
    from .._base_.default_runtime import *  # noqa
# model settings
norm_cfg = dict(type=SyncBatchNorm, requires_grad=True)
data_preprocessor = dict(
    type=SegDataPreProcessor,
    mean=None,
    std=None,
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=(512, 512))
model = dict(
    type=EncoderDecoder,
    data_preprocessor=data_preprocessor,
    backbone=dict(type=Identity),
    decode_head=dict(
        type=UNet,
        in_channels=1,
        channels=3,
        resize_mode='bilinear',
        align_corners=False,
        num_classes=9,
        # loss_decode=[
        #     dict(
        #         type=CrossEntropyLoss, use_sigmoid=False, loss_weight=1.2),
        #     dict(
        #         type=DiceLoss, loss_weight=0.8)]
        loss_decode=[
            dict(
                type=CrossEntropyLoss, use_sigmoid=False, loss_weight=1.0),
            dict(
                type=MemoryEfficientSoftDiceLoss, loss_weight=1.0)]
    ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)
train_dataloader.update(dict(batch_size=2, num_workers=2))
val_dataloader.update(dict(batch_size=1, num_workers=4))
test_dataloader = val_dataloader

default_hooks.update(dict(
    checkpoint=dict(
        type=MyCheckpointHook,
        by_epoch=False,
        interval=8000,
        max_keep_ckpts=1,
        save_best=['mDice'], rule='greater')))

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='synapse', name='unet-80k'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(type=SegLocalVisualizer,
                  vis_backends=vis_backends,
                  name='visualizer')
