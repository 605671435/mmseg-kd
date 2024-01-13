# import modules
from torch.nn import LeakyReLU, InstanceNorm2d, BatchNorm2d
from mmseg.models import SegDataPreProcessor
from seg.models.segmentors import EncoderDecoder
from mmseg.models.backbones import ResNet
from seg.models.necks.RAUNet import RAUNet
from seg.models.decode_heads import FCNHead
from seg.models.utils.conv_transpose import InterpConv
from mmseg.models.losses import CrossEntropyLoss, DiceLoss
from seg.models.losses.dice import MemoryEfficientSoftDiceLoss

# model settings
norm_cfg = dict(type=BatchNorm2d, requires_grad=True)
data_preprocessor = dict(
    type=SegDataPreProcessor,
    mean=None,
    std=None,
    pad_val=0,
    seg_pad_val=255)
model = dict(
    type=EncoderDecoder,
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type=ResNet,
        depth=34,
        in_channels=3,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch'),
    neck=dict(
        type=RAUNet,
        in_channels=[64, 128, 256, 512],
        norm_cfg=norm_cfg,
        act_cfg=dict(type=LeakyReLU, inplace=True)),
    decode_head=dict(
        type=FCNHead,
        in_channels=32,
        in_index=0,
        channels=32,
        num_convs=0,
        concat_input=False,
        dropout_ratio=0.,
        num_classes=9,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=[
            dict(
                type=CrossEntropyLoss, use_sigmoid=False, loss_weight=1.0),
            dict(
                type=MemoryEfficientSoftDiceLoss, loss_weight=1.0)]),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
