# import modules
from torch.nn import LeakyReLU, InstanceNorm2d
from mmseg.models import SegDataPreProcessor
from seg.models.segmentors import EncoderDecoder
from mmseg.models.backbones import ResNetV1c
from seg.models.necks.unet import UNet_Neck
from seg.models.decode_heads import FCNHead
from seg.models.utils.conv_transpose import InterpConv
from mmseg.models.losses import CrossEntropyLoss
from seg.models.losses.dice import MemoryEfficientSoftDiceLoss

# import numpy as np
# num_decoders = 4
# weights = np.array([1 / (2 ** i) for i in range(num_decoders)])
# weights[-1] = 0
# weights = weights / weights.sum()
#
# # [0.53333333, 0.26666667, 0.13333333, 0.06666667, 0.]
# loss_weights = weights.tolist()
# print(loss_weights)
loss_weights = [0.5714285714285714, 0.2857142857142857, 0.14285714285714285, 0.0]
# model settings
norm_cfg = dict(type=InstanceNorm2d, requires_grad=True)
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
        type=ResNetV1c,
        depth=50,
        in_channels=3,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    neck=dict(
        type=UNet_Neck,
        base_channels=256,
        norm_cfg=norm_cfg,
        act_cfg=dict(type=LeakyReLU),
        upsample_cfg=dict(type=InterpConv)),
    decode_head=dict(
        type=FCNHead,
        in_channels=256,
        in_index=3,
        channels=256,
        num_convs=0,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=9,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=[
            dict(
                type=CrossEntropyLoss, use_sigmoid=False, loss_weight=loss_weights[0]),
            dict(
                type=MemoryEfficientSoftDiceLoss, loss_weight=loss_weights[0])]),
    auxiliary_head=[
        dict(
            type=FCNHead,
            in_channels=512,
            channels=512,
            num_convs=0,
            num_classes=9,
            in_index=2,
            norm_cfg=norm_cfg,
            concat_input=False,
            align_corners=False,
            upsample_label=True,
            loss_decode=[
                dict(
                    type=CrossEntropyLoss, use_sigmoid=False, loss_weight=loss_weights[1]),
                dict(
                    type=MemoryEfficientSoftDiceLoss, loss_weight=loss_weights[1])]),
        dict(
            type=FCNHead,
            in_channels=1024,
            channels=1024,
            num_convs=0,
            num_classes=9,
            in_index=1,
            norm_cfg=norm_cfg,
            concat_input=False,
            align_corners=False,
            upsample_label=True,
            loss_decode=[
                dict(
                    type=CrossEntropyLoss, use_sigmoid=False, loss_weight=loss_weights[2]),
                dict(
                    type=MemoryEfficientSoftDiceLoss, loss_weight=loss_weights[2])]),
    ],
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
