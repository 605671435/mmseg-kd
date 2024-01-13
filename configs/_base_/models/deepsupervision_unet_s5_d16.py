# import modules
from torch.nn import Conv2d, LeakyReLU, InstanceNorm2d
from mmseg.models import SegDataPreProcessor
from seg.models.segmentors import EncoderDecoder
from seg.models.decode_heads import FCNHead
from seg.models.backbones.resnet_unet import PlainConvUNet
from mmseg.models.losses import CrossEntropyLoss
from seg.models.losses.dice_loss import DiceLoss
from seg.models.losses.dice import MemoryEfficientSoftDiceLoss

# import numpy as np
# num_decoders = 5
# weights = np.array([1 / (2 ** i) for i in range(num_decoders)])
# weights[-1] = 0
# weights = weights / weights.sum()
#
# # [0.53333333, 0.26666667, 0.13333333, 0.06666667, 0.]
# loss_weights = weights.tolist()
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
        type=PlainConvUNet,
        input_channels=1,
        n_stages=5,
        features_per_stage=(32, 64, 128, 256, 512),
        conv_op=Conv2d,
        kernel_sizes=3,
        strides=(1, 2, 2, 2, 2),
        n_conv_per_stage=(2, 2, 2, 2, 2),
        n_conv_per_stage_decoder=(2, 2, 2, 2),
        conv_bias=True,
        num_classes=9,
        norm_op=InstanceNorm2d,
        norm_op_kwargs={},
        dropout_op=None,
        nonlin=LeakyReLU,
        nonlin_kwargs={'inplace': True},
        deep_supervision=True),
    decode_head=dict(
        type=FCNHead,
        in_channels=32,
        in_index=0,
        channels=32,
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
            in_channels=64,
            channels=64,
            num_convs=0,
            num_classes=9,
            in_index=1,
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
            in_channels=128,
            channels=128,
            num_convs=0,
            num_classes=9,
            in_index=2,
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
