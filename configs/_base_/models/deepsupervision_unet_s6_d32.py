# import modules
from torch.nn import SyncBatchNorm, LeakyReLU, InstanceNorm2d
from mmseg.models import SegDataPreProcessor
from seg.models.segmentors import EncoderDecoder
from seg.models.decode_heads import FCNHead
from seg.models.backbones.unet import UNet
from seg.models.utils.conv_transpose import InterpConv
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
loss_weighs = [0.53333333, 0.26666667, 0.13333333, 0.06666667, 0.]

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
        type=UNet,
        in_channels=3,
        base_channels=32,
        num_stages=6,
        strides=(1, 2, 2, 2, 2, 2),
        enc_num_convs=(2, 2, 2, 2, 2, 2),
        dec_num_convs=(2, 2, 2, 2, 2),
        downsamples=(True, True, True, True, True),
        enc_dilations=(1, 1, 1, 1, 1, 1),
        dec_dilations=(1, 1, 1, 1, 1),
        with_cp=False,
        conv_cfg=None,
        norm_cfg=norm_cfg,
        act_cfg=dict(type=LeakyReLU),
        upsample_cfg=dict(type=InterpConv),
        norm_eval=False),
    decode_head=dict(
        type=FCNHead,
        in_channels=32,
        in_index=5,
        channels=32,
        num_convs=0,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=9,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=[
            dict(
                type=CrossEntropyLoss, use_sigmoid=False, loss_weight=loss_weighs[0]),
            dict(
                type=MemoryEfficientSoftDiceLoss, loss_weight=loss_weighs[0])]),
            # dict(
            #     type=DiceLoss, ignore_index=0,
            #     naive_dice=True, eps=1e-5, use_sigmoid=False, loss_weight=loss_weights[0])]),
    auxiliary_head=[
        dict(
            type=FCNHead,
            in_channels=64,
            channels=64,
            num_convs=0,
            num_classes=9,
            in_index=4,
            norm_cfg=norm_cfg,
            concat_input=False,
            align_corners=False,
            upsample_label=True,
            loss_decode=[
                dict(
                    type=CrossEntropyLoss, use_sigmoid=False, loss_weight=loss_weighs[1]),
                dict(
                    type=MemoryEfficientSoftDiceLoss, loss_weight=loss_weighs[1])]),
                # dict(
                #     type=DiceLoss, ignore_index=0,
                #     naive_dice=True, eps=1e-5, use_sigmoid=False, loss_weight=loss_weights[1])]),
        dict(
            type=FCNHead,
            in_channels=128,
            channels=128,
            num_convs=0,
            num_classes=9,
            in_index=3,
            norm_cfg=norm_cfg,
            concat_input=False,
            align_corners=False,
            upsample_label=True,
            loss_decode=[
                dict(
                    type=CrossEntropyLoss, use_sigmoid=False, loss_weight=loss_weighs[2]),
                dict(
                    type=MemoryEfficientSoftDiceLoss, loss_weight=loss_weighs[2])]),
                # dict(
                #     type=DiceLoss, ignore_index=0,
                #     naive_dice=True, eps=1e-5, use_sigmoid=False, loss_weight=loss_weights[2])]),
        dict(
            type=FCNHead,
            in_channels=256,
            channels=256,
            num_convs=0,
            num_classes=9,
            in_index=2,
            norm_cfg=norm_cfg,
            concat_input=False,
            align_corners=False,
            upsample_label=True,
            loss_decode=[
                dict(
                    type=CrossEntropyLoss, use_sigmoid=False, loss_weight=loss_weighs[3]),
                dict(
                    type=MemoryEfficientSoftDiceLoss, loss_weight=loss_weighs[3])]),
                # dict(
                #     type=DiceLoss, ignore_index=0,
                #     naive_dice=True, eps=1e-5, use_sigmoid=False, loss_weight=loss_weights[3])]),
    ],
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
