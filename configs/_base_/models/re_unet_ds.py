# import modules
from torch.nn import SyncBatchNorm, LeakyReLU, InstanceNorm2d, Conv2d
from mmseg.models import SegDataPreProcessor
from seg.models.segmentors import EncoderDecoder
from seg.models.unet.residual_encoder_unet import ResidualEncoderUNet
from seg.models.decode_heads import FCNHead
from mmseg.models.losses import CrossEntropyLoss
from seg.models.losses.dice import MemoryEfficientSoftDiceLoss

# import numpy as np
# num_decoders = 6
# weights = np.array([1 / (2 ** i) for i in range(num_decoders)])
# weights[-1] = 0
# weights = weights / weights.sum()
#
# # [0.53333333, 0.26666667, 0.13333333, 0.06666667, 0.]
# loss_weights = weights.tolist()
loss_weighs = [0.5161290322580645, 0.25806451612903225,
               0.12903225806451613, 0.06451612903225806, 0.03225806451612903, 0.0]
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
        type=ResidualEncoderUNet,
        input_channels=1,
        n_stages=7,
        features_per_stage=(32, 64, 128, 256, 512, 512, 512),
        conv_op=Conv2d,
        kernel_sizes=3,
        strides=(1, 2, 2, 2, 2, 2, 2),
        n_blocks_per_stage=(1, 3, 4, 6, 6, 6, 6),
        n_conv_per_stage_decoder=(1, 1, 1, 1, 1, 1),
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
        in_channels=9,
        in_index=0,
        channels=9,
        num_convs=0,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=9,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=[
            dict(
                type=CrossEntropyLoss, use_sigmoid=False, loss_weight=loss_weighs[0]),
            # dict(
            #     type=MemoryEfficientSoftDiceLoss, loss_weight=loss_weighs[0])]),
            dict(
                type=MemoryEfficientSoftDiceLoss, loss_weight=loss_weighs[0])]),
    auxiliary_head=[
        dict(
            type=FCNHead,
            in_channels=9,
            channels=9,
            num_convs=0,
            num_classes=9,
            in_index=1,
            norm_cfg=norm_cfg,
            concat_input=False,
            align_corners=False,
            upsample_label=True,
            loss_decode=[
                dict(
                    type=CrossEntropyLoss, use_sigmoid=False, loss_weight=loss_weighs[1]),
                dict(
                    type=MemoryEfficientSoftDiceLoss, loss_weight=loss_weighs[1])]),
        dict(
            type=FCNHead,
            in_channels=9,
            channels=9,
            num_convs=0,
            num_classes=9,
            in_index=2,
            norm_cfg=norm_cfg,
            concat_input=False,
            align_corners=False,
            upsample_label=True,
            loss_decode=[
                dict(
                    type=CrossEntropyLoss, use_sigmoid=False, loss_weight=loss_weighs[2]),
                dict(
                    type=MemoryEfficientSoftDiceLoss, loss_weight=loss_weighs[2])]),
        dict(
            type=FCNHead,
            in_channels=9,
            channels=9,
            num_convs=0,
            num_classes=9,
            in_index=3,
            norm_cfg=norm_cfg,
            concat_input=False,
            align_corners=False,
            upsample_label=True,
            loss_decode=[
                dict(
                    type=CrossEntropyLoss, use_sigmoid=False, loss_weight=loss_weighs[3]),
                dict(
                    type=MemoryEfficientSoftDiceLoss, loss_weight=loss_weighs[3])]),
        dict(
            type=FCNHead,
            in_channels=9,
            channels=9,
            num_convs=0,
            num_classes=9,
            in_index=4,
            norm_cfg=norm_cfg,
            concat_input=False,
            align_corners=False,
            upsample_label=True,
            loss_decode=[
                dict(
                    type=CrossEntropyLoss, use_sigmoid=False, loss_weight=loss_weighs[4]),
                dict(
                    type=MemoryEfficientSoftDiceLoss, loss_weight=loss_weighs[4])]),
    ],
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
