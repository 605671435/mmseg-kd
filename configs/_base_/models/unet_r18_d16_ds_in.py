# import modules
from torch.nn import LeakyReLU, InstanceNorm2d, SyncBatchNorm, ReLU
from mmseg.models import SegDataPreProcessor
from seg.models.segmentors import EncoderDecoder
# from mmseg.models.backbones import ResNetV1c
from seg.models.backbones import ResNet
from seg.models.necks.unet import UNet_Neck
from seg.models.decode_heads import FCNHead
from mmseg.models.losses import CrossEntropyLoss, DiceLoss
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
act_cfg = dict(type=LeakyReLU)
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
        depth=18,
        in_channels=3,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        # dilations=(1, 1, 2, 4),
        # strides=(1, 2, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        # init_cfg=dict(
        #     type='Pretrained',
        #     checkpoint='https://download.openmmlab.com/mmclassification/v0/resnet/resnet18_8xb32_in1k_20210831-fbbb1da6.pth',
        #     prefix='backbone'),
        contract_dilation=True),
    neck=dict(
        type=UNet_Neck,
        base_channels=64,
        norm_cfg=norm_cfg,
        act_cfg=act_cfg),
    decode_head=dict(
        type=FCNHead,
        in_channels=64,
        channels=64,
        in_index=3,
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
                    type=CrossEntropyLoss, use_sigmoid=False, loss_weight=loss_weights[1]),
                dict(
                    type=MemoryEfficientSoftDiceLoss, loss_weight=loss_weights[1])]),
        dict(
            type=FCNHead,
            in_channels=256,
            channels=256,
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
