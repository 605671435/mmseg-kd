# import modules
from torch.nn import SyncBatchNorm, LeakyReLU, ConvTranspose2d
from mmseg.models import SegDataPreProcessor
from seg.models.segmentors import EncoderDecoder
from seg.models.decode_heads import FCNHead
from mmseg.models.backbones import UNet
from mmseg.models.backbones.unet import InterpConv
from mmseg.models.losses import CrossEntropyLoss
from seg.models.losses.dice_loss import DiceLoss
# model settings
norm_cfg = dict(type=SyncBatchNorm, requires_grad=True)
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
        base_channels=64,
        num_stages=5,
        strides=(1, 1, 1, 1, 1),
        enc_num_convs=(2, 2, 2, 2, 2),
        dec_num_convs=(2, 2, 2, 2),
        downsamples=(True, True, True, True),
        enc_dilations=(1, 1, 1, 1, 1),
        dec_dilations=(1, 1, 1, 1),
        with_cp=False,
        conv_cfg=None,
        norm_cfg=norm_cfg,
        act_cfg=dict(type=LeakyReLU),
        upsample_cfg=dict(type=InterpConv),
        norm_eval=False),
    decode_head=dict(
        type=FCNHead,
        in_channels=64,
        in_index=4,
        channels=64,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=9,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=[
            dict(
                type=CrossEntropyLoss, use_sigmoid=False, loss_weight=1.0),
            dict(
                type=DiceLoss, ignore_index=0, reduction='sum',
                naive_dice=True, eps=1e-5, use_sigmoid=False, loss_weight=1.0)]
    ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
