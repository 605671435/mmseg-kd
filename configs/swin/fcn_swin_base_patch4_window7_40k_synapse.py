from mmengine.config import read_base
from torch.nn import SyncBatchNorm, LayerNorm, GELU
from mmseg.models import SegDataPreProcessor
from mmseg.models.backbones import SwinTransformer
from seg.models.segmentors import EncoderDecoder
from seg.models.decode_heads import FCNHead
from mmseg.models.losses import CrossEntropyLoss
from seg.models.losses.dice_loss import DiceLoss
from torch.optim import AdamW
from mmengine.optim.scheduler import LinearLR
with read_base():
    from .._base_.datasets.synapse import *  # noqa
    from .._base_.schedules.schedule_40k import *  # noqa
    from .._base_.default_runtime import *  # noqa

checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_base_patch4_window7_224_20220317-e9b98025.pth'  # noqa

# model settings
backbone_norm_cfg = dict(type=LayerNorm, requires_grad=True)
norm_cfg = dict(type=SyncBatchNorm, requires_grad=True)

crop_size = (512, 512)
data_preprocessor = dict(
    type=SegDataPreProcessor,
    mean=None,
    std=None,
    # mean=[123.675, 116.28, 103.53],
    # std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size)
model = dict(
    type=EncoderDecoder,
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type=SwinTransformer,
        in_channels=3,
        pretrain_img_size=224,
        embed_dims=128,
        patch_size=4,
        window_size=7,
        mlp_ratio=4,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        strides=(4, 2, 2, 2),
        out_indices=(0, 1, 2, 3),
        qkv_bias=True,
        qk_scale=None,
        patch_norm=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        use_abs_pos_embed=False,
        act_cfg=dict(type=GELU),
        norm_cfg=backbone_norm_cfg,
        init_cfg=dict(type='Pretrained',
                      checkpoint=checkpoint_file)
    ),
    decode_head=dict(
        type=FCNHead,
        in_channels=1024,
        in_index=3,
        channels=512,
        num_convs=2,
        concat_input=True,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        resize_mode='bilinear',
        align_corners=False,
        loss_decode=[
            dict(
                type=CrossEntropyLoss, use_sigmoid=False, loss_weight=1.0),
            # dict(
            #     type=DiceLoss, loss_weight=0.8)]),
            dict(
                type=DiceLoss, naive_dice=True, eps=1e-5, use_sigmoid=False, loss_weight=1.0)]),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

train_pipeline = [
    dict(type=LoadImageFromFile),
    dict(type=LoadAnnotations),
    dict(type=Resize, scale=img_scale, keep_ratio=True),
    dict(type=RandomRotFlip, rotate_prob=0.5, flip_prob=0.5, degree=20),
    dict(type=PackSegInputs)
]
test_pipeline = [
    dict(type=LoadImageFromFile),
    dict(type=Resize, scale=img_scale, keep_ratio=True),
    dict(type=LoadAnnotations),
    dict(type=PackSegInputs)
]
train_dataloader['dataset']['pipeline'] = train_pipeline
val_dataloader['dataset']['pipeline'] = test_pipeline
test_dataloader = val_dataloader

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optim_wrapper = dict(
    type=OptimWrapper,
    optimizer=dict(
        type=AdamW, lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

param_scheduler = [
    dict(
        type=LinearLR, start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type=PolyLR,
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=40000,
        by_epoch=False,
    )
]

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='synapse', name='swin_b-80k'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(type=SegLocalVisualizer,
                  vis_backends=vis_backends,
                  name='visualizer')
