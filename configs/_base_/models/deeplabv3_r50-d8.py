# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=None,
    std=None,
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type='ResNetV1c',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://resnet50_v1c')),
    decode_head=dict(
        type='ASPPHead',
        in_channels=2048,
        in_index=3,
        channels=512,
        dilations=(1, 12, 24, 36),
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=[
            dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.2),
            dict(
                type='DiceLoss', loss_weight=0.8)]),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=1024,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=[
            dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.2),
            dict(
                type='DiceLoss', loss_weight=0.8)]),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))