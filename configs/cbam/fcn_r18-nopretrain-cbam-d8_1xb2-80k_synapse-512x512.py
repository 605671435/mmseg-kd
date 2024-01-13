_base_ = ['../resnet/fcn_r18-no-pretrain-d8_1xb2-80k_synapse-512x512.py']

model = dict(
    backbone=dict(
        type='ResNetV1c_KD',
        depth=18,
        stage_plugin=dict(type='CBAM'),
        out_channels=[64, 128, 256, 512],
        init_cfg=None),
    decode_head=dict(
        in_channels=512,
        channels=128))

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            project='synapse', name='fcn-r18-nopretrain-cbam-80k'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')
