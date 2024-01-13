_base_ = ['./fcn_r50-d8_1xb2-80k_lits17-256x256.py']

model = dict(
    backbone=dict(
        type='ResNetV1c_KD',
        depth=18,
        out_channels=[64, 128, 256, 512],
        # init_cfg=dict(
        #     type='Pretrained',
        #     checkpoint='open-mmlab://resnet18_v1c'),
        init_cfg=None),
    decode_head=dict(
        in_channels=512,
        channels=128))
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            project='lits17', name='fcn-r18-kd-80k'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(type='SegLocalVisualizer',
                  vis_backends=vis_backends,
                  name='visualizer')
