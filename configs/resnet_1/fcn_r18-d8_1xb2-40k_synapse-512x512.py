_base_ = ['./fcn_r50-d8_1xb2-40k_synapse-512x512.py']

model = dict(
    backbone=dict(
        depth=18,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://resnet18_v1c')),
    decode_head=dict(
        in_channels=512,
        channels=128))
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            project='synapse', name='fcn-r18-40k'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(type='SegLocalVisualizer',
                  vis_backends=vis_backends,
                  name='visualizer')
