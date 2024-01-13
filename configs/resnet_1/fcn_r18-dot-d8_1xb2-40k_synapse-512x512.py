_base_ = ['./fcn_r50-d8_1xb2-40k_synapse-512x512.py']

model = dict(
    backbone=dict(
        depth=10,
        init_cfg=None,
        plugins=[
            dict(cfg=dict(type='EX_KD'),
                 stages=(True, True, True, True),
                 position='after_conv2')]),
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
