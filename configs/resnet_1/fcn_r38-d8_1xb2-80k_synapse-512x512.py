_base_ = ['./fcn_r50-d8_1xb2-80k_synapse-512x512.py']

model = dict(
    backbone=dict(
        type='ResNetV1c_KD',
        depth=38,
        init_cfg=None))
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            project='synapse', name='fcn-r38-80k'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(type='SegLocalVisualizer',
                  vis_backends=vis_backends,
                  name='visualizer')
