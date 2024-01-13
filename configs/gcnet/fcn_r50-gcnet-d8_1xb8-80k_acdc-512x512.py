_base_ = ['../resnet/fcn_r50-d8_1xb2-80k_acdc-256x256.py']

model = dict(
    backbone=dict(plugins=[dict(cfg=dict(type='ContextBlock',
                                         ratio=1 / 2.),
                                stages=(True, True, True, True),
                                position='after_conv1')]))

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            project='acdc', name='fcn-r50-gcnet-80k'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# optimizer = dict(type='SGD', lr=0.05, momentum=0.9, weight_decay=0.0005)
# optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)
