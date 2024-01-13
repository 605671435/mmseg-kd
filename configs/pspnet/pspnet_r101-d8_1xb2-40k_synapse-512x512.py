_base_ = ['./pspnet_r50-d8_1xb2-40k_synapse-512x512.py']
# model settings
model = dict(pretrained='open-mmlab://resnet101_v1c', backbone=dict(depth=101))
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            project='synapse', name='pspnet-r101-40k'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(type='SegLocalVisualizer',
                  vis_backends=vis_backends,
                  name='visualizer')

