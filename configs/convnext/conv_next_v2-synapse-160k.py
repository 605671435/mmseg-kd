_base_ = ['./conv_next.py']
model = dict(
    backbone=dict(
        layer_scale_init_value=0.,
        use_grn=True)
)

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            project='synapse', name='convnextv2-160k'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(type='SegLocalVisualizer',
                  vis_backends=vis_backends,
                  name='visualizer')
