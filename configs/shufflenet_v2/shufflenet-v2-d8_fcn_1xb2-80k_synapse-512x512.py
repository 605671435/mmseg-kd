_base_ = '../resnet/fcn_r18-d8_1xb2-80k_synapse-512x512.py'
model = dict(
    pretrained=None,
    backbone=dict(
        _delete_=True,
        type='ShuffleNetV2',
        out_indices=(0, 1, 2, 3),
        widen_factor=1.),
    decode_head=dict(in_channels=1024))

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            project='synapse', name='shufflenet-v2-d8_fcn-80k'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(type='SegLocalVisualizer',
                  vis_backends=vis_backends,
                  name='visualizer')