from mmengine.config import read_base
with read_base():
    from .unet_r18v1c_d8_40k_synapse import *  # noqa
# model settings
model.update(
    dict(backbone=dict(
        depth=34,
    )))

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='synapse', name='unet-r34v1c-40k'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(type=SegLocalVisualizer,
                  vis_backends=vis_backends,
                  name='visualizer')
