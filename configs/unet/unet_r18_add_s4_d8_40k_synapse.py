from mmengine.config import read_base
with read_base():
    from .unet_r18_s4_d8_40k_synapse import * # noqa
# model settings
model.update(
    dict(
        neck=dict(fusion_type='add')))

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='synapse', name='r18-unet-add-40k'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(type=SegLocalVisualizer,
                  vis_backends=vis_backends,
                  name='visualizer')
