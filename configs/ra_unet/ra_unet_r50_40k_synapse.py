from mmengine import read_base
from seg.models.necks.RAUNet import AAM
with read_base():
    from ..unet.unet_r50_s4_d8_40k_synapse import *  # noqa

model['neck'].update(
    dict(fusion_cfg=dict(type=AAM)))

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='synapse', name='ra-unet-r18-40k'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(type=SegLocalVisualizer,
                  vis_backends=vis_backends,
                  name='visualizer')