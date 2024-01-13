from mmengine.config import read_base
from seg.models.backbones.encoder import EncoderNetV1c
with read_base():
    from ..unet.unet_r18_s4_d8_40k_synapse import * # noqa

model.update(dict(
    backbone=dict(
        type=EncoderNetV1c)))

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='synapse', name='unet-e18-40k'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer.update(
    dict(vis_backends=vis_backends,
         name='visualizer'))
