from mmengine.config import read_base
from seg.models.utils.hamburger import Ham


with read_base():
    from ..unet.unet_r18v1c_d8_40k_synapse import * # noqa

model.update(dict(
    backbone=dict(
        final_plugins=dict(type=Ham))))

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='synapse', name='unet-r18v1c-hamnet'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(
    type=SegLocalVisualizer, vis_backends=vis_backends, name='visualizer')
