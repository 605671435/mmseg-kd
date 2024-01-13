from mmengine.config import read_base
from seg.models.decode_heads.dsnet_head import DSNetHeadV2

with read_base():
    from .unet_segnext_b_160k_synapse import *  # noqa

model.update(dict(
    decode_head=dict(
        type=DSNetHeadV2,
        ratio=16)))

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='synapse', name='dsnetv3-unet-segnext_b-160k'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(type=SegLocalVisualizer,
                  vis_backends=vis_backends,
                  name='visualizer')
