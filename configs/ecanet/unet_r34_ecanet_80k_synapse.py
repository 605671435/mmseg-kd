from mmengine.config import read_base
from seg.models.utils import eca_layer


with read_base():
    from ..unet.unet_r34_s4_d8_80k_synapse import * # noqa

model.update(dict(
    backbone=dict(
        plugins=[dict(
                        cfg=dict(type=eca_layer),
                        stages=(True, True, True, True),
                        position='after_conv2')
                    ]
    )
))

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='synapse', name='unet-r34-ecanet-80k'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(
    type=SegLocalVisualizer, vis_backends=vis_backends, name='visualizer')
