from mmengine.config import read_base
from seg.models.utils.se_layer import SqueezeExcite


with read_base():
    from ..residual_encoder_unet.re_unet_ds_40k_synapse import * # noqa

model.update(dict(
    backbone=dict(
        plugin=dict(type=SqueezeExcite),
    )
))
vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='synapse', name='re-unet-se-40k'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(
    type=SegLocalVisualizer, vis_backends=vis_backends, name='visualizer')

# optimizer = dict(type='SGD', lr=0.05, momentum=0.9, weight_decay=0.0005)
# optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)
