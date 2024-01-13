from mmengine.config import read_base
from pretrain.models.backbones.freqnet import FreqNet

with read_base():
    from ..resnet.fcn_r50_d8_40k_synapse import * # noqa

model['backbone'] = dict(
    type=FreqNet,
    depth=50,
    num_stages=4,
    out_indices=(0, 1, 2, 3),
    dilations=(1, 1, 2, 4),
    strides=(1, 2, 1, 1),
    norm_cfg=norm_cfg,
    norm_eval=False,
    style='pytorch')

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='synapse', name='fcn-fcanet50'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(
    type=SegLocalVisualizer, vis_backends=vis_backends, name='visualizer')

