from mmengine.config import read_base
from seg.models.backbones import ShuffleNetV1
with read_base():
    from ..resnet.fcn_r18_d8_80k_synapse import *  # noqa

model['backbone'] = dict(
        type=ShuffleNetV1,
        out_indices=(0, 1, 2),
        widen_factor=1.)

model.update(dict(
    pretrained=None,
    decode_head=dict(in_channels=960, in_index=2)))

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='synapse', name='shufflenet-v2-d8_fcn-80k'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(type=SegLocalVisualizer,
                  vis_backends=vis_backends,
                  name='visualizer')
