from mmengine.config import read_base
from seg.models.transunet.vit_seg_modeling import VisionTransformer
from seg.models.transunet.vit_seg_modeling_resnet_skip import ResNetV2
from seg.models.decode_heads.naive_head import NaiveHead
with read_base():
    from .._base_.models.fcn_r50_d8 import *  # noqa
    from .._base_.datasets.synapse import *  # noqa
    from .._base_.schedules.schedule_40k import *  # noqa
    from .._base_.default_runtime import *  # noqa
# model settings
crop_size = (512, 512)
data_preprocessor.update(dict(size=crop_size))
model.update(
    dict(data_preprocessor=data_preprocessor,
         pretrained=None,
         neck=None,
         decode_head=dict(
             in_index=0,
             in_channels=1024,
             channels=256,
             num_classes=9),
         auxiliary_head=None,
         test_cfg=dict(mode='whole')))
model['backbone'] = dict(
    type=ResNetV2,
    in_channels=1,
    block_units=(3, 4, 9),
    width_factor=1)

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='synapse', name='r50v2-40k'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(type=SegLocalVisualizer,
                  vis_backends=vis_backends,
                  name='visualizer')
