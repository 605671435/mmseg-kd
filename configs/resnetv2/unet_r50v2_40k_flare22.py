from mmengine.config import read_base
from seg.models.medical_seg.transunet.vit_seg_modeling_resnet_skip import ResNetV2
with read_base():
    from .._base_.models.unet_r18_d16_ds import *  # noqa
    from .._base_.datasets.flare22 import *  # noqa
    from .._base_.schedules.schedule_40k import *  # noqa
    from .._base_.default_runtime import *  # noqa
# model settings
crop_size = (512, 512)
data_preprocessor.update(dict(size=crop_size))
model.update(
    dict(data_preprocessor=data_preprocessor,
         pretrained=None,
         neck=dict(
            # base_channels=128,
            in_channels=[64, 256, 512, 1024],
            dec_num_convs=(2, 2, 2),
            norm_cfg=dict(type=InstanceNorm2d, requires_grad=True),
            act_cfg=dict(type=LeakyReLU)),
         decode_head=dict(
             in_channels=64,
             channels=64,
             num_classes=14,
             norm_cfg=dict(type=InstanceNorm2d, requires_grad=True)),
         auxiliary_head=None,
         test_cfg=dict(mode='whole')))
model['backbone'] = dict(
    type=ResNetV2,
    in_channels=1,
    block_units=(3, 4, 9),
    width_factor=1,
    output_tuple=True)

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='flare22', name='r50v2-40k'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(type=SegLocalVisualizer,
                  vis_backends=vis_backends,
                  name='visualizer')
