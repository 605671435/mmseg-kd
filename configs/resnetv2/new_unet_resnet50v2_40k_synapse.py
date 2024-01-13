from mmengine.config import read_base
from seg.models.medical_seg.transunet.unet_resnet import UNetResNet
from seg.models.decode_heads.decode_head import LossHead
with read_base():
    from .._base_.models.unet_r18_d16_ds import *  # noqa
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
         decode_head=dict(num_classes=9),
         auxiliary_head=None,
         test_cfg=dict(mode='whole')))
model['backbone'] = dict(
    type=UNetResNet,
    num_classes=9,
    config=dict(
        hidden_size=1024,
        resnet=dict(num_layers=(3, 4, 9), width_factor=1),
        classifier='seg',
        decoder_channels=(512, 256, 128, 32),
        skip_channels=[512, 256, 64, 16],
        n_classes=9,
        n_skip=3,
        activation='softmax'))
model['decode_head'] = dict(
    type=LossHead,
    num_classes=9,
    loss_decode=[
        dict(
            type=CrossEntropyLoss, use_sigmoid=False, loss_weight=1.0),
        dict(
            type=MemoryEfficientSoftDiceLoss, loss_weight=1.0)])

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='synapse', name='transunet-40k'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(type=SegLocalVisualizer,
                  vis_backends=vis_backends,
                  name='visualizer')
