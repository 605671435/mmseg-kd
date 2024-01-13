from mmengine.config import read_base
from seg.models.nnformer.nnformer import nnFormer
from torch.optim import AdamW
from mmengine.optim.scheduler import LinearLR, CosineAnnealingLR
from seg.models.decode_heads.naive_head import NaiveHead
from torch.nn import Conv2d
with read_base():
    from .._base_.models.unet_r18_s4_d8 import *  # noqa
    from .._base_.datasets.FLARE22 import *  # noqa
    from .._base_.schedules.schedule_40k import *  # noqa
    from .._base_.default_runtime import *  # noqa
# model settings
crop_size = (512, 512)
data_preprocessor.update(dict(size=crop_size))
model.update(
    dict(data_preprocessor=data_preprocessor,
         pretrained=None,
         neck=None,
         decode_head=dict(num_classes=14),
         auxiliary_head=None,
         test_cfg=dict(mode='whole')))
model['backbone'] = dict(
    type=nnFormer,
    conv_op=Conv2d,
    crop_size=crop_size,
    patch_size=(4, 4),
    window_size=(4, 4, 8, 4),
    deep_supervision=False)
model['decode_head'] = dict(
    type=NaiveHead,
    in_channels=14,
    channels=14,
    num_classes=14,
    loss_decode=[
        dict(
            type=CrossEntropyLoss, use_sigmoid=False, loss_weight=1.0),
        dict(
            type=DiceLoss, naive_dice=True, eps=1e-5, use_sigmoid=False, loss_weight=1.0)])

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='flare22', name='swin-unetr-40k'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(type=SegLocalVisualizer,
                  vis_backends=vis_backends,
                  name='visualizer')
