from mmengine.config import read_base
from monai.losses import DiceCELoss
from monai.networks.nets import AttentionUnet
from seg.models.segmentors.monai_model import MonaiModel

with read_base():
    from .._base_.datasets.synapse_3d import *  # noqa
    from .._base_.schedules.schedule_1000e_sgd import *  # noqa
    from .._base_.monai_runtime import *  # noqa

# model settings
model = dict(
    type=MonaiModel,
    num_classes=14,
    model_cfg=dict(
        type=AttentionUnet,
        spatial_dims=3,
        in_channels=1,
        out_channels=14,
        channels=(3, 4, 6, 3),
        strides=(2, 2, 2)),
    loss_function=dict(
        type=DiceCELoss, to_onehot_y=True, softmax=True),
    infer_cfg=dict(
        inf_size=roi,
        sw_batch_size=4,    # number of sliding window batch size
        infer_overlap=0.5   # sliding window inference overlap
    ))

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='synapse', name='swin-unetr-40k'),
        define_metric_cfg=dict(loss='min'))
]
visualizer = dict(type=SegLocalVisualizer,
                  vis_backends=vis_backends,
                  name='visualizer')
