from mmpretrain.models.classifiers import ImageClassifier
from mmpretrain.models.backbones import ResNet
from mmpretrain.models.necks import GlobalAveragePooling
from mmpretrain.models.losses import CrossEntropyLoss
from mmpretrain.models.heads import LinearClsHead

# model settings
model = dict(
    type=ImageClassifier,
    backbone=dict(
        type=ResNet,
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
    neck=dict(type=GlobalAveragePooling),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=2048,
        loss=dict(type=CrossEntropyLoss, loss_weight=1.0),
        topk=(1, 5),
    ))