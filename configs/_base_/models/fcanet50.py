from mmpretrain.models.classifiers import ImageClassifier
from mmpretrain.models.necks import GlobalAveragePooling
from mmpretrain.models.heads import LinearClsHead
from mmpretrain.models.losses import CrossEntropyLoss
from pretrain.models.backbones import fcanet50
# model settings
model = dict(
    type=ImageClassifier,
    backbone=dict(type=fcanet50),
    neck=dict(type=GlobalAveragePooling),
    head=dict(
        type=LinearClsHead,
        num_classes=1000,
        in_channels=2048,
        loss=dict(type=CrossEntropyLoss, loss_weight=1.0),
        topk=(1, 5),
    ))
