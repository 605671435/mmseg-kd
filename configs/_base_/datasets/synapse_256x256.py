from seg.datasets import SynapseDataset
from mmseg.datasets.transforms.loading import LoadImageFromFile, LoadAnnotations
from seg.datasets.transforms.transforms import BioMedicalRandomGamma
from mmcv.transforms.processing import Resize
from mmseg.datasets.transforms.transforms import RandomRotFlip, \
    BioMedicalGaussianNoise, BioMedicalGaussianBlur
from mmseg.datasets.transforms.formatting import PackSegInputs
from mmengine.dataset.sampler import InfiniteSampler, DefaultSampler
from seg.evaluation.metrics import IoUMetric

dataset_type = SynapseDataset
data_root = 'data/synapse9/'
img_scale = (256, 256)
train_pipeline = [
    dict(type=LoadImageFromFile, color_type='grayscale'),
    dict(type=BioMedicalGaussianNoise),
    dict(type=BioMedicalGaussianBlur, different_sigma_per_axis=False),
    # dict(type=BioMedicalRandomGamma, prob=1.0, gamma_range=(0.7, 1.5),
    #      invert_image=True, per_channel=True, retain_stats=True),
    # dict(type=BioMedicalRandomGamma, prob=1.0, gamma_range=(0.7, 1.5),
    #      invert_image=False, per_channel=True, retain_stats=True),
    dict(type=LoadAnnotations),
    dict(type=Resize, scale=img_scale, keep_ratio=True),
    dict(type=RandomRotFlip, rotate_prob=0.5, flip_prob=0.5, degree=20),
    dict(type=PackSegInputs)
]
test_pipeline = [
    dict(type=LoadImageFromFile, color_type='grayscale'),
    dict(type=Resize, scale=img_scale, keep_ratio=True),
    dict(type=LoadAnnotations),
    dict(type=PackSegInputs)
]
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type=InfiniteSampler, shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='img_dir/train', seg_map_path='ann_dir/train'),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='img_dir/val', seg_map_path='ann_dir/val'),
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(type=IoUMetric, iou_metrics=['mDice', 'mIoU'], ignore_index=0)
test_evaluator = val_evaluator
# test_evaluator = dict(type=CaseMetric, iou_metrics=['mDice'], hd_metric=True, ignore_index=0)
