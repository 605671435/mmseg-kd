from mmseg.datasets import SynapseDataset
from seg.datasets.transforms.formatting import PackSynapseInputs
from seg.datasets.transforms.loading import LoadSynapseFromFile
from mmcv.transforms.processing import Resize
from mmseg.datasets.transforms.transforms import RandomRotFlip
from mmseg.datasets.transforms.formatting import PackSegInputs
from mmengine.dataset.sampler import InfiniteSampler, DefaultSampler
from seg.evaluation.metrics import IoUMetric, CaseMetric

dataset_type = SynapseDataset
data_root = 'data/synapse9/'
img_scale = (512, 512)
train_pipeline = [
    dict(type=LoadSynapseFromFile),
    dict(type=Resize, scale=img_scale, keep_ratio=True),
    dict(type=RandomRotFlip, rotate_prob=0.5, flip_prob=0.5, degree=20),
    dict(type=PackSegInputs)
]
val_pipeline = [
    dict(type=LoadSynapseFromFile),
    dict(type=PackSegInputs)
]
test_pipeline = [
    dict(type=LoadSynapseFromFile, h5=True),
    dict(type=PackSynapseInputs)
]
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type=InfiniteSampler, shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        img_suffix='npz',
        seg_map_suffix='npz',
        data_prefix=dict(
            img_path='train_npz', seg_map_path='train_npz'),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        img_suffix='npz',
        seg_map_suffix='npz',
        data_prefix=dict(img_path='test_npz', seg_map_path='test_npz'),
        pipeline=val_pipeline))

test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        img_suffix='h5',
        seg_map_suffix='h5',
        data_prefix=dict(img_path='test_vol_h5', seg_map_path='test_vol_h5'),
        pipeline=test_pipeline))

val_evaluator = dict(type=IoUMetric, iou_metrics=['mDice', 'mIoU'], ignore_index=0)
test_evaluator = dict(type=CaseMetric, case_metrics=['Dice', 'Jaccard', 'HD95', 'ASD'])
