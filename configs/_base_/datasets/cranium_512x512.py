from seg.datasets.cranium_dataset import CraniumDataset
from mmseg.datasets.transforms.loading import LoadImageFromFile, LoadAnnotations
from mmcv.transforms.processing import Resize, RandomFlip
from mmseg.datasets.transforms.transforms import PhotoMetricDistortion
from mmseg.datasets.transforms.formatting import PackSegInputs
from mmengine.dataset.sampler import InfiniteSampler, DefaultSampler
from seg.evaluation.metrics import IoUMetric

dataset_type = CraniumDataset
data_root = 'data/Cranium/'
img_scale = (512, 512)
train_pipeline = [
    dict(type=LoadImageFromFile),
    dict(type=LoadAnnotations),
    dict(type=Resize, scale=img_scale, keep_ratio=False),
    dict(type=RandomFlip, prob=0.5),
    dict(type=PhotoMetricDistortion),
    dict(type=PackSegInputs)
]
test_pipeline = [
    dict(type=LoadImageFromFile),
    dict(type=Resize, scale=img_scale, keep_ratio=False),
    dict(type=LoadAnnotations),
    dict(type=PackSegInputs)
]
train_list = ['120', '074', '073', '065', '077', '063', '097', '124', '059', '081', '060', '090', '070', '107', '083',
              '054', '108', '058', '112', '088', '092', '102', '055', '116', '096', '084', '079', '093', '078', '099',
              '113', '072', '122', '128', '129', '085', '061', '051', '089', '087', '114', '075', '119', '071', '117',
              '053', '062', '067', '064', '121', '049', '068', '080', '082', '106', '126', '118', '056', '125', '095',
              '052', '100']


val_list = ['104', '123', '109', '091', '066', '076', '050', '130', '069', '098', '110', '086', '101', '115', '105',
            '103', '127', '057', '094', '111']

train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type=InfiniteSampler, shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        case_list=train_list,
        data_prefix=dict(img_path='images/train', seg_map_path='masks/train'),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        case_list=val_list,
        data_prefix=dict(img_path='images/train', seg_map_path='masks/train'),
        pipeline=test_pipeline))
test_dataloader = val_dataloader
val_evaluator = dict(type=IoUMetric, iou_metrics=['mIoU', 'mDice'])
test_evaluator = dict(type=IoUMetric, iou_metrics=['mIoU', 'mDice'])
