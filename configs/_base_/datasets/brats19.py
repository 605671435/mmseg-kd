dataset_type = 'BraTS19'
data_root = 'data/brats19/'
img_scale = (240, 240, 160)
train_pipeline = [
    dict(type='LoadMultiModalitiesData', with_seg=True),
    dict(type='BioMedical3DPad', pad_shape=img_scale),
    dict(type='BioMedical3DRandomCrop', crop_shape=128),
    dict(type='BioMedical3DRandomFlip', prob=0.5, axes=(0, 1, 2)),
    dict(type='RandomIntencityShift'),
    dict(type='ToTensor')
]
test_pipeline = [
    dict(type='LoadMultiModalitiesData', with_seg=True),
    dict(type='BioMedical3DPad', pad_shape=img_scale),
    dict(type='ToTensor')
]
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train/train.txt',
        with_seg=True,
        data_prefix=dict(
            img_path='train'),
        pipeline=train_pipeline))
# val_dataloader = dict(
#     batch_size=1,
#     num_workers=2,
#     persistent_workers=True,
#     sampler=dict(type='DefaultSampler', shuffle=False),
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         ann_file='train/train.txt',
#         with_seg=True,
#         # indices=10,
#         data_prefix=dict(
#             img_path='train'),
#         pipeline=test_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='val/valid.txt',
        with_seg=False,
        data_prefix=dict(img_path='val'),
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mDice'], ignore_index=3)
test_evaluator = val_evaluator
