dataset_type = 'BraTS19_2D'
data_root = 'data/brats19_2d/'
img_scale = (240, 240, 160)
train_pipeline = [
    dict(type='LoadBraTSData2D', with_seg=True),
    dict(type='RandomCrop', crop_size=128),
    dict(type='RandomFlip', prob=0.5, direction=['horizontal', 'vertical']),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadBraTSData2D', with_seg=True),
    dict(type='PackSegInputs')
]
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        with_seg=True,
        ann_file='img_dir/train/train.txt',
        data_prefix=dict(
            img_path='img_dir/train'),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        with_seg=True,
        # indices=10,
        data_prefix=dict(
            img_path='img_dir/train'),
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(type='PerCaseMetric', iou_metrics=['mDice'], split_for_case='_slice')
test_evaluator = val_evaluator
