_base_ = [
    '../../_base_/datasets/synapse.py',
    '../../_base_/schedules/schedule_80k.py',
    '../../_base_/default_runtime.py'
]

teacher_ckpt = '/home/jz207/workspace/zhangdw/ex_kd/work_dirs/fcn_r50-d8_1xb2-80k_synapse-512x512/5-run_20230526_010951/run2/best_mDice_84-09_iter_32000.pth'  # noqa: E501
teacher_cfg_path = 'configs/resnet/fcn_r50-d8_1xb2-80k_synapse-512x512.py'  # noqa: E501
student_cfg_path = 'configs/resnet/fcn_r18-d8_1xb2-80k_synapse-512x512.py'  # noqa: E501
model = dict(
    _scope_='razor',
    type='SingleTeacherDistill',
    architecture=dict(cfg_path=student_cfg_path, pretrained=False),
    teacher=dict(cfg_path=teacher_cfg_path, pretrained=False),
    teacher_ckpt=teacher_ckpt,
    distiller=dict(
        type='ConfigurableDistiller',
        student_recorders=dict(
            logits=dict(type='ModuleOutputs', source='decode_head.conv_seg')),
        teacher_recorders=dict(
            logits=dict(type='ModuleOutputs', source='decode_head.conv_seg')),
        distill_losses=dict(
            loss_dist=dict(type='DISTLoss', tau=1, loss_weight=3)),
        loss_forward_mappings=dict(
            loss_dist=dict(
                preds_S=dict(from_student=True, recorder='logits'),
                preds_T=dict(from_student=False, recorder='logits')))))

find_unused_parameters = True

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            project='synapse', name='dist_fcn_r50_fcn_r18-80k'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(type='SegLocalVisualizer',
                  vis_backends=vis_backends,
                  name='visualizer')

default_hooks = dict(
    checkpoint=dict(type='MyCheckpointHook',
                    by_epoch=False,
                    interval=8000,
                    max_keep_ckpts=1,
                    save_best=['mDice'], rule='greater'))
