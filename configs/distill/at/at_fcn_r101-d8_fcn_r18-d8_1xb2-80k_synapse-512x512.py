_base_ = [
    '../../_base_/datasets/synapse.py',
    '../../_base_/schedules/schedule_80k.py',
    '../../_base_/default_runtime.py'
]

teacher_ckpt = 'ckpts/fcn_r101-d8_1xb2-80k_synapse-512x512/best_mDice_85-27_iter_72000.pth'  # noqa: E501
teacher_cfg_path = 'configs/resnet/fcn_r101-d8_1xb2-80k_synapse-512x512.py'  # noqa: E501
student_cfg_path = 'configs/resnet/fcn_r18-no-pretrain-d8_1xb2-80k_synapse-512x512.py'  # noqa: E501
model = dict(
    _scope_='razor',
    type='SingleTeacherDistill',
    architecture=dict(cfg_path=student_cfg_path, pretrained=False),
    teacher=dict(cfg_path=teacher_cfg_path, pretrained=False),
    teacher_ckpt=teacher_ckpt,
    distiller=dict(_scope_='razor',
        type='ConfigurableDistiller',
        student_recorders=dict(
            bb_s1=dict(type='ModuleOutputs', source='backbone.layer1.1.relu'),
            bb_s2=dict(type='ModuleOutputs', source='backbone.layer2.1.relu'),
            bb_s3=dict(type='ModuleOutputs', source='backbone.layer3.1.relu'),
            bb_s4=dict(type='ModuleOutputs', source='backbone.layer4.1.relu')),
        teacher_recorders=dict(
            bb_s1=dict(type='ModuleOutputs', source='backbone.layer1.2.relu'),
            bb_s2=dict(type='ModuleOutputs', source='backbone.layer2.3.relu'),
            bb_s3=dict(type='ModuleOutputs', source='backbone.layer3.22.relu'),
            bb_s4=dict(type='ModuleOutputs', source='backbone.layer4.2.relu')),
        distill_losses=dict(
            loss_s1=dict(type='ATLoss', loss_weight=250.0),
            loss_s2=dict(type='ATLoss', loss_weight=250.0),
            loss_s3=dict(type='ATLoss', loss_weight=250.0),
            loss_s4=dict(type='ATLoss', loss_weight=250.0)),
        loss_forward_mappings=dict(
            loss_s1=dict(
                s_feature=dict(
                    from_student=True, recorder='bb_s1', record_idx=1),
                t_feature=dict(
                    from_student=False, recorder='bb_s1', record_idx=2)),
            loss_s2=dict(
                s_feature=dict(
                    from_student=True, recorder='bb_s2', record_idx=1),
                t_feature=dict(
                    from_student=False, recorder='bb_s2', record_idx=2)),
            loss_s3=dict(
                s_feature=dict(
                    from_student=True, recorder='bb_s3', record_idx=1),
                t_feature=dict(
                    from_student=False, recorder='bb_s3', record_idx=2)),
            loss_s4=dict(
                s_feature=dict(
                    from_student=True, recorder='bb_s4', record_idx=1),
                t_feature=dict(
                    from_student=False, recorder='bb_s4', record_idx=2)))))

find_unused_parameters = True

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            project='synapse', name='at_fcn_r101_fcn_r18-80k'),
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
