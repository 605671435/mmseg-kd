_base_ = [
    '../../_base_/datasets/synapse.py',
    '../../_base_/schedules/schedule_80k.py',
    '../../_base_/default_runtime.py'
]

teacher_ckpt = 'ckpts/fcn_r50-d8_1xb2-80k_synapse-512x512/best_mDice_84-53_iter_72000.pth'  # noqa: E501
teacher_cfg_path = 'configs/resnet/fcn_r50-d8_1xb2-80k_synapse-512x512.py'  # noqa: E501
student_cfg_path = 'configs/resnet/fcn_r18-no-pretrain-d8_1xb2-80k_synapse-512x512.py'  # noqa: E501
model = dict(
    _scope_='razor',
    type='SingleTeacherDistill',
    architecture=dict(cfg_path=student_cfg_path, pretrained=False),
    teacher=dict(cfg_path=teacher_cfg_path, pretrained=False),
    teacher_ckpt=teacher_ckpt,
    distiller=dict(_scope_='razor',
        type='ReviewKDDistiller',
        student_recorders=dict(
            bb_s1=dict(type='ModuleInputs', source='backbone.layer1.1.relu'),
            bb_s2=dict(type='ModuleInputs', source='backbone.layer2.1.relu'),
            bb_s3=dict(type='ModuleInputs', source='backbone.layer3.1.relu'),
            bb_s4=dict(type='ModuleInputs', source='backbone.layer4.1.relu')),
        teacher_recorders=dict(
            bb_s1=dict(type='ModuleInputs', source='backbone.layer1.2.relu'),
            bb_s2=dict(type='ModuleInputs', source='backbone.layer2.3.relu'),
            bb_s3=dict(type='ModuleInputs', source='backbone.layer3.5.relu'),
            bb_s4=dict(type='ModuleInputs', source='backbone.layer4.2.relu')),
        distill_losses=dict(
            loss_hcl_s1=dict(type='HCL', loss_weight=1),
            loss_hcl_s2=dict(type='HCL', loss_weight=1),
            loss_hcl_s3=dict(type='HCL', loss_weight=1),
            loss_hcl_s4=dict(type='HCL', loss_weight=1)),
        connectors=dict(
           loss_s4_sfeat=dict(_scope_='razor',
                              type='ABFConnector',
                              student_channels=512,
                              mid_channel=512,
                              teacher_channels=2048,
                              teacher_shapes=64,
                              student_shapes=64,
                              fuse=False),
           loss_s3_sfeat=dict(_scope_='razor',
                              type='ABFConnector',
                              student_channels=256,
                              mid_channel=512,
                              teacher_channels=1024,
                              teacher_shapes=64,
                              student_shapes=64,
                              fuse=True),
           loss_s2_sfeat=dict(_scope_='razor',
                              type='ABFConnector',
                              student_channels=128,
                              mid_channel=512,
                              teacher_channels=512,
                              teacher_shapes=64,
                              student_shapes=64,
                              fuse=True),
           loss_s1_sfeat=dict(_scope_='razor',
                              type='ABFConnector',
                              student_channels=64,
                              mid_channel=512,
                              teacher_channels=256,
                              teacher_shapes=128,
                              student_shapes=128,
                              fuse=True)),
        loss_forward_mappings=dict(
            loss_hcl_s4=dict(
                s_feature=dict(from_student=True,
                               recorder='bb_s4',
                               record_idx=1,
                               data_idx=0,
                               connector='loss_s4_sfeat'),
                t_feature=dict(from_student=False, recorder='bb_s4', record_idx=2, data_idx=0)),
            loss_hcl_s3=dict(
                s_feature=dict(from_student=True,
                               recorder='bb_s3',
                               record_idx=1,
                               data_idx=0,
                               connector='loss_s3_sfeat'),
                t_feature=dict(from_student=False, recorder='bb_s3', record_idx=2, data_idx=0)),
            loss_hcl_s2=dict(
                s_feature=dict(from_student=True,
                               recorder='bb_s2',
                               record_idx=1,
                               data_idx=0,
                               connector='loss_s2_sfeat'),
                t_feature=dict(from_student=False, recorder='bb_s2', record_idx=2, data_idx=0)),
            loss_hcl_s1=dict(
                s_feature=dict(from_student=True,
                               recorder='bb_s1',
                               record_idx=1,
                               data_idx=0,
                               connector='loss_s1_sfeat'),
                t_feature=dict(from_student=False, recorder='bb_s1', record_idx=2, data_idx=0)))))

find_unused_parameters = True

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            project='synapse', name='reviewkd_fcn_r50_fcn_r18-80k'),
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
