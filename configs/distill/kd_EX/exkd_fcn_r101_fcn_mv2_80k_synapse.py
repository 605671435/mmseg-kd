from mmengine.config import read_base
from razor.models.algorithms import SingleTeacherDistill
from razor.models.distillers import EXDistiller
from razor.models.architectures.connectors import EXKDConnector
from mmrazor.models.task_modules.recorder import ModuleOutputsRecorder
from mmrazor.models.losses import L2Loss
from seg.engine.hooks import MyCheckpointHook

with read_base():
    from ..._base_.datasets.synapse import *  # noqa
    from ..._base_.schedules.schedule_80k import *  # noqa
    from ..._base_.default_runtime import *  # noqa
    # import teacher model and student model
    from ...resnet.fcn_r101_d8_80k_synapse import model as teacher_model  # noqa
    from ...mobilenet_v2.mobilenet_v2_kd_fcn_80k_synapse import model as student_model  # noqa

teacher_ckpt = 'ckpts/fcn_r101-d8_1xb2-80k_synapse-512x512/best_mDice_85-27_iter_72000.pth'  # noqa: E501
model = dict(
    type=SingleTeacherDistill,
    architecture=dict(cfg_path=student_model, pretrained=False),
    teacher=dict(cfg_path=teacher_model, pretrained=False),
    teacher_ckpt=teacher_ckpt,
    distiller=dict(
        type=EXDistiller,
        student_recorders=dict(
            bb_l1_1_bn3=dict(type=ModuleOutputsRecorder, source='backbone.layer2.2.conv1'),
            bb_l2_1_bn3=dict(type=ModuleOutputsRecorder, source='backbone.layer3.3.conv1'),
            bb_l3_1_bn3=dict(type=ModuleOutputsRecorder, source='backbone.layer5.3.conv1'),
            bb_l4_1_bn3=dict(type=ModuleOutputsRecorder, source='backbone.layer7.1.conv1')),
        teacher_recorders=dict(
            bb_l1_2_bn3=dict(type=ModuleOutputsRecorder, source='backbone.layer1.2.conv3'),

            bb_l2_3_bn3=dict(type=ModuleOutputsRecorder, source='backbone.layer2.3.conv3'),

            bb_l3_3_bn3=dict(type=ModuleOutputsRecorder, source='backbone.layer3.3.conv3'),
            bb_l3_4_bn3=dict(type=ModuleOutputsRecorder, source='backbone.layer3.4.conv3'),
            bb_l3_5_bn3=dict(type=ModuleOutputsRecorder, source='backbone.layer3.5.conv3'),
            bb_l3_6_bn3=dict(type=ModuleOutputsRecorder, source='backbone.layer3.6.conv3'),
            bb_l3_7_bn3=dict(type=ModuleOutputsRecorder, source='backbone.layer3.7.conv3'),
            bb_l3_8_bn3=dict(type=ModuleOutputsRecorder, source='backbone.layer3.8.conv3'),
            bb_l3_9_bn3=dict(type=ModuleOutputsRecorder, source='backbone.layer3.9.conv3'),
            bb_l3_10_bn3=dict(type=ModuleOutputsRecorder, source='backbone.layer3.10.conv3'),
            bb_l3_11_bn3=dict(type=ModuleOutputsRecorder, source='backbone.layer3.11.conv3'),
            bb_l3_12_bn3=dict(type=ModuleOutputsRecorder, source='backbone.layer3.12.conv3'),
            bb_l3_13_bn3=dict(type=ModuleOutputsRecorder, source='backbone.layer3.13.conv3'),
            bb_l3_14_bn3=dict(type=ModuleOutputsRecorder, source='backbone.layer3.14.conv3'),
            bb_l3_15_bn3=dict(type=ModuleOutputsRecorder, source='backbone.layer3.15.conv3'),
            bb_l3_16_bn3=dict(type=ModuleOutputsRecorder, source='backbone.layer3.16.conv3'),
            bb_l3_17_bn3=dict(type=ModuleOutputsRecorder, source='backbone.layer3.17.conv3'),
            bb_l3_18_bn3=dict(type=ModuleOutputsRecorder, source='backbone.layer3.18.conv3'),
            bb_l3_19_bn3=dict(type=ModuleOutputsRecorder, source='backbone.layer3.19.conv3'),
            bb_l3_20_bn3=dict(type=ModuleOutputsRecorder, source='backbone.layer3.20.conv3'),
            bb_l3_21_bn3=dict(type=ModuleOutputsRecorder, source='backbone.layer3.21.conv3'),
            bb_l3_22_bn3=dict(type=ModuleOutputsRecorder, source='backbone.layer3.22.conv3'),

            bb_l4_1_bn3=dict(type=ModuleOutputsRecorder, source='backbone.layer4.1.conv3'),
            bb_l4_2_bn3=dict(type=ModuleOutputsRecorder, source='backbone.layer4.2.conv3')),
        distill_losses=dict(
            loss_1=dict(type=L2Loss, normalize=True, loss_weight=0.5, dist=True),
            loss_2=dict(type=L2Loss, normalize=True, loss_weight=0.5, dist=True),
            loss_3=dict(type=L2Loss, normalize=True, loss_weight=0.5, dist=True),
            loss_4=dict(type=L2Loss, normalize=True, loss_weight=0.5, dist=True)),
        connectors=dict(
            loss_s1_sfeat=dict(type=EXKDConnector,
                               student_channels=24,
                               teacher_channels=256),
            loss_s2_sfeat=dict(type=EXKDConnector,
                               student_channels=32,
                               teacher_channels=512),
            loss_s3_sfeat=dict(type=EXKDConnector,
                               student_channels=96,
                               teacher_channels=1024),
            loss_s4_sfeat=dict(type=EXKDConnector,
                               student_channels=320,
                               teacher_channels=2048)),
        loss_forward_mappings=dict(
            loss_1=dict(
                s_feature=dict(
                    recorder='bb_l1_1_bn3',
                    from_student=True,
                    connector='loss_s1_sfeat'),
                t_feature=dict(recorder='bb_l1_2_bn3', from_student=False)),
            loss_2=dict(
                s_feature=dict(
                    recorder='bb_l2_1_bn3',
                    from_student=True,
                    connector='loss_s2_sfeat'),
                t_feature=dict(recorder='bb_l2_3_bn3',
                               from_student=False)),
            loss_3=dict(
                s_feature=dict(
                    recorder='bb_l3_1_bn3',
                    from_student=True,
                    connector='loss_s3_sfeat'),
                t_feature=dict(recorder=['bb_l3_3_bn3',
                                         'bb_l3_4_bn3',
                                         'bb_l3_5_bn3',
                                         'bb_l3_6_bn3',
                                         'bb_l3_7_bn3',
                                         'bb_l3_8_bn3',
                                         'bb_l3_9_bn3',
                                         'bb_l3_10_bn3',
                                         'bb_l3_11_bn3',
                                         'bb_l3_12_bn3',
                                         'bb_l3_13_bn3',
                                         'bb_l3_14_bn3',
                                         'bb_l3_15_bn3',
                                         'bb_l3_16_bn3',
                                         'bb_l3_17_bn3',
                                         'bb_l3_18_bn3',
                                         'bb_l3_19_bn3',
                                         'bb_l3_20_bn3',
                                         'bb_l3_21_bn3',
                                         'bb_l3_22_bn3'],
                               from_student=False)),
            loss_4=dict(
                s_feature=dict(
                    recorder='bb_l4_1_bn3',
                    from_student=True,
                    connector='loss_s4_sfeat'),
                t_feature=dict(recorder=['bb_l4_1_bn3', 'bb_l4_2_bn3'], from_student=False)))))

find_unused_parameters = True

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='synapse', name='exkd_fcn_r101_fcn_r18-80k'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(type=SegLocalVisualizer,
                  vis_backends=vis_backends,
                  name='visualizer')

default_hooks.update(dict(
    checkpoint=dict(type=MyCheckpointHook,
                    by_epoch=False,
                    interval=8000,
                    max_keep_ckpts=1,
                    save_best=['mDice'], rule='greater')))
