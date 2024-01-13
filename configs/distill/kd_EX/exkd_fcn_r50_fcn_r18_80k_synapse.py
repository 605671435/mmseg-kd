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
    from ...resnet.fcn_r50_d8_80k_synapse import model as teacher_model  # noqa
    from ...resnet.fcn_r18_kd_80k_synapse import model as student_model  # noqa

teacher_ckpt = 'ckpts/fcn_r50-d8_1xb2-80k_synapse-512x512/best_mDice_84-53_iter_72000.pth'  # noqa: E501

model = dict(
    type=SingleTeacherDistill,
    architecture=dict(cfg_path=student_model, pretrained=False),
    teacher=dict(cfg_path=teacher_model, pretrained=False),
    teacher_ckpt=teacher_ckpt,
    distiller=dict(
        type=EXDistiller,
        student_recorders=dict(
            bb_l1_1_bn3=dict(type=ModuleOutputsRecorder, source='backbone.layer1.2.conv1'),
            bb_l2_1_bn3=dict(type=ModuleOutputsRecorder, source='backbone.layer2.2.conv1'),
            bb_l3_1_bn3=dict(type=ModuleOutputsRecorder, source='backbone.layer3.2.conv1'),
            bb_l4_1_bn3=dict(type=ModuleOutputsRecorder, source='backbone.layer4.2.conv1')),
        teacher_recorders=dict(
            bb_l1_2_bn3=dict(type=ModuleOutputsRecorder, source='backbone.layer1.2.conv3'),

            bb_l2_2_bn3=dict(type=ModuleOutputsRecorder, source='backbone.layer2.2.conv3'),
            bb_l2_3_bn3=dict(type=ModuleOutputsRecorder, source='backbone.layer2.3.conv3'),

            bb_l3_2_bn3=dict(type=ModuleOutputsRecorder, source='backbone.layer3.2.conv3'),
            bb_l3_3_bn3=dict(type=ModuleOutputsRecorder, source='backbone.layer3.3.conv3'),
            bb_l3_4_bn3=dict(type=ModuleOutputsRecorder, source='backbone.layer3.4.conv3'),
            bb_l3_5_bn3=dict(type=ModuleOutputsRecorder, source='backbone.layer3.5.conv3'),

            bb_l4_2_bn3=dict(type=ModuleOutputsRecorder, source='backbone.layer4.2.conv3')),
        distill_losses=dict(
            loss_1=dict(type=L2Loss, normalize=True, loss_weight=0.5, dist=True),
            loss_2=dict(type=L2Loss, normalize=True, loss_weight=0.5, dist=True),
            loss_3=dict(type=L2Loss, normalize=True, loss_weight=0.5, dist=True),
            loss_4=dict(type=L2Loss, normalize=True, loss_weight=0.5, dist=True)),
        connectors=dict(
            loss_s1_sfeat=dict(type=EXKDConnector,
                               student_channels=64,
                               teacher_channels=256),
            loss_s2_sfeat=dict(type=EXKDConnector,
                               student_channels=128,
                               teacher_channels=512),
            loss_s3_sfeat=dict(type=EXKDConnector,
                               student_channels=256,
                               teacher_channels=1024),
            loss_s4_sfeat=dict(type=EXKDConnector,
                               student_channels=512,
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
                t_feature=dict(recorder=['bb_l2_2_bn3',
                                         'bb_l2_3_bn3'],
                               from_student=False)),
            loss_3=dict(
                s_feature=dict(
                    recorder='bb_l3_1_bn3',
                    from_student=True,
                    connector='loss_s3_sfeat'),
                t_feature=dict(recorder=['bb_l3_2_bn3',
                                         'bb_l3_3_bn3',
                                         'bb_l3_4_bn3',
                                         'bb_l3_5_bn3'],
                               from_student=False)),
            loss_4=dict(
                s_feature=dict(
                    recorder='bb_l4_1_bn3',
                    from_student=True,
                    connector='loss_s4_sfeat'),
                t_feature=dict(recorder='bb_l4_2_bn3', from_student=False)))))

find_unused_parameters = True

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='synapse', name='exkd_fcn_r50_fcn_r18-80k'),
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
