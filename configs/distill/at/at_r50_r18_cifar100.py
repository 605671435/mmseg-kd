from mmengine.config import read_base
from razor.models.algorithms import SingleTeacherDistill
from razor.models.distillers import ConfigurableDistiller
from mmpretrain.visualization import UniversalVisualizer
from mmrazor.models.task_modules.recorder import ModuleOutputsRecorder
from mmrazor.models.losses import ATLoss

with read_base():
    from ..._base_.datasets.cifar100 import *  # noqa
    from ..._base_.schedules.cifar10_bs128 import *  # noqa
    from ..._base_.default_runtime import *  # noqa
    # import teacher model and student model
    from ...resnet.resnet50_cifar100 import model as teacher_model  # noqa
    from ...resnet.resnet18_cifar100 import model as student_model  # noqa

optim_wrapper.update(dict(optimizer=dict(weight_decay=0.0005)))
param_scheduler = dict(
    type=MultiStepLR,
    by_epoch=True,
    milestones=[60, 120, 160],
    gamma=0.2,
)

teacher_ckpt = 'ckpts/resnet50_cifar100/top1_80-80_epoch_178.pth'  # noqa: E501
model = dict(
    type=SingleTeacherDistill,
    architecture=dict(cfg_path=student_model, pretrained=False),
    teacher=dict(cfg_path=teacher_model, pretrained=False),
    teacher_ckpt=teacher_ckpt,
    distiller=dict(
        type=ConfigurableDistiller,
        student_recorders=dict(
            bb_s1=dict(type=ModuleOutputsRecorder, source='backbone.layer1.1.relu'),
            bb_s2=dict(type=ModuleOutputsRecorder, source='backbone.layer2.1.relu'),
            bb_s3=dict(type=ModuleOutputsRecorder, source='backbone.layer3.1.relu'),
            bb_s4=dict(type=ModuleOutputsRecorder, source='backbone.layer4.1.relu')),
        teacher_recorders=dict(
            bb_s1=dict(type=ModuleOutputsRecorder, source='backbone.layer1.2.relu'),
            bb_s2=dict(type=ModuleOutputsRecorder, source='backbone.layer2.3.relu'),
            bb_s3=dict(type=ModuleOutputsRecorder, source='backbone.layer3.5.relu'),
            bb_s4=dict(type=ModuleOutputsRecorder, source='backbone.layer4.2.relu')),
        distill_losses=dict(
            loss_s1=dict(type=ATLoss, loss_weight=1e+3),
            loss_s2=dict(type=ATLoss, loss_weight=1e+3),
            loss_s3=dict(type=ATLoss, loss_weight=1e+3),
            loss_s4=dict(type=ATLoss, loss_weight=1e+3)),
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
log_processor = dict(by_epoch=True)
vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='cifar100', name='at_r50_r18'))
]
visualizer = dict(type=UniversalVisualizer,
                  vis_backends=vis_backends,
                  name='visualizer')
