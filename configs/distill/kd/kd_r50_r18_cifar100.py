from mmengine.config import read_base
from razor.models.algorithms import SingleTeacherDistill
from razor.models.distillers import ConfigurableDistiller
from mmrazor.models.task_modules.recorder import ModuleOutputsRecorder
from mmrazor.models.losses import KLDivergence
from mmpretrain.visualization import UniversalVisualizer
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
            fc=dict(type=ModuleOutputsRecorder, source='head.fc')),
        teacher_recorders=dict(
            fc=dict(type=ModuleOutputsRecorder, source='head.fc')),
        distill_losses=dict(
            loss_kl=dict(type=KLDivergence, tau=1, loss_weight=3.0)),
        loss_forward_mappings=dict(
            loss_kl=dict(
                preds_S=dict(from_student=True, recorder='fc'),
                preds_T=dict(from_student=False, recorder='fc')))))

find_unused_parameters = True

log_processor = dict(by_epoch=True)
vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='cifar100', name='kd_r50_r18'))
]
visualizer.update(
    dict(type=UniversalVisualizer,
         vis_backends=vis_backends,
         name='visualizer'))
