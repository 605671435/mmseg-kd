import torch
import torch.nn as nn
from mmseg.models.utils import resize

def cosine_similarity(x, y, eps=1e-8):
    return (x * y).sum(1) / (x.norm(dim=1) * y.norm(dim=1) + eps)


def pearson_correlation(x, y, eps=1e-8):
    return cosine_similarity(x - x.mean(1).unsqueeze(1), y - y.mean(1).unsqueeze(1), eps)


def inter_class_relation(y_s, y_t):
    return 1 - pearson_correlation(y_s, y_t).mean()


def intra_class_relation(y_s, y_t):
    return inter_class_relation(y_s.transpose(0, 1), y_t.transpose(0, 1))


class DIST(nn.Module):
    def __init__(self,
                 inter_loss_weight=1.,
                 intra_loss_weight=1.,
                 tau=1.0,
                 loss_weight: float = 1.0,
                 teacher_detach: bool = True):
        super(DIST, self).__init__()
        self.inter_loss_weight = inter_loss_weight
        self.intra_loss_weight = intra_loss_weight
        self.tau = tau

        self.loss_weight = loss_weight
        self.teacher_detach = teacher_detach

    def forward(self, logits_S, logits_T: torch.Tensor):  # noqa
        assert logits_S.ndim in (2, 4)
        if self.teacher_detach:
            logits_T = logits_T.detach()  # noqa
        if logits_S.shape[-2:] != logits_T.shape[-2:]:
            logits_S = resize(   # noqa
                input=logits_S,
                size=logits_T.shape[-2:],
                mode='bilinear')
        if logits_S.ndim == 4:
            num_classes = logits_S.shape[1]
            logits_S = logits_S.transpose(1, 3).reshape(-1, num_classes)  # noqa
            logits_T = logits_T.transpose(1, 3).reshape(-1, num_classes)  # noqa
        logits_S = (logits_S / self.tau).softmax(dim=1)  # noqa
        logits_T = (logits_T / self.tau).softmax(dim=1)  # noqa
        inter_loss = self.tau**2 * inter_class_relation(logits_S, logits_T)
        intra_loss = self.tau**2 * intra_class_relation(logits_S, logits_T)
        loss = self.inter_loss_weight * inter_loss + self.intra_loss_weight * intra_loss
        return loss


