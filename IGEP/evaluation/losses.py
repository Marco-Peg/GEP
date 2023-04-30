import torch
import torch.nn.functional as F
from torchmetrics import Metric


class BCE(Metric):
    is_differentiable = True
    higher_is_better = False
    full_state_update = True

    def __init__(self, **kwargs):
        super().__init__()
        self.add_state("loss", default=torch.tensor(.0, requires_grad=True), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        self.loss += F.binary_cross_entropy_with_logits(preds, target, pos_weight=torch.FloatTensor(
            [(target.size(1) - float(target.cpu().sum()))
             / float(target.cpu().sum())]).to(target.device))
        self.total += target.size(0)

    def compute(self):
        return self.loss / self.total

    def reset(self) -> None:
        super().reset()
        self.loss.requires_grad = True

def compute_losses(losses, data, pred):
    preds = torch.cat((pred["ab"], pred["ag"]), 1)
    target = torch.cat((data["ab_labels"], data["ag_labels"]), -1).unsqueeze(-1).to(preds.dtype)
    losses_vals = dict()
    if "BCE" in losses:
        losses_vals["BCE"] = F.binary_cross_entropy_with_logits(preds, target, pos_weight=torch.FloatTensor(
            [((target.size(1) * target.size(0)) - float(target.cpu().sum()))
             / float(target.cpu().sum())]).to(target.device))
    return losses_vals


def update_metrics(metrics, target, choice):
    loss_select = list(metrics.keys())
    for x in ["accuracy", "precision", "recall"]:
        if x in loss_select:
            metrics[x].update(choice, target.to(torch.int64))
    if "auroc" in loss_select:
        metrics["auroc"].update(choice.view(-1), target.to(torch.int64).view(-1))
    return metrics


def compute_metrics(metrics):
    computed_loss = dict()
    for metric in metrics:
        computed_loss[metric] = metrics[metric].compute()
    return computed_loss


def reset_metrics(metrics):
    for metric in metrics:
        metrics[metric].reset()
