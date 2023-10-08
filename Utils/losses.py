import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy, Precision, AUROC, Recall, MatthewsCorrCoef
from torchmetrics import Metric

metrics_dict = {"MCC": MatthewsCorrCoef, "auroc": AUROC, "accuracy": Accuracy, "precision": Precision,
                "recall": Recall, }

class BCE(Metric):
    """
    Binary Cross Entropy

    Args:
        preds: Predictions from the model
        target: Ground truth labels

    Returns:
        BCE loss
    """
    is_differentiable = True
    higher_is_better = False
    full_state_update = True

    def __init__(self, **kwargs):
        """     Initialize BCE loss
        """
        super().__init__()
        self.add_state("loss", default=torch.tensor(.0, requires_grad=True), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """     Update BCE loss
        :param preds:   Predictions from the model
        :param target:  Ground truth labels
        :return:        BCE loss
        """

        self.loss += F.binary_cross_entropy_with_logits(preds, target, pos_weight=torch.FloatTensor(
            [(target.size(1) - float(target.cpu().sum()))
             / float(target.cpu().sum())]).to(target.device))
        self.total += target.size(0)

    def compute(self):
        return self.loss / self.total

    def reset(self) -> None:
        super().reset()
        self.loss.requires_grad = True


def gk(x):
    """
    Gaussian Kernel
    :param x:   Input
    :return:    Gaussian Kernel
    """
    cen = torch.nn.Parameter(torch.tensor([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]).float(),
                             requires_grad=False).to(x.device)

    xmat = x.float() - cen
    sigma = 200
    y = torch.sum(torch.sigmoid(sigma * (xmat + 0.1 / 2)) - torch.sigmoid(sigma * (xmat - 0.1 / 2)), dim=0)
    y = y / torch.sum(y)
    return y


def hist_loss(br, bl, verts_r, verts_l, max_subsample=5000, empty_penalty=1e3, lloss=1):
    """    Histogram Loss
    distance distributions (represented by histograms) of subsets of random points in the positive labeled point clouds
    Act as regulizer to enforce complementary shapes

    :param br: prediction on Receptor surface [ n_verts, 1]
    :param bl: prediction on Ligand surface [ n_verts, 1]
    :param verts_r: coordinates of Receptor surface [ n_verts, 3]
    :param verts_l: coordinates of Ligand surface [ n_verts, 3]
    :param max_subsample: maximum vertices to consider
    :return: loss as scalar
    """

    br = (torch.gt(br, 0.5) == 1).nonzero()
    br = torch.squeeze(br, -1)
    bl = (torch.gt(bl, 0.5) == 1).nonzero()
    bl = torch.squeeze(bl, -1)
    # if they are not empty
    if br.size(0) != 0 and bl.size(0) != 0:
        if br.size(0) > max_subsample:
            brs = np.random.choice(br.size(0), max_subsample, replace=False)
            pr = verts_r[br[brs], :]
        else:
            pr = verts_r[br, :]  # Nr,3

        if bl.size(0) > max_subsample:
            bls = np.random.choice(bl.size(0), max_subsample, replace=False)
            pl = verts_l[bl[bls], :]
        else:
            pl = verts_l[bl, :]
        dr = torch.cdist(pr, pr)  # Nr,Nr
        dl = torch.cdist(pl, pl)
        diam = torch.max(torch.tensor([torch.max(dr), torch.max(dl)]))
        dr = dr / diam
        dl = dl / diam

        if lloss == 1:
            loss = F.l1_loss(gk(dr.view(-1, 1)), gk(dl.view(-1, 1)))
        else:
            loss = F.mse_loss(gk(dr.view(-1, 1)), gk(dl.view(-1, 1)))
        return loss
    else:
        return torch.tensor(empty_penalty)


class HistLoss(Metric):
    """ Histogram Loss
    distance distributions (represented by histograms) of subsets of random points in the positive labeled point clouds
    Act as regulizer to enforce complementary shapes
    """
    is_differentiable = True
    higher_is_better = False
    full_state_update = True

    def __init__(self, weight=1.0, **kwargs):
        """    Initialize BCE loss
        """

        super().__init__()
        self.add_state("loss", default=torch.tensor(.0, requires_grad=True), dist_reduce_fx="sum", )
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.weight = weight

    def update(self, choice1, choice2, verts1, verts2):
        """     Update Histogram loss
        :param choice1:
        :param choice2:
        :param verts1:
        :param verts2:
        :return:            Histogram loss
        """

        self.loss += hist_loss(choice1.view(-1, 1), choice2.view(-1, 1),
                               verts1.view(-1, 3), verts2.view(-1, 3), empty_penalty=1e2)
        self.total += choice1.size(0)

    def compute(self):
        """     Compute Histogram loss
        :return:    Histogram loss
        """
        return self.weight * self.loss / self.total

    def reset(self) -> None:
        """ Reset loss
        """
        super().reset()
        self.loss.requires_grad = True

def compute_BCE(pred, target):
    """
    Compute Binary Cross Entropy Loss

    :param pred:  prediction
    :param target:  target
    :return:      loss
    """
    loss = F.binary_cross_entropy_with_logits(pred, target, pos_weight=torch.FloatTensor(
        [((target.size(1) * target.size(0)) - float(target.cpu().sum()))
         / float(target.cpu().sum())]).to(target.device))

    return loss

def compute_losses(losses, data, pred):
    """
    Compute losses

    :param losses:  list of losses to compute
    :param data:    data dict
    :param pred:    prediction dict
    :return:
    """
    losses_vals = dict()
    if "BCE" in losses:
        preds = torch.cat((pred["ab"], pred["ag"]), 1)
        target = torch.cat((data["ab_labels"], data["ag_labels"]), -1).unsqueeze(-1).to(preds.dtype)
        losses_vals["BCE"] = compute_BCE(preds, target.unsqueeze(-1).to(preds.dtype))
    if "BCE_ag" in losses:
        losses_vals["BCE_ag"] = compute_BCE(pred["ag"], data["ag_labels"].unsqueeze(-1).to(pred["ag"].dtype))
    if "BCE_ab" in losses:
        losses_vals["BCE_ab"] = compute_BCE(pred["ab"], data["ab_labels"].unsqueeze(-1).to(pred["ab"].dtype))
    if "hist" in losses:
        preds = torch.cat((pred["ab"], pred["ag"]), 1)
        # target = torch.cat((data["ab_labels"], data["ag_labels"]), -1).unsqueeze(-1).to(preds.dtype)

        choice = {"ab": [], "ag": []}
        choice["ab"] = torch.sigmoid(pred["ab"].data).long().cpu()
        choice["ag"] = torch.sigmoid(pred["ag"].data).long().cpu()
        losses_vals["hist"] = hist_loss(choice["ab"].view(-1, 1), choice["ag"].view(-1, 1),
                                        data["ab_verts"].view(-1, 3), data["ag_verts"].view(-1, 3), empty_penalty=1e2)
    return losses_vals


def update_metrics(metrics, target, choice):
    """     Update metrics
    :param metrics:    Dictionary of metrics
    :param target:     Target labels
    :param choice:
    :return:      Updated metrics
    """
    for metric in metrics:
        metrics[metric].update(choice, target.to(torch.int64))
    return metrics


def compute_metrics(metrics):
    """    Compute metrics
    :param metrics:     Dictionary of metrics
    :return:        Computed metrics
    """

    computed_loss = dict()
    for metric in metrics:
        computed_loss[metric] = metrics[metric].compute()
    return computed_loss


def reset_metrics(metrics):
    """    Reset metrics
    :param metrics:     Dictionary of metrics
    :return:        Reset metrics
    """

    for metric in metrics:
        metrics[metric].reset()
