import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, Precision, AUROC, Recall, ROC, PrecisionRecallCurve, MatthewsCorrCoef
from tqdm import tqdm

metrics_dict = {"MCC": MatthewsCorrCoef, "auroc": AUROC, "accuracy": Accuracy, "precision": Precision,
                "recall": Recall, }

HERE = Path(os.getcwd())
sys.path.append(str(HERE / '/../'))
from IGEP.constants.epi_constants import *


def batch_edge_index(dist_mats):
    """
    Creates edge index for batched graphs

    :param dist_mats:   distance matrices
    :return:        edge index
    """

    # get total number of cdr nodes
    n_cdr_nodes = dist_mats.size()[0]
    dim_x = 0
    dim_y = n_cdr_nodes

    # get graph
    graph = dist_mats
    graph = graph.cpu()
    # identify rows
    graph = graph.detach().numpy()
    graph_edge = np.nonzero(graph)
    graph_edge = np.asarray(graph_edge)
    # Add dimensions
    graph_edge[0, :] = graph_edge[0, :] + dim_x
    graph_edge[1, :] = graph_edge[1, :] + dim_y
    # Turn into tensor and add to list
    graph_edge = torch.tensor(graph_edge)
    # add to dim_x,dim_y
    reflected_edge_index = torch.stack((graph_edge[1, :], graph_edge[0, :]))
    edge_index = torch.cat((graph_edge, reflected_edge_index), dim=1)

    return edge_index


def update_metrics(metrics, target, choice):
    """ Updates the metrics dictionary

    Args:
       metrics (dict): dictionary of metrics
       target (torch.Tensor): target tensor
       choice (torch.Tensor): choice tensor
    """
    for metric in metrics:
       metrics[metric].update(choice, target.to(torch.int64))
    return metrics


def compute_metrics(metrics):
    """ Computes the metrics

    Args: metrics (dict): dictionary of metrics

    """
    computed_loss = dict()
    for metric in metrics:
        computed_loss[metric] = metrics[metric].compute()
    return computed_loss


def reset_metrics(metrics):
    """ Resets the metrics

   Args: metrics (dict): dictionary of metrics

    """
    for metric in metrics:
       metrics[metric].reset()


def fix_seed(seed):
    """
    Fixes the seed for reproducibility
    Args: seed (int): seed to be fixed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


def reset_all_weights(model: nn.Module) -> None:
    """
    Resets all weights in a model

    Args:
        model (nn.Module): model to reset weights for

    refs:
        - https://discuss.pytorch.org/t/how-to-re-set-alll-parameters-in-a-network/20819/6
        - https://stackoverflow.com/questions/63627997/reset-parameters-of-a-neural-network-in-pytorch
        - https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    """

    @torch.no_grad()
    def weight_reset(m: nn.Module):
        """
        Resets the weights of a module
        :param m:   module
        """
        # - check if the current module has reset_parameters & if it's callabed called it on m
        reset_parameters = getattr(m, "reset_parameters", None)
        if callable(reset_parameters):
            m.reset_parameters()

    # Applies fn recursively to every submodule see: https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    model.apply(fn=weight_reset)


@torch.no_grad()
def weight_reset(m: nn.Module):
    """
    Resets the weights of a module

    :param m:   module
    """
    # - check if the current module has reset_parameters & if it's callabed called it on m
    reset_parameters = getattr(m, "reset_parameters", None)
    if callable(reset_parameters):
        m.reset_parameters()


import torch.nn.functional as F


def compute_BCE(pred, target):
    """
    Computes the binary cross entropy loss
    :param pred:    predicted values
    :param target:  target values
    :return:    loss

    """
    loss = F.binary_cross_entropy_with_logits(pred, target, pos_weight=torch.FloatTensor(
        [((target.size(1) * target.size(0)) - float(target.cpu().sum()))
         / float(target.cpu().sum())]).to(target.device))
    return loss


class Evaluator:
    """
    Class for evaluating the model
    """

    def __init__(self, model, args, dtype=torch.float64):
        """
        Args:

        :param model:   model to be evaluated
        :param args:    arguments
        :param dtype:   data type

        """
        self.model = model
        self.pytorch_total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("Total number of parameters: ", self.pytorch_total_params)

        self.args = args
        self.dtype = dtype

        if self.args.cpu:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.model = self.model.to(self.device)

    def load_state_dict(self, state_dict, key="model_state_dict"):
        """
        Loads the state dict
        :param state_dict:  state dict
        :param key:    key to be loaded
        """
        self.model.load_state_dict(torch.load(state_dict)[key])

    def reset(self, random_seed):
        """
        Resets the model
        :param random_seed:     random seed
        """
        fix_seed(random_seed)
        self.model.apply(fn=weight_reset)

    def define_optimizer(self):
        """
        Defines the optimizer
        """
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)

    def define_scheduler(self, optimizer):
        """
        Defines the scheduler
        :param optimizer:   optimizer
        """
        if self.args.scheduler:
            self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.decay_every,
                                                             gamma=self.args.decay_rate)

    def run_model(self, data):
        raise NotImplementedError()

    def run_epoch(self, protein_list, metrics=None,
                  optimizer=None, train=False):
        """
        Runs the epoch
        :param protein_list:    list of proteins
        :param metrics:     metrics
        :param optimizer:   optimizer
        :param train:   train or not
        :return:    computed loss
        """
        if train:
            self.model.train()
            self.optimizer.zero_grad()
        else:
            self.model.eval()
        total_loss = 0
        total_epitope_loss = 0
        total_paratope_loss = 0
        pred = {"ab": [], "ag": []}

        for i, protein in tqdm(enumerate(protein_list, 0), total=len(protein_list)):
            lbls = torch.unsqueeze(protein["cdr_lbls"], dim=1).to(self.device)
            ag_lbls = torch.unsqueeze(protein["ag_lbls"], dim=1).to(self.device)

            pred["ab"], pred["ag"] = self.run_model(protein)  # Pass Batch

            # calculate paratope loss
            paratope_loss = compute_BCE(pred["ab"], lbls.float())
            # calculate epitope loss
            epitope_loss = compute_BCE(pred["ag"], ag_lbls.float())

            loss = paratope_loss + 2 * epitope_loss
            if train:
                loss.backward()  # Calculate Gradients

            total_loss += loss.detach().item()
            total_epitope_loss += epitope_loss
            total_paratope_loss += paratope_loss
            data = {"ab_labels": lbls.cpu(), "ag_labels": ag_lbls.cpu()}
            if metrics is not None:
                choice = {"ab": [], "ag": []}
                choice["ab"] = torch.sigmoid(pred["ab"].data).cpu()
                choice["ag"] = torch.sigmoid(pred["ag"].data).cpu()
                with torch.no_grad():
                    for mol in ["ab", "ag"]:
                        update_metrics(metrics[mol], data[mol + "_labels"], choice[mol])

            if ((i + 1) % self.args.sub_batch == 0) or (i == len(protein_list) - 1):
                tqdm.write('\t[%d/%d] loss: %f' % (i, len(protein_list), total_loss / (i + 1)))
                # Step the optimizer
                if train:
                    optimizer.step()
                    optimizer.zero_grad()
        losses = {"total": total_loss / i, "ag": total_epitope_loss / i, "ab": total_paratope_loss / i}

        if metrics is not None:
            metrics_vals = dict()
            for mol in ["ab", "ag"]:
                metrics_vals[mol] = compute_metrics(metrics[mol])
                reset_metrics(metrics[mol])
            return losses, metrics_vals
        return losses

    def train_epitope(self, protein_list_train, protein_list_val, protein_list_test, model_save_path=None, n_run=None):
        """
        Evaluate epitope multitask model on test set, only outputs the final auc pr and auc roc

          :param protein_list_train:  list of proteins for training
          :param protein_list_val:    list of proteins for validation
          :param protein_list_test:   list of proteins for testing
          :param model_save_path:     path to save the model
          :param n_run:   run number
          :return:    metrics

        """

        dateTimeObj = datetime.now()
        if model_save_path is None:
            model_save_path = os.path.join('logs', self.args.path, dateTimeObj.strftime("%d%b_%H_%M") + '.pth')
        os.makedirs(os.path.split(model_save_path)[0], exist_ok=True)
        with open(os.path.join('logs', self.args.path, dateTimeObj.strftime("%d%b_%H_%M") + '_params.json'),
                  'w') as outfile:
            json.dump(vars(self.args), outfile)

        dataset_loader = {"train": DataLoader(protein_list_train, batch_size=None, shuffle=True),
                          "val": DataLoader(protein_list_val, batch_size=None, shuffle=False)}


        self.define_optimizer()
        if self.args.scheduler:
            self.define_scheduler(self.optimizer)

        metrics = self.define_metrics()
        for epoch in range(self.args.epochs):
            print("Train")
            train_losses, train_metrics = self.run_epoch(dataset_loader["train"], metrics=metrics,
                                                         optimizer=self.optimizer, train=True)
            if self.args.scheduler:
                self.scheduler.step()
            print("val")
            with torch.no_grad():
                test_losses, test_metrics = self.run_epoch(dataset_loader["val"], metrics=metrics, train=False)
            print(
                "Epoch {} - Train loss: {:06.3f}  val loss: {:06.3f}".format(epoch, train_losses["total"],
                                                                             test_losses["total"]))
            buffer = "val metrics\n"
            for mol in ["ab", "ag"]:
                buffer += "\n ".join(
                    [mol + "_" + metric + f": {test_metrics[mol][metric].item():.2f}" for metric in test_metrics[mol]])
                buffer += "\n"
            print(buffer)

            if epoch == 0 or best_loss > test_losses["total"]:
                print("New best found")
                best_loss = test_losses["total"]
                best_metrics = test_metrics.copy()
                torch.save(self.model.state_dict(), model_save_path)

        return best_metrics

    def define_metrics(self):
        """
        Define metrics for epitope multitask model

        :return: metrics dictionary

        """
        metrics = {"ag": dict(), "ab": dict()}
        for metrics_name in self.args.metrics:
            metrics["ag"][metrics_name] = metrics_dict[metrics_name](task="binary").set_dtype(
                self.dtype)
            metrics["ab"][metrics_name] = metrics_dict[metrics_name](task="binary").set_dtype(
                self.dtype)

        return metrics

    def test_epitope(self, protein_list_test, thresholds=None):
        """
        Evaluate epitope multitask model on test set, only outputs the final auc pr and auc roc

        :param protein_list_test: list of proteins to evaluate
        :param thresholds: thresholds for curves
        :return: metrics dictionary


        """
        metrics = self.define_metrics()
        curves_dict = {"ROC": ROC, "PrecRecall": PrecisionRecallCurve}
        for metrics_name in curves_dict:
            metrics["ag"][metrics_name] = curves_dict[metrics_name](task="binary", thresholds=thresholds).set_dtype(
                self.dtype)
            metrics["ab"][metrics_name] = curves_dict[metrics_name](task="binary", thresholds=thresholds).set_dtype(
                self.dtype)

        dataset_loader = DataLoader(protein_list_test, batch_size=None, shuffle=False)

        print("Test")
        with torch.no_grad():
            test_losses, test_metrics = self.run_epoch(dataset_loader, metrics=metrics, train=False)
        # remove curves
        curves = {"ag": dict(), "ab": dict()}
        for curves_name in curves_dict:
            curves["ag"][curves_name] = test_metrics["ag"].pop(curves_name)
            curves["ab"][curves_name] = test_metrics["ab"].pop(curves_name)

        buffer = "Test metrics\n"
        for mol in ["ab", "ag"]:
            buffer += "\n ".join(
                [mol + "_" + metric + f": {test_metrics[mol][metric].item():.2f}" for metric in test_metrics[mol]])
            buffer += "\n"
        print(buffer)

        dict_vals = {mol: pd.Series(data=[test_metrics[mol][metric].item() for metric in test_metrics[mol]],
                                    index=list(test_metrics[mol].keys())) for mol in ["ag", "ab"]}
        df = pd.DataFrame(dict_vals)
        print(df.to_string(float_format='{:.2f}'.format))
        return test_metrics, df, curves


from IGEP.models.epitope_model_egnn import EpiEPMP as egnnEPMP


class EGNNEvaluator(Evaluator):
    """
    Evaluator for epitope model with EGNNS
    """
    def __init__(self, args, num_cdr_feats, num_ag_feats, dtype=torch.float64):

        super().__init__(egnnEPMP(num_cdr_feats, num_ag_feats, inner_dim=args.inner_dim, use_adj=args.use_adj,
                                      update_coors=args.update_coors, dropout=args.dropout, num_egnn=args.num_egnns),
                             args, dtype)

    def define_optimizer(self):
        """
        Define optimizer for epitope model with EGNNS
        """
        ignored_params = list(map(id, [self.model.agfc.weight, self.model.agfc.bias]))
        base_params = filter(lambda p: id(p) not in ignored_params, self.model.parameters())
        ignored_params = filter(lambda p: id(p) in ignored_params, self.model.parameters())
        self.optimizer = optim.Adam([
            {'params': base_params}, {'params': ignored_params, 'lr': 0.00001}], lr=self.args.lr)

    def run_model(self, protein):
        cdrs, ags, edge_index_ab, edge_index_ag = (
            protein["cdrs"].to(self.device), protein["ags"].to(self.device),
            protein["edge_index_cdr"].to(self.device), protein["edge_index_ag"].to(self.device),
        )
        coords_ab = protein["coords_cdr"].to(self.device)
        coords_ag = protein["coords_ag"].to(self.device)
        dist_mat_train = protein['dist_mat'].to(self.device)
        edge_index_d = batch_edge_index(dist_mat_train).to(self.device)

        preds_ab, preds_ag = self.model(cdrs.float(), ags.float(), edge_index_ab, edge_index_ag, edge_index_d,
                                        coords_ab,
                                        coords_ag)
        return preds_ab, preds_ag

from IGEP.models.epitope_model import EpiEPMP


class EPMPEvaluator(Evaluator):
    """
    Evaluator for the EPMPEpitopeModel
    """
    def __init__(self, args, num_cdr_feats, num_ag_feats, dtype=torch.float64):
        super().__init__(EpiEPMP(num_cdr_feats, num_ag_feats, inner_dim=args.inner_dim), args, dtype)

    def define_optimizer(self):
        ignored_params = list(map(id, [self.model.aggcn.lin.weight, self.model.aggcn.bias,
                                       self.model.agfc.weight, self.model.agfc.bias]))
        base_params = filter(lambda p: id(p) not in ignored_params, self.model.parameters())
        ignored_params = filter(lambda p: id(p) in ignored_params, self.model.parameters())
        self.optimizer = optim.Adam([
            {'params': base_params}, {'params': ignored_params, 'lr': 0.00001}], lr=0.001)

    def run_model(self, protein):

        cdrs, ags, edge_index_ab, edge_index_ag = (
            protein["cdrs"].to(self.device), protein["ags"].to(self.device),
            protein["edge_index_cdr"].to(self.device), protein["edge_index_ag"].to(self.device),
        )
        dist_mat_train = protein['dist_mat'].to(self.device)
        edge_index_d = batch_edge_index(dist_mat_train).to(self.device)

        preds_ab, preds_ag = self.model(cdrs.float(), ags.float(), edge_index_ab, edge_index_ag,
                                        edge_index_d)
        return preds_ab, preds_ag