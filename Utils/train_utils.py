import torch
from tqdm import tqdm

from Utils.dataset import to_device
from Utils.losses import compute_losses, update_metrics, reset_metrics, compute_metrics


class NaNError(Exception):
    pass

# handle exception: None items
def collate_None(batch):
    # check if empty or None
    out_indeces = []
    for i_b in range(len(batch) - 1, -1, -1):
        if batch[i_b] is None:
            batch.pop(i_b)
    #         out_indeces.append(i_b)
    # for i_b in range(len(out_indeces),0,-1):
    #     batch.pop(i_b)
    if len(batch) == 0:
        return None
    return torch.utils.data.default_collate(batch)


# def collate_faces(batch):
#
#     return {key: collate([d[key] for d in batch], collate_fn_map=collate_fn_map) for key in elem}


def run_epoch(model, data_loader, losses_weights, metrics=None, device="cpu",
              optimizer=None, run_model=None, train=False, sub_batch=1, logger=None):
    if train:
        model.train()
        optimizer.zero_grad()
    else:
        model.eval()

    losses = list(losses_weights.keys())
    loss_total = {"total": 0.0}
    for loss in losses_weights:
        loss_total[loss] = 0
    # if metrics is not None:
    #     metrics_val = {"ag":dict(),"cdr":dict()}
    #     for metric in metrics:
    #         metrics_val["ag"][metric] = 0.0
    #         metrics_val["cdr"][metric] = 0.0

    for i, data in tqdm(enumerate(data_loader, 0), total=len(data_loader)):
        try:
            if data is None:
                continue
            if "ab_L" in data.keys() and "ag_L" in data.keys():
                if data["ab_L"].numel() == 0 or data["ag_L"].numel() == 0:
                    raise NaNError("Skip this batch")
                    # tqdm.write("Skip this batch")
            # NB: dataloader returns a sample at the time with no batch size
            data = to_device(data, device)
            preds = run_model(model, data)

            pred = {"ab": [], "ag": []}
            pred["ab"] = preds[:, 0:data["ab_feats"].size(1), :].cpu()
            pred["ag"] = preds[:, data["ab_feats"].size(1):, :].cpu()
            # apply cdr mask
            # data["ab_labels"][torch.logical_not(data["cdr_mask"])] = 0
            # pred["ab"][torch.logical_not(data["cdr_mask"].view(pred["ab"].shape))] = -1e16
            data["ab_labels"] = data["ab_labels"][data["cdr_mask"]]
            data = to_device(data, "cpu")
            pred["ab"] = pred["ab"][data["cdr_mask"]]

            computed_losses = compute_losses(losses, data, pred)
            epoch_loss = None
            for loss in computed_losses:
                loss_total[loss] += computed_losses[loss].cpu().item()
                if epoch_loss is None:
                    epoch_loss = losses_weights[loss] * computed_losses[loss]
                else:
                    epoch_loss += losses_weights[loss] * computed_losses[loss]
            loss_total["total"] += epoch_loss.cpu().detach().item()
            if train:
                epoch_loss.backward()

            if metrics is not None:
                choice = {"ab": [], "ag": []}
                choice["ab"] = torch.sigmoid(pred["ab"].data).cpu()
                choice["ag"] = torch.sigmoid(pred["ag"].data).cpu()
                with torch.no_grad():
                    for mol in ["ab", "ag"]:
                        update_metrics(metrics[mol], data[mol + "_labels"].unsqueeze(-1), choice[mol])
        except NaNError as rerr:
            print(rerr)
        finally:
            if ((i + 1) % sub_batch == 0) or (i == len(data_loader) - 1):
                tqdm.write('\t[%d/%d] loss: %f' % (i, len(data_loader), loss_total["total"] / (i + 1)))
                # Step the optimizer
                if train:
                    optimizer.step()
                    optimizer.zero_grad()
                    # if scheduler is not None:
                    #     scheduler.step()

    for loss in loss_total:
        loss_total[loss] /= len(data_loader)
    # if train:
    #     if scheduler is not None:
    #         scheduler.step()

    if metrics is not None:
        metrics_vals = dict()
        for mol in ["ab", "ag"]:
            metrics_vals[mol] = compute_metrics(metrics[mol])
            reset_metrics(metrics[mol])
        return loss_total, metrics_vals
    return loss_total


from torchmetrics import Accuracy, Precision, AUROC, Recall, ROC, PrecisionRecallCurve

metrics_dict = {"auroc": AUROC, "accuracy": Accuracy, "precision": Precision,
                "recall": Recall}
curves_dict = {"ROC": ROC, "PrecRecall": PrecisionRecallCurve}


def test_model(model, data_loader, metrics_dict=metrics_dict, curves_dict=curves_dict, thresholds=None, threshold=0.5,
               device="cpu", reduce_res='mean',
               run_model=None, dtype=torch.float64):
    model.eval()
    # for metric in metrics_dict:

    # metrics_val = {"ag":dict(),"ab":dict()}
    metrics = {"ag": dict(), "ab": dict()}
    curves = {"ag": dict(), "ab": dict()}
    for curves_name in curves_dict:
        curves["ag"][curves_name] = curves_dict[curves_name](task="binary", thresholds=thresholds).set_dtype(
            dtype)  # to(device=device).
        curves["ab"][curves_name] = curves_dict[curves_name](task="binary", thresholds=thresholds).set_dtype(
            dtype)  # to(device=device).
    for metrics_name in metrics_dict:
        metrics["ag"][metrics_name] = metrics_dict[metrics_name](task="binary", threshold=threshold).set_dtype(
            dtype)  # to(device=device).
        metrics["ab"][metrics_name] = metrics_dict[metrics_name](task="binary", threshold=threshold).set_dtype(
            dtype)  # to(device=device).
        # metrics_val["ag"][metrics_name] = list()
        # metrics_val["ab"][metrics_name] = list()
    metrics_res = {"ag": dict(), "ab": dict()}
    curves_res = {"ag": dict(), "ab": dict()}
    for curves_name in curves_dict:
        curves_res["ag"][curves_name] = curves_dict[curves_name](task="binary", thresholds=thresholds).set_dtype(
            dtype)  # to(device=device).
        curves_res["ab"][curves_name] = curves_dict[curves_name](task="binary", thresholds=thresholds).set_dtype(
            dtype)  # to(device=device).
    for metrics_name in metrics_dict:
        metrics_res["ag"][metrics_name] = metrics_dict[metrics_name](task="binary", threshold=threshold).set_dtype(
            dtype)  # to(device=device).
        metrics_res["ab"][metrics_name] = metrics_dict[metrics_name](task="binary", threshold=threshold).set_dtype(
            dtype)  # to(device=device).

    with torch.no_grad():
        for i, data in tqdm(enumerate(data_loader, 0), total=len(data_loader)):
            try:
                if data is None:
                    continue
                if "ab_L" in data.keys() and "ag_L" in data.keys():
                    if data["ab_L"].numel() == 0 or data["ag_L"].numel() == 0:
                        raise NaNError("Skip this batch")
                        # tqdm.write("Skip this batch")
                # NB: dataloader returns a sample at the time with no batch size
                data = to_device(data, device)
                preds = run_model(model, data)

                pred = {"ab": [], "ag": []}
                pred["ab"] = preds[:, 0:data["ab_feats"].size(1), :].cpu()
                pred["ag"] = preds[:, data["ab_feats"].size(1):, :].cpu()
                # apply cdr mask
                data["ab_labels"][torch.logical_not(data["cdr_mask"])] = 0
                pred["ab"][torch.logical_not(data["cdr_mask"].view(pred["ab"].shape))] = torch.tensor(-float('inf'),
                                                                                                      dtype=pred[
                                                                                                          "ab"].dtype)
                data = to_device(data, "cpu")

                choice = {"ab": [], "ag": []}
                choice["ab"] = torch.sigmoid(pred["ab"].data).cpu()
                choice["ag"] = torch.sigmoid(pred["ag"].data).cpu()

                choice["res_ab"] = torch.scatter_reduce(
                    torch.zeros((choice["ab"].shape[0], data[f"lbls_ab_res"].shape[1] + 1, choice["ab"].shape[2]),
                                dtype=torch.float64),
                    1,
                    torch.tensor(data["nearest_res_ab"].unsqueeze(-1) + 1, dtype=int),
                    torch.tensor(choice["ab"]),
                    reduce=reduce_res, include_self=False)[:, 1:, :]
                choice["res_ag"] = torch.scatter_reduce(
                    torch.zeros((choice["ag"].shape[0], data[f"lbls_ag_res"].shape[1], choice["ag"].shape[2]),
                                dtype=torch.float64),
                    1,
                    torch.tensor(data["nearest_res_ag"].unsqueeze(-1), dtype=int),
                    torch.tensor(choice["ag"]),
                    reduce=reduce_res, include_self=False)

                choice["ab"] = choice["ab"][data["cdr_mask"]]
                data["ab_labels"] = data["ab_labels"][data["cdr_mask"]]

                for mol in ["ab", "ag"]:
                    update_metrics(metrics[mol], data[mol + "_labels"].unsqueeze(-1), choice[mol])
                    update_metrics(curves[mol], data[mol + "_labels"].unsqueeze(-1), choice[mol])
                    update_metrics(metrics_res[mol], data[f"lbls_{mol}_res"].unsqueeze(-1), choice[f"res_{mol}"])
                    update_metrics(curves_res[mol], data[f"lbls_{mol}_res"].unsqueeze(-1), choice[f"res_{mol}"])
            except NaNError as rerr:
                print(rerr)
            finally:
                pass

    metrics_vals = dict()
    curves_vals = dict()
    metrics_res_vals = dict()
    curves_res_vals = dict()
    for mol in ["ab", "ag"]:
        metrics_vals[mol] = compute_metrics(metrics[mol])
        curves_vals[mol] = compute_metrics(curves[mol])
        reset_metrics(metrics[mol])
        reset_metrics(curves[mol])
        metrics_res_vals[mol] = compute_metrics(metrics_res[mol])
        curves_res_vals[mol] = compute_metrics(curves_res[mol])
        reset_metrics(metrics_res[mol])
        reset_metrics(curves_res[mol])
    return metrics_vals, curves_vals, metrics_res_vals, curves_res_vals
