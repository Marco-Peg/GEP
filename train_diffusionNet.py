import argparse
import json
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchinfo import summary

import wandb
from OGEP.Models.models import DiffNet_AbAgPredictor
from Utils.dataset import datasetCreator
from Utils.losses import metrics_dict
from Utils.train_utils import run_epoch, collate_None


def run_DiffNet(model, data):
    preds = model(data["ab_feats"], data["ab_massvec"], data["ag_feats"], data["ag_massvec"],
                  L1=data["ab_L"], evals1=data["ab_evals"], evecs1=data["ab_evecs"], gradX1=data["ab_gradX"],
                  gradY1=data["ab_gradY"], faces1=data.get("ab_faces", None),
                  L2=data["ag_L"], evals2=data["ag_evals"], evecs2=data["ag_evecs"], gradX2=data["ag_gradX"],
                  gradY2=data["ag_gradY"], faces2=data.get("ag_faces", None)).cpu()
    if preds.dim() < 3:
        preds = preds.view(1, -1, 1)  # batch x n_points x 1
    return preds

def parse_params():
    parser = argparse.ArgumentParser(description='Train diffusionNet on AbAg prediction.')
    parser.add_argument('-p', '--params-files', dest='params_file', default='Param_runs/diffNet_mesh_split.json',
                        type=str, help='json file containing the params of the training')
    parser.add_argument('-s', '--seed', default=42, type=int, help='random seed')
    args = parser.parse_args()

    # load model params
    with open(args.params_file, 'r') as f:
        params = json.load(f)

    params["seed"] = args.seed

    return args, params


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    args, params = parse_params()
    dataset_name = params["dataset_name"]
    # system things
    print(torch.zeros(1).cuda())
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.cuda.current_device())
    print(torch.cuda.device(0))
    print(torch.cuda.get_device_name(0))
    if torch.cuda.is_available():
        device = params["device"]
    else:
        print("cuda not available, using cpu as deafult")
        device = torch.device("cpu")
    dtype = torch.float64
    # REPRODUCIBILITY
    fix_seed(params["seed"])
    # model
    input_features = params["input_features"]  # one of ['xyz', 'hks']
    precompute_data = params["precompute_data"]

    # training settings
    n_epoch = int(params["n_epoch"])
    lr = params["lr"]
    decay_every = params["decay_every"]
    decay_rate = params["decay_rate"]
    augment_random_rotate = (input_features == 'xyz')

    # loss params
    batchSize = None if params["batch_size"] == 0 else params["batch_size"]
    bs2 = params["sub_batch_size"]  # sub batch number

    # Important paths
    base_path = os.path.dirname(__file__)
    pretrain_path = os.path.join(base_path, "pretrained_models/diffNet/{}/{}.pth".format(dataset_name, input_features))
    model_save_path = os.path.join(base_path, "saved_models/diffNet/{}/".format(dataset_name))
    os.makedirs(model_save_path, exist_ok=True)
    model_save_path = os.path.join(model_save_path, "{}.pth".format(input_features))

    # === Load datasets
    print("Load datasets")
    dataset = {"train": [], "test": [], "validation": []}
    dataset_loader = {"train": [], "test": [], "validation": []}
    dataset_params = {"as_mesh": params["as_mesh"], "npoints": params["n_points"], "centered": params["centered"],
                      "data_augmentation": augment_random_rotate, "features": params["input_features"],
                      "hks_dim": params["hks_dim"], "load_submesh": params.get("load_submesh", False),
                      "need_operators": True, "k_eig": params["k_eig"], "precompute_data": params["precompute_data"]}
    dataset["train"] = datasetCreator.get_dataset(dataset_name, split="train", dtype=dtype, **dataset_params)
    dataset_loader["train"] = DataLoader(dataset["train"], batch_size=batchSize, shuffle=True, drop_last=True,
                                         collate_fn=collate_None)
    dataset["val"] = datasetCreator.get_dataset(dataset_name, split="validation", dtype=dtype, **dataset_params)
    dataset_loader["val"] = DataLoader(dataset["val"], batch_size=batchSize, shuffle=False, drop_last=False,
                                       collate_fn=collate_None)
    num_batch = len(dataset_loader["train"])
    # === Create the model
    feat_dim = dataset["train"][0]["ab_feats"].size(-1)
    model = DiffNet_AbAgPredictor(feat_dim, 1, diffN_widths=params["diffN_widths"],
                                  diffN_dropout=True,
                                  seg_widths=params["diffN_widths"],
                                  pdrop=params["pdrop"],
                                  with_gradient_features=True,
                                  with_gradient_rotations=not params["as_mesh"],
                                  # set to False because we work with point clouds (suggested in the paper)
                                  diffusion_method=params["diffusion_method"],
                                  device=device,
                                  shared_diff=params.get("shared_diff", True))
    model.to(dtype)

    # === Optimize
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=params["decay_every"], gamma=params["decay_rate"])
    print("Number of Input features", feat_dim)
    model_stats = summary(model)


    # === train
    print("Training...")
    metrics = {"ag": dict(), "ab": dict()}
    for metrics_name in params["metrics"]:
        metrics["ag"][metrics_name] = metrics_dict[metrics_name](num_classes=2, task="binary").set_dtype(
            dtype)  # to(device=device).
        metrics["ab"][metrics_name] = metrics_dict[metrics_name](num_classes=2, task="binary").set_dtype(
            dtype)  # to(device=device).
    for epoch in range(n_epoch):
        if params["wandb_on"]:
            wandb_logs = {}
        # train_acc = train_epoch(epoch, wandb_logs)
        print("Train")
        train_losses, train_metrics = run_epoch(model, dataset_loader["train"], params["losses_weight"],
                                                metrics=metrics, device=device, run_model=run_DiffNet,
                                                optimizer=optimizer, train=True, sub_batch=bs2)
        scheduler.step()
        print("Validation")
        with torch.no_grad():
            test_losses, test_metrics = run_epoch(model, dataset_loader["val"], params["losses_weight"],
                                                  metrics=metrics, device=device, run_model=run_DiffNet,
                                                  train=False, sub_batch=bs2)
        print(
            "Epoch {} - Train loss: {:06.3f}  val loss: {:06.3f}".format(epoch, train_losses["total"],
                                                                         test_losses["total"]))
        buffer = "val metrics\n"
        for mol in ["ab", "ag"]:
            buffer += "\n ".join(
                [mol + "_" + metric + f": {test_metrics[mol][metric].item():.2f}" for metric in test_metrics[mol]])
            buffer += "\n"
        print(buffer)
        metric_val = test_metrics["ab"]["MCC"] + test_metrics["ag"]["MCC"] + test_metrics["ab"]["auroc"] + \
                     test_metrics["ag"]["auroc"]

        if epoch == 0 or best_loss > test_losses["total"]:
            print("New best found")
            best_loss = test_losses["total"]
            torch.save(model.state_dict(), model_save_path)

    print(" ==> saving last model to " + model_save_path)
    if params["wandb_on"]:
        # save model state dict
        print("Saving the model state dict...", end='')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()}, os.path.join(wandb.run.dir, "state_dict.pth"))
        print("...done!")

        # Final test
        device = torch.device("cpu")
        batchSize = 1
        augment_random_rotate = True
        dataset = {"test": []}  # {"train": [], "test": [], "validation": []}
        dataset_loader = {"test": []}  # {"train": [], "test": [], "validation": []}
        dataset_params = {"as_mesh": True, "npoints": 2000, "centered": params["centered"],
                          "data_augmentation": augment_random_rotate, "features": params["input_features"],
                          "hks_dim": params["hks_dim"], "load_submesh": True, 'load_residuals': True, "get_faces": True,
                          "need_operators": True, "precompute_data": params["precompute_data"]}
        for split in dataset.keys():
            dataset[split] = datasetCreator.get_dataset(dataset_name, split=split, dtype=dtype, **dataset_params)
            dataset_loader[split] = DataLoader(dataset[split], batch_size=batchSize, shuffle=split == "train",
                                               collate_fn=collate_None)
        model.to(device)

        model.load_state_dict(
            torch.load(os.path.join(model_save_path, "best_state_dict.pth"), map_location=device)["model_state_dict"])
        model.eval()

        from Utils.train_utils import test_model

        test_metrics, curves_vals, test_metrics_res, curves_res_vals = test_model(model, dataset_loader["test"],
                                                                                  metrics_dict=metrics_dict,
                                                                                  device=device,
                                                                                  run_model=run_DiffNet, dtype=dtype,
                                                                                  threshold=0.8,
                                                                                  reduce_res="mean")
        df_dict = dict()
        for mol in ["ag", "ab"]:
            df_dict.update(
                {f"Test_{mol}_{metric}": test_metrics[mol][metric] for metric in test_metrics[mol]})
            df_dict.update(
                {f"Test_res_{mol}_{metric}": test_metrics_res[mol][metric] for metric in
                 test_metrics_res[mol]})
        print(df_dict)
