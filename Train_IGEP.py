# Run training and testing on all GNN flavors
import json
import os
import sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import statistics
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd

from IGEP.evaluation.evaluate import EPMPEvaluator, EGNNEvaluator
from IGEP.preprocessing.data_utils import epipredDataset

models = {"EpiEPMP": EPMPEvaluator, "egnn": EGNNEvaluator}


def run_test(args):

    """
    Run file for evaluating EpiEPMP on test set
    :param args: arguments for running the evaluation
    :return: None
    """

    path_results = args.path
    dir = Path(os.getcwd())
    path_new_directory = str(dir) + '/logs/' + str(path_results)
    os.makedirs(path_new_directory, exist_ok=True)

    # **Load data**
    protein_list_train = epipredDataset(args.train_path, feats=args.feats, random_rotation=True, centered=args.centered)
    protein_list_val = epipredDataset(args.val_path, feats=args.feats, random_rotation=True, centered=args.centered)
    protein_list_test = epipredDataset(args.test_path, feats=args.feats, random_rotation=False, centered=args.centered)

    # Run Evaluation
    args.num_cdr_feats = protein_list_train[0]["cdrs"].shape[-1]
    args.num_ag_feats = protein_list_train[0]["ags"].shape[-1]
    evaluator = models[args.model](args, args.num_cdr_feats, args.num_ag_feats)
    metrics_vals = []
    df = None
    for i in range(args.runs):
        print(f"### run {i} ###")
        evaluator.reset(args.seeds[i])
        best_metrics = evaluator.train_epitope(protein_list_train, protein_list_val, protein_list_test,
                                               model_save_path=os.path.join(path_new_directory, "r" + str(i),
                                                                            "best.pth"), n_run=i)
        metrics_vals.append(best_metrics)
        print("Finished run " + str(i))
        df_dict = {}
        for mol in ["ab", "ag"]:
            df_dict.update(
                {"test_" + mol + "_" + metric: [best_metrics[mol][metric].item()] for metric in best_metrics[mol]})
        if df is None:
            df = pd.DataFrame(df_dict, index=[str(i)])
        else:
            df = pd.concat([df, pd.DataFrame(df_dict, index=[str(i)])])
    file_path = os.path.join(path_new_directory, f"final_stats.csv")
    print("Saving csv file to: ", file_path)
    df.to_csv(file_path, index=True, float_format='%f')
    ag_pr = [val["ag"]["MCC"].item() for val in metrics_vals]
    print(f"ag_MCC: {statistics.mean(ag_pr):.2f} \pm {statistics.stdev(ag_pr):.2f}")
    ag_roc = [val["ag"]["auroc"].item() for val in metrics_vals]
    print(f"ag_roc: {statistics.mean(ag_roc):.2f} \pm {statistics.stdev(ag_roc):.2f}")

def json_load(args):
    """
    Load JSON file into object (args)
    :param args: args object from arg parser "
    :return: None (args object is updated)
    """
    json_file_dict = args.json_file
    json_keys = args.json_keys
    print("Processing file %s" % json_file_dict)
    if not json_keys:
        print('--json_load : no keys specified in line for file %s ' % json_file_dict)
        sys.exit(1)
    try:
        json_dict = json.load(open(json_file_dict))
    except FileNotFoundError:
        print('--json_load : file not found %s ' % json_file_dict)
        sys.exit(1)
    for json_key in json_keys:
        print("Processing key %s" % json_key)
        if json_key not in json_dict:
            print('Key %s does not exist in %s' % (json_key, json_file_dict))
            print('Valid keys are :%s ' % (','.join(json_dict)))
            sys.exit(1)
        vars(args).update(json_dict[json_key])


def input_parser():
    """
    Parse input arguments
    :return: args object
    """
    config_parser = argparse.ArgumentParser(add_help=False)
    # JSON support
    config_parser.add_argument('--json-file', help='Configuration JSON file', default=None)
    config_parser.add_argument('--json-keys', nargs='+', help='JSON keys', default=None)
    parser = argparse.ArgumentParser(parents=[config_parser])
    parser.add_argument("--path", default=f"")
    parser.add_argument("--train-path",default = 'Data/data_epipred/data_test/processed-dataset.p', type=str)
    parser.add_argument("--test-path",default = 'Data/data_epipred/data_test/processed-dataset.p', type=str)
    parser.add_argument("--val-path",default = 'Data/data_epipred/data_val/processed-dataset.p', type=str)
    parser.add_argument("--model", default=f"EpiEPMP", type=str)
    parser.add_argument('--feats', nargs='+', default=["bio"])
    parser.add_argument("--epochs", default=500, type=int)
    parser.add_argument("--batch", default=1, type=int)
    parser.add_argument("--sub_batch", default=1, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--scheduler", action='store_true')
    parser.add_argument("--decay-rate", default=0.9, type=float)
    parser.add_argument("--decay-every", default=500, type=int)
    parser.add_argument("--runs", default=5, type=int)
    parser.add_argument('-s', '--seeds', nargs='+', default=[42, 40, 63, 64, 65])
    parser.add_argument('-m', '--metrics', nargs='+', default=["auroc", "accuracy", "precision", "recall"])
    parser.add_argument("--centered", action='store_true')
    parser.add_argument("--dropout", default=0.5, type=float)
    parser.add_argument("--use-adj", action='store_true')
    parser.add_argument("--update-coors", default=True, type=bool)
    parser.add_argument("--inner-dim", default=None, type=int)
    parser.add_argument("--num-egnns", default=1, type=int)
    parser.add_argument("--cpu", default=True, type=bool)

    args, left_argv = config_parser.parse_known_args()
    if args.json_file is not None:
        json_load(args)
    parser.parse_args(left_argv, args)
    return args

if __name__ == "__main__":
    dateTimeObj = datetime.now()
    args = input_parser()
    run_test(args)
