import argparse
import json
import os
import torch
import torchvision
import torchvision.transforms as tf
import torch.utils.data as data
import torch.nn.functional as F
import math

import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from datetime import datetime
import pickle

import numpy as np

from tqdm import tqdm, trange
import matplotlib.pyplot as plt

from bnn.nn.nets import make_network
import bnn
from laplace import Laplace
import socket
from helper_functions import validate_laplace, validate_bayesian, validate_deterministic

STATS = {
    "CIFAR10": {"mean": (0.49139968, 0.48215841, 0.44653091), "std": (0.24703223, 0.24348513, 0.26158784)},
    "CIFAR100": {"mean": (0.50707516, 0.48654887, 0.44091784), "std": (0.26733429, 0.25643846, 0.27615047)}
}
ROOT = os.environ.get("DATASETS_PATH", "./data")
NUM_BINS = 10

def save_metrics(opt, top1, top5, repetitions, bnn_samples, filename="metrics.txt"):
    os.makedirs(opt.teacher_path, exist_ok=True)

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    save_path = os.path.join(opt.teacher_path, filename)

    lines = [
        f"Timestamp:        {ts}",
        f"Top-1 Accuracy:   {top1:.3f}%",
        f"Top-5 Accuracy:   {top5:.3f}%",
        f"Repetitions:      {repetitions}",
        f"BNN samples/run:  {bnn_samples}",
    ]

    with open(save_path, "w") as f:
        f.write("\n".join(lines) + "\n")

def parse_option():
    hostname = socket.gethostname()

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument("--teacher_path", type=str, required=True)
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100'], help='dataset')
    parser.add_argument('--bnn_test_samples', type=int, default=5,
                        help='number of test samples for variational inference')
    parser.add_argument('--bnn_total_samples', type=int, default=20,
                        help='number of total samples to average over')
    parser.add_argument('--batch_size', type=int, default=100, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')


    opt = parser.parse_args()

    return opt


def main():
    opt = parse_option()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_name = f"CIFAR100"
    dataset_cls = getattr(torchvision.datasets, dataset_name)
    root = f"{ROOT}/{dataset_name.lower()}"
    print(f"Loading dataset {dataset_cls} from {root}")

    norm_tf = [tf.ToTensor(), tf.Normalize(**STATS[dataset_name])]
    test_data = dataset_cls(root, train=False, transform=tf.Compose(norm_tf), download=True)

    test_loader = data.DataLoader(test_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers, pin_memory=True)

    parts = opt.teacher_path.split(os.sep)
    model_type = parts[3]
    model_name = parts[4]
    trial_part = parts[5]

    if model_type == 'Deterministic' or model_type == 'MSE' or model_type == 'MCMI':
        model = make_network(model_name, kernel_size=3, remove_maxpool=True, out_features=100)

        teacher_snapshot = os.path.join(opt.teacher_path, "snapshot_sd.pt")
        state_dict = torch.load(teacher_snapshot, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()

        top_1_acc, top_5_acc = validate_deterministic(test_loader,model,device)

    if model_type == 'Bayesian':

        model = make_network(model_name, kernel_size=3, remove_maxpool=True, out_features=100)

        with open('../configs/ffg_u_cifar100.json') as f:
            cfg = json.load(f)
        bnn.bayesianize_(model, **cfg)

        teacher_snapshot = os.path.join(opt.teacher_path, "snapshot_sd.pt")
        state_dict = torch.load(teacher_snapshot,
                                map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()

        top_1_acc, top_5_acc = validate_bayesian(test_loader, model, device, opt.bnn_test_samples, opt.bnn_total_samples)

    if model_type == 'Laplace':

        tokens = trial_part.split("_")

        trial_idx = None
        for i, tok in enumerate(tokens):
            if tok.startswith("trial"):
                trial_idx = i

        trial_part_back = "_".join(tokens[:trial_idx + 2])

        deterministic_path = os.path.join(
            parts[0], parts[1], parts[2],
            "Deterministic",
            parts[4],
            trial_part_back,
            "snapshot_sd.pt"
        )

        model = make_network(model_name, kernel_size=3, remove_maxpool=True, out_features=100)

        state_dict = torch.load(deterministic_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        model.load_state_dict(state_dict, strict=False)
        model.to(device)

        tokens = trial_part.split("_")

        trial_idx = None
        for i, tok in enumerate(tokens):
            if tok.startswith("trial"):
                trial_idx = i
                break

        after_trial = tokens[trial_idx + 2:]

        subs_weights = "_".join(after_trial[0:2])
        hess_struct = after_trial[2]
        la_method = after_trial[3]
        pred_type = after_trial[4]
        link_approx = after_trial[5]

        la = Laplace(
            model,
            "classification",
            subset_of_weights=subs_weights,
            hessian_structure=hess_struct
        )

        teacher_snapshot = os.path.join(opt.teacher_path, "snapshot_sd.pt")
        laplace_state_dict = torch.load(teacher_snapshot, map_location=device)

        la.load_state_dict(laplace_state_dict)

        model.eval()

        top_1_acc, top_5_acc = validate_laplace(test_loader, la, device, opt.bnn_test_samples, opt.bnn_total_samples,pred_type,link_approx)

    repetitions = math.ceil(opt.bnn_total_samples / opt.bnn_test_samples)
    save_metrics(opt, top_1_acc, top_5_acc, repetitions, opt.bnn_test_samples)

if __name__ == '__main__':
    main()