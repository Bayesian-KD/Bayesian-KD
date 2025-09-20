import argparse
from collections import defaultdict
import json
import os
import pickle

import numpy as np

from tqdm import tqdm, trange
import matplotlib.pyplot as plt

import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as tf

import bnn
from laplace import Laplace
import socket
import pickle


STATS = {
    "CIFAR10": {"mean": (0.49139968, 0.48215841, 0.44653091), "std": (0.24703223, 0.24348513, 0.26158784)},
    "CIFAR100": {"mean": (0.50707516, 0.48654887, 0.44091784), "std": (0.26733429, 0.25643846, 0.27615047)}
}
ROOT = os.environ.get("DATASETS_PATH", "./data")


def parse_option():
    hostname = socket.gethostname()

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument("--teacher_path", type=str, required=True)
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100'], help='dataset')
    parser.add_argument('--batch_size', type=int, default=20, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--subs_weights', type=str, default='last_layer', help='subset_of_weights')
    parser.add_argument('--hess_struct', type=str, default='diag', help='hessian_structure')
    parser.add_argument('--la_method', type=str, default='gridsearch', help='method')
    parser.add_argument('--pred_type', type=str, default='glm', help='pred_type')
    parser.add_argument('--link_approx', type=str, default='probit', help='link_approx')

    opt = parser.parse_args()

    if hostname.startswith('visiongpu'):
        opt.model_path = '/path/to/my/model'
        opt.tb_path = '/path/to/my/tensorboard'
    else:
        opt.model_path = '../save/teachers'
        opt.tb_path = '../save/tensorboard'

    parts = opt.teacher_path.split(os.sep)
    parts[3] = "Laplace"
    model_name = parts[4]
    trial_part = parts[5]

    extra_params = [opt.subs_weights, opt.hess_struct, opt.la_method, opt.pred_type, opt.link_approx]

    new_trial_part = trial_part + "_" + "_".join(extra_params)
    new_path = os.path.join(*parts[:5], new_trial_part)

    opt.save_folder = new_path

    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder, exist_ok=True)

    return opt


def main():
    opt = parse_option()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_name = f"CIFAR100"
    dataset_cls = getattr(torchvision.datasets, dataset_name)
    root = f"{ROOT}/{dataset_name.lower()}"
    print(f"Loading dataset {dataset_cls} from {root}")

    aug_tf = [tf.RandomCrop(32, padding=4, padding_mode="reflect"), tf.RandomHorizontalFlip()]
    norm_tf = [tf.ToTensor(), tf.Normalize(**STATS[dataset_name])]

    train_data = dataset_cls(root, train=True, transform=tf.Compose(aug_tf + norm_tf), download=True)
    test_data = dataset_cls(root, train=False, transform=tf.Compose(norm_tf), download=True)

    train_loader = data.DataLoader(train_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers, pin_memory=True)
    test_loader = data.DataLoader(test_data, batch_size=50)

    parts = opt.teacher_path.split(os.sep)
    teacher_model = bnn.nn.nets.make_network(parts[4], kernel_size=3, remove_maxpool=True, out_features=100)

    teacher_snapshot = os.path.join(opt.teacher_path, "snapshot_sd.pt")
    state_dict = torch.load(teacher_snapshot, map_location=device)
    teacher_model.load_state_dict(state_dict, strict=False)
    teacher_model.to(device)

    la = Laplace(teacher_model, "classification",
                 subset_of_weights=opt.subs_weights,
                 hessian_structure=opt.hess_struct)

    la.fit(train_loader)

    la.optimize_prior_precision(
        method=opt.la_method,
        pred_type=opt.pred_type,
        link_approx=opt.link_approx,
        val_loader=train_loader
    )

    snapshot_sd_path = os.path.join(opt.save_folder, "snapshot_sd.pt")
    torch.save(la.state_dict(), snapshot_sd_path)
    print(f"Laplace approximation state dict saved to {snapshot_sd_path}")

if __name__ == '__main__':
    main()