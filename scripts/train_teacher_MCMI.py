import argparse
from collections import defaultdict
import json
import os
import pickle

import numpy as np
import torch.nn.functional as F
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler

import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as tf

import bnn
import socket
import pickle
from centroid import Centroid

STATS = {
    "CIFAR10": {"mean": (0.49139968, 0.48215841, 0.44653091), "std": (0.24703223, 0.24348513, 0.26158784)},
    "CIFAR100": {"mean": (0.50707516, 0.48654887, 0.44091784), "std": (0.26733429, 0.25643846, 0.27615047)}
}
ROOT = os.environ.get("DATASETS_PATH", "./data")

def parse_option():
    hostname = socket.gethostname()

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument("--teacher_path", type=str, required=True)
    parser.add_argument('--progress_bar', action='store_true', help='if to print progress')
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100'], help='dataset')
    parser.add_argument('--batch_size', type=int, default=100, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=20, help='number of training epochs')
    parser.add_argument('--CentroidSampleSize', type=int, default=8, help='num of workers to use')
    parser.add_argument('--mcmiparam', type=float, default=0.20, help='MCMI hyperparameter')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='100', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')


    opt = parser.parse_args()

    if hostname.startswith('visiongpu'):
        opt.model_path = '/path/to/my/model'
        opt.tb_path = '/path/to/my/tensorboard'
    else:
        opt.model_path = '../save/teachers'
        opt.tb_path = '../save/tensorboard'

    parts = opt.teacher_path.split(os.sep)
    parts[3] = "MCMI"
    model_name = parts[4]
    trial_part = parts[5]

    extra_params = [str(opt.mcmiparam), str(opt.epochs), str(opt.learning_rate), str(opt.batch_size), str(opt.momentum)]

    new_trial_part = trial_part + "_" + "_".join(extra_params)

    new_path = os.path.join(*parts[:5], new_trial_part)

    opt.save_folder = new_path

    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder, exist_ok=True)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))


    return opt


def main():
    best_acc = 0

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

    samples_data, samples_tar = [], []
    train_label_list = torch.tensor(train_loader.dataset.targets)
    for Class in tqdm(range(100)):
        idx = (train_label_list == Class).nonzero().squeeze().numpy()
        sampler = SubsetRandomSampler(idx)
        class_loader = data.DataLoader(train_loader.dataset, batch_size = 500, sampler = sampler, pin_memory = True)
        img, tar = next(iter(class_loader))
        samples_data.append(img.numpy())
        samples_tar.append(tar)

    samples_data = np.array(samples_data)
    samples_tar = torch.cat(samples_tar, 0)

    centroids = Centroid(100, samples_data, samples_tar, 1, CdecayFactor=0.9999)
    centroids.update_epoch(teacher_model, train_loader)

    optim = torch.optim.Adam(teacher_model.parameters(), opt.learning_rate)

    metrics = defaultdict(list)

    snapshot_sd_path = os.path.join(opt.save_folder, "snapshot_sd.pt")
    snapshot_optim_path = os.path.join(opt.save_folder, "snapshot_optim.sd")
    metrics_path = os.path.join(opt.save_folder, "metrics.pkl")

    last_epoch = len(metrics["acc"]) - 1

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim,opt.epochs)

    epoch_iter = trange(last_epoch + 1, opt.epochs, desc="Epochs") if opt.progress_bar else range(last_epoch + 1,
                                                                                              opt.epochs)
    for i in epoch_iter:
        teacher_model.train()
        batch_iter = tqdm(iter(train_loader), desc="Batches") if opt.progress_bar else iter(train_loader)
        for j, (x, y) in enumerate(batch_iter):
            x = x.to(device)
            y = y.to(device)

            optim.zero_grad()
            yhat = teacher_model(x)
            Cs = centroids.get_centroids(y).to(device)

            CE = torch.nn.functional.cross_entropy(yhat, y)
            KL = F.kl_div(Cs.log(), F.log_softmax(yhat,1), reduction="batchmean", log_target=True)
            loss = 1*CE+opt.mcmiparam*KL

            loss.backward()
            optim.step()

            metrics["loss"].append(loss.item())

        if scheduler is not None:
            scheduler.step()

        teacher_model.eval()
        with torch.no_grad():
            correct, total = 0, 0
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                yhat = teacher_model(x).argmax(dim=1)
                correct += (yhat == y).sum().item()
                total += y.size(0)

            acc = correct / total
            metrics["acc"].append(acc)

            print(f"Epoch {i} -- Accuracy: {100 * acc:.2f}%")

        if opt.save_folder is not None:
            torch.save(teacher_model.state_dict(), snapshot_sd_path)
            torch.save(optim.state_dict(), snapshot_optim_path)
            with open(metrics_path, "wb") as fn:
                pickle.dump(metrics, fn)


    print(f"Final test accuracy: {100 * metrics['acc'][-1]:.2f}")

if __name__ == '__main__':
    main()