import argparse
from collections import defaultdict
import json
import os
import pickle
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm, trange
import matplotlib.pyplot as plt

import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
import torch.utils.data as data
import torch.distributions as dist
import torchvision
import torchvision.transforms as tf

import bnn
from bnn.calibration import calibration_curve, expected_calibration_error as ece
import socket

STATS = {
    "CIFAR10": {"mean": (0.49139968, 0.48215841, 0.44653091), "std": (0.24703223, 0.24348513, 0.26158784)},
    "CIFAR100": {"mean": (0.50707516, 0.48654887, 0.44091784), "std": (0.26733429, 0.25643846, 0.27615047)}
}
ROOT = os.environ.get("DATASETS_PATH", "./data")
NUM_BINS = 10

def parse_option():
    hostname = socket.gethostname()

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--progress_bar', action='store_true', help='if to print progress')
    parser.add_argument('--save_freq', type=int, default=40, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=100, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=200, help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='100', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # dataset
    parser.add_argument('--model', type=str, default='resnet18',
                        choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
                                 'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2',
                                 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'vgg8_bn', 'vgg11_bn', 'vgg13_bn', 'vgg16_bn', 'vgg19_bn',
                                 'MobileNetV2', 'ShuffleV1', 'ShuffleV2', 'vgg_c8', 'vgg_c13'])
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100'], help='dataset')

    parser.add_argument('-t', '--trial', type=int, default=0, help='the experiment id')

    parser.add_argument('--bayesianize', action='store_true', help='train Bayesian teacher')
    parser.add_argument('--mse', action='store_true', help='train MSE teacher')
    parser.add_argument('--bnn_ml_epochs', type=int, default=100,
                        help='number of epochs for training with nll loss only')
    parser.add_argument('--bnn_annealing_epochs', type=int, default=50, help='number of epochs for gradual annealing')
    parser.add_argument('--bnn_test_samples', type=int, default=1,
                        help='number of test samples for variational inference')

    opt = parser.parse_args()

    if opt.model in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
        opt.learning_rate = 0.01

    if hostname.startswith('visiongpu'):
        opt.model_path = '/path/to/my/model'
        opt.tb_path = '/path/to/my/tensorboard'
    else:
        opt.model_path = '../save/teachers'
        opt.tb_path = '../save/tensorboard'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    if opt.bayesianize:
        opt.file_name = os.path.join('Bayesian', opt.model)
    elif opt.mse:
        opt.file_name = os.path.join('MSE', opt.model)
    else:
        opt.file_name = os.path.join('Deterministic', opt.model)

    opt.model_name = '{}_{}_{}_{}_{}_{}_trial_{}'.format(opt.epochs, opt.learning_rate, opt.lr_decay_rate,
                                                            opt.lr_decay_epochs, opt.bnn_annealing_epochs, opt.bnn_ml_epochs, opt.trial)

    opt.tb_folder = os.path.join(opt.tb_path, opt.file_name, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder, exist_ok=True)

    opt.save_folder = os.path.join(opt.model_path, opt.file_name, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder, exist_ok=True)

    return opt

def reset_cache(module):
    if hasattr(module, "reset_cache"):
        module.reset_cache()

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

    train_loader = data.DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers, pin_memory=True)
    test_loader = data.DataLoader(test_data, batch_size=1000)

    net = bnn.nn.nets.make_network(f"{opt.model}", kernel_size=3, remove_maxpool=True, out_features=100)

    if opt.bayesianize:
        with open('../configs/ffg_u_cifar100.json') as f:
            cfg = json.load(f)
        bnn.bayesianize_(net, **cfg)

    print(net)
    net.to(device)

    optim = torch.optim.Adam(net.parameters(), opt.learning_rate)
    metrics = defaultdict(list)

    snapshot_sd_path = os.path.join(opt.save_folder, "snapshot_sd.pt")
    snapshot_optim_path = os.path.join(opt.save_folder, "snapshot_optim.sd")
    metrics_path = os.path.join(opt.save_folder, "metrics.pkl")
    if os.path.isfile(snapshot_sd_path):
        net.load_state_dict(torch.load(snapshot_sd_path, map_location=device))
        optim.load_state_dict(torch.load(snapshot_optim_path, map_location=device))
        with open(metrics_path, "rb") as f:
            metrics = pickle.load(f)
    else:
        torch.save(net.state_dict(), os.path.join(opt.save_folder, "initial_sd.pt"))

    last_epoch = len(metrics["acc"]) - 1

    if opt.lr_decay_epochs is not None:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, opt.lr_decay_epochs, gamma=opt.lr_decay_rate, last_epoch=last_epoch)
    else:
        scheduler = None

    kl_factor = 0. if opt.bnn_ml_epochs > 0 or opt.bnn_annealing_epochs > 0 else 1.
    annealing_rate = opt.bnn_annealing_epochs ** -1 if opt.bnn_annealing_epochs > 0 else 1.

    epoch_iter = trange(last_epoch + 1, opt.epochs, desc="Epochs") if opt.progress_bar else range(last_epoch + 1,
                                                                                              opt.epochs)
    for i in epoch_iter:
        net.train()
        if opt.bayesianize:
            net.apply(reset_cache)
        batch_iter = tqdm(iter(train_loader), desc="Batches") if opt.progress_bar else iter(train_loader)
        for j, (x, y) in enumerate(batch_iter):
            x = x.to(device)
            y = y.to(device)

            optim.zero_grad()
            if opt.bayesianize:
                avg_nll = 0.
                yhat = net(x)
                nll = -dist.Categorical(logits=yhat).log_prob(y).mean()
                kl = torch.tensor(0., device=device)
                for module in net.modules():
                    if hasattr(module, "parameter_loss"):
                        kl = kl + module.parameter_loss().sum()
                metrics["kl"].append(kl.item())
                loss = nll + kl * kl_factor / len(train_data)

                avg_nll += nll.item()
                loss.backward(retain_graph=False)

                optim.step()

                net.apply(reset_cache)

                metrics["nll"].append(avg_nll)
            elif opt.mse:
                yhat = net(x)
                loss = torch.nn.functional.mse_loss(yhat, F.one_hot(y, num_classes=100).float())
                loss.backward()
                optim.step()

                metrics["loss"].append(loss.item())
            else:
                yhat = net(x)
                loss = torch.nn.functional.cross_entropy(yhat, y)
                loss.backward()
                optim.step()

                metrics["loss"].append(loss.item())

        if scheduler is not None:
            scheduler.step()

        net.eval()
        with torch.no_grad():
            if opt.bayesianize:
                probs, targets = map(torch.cat, zip(*(
                    (sum(net(x.to(device)).softmax(-1) for _ in range(opt.bnn_test_samples)).div(opt.bnn_test_samples).to("cpu"), y)
                    for x, y in iter(test_loader)
                )))

                acc = probs.argmax(-1).eq(targets).float().mean().item()
                metrics["acc"].append(acc)

                del probs
                del targets

                torch.cuda.empty_cache()

            else:
                correct, total = 0, 0
                for x, y in test_loader:
                    x, y = x.to(device), y.to(device)
                    yhat = net(x).argmax(dim=1)
                    correct += (yhat == y).sum().item()
                    total += y.size(0)

                acc = correct / total
                metrics["acc"].append(acc)

            print(f"Epoch {i} -- Accuracy: {100 * acc:.2f}%")


        if opt.save_folder is not None:
            torch.save(net.state_dict(), snapshot_sd_path)
            torch.save(optim.state_dict(), snapshot_optim_path)
            with open(metrics_path, "wb") as fn:
                pickle.dump(metrics, fn)

        if i >= opt.bnn_ml_epochs:
            kl_factor = min(1., kl_factor + annealing_rate)

    print(f"Final test accuracy: {100 * metrics['acc'][-1]:.2f}")

if __name__ == '__main__':
    main()