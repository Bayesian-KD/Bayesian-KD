import argparse
from collections import defaultdict
import json
import os
import pickle
from datetime import datetime
import numpy as np
import copy

from tqdm import tqdm, trange
import matplotlib.pyplot as plt

import sys, os as _os
ROOT_SYS = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..'))
if ROOT_SYS not in sys.path:
    sys.path.insert(0, ROOT_SYS)

import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as tf

import bnn
from bnn.calibration import calibration_curve, expected_calibration_error as ece
import socket
from helper_functions import validate_deterministic

STATS = {
    "CIFAR10": {"mean": (0.49139968, 0.48215841, 0.44653091), "std": (0.24703223, 0.24348513, 0.26158784)},
    "CIFAR100": {"mean": (0.50707516, 0.48654887, 0.44091784), "std": (0.26733429, 0.25643846, 0.27615047)}
}
DATA_ROOT = os.environ.get("DATASETS_PATH", "./data")
NUM_BINS = 10


def parse_option():
    hostname = socket.gethostname()

    parser = argparse.ArgumentParser('argument for training (samples experiment)')

    parser.add_argument('--progress_bar', action='store_true', help='if to print progress')
    parser.add_argument('--save_freq', type=int, default=40, help='save frequency (epochs)')
    parser.add_argument('--batch_size', type=int, default=100, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=200, help='number of training epochs')

    parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='100', help='where to decay lr, comma list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')

    parser.add_argument('--model', type=str, default='resnet18',
                        choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
                                 'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2',
                                 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19',
                                 'MobileNetV2', 'ShuffleV1', 'ShuffleV2', 'vgg_c8', 'vgg_c13'])
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100'], help='dataset')

    parser.add_argument('--alpha', type=float, default=1.0, help='weight balance for KD (teacher vs CE) -> fixed 1.0')
    parser.add_argument('--kd_T', type=float, default=1.0, help='teacher temperature -> fixed 1.0')
    parser.add_argument('--kd_S', type=float, default=1.0, help='student temperature -> fixed 1.0')
    parser.add_argument('--num_chunk', type=int, default=100, help='(kept) chunks for potential batched eval (unused here)')

    parser.add_argument('--teacher_path', type=str, required=True)
    parser.add_argument('--bnn_samples', type=int, default=12, help='Max MC samples from Bayesian teacher; trains S=1..this')

    parser.add_argument('--students', type=str, default='', help='comma-separated student nets (e.g., "resnet50,resnet34,resnet18")')
    parser.add_argument('--return_val_loss', action='store_true', help='save validation loss in metrics')

    opt = parser.parse_args()

    if hostname.startswith('visiongpu'):
        opt.model_path = '../save/samples_experiment'
        opt.tb_path = '../save/tensorboard'
    else:
        opt.model_path = '../save/samples_experiment'
        opt.tb_path = '../save/tensorboard'

    iterations = opt.lr_decay_epochs.split(',') if opt.lr_decay_epochs else []
    opt.lr_decay_epochs = [int(it) for it in iterations if it]

    parts = os.path.normpath(opt.teacher_path).split(os.sep)

    opt.teacher_type, opt.teacher_arch, opt.trial_part = parts[-3], parts[-2], parts[-1]

    opt.dist_type = f"T:{opt.teacher_arch}_S:{opt.model}"
    temp_student_name = 'S:{}_{}_{}_{}_{}_{}_{}_{}'.format(
        opt.epochs, opt.learning_rate, opt.lr_decay_rate, opt.lr_decay_epochs,
        opt.alpha, opt.kd_T, opt.kd_S, opt.bnn_samples)
    opt.student_name = f"T:{opt.trial_part}_{temp_student_name}"
    opt.lamb_temps = 'lambd_{}tempt_{}temps_{}'.format(opt.alpha, opt.kd_T, opt.kd_S)
    opt.save_folder = os.path.join(opt.model_path, opt.dist_type, opt.teacher_type, opt.lamb_temps, opt.student_name)
    opt.tb_folder = os.path.join(opt.tb_path, opt.dist_type, opt.teacher_type, opt.lamb_temps, opt.student_name)

    os.makedirs(opt.tb_folder, exist_ok=True)
    os.makedirs(opt.save_folder, exist_ok=True)

    return opt

def parse_students(students_str, default_model):
    if not students_str:
        return [default_model]
    return [s.strip() for s in students_str.split(',') if s.strip()]


def reset_cache(module):
    if hasattr(module, "reset_cache"):
        module.reset_cache()


class Experiment:
    def __init__(self, base_opt, student_arch, samples_used, device):
        self.opt = copy.deepcopy(base_opt)
        self.opt.model = student_arch
        self.opt.bnn_samples = int(samples_used)

        self.opt.dist_type = f"T:{self.opt.teacher_arch}_S:{self.opt.model}"
        temp_student_name = 'S:{}_{}_{}_{}_{}_{}_{}_{}'.format(
            self.opt.epochs, self.opt.learning_rate, self.opt.lr_decay_rate,
            self.opt.lr_decay_epochs, self.opt.alpha, self.opt.kd_T, self.opt.kd_S, self.opt.bnn_samples)
        self.opt.student_name = f"T:{self.opt.trial_part}_{temp_student_name}"
        self.opt.lamb_temps = 'lambd_{}tempt_{}temps_{}'.format(self.opt.alpha, self.opt.kd_T, self.opt.kd_S)
        self.opt.save_folder = os.path.join(
            self.opt.model_path, self.opt.dist_type, self.opt.teacher_type, self.opt.lamb_temps, self.opt.student_name)
        self.opt.tb_folder = os.path.join(
            self.opt.tb_path, self.opt.dist_type, self.opt.teacher_type, self.opt.lamb_temps, self.opt.student_name)
        os.makedirs(self.opt.save_folder, exist_ok=True)
        os.makedirs(self.opt.tb_folder, exist_ok=True)

        self.net = bnn.nn.nets.make_network(f"{student_arch}", kernel_size=3, remove_maxpool=True, out_features=100).to(device)
        self.optim = torch.optim.Adam(self.net.parameters(), self.opt.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optim, self.opt.lr_decay_epochs, gamma=self.opt.lr_decay_rate) if self.opt.lr_decay_epochs else None

        self.metrics = defaultdict(list)
        self.snapshot_sd_path    = os.path.join(self.opt.save_folder, "snapshot_sd.pt")
        self.snapshot_optim_path = os.path.join(self.opt.save_folder, "snapshot_optim.sd")
        self.metrics_path        = os.path.join(self.opt.save_folder, "metrics.pkl")

        if os.path.isfile(self.snapshot_sd_path):
            self.net.load_state_dict(torch.load(self.snapshot_sd_path, map_location='cpu'))
            if os.path.isfile(self.snapshot_optim_path):
                self.optim.load_state_dict(torch.load(self.snapshot_optim_path, map_location='cpu'))
            if os.path.isfile(self.metrics_path):
                with open(self.metrics_path, "rb") as f:
                    self.metrics = pickle.load(f)
            prev_epochs = len(self.metrics.get("acc", []))
            if self.scheduler is not None:
                for _ in range(prev_epochs):
                    self.scheduler.step()
        else:
            torch.save(self.net.state_dict(), os.path.join(self.opt.save_folder, "initial_sd.pt"))

def main():
    best_acc1 = 0
    best_acc5 = 0

    opt = parse_option()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_name = f"CIFAR100"
    dataset_cls = getattr(torchvision.datasets, dataset_name)
    root = f"{DATA_ROOT}/{dataset_name.lower()}"
    print(f"Loading dataset {dataset_cls} from {root}")
    aug_tf = [tf.RandomCrop(32, padding=4, padding_mode="reflect"), tf.RandomHorizontalFlip()]
    norm_tf = [tf.ToTensor(), tf.Normalize(**STATS[dataset_name])]
    train_data = dataset_cls(root, train=True, transform=tf.Compose(aug_tf + norm_tf), download=True)
    test_data = dataset_cls(root, train=False, transform=tf.Compose(norm_tf), download=True)

    train_loader = data.DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers, pin_memory=True)
    test_loader = data.DataLoader(test_data, batch_size=1000)

    if opt.teacher_type != 'Bayesian':
        raise ValueError(f"This script expects a Bayesian teacher. Got: {opt.teacher_type}")

    teacher_model = bnn.nn.nets.make_network(opt.teacher_arch, kernel_size=3, remove_maxpool=True, out_features=100)
    with open('../configs/ffg_u_cifar100.json') as f:
        cfg = json.load(f)
    bnn.bayesianize_(teacher_model, **cfg)
    teacher_snapshot = os.path.join(opt.teacher_path, "snapshot_sd.pt")
    state_dict = torch.load(teacher_snapshot, map_location=device)
    teacher_model.load_state_dict(state_dict, strict=False)
    teacher_model.to(device)
    teacher_model.eval()

    students = parse_students(opt.students, opt.model)
    S_max = int(opt.bnn_samples)
    experiments = [Experiment(opt, s_arch, S_used, device) for s_arch in students for S_used in range(1, S_max + 1)]
    if not experiments:
        raise RuntimeError("No experiments constructed.")

    start_epoch = min((len(exp.metrics.get("acc", [])) for exp in experiments), default=0)

    epoch_iter = trange(start_epoch, opt.epochs, desc="Epochs") if opt.progress_bar else range(start_epoch, opt.epochs)
    for epoch in epoch_iter:
        for exp in experiments:
            exp.net.train()

        batch_iter = tqdm(iter(train_loader), desc="Batches") if opt.progress_bar else iter(train_loader)
        for j, (x, y) in enumerate(batch_iter):
            x = x.to(device)
            y = y.to(device)

            with torch.no_grad():
                logits_samples = []
                for _ in range(S_max):
                    logits_samples.append(teacher_model(x))
                logits_mc = torch.stack(logits_samples, dim=0)
                probs_samples = torch.softmax(logits_mc, dim=2)

                cumsum = probs_samples.cumsum(dim=0)
                denom = torch.arange(1, S_max + 1, device=cumsum.device, dtype=cumsum.dtype).view(-1, 1, 1)
                cum_means = cumsum / denom

            for exp in experiments:
                exp.optim.zero_grad(set_to_none=True)

                student_logits = exp.net(x)

                S_used = exp.opt.bnn_samples
                teacher_probs = cum_means[S_used - 1]

                kd_loss = torch.nn.functional.kl_div(
                    torch.nn.functional.log_softmax(student_logits / opt.kd_S, dim=1),
                    teacher_probs,
                    reduction='batchmean'
                ) * (opt.kd_T * opt.kd_S)

                loss = opt.alpha * kd_loss

                loss.backward()
                exp.optim.step()

                exp.metrics["loss"].append(loss.item())

            del logits_mc, probs_samples, cumsum, denom, cum_means

        for exp in experiments:
            if exp.scheduler is not None:
                exp.scheduler.step()

        for exp in experiments:
            exp.net.eval()
            with torch.no_grad():
                if opt.return_val_loss == True:
                    top1_acc, top5_acc, curr_val_loss = validate_deterministic(test_loader, exp.net, device, return_loss=True)
                else:
                    top1_acc, top5_acc = validate_deterministic(test_loader, exp.net, device)
                    curr_val_loss = None

            exp.metrics["acc"].append(top1_acc)
            if opt.return_val_loss == True and curr_val_loss is not None:
                exp.metrics["val_loss"].append(curr_val_loss)

            best_acc1 = max(best_acc1, top1_acc)
            best_acc5 = max(best_acc5, top5_acc)

            print(f"[{exp.opt.model} | S_used={exp.opt.bnn_samples}] "
                  f"Epoch {epoch}: Top-1={top1_acc:.2f}%, Top-5={top5_acc:.2f}%")

            torch.save(exp.net.state_dict(), exp.snapshot_sd_path)
            torch.save(exp.optim.state_dict(), exp.snapshot_optim_path)
            with open(exp.metrics_path, "wb") as fn:
                pickle.dump(exp.metrics, fn)

    for exp in experiments:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        save_path1 = os.path.join(exp.opt.save_folder, "metrics.txt")
        last_acc = exp.metrics["acc"][-1] if exp.metrics["acc"] else float('nan')
        lines = [
            f"Timestamp:        {ts}",
            f"Top-1 Final Accuracy: {last_acc:.3f}%",
        ]
        with open(save_path1, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"Saved metrics to {save_path1}")


if __name__ == '__main__':
    main()
