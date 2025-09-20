import argparse
from collections import defaultdict
import json
import os
import pickle
from datetime import datetime
import numpy as np
import copy

from tqdm import tqdm, trange

import sys, os as _os
ROOT_SYS = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..'))
if ROOT_SYS not in sys.path:
    sys.path.insert(0, ROOT_SYS)

import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as tf

from laplace import Laplace
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

def load_fewshot_loaders(percent: int, *, batch_size: int, num_workers: int, pin_memory: bool,
                         fewshot_root: str = "../data/fewshot"):
    try:
        from create_cifar_subsets import get_fewshot_loaders
        return get_fewshot_loaders(
            percent=percent,
            data_root=DATA_ROOT,
            out_root=fewshot_root,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=True,
        )
    except Exception as e:
        dataset_name = "CIFAR100"
        dataset_cls = getattr(torchvision.datasets, dataset_name)
        root = f"{DATA_ROOT}/{dataset_name.lower()}"
        aug_tf = [tf.RandomCrop(32, padding=4, padding_mode="reflect"), tf.RandomHorizontalFlip()]
        norm_tf = [tf.ToTensor(), tf.Normalize(**STATS[dataset_name])]
        train_base = dataset_cls(root, train=True, transform=tf.Compose(aug_tf + norm_tf), download=True)
        test_data = dataset_cls(root, train=False, transform=tf.Compose(norm_tf), download=True)

        import pathlib, json as _json
        manifest_path = pathlib.Path(fewshot_root) / f"cifar100_{percent}" / "manifest.json"

        with open(manifest_path, "r") as f:
            mf = _json.load(f)
        indices = []
        for c in range(100):
            indices.extend(mf["class_to_indices"][str(c)])
        subset = torch.utils.data.Subset(train_base, indices)
        train_loader = data.DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
        test_loader = data.DataLoader(test_data, batch_size=1000)
        return train_loader, test_loader

def parse_option():
    hostname = socket.gethostname()

    parser = argparse.ArgumentParser('argument for training (few-shot, multiple students, one KD triple each)')

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

    parser.add_argument('--alpha', type=float, default=1.0, help='weight balance for KD (teacher vs CE)')
    parser.add_argument('--kd_T', type=float, default=1.0, help='teacher temperature')
    parser.add_argument('--kd_S', type=float, default=1.0, help='student temperature')
    parser.add_argument('--num_chunk', type=int, default=100, help='chunks for Laplace teacher batched eval')

    parser.add_argument('--teacher_path', type=str, required=True)
    parser.add_argument('--bnn_samples', type=int, default=10, help='MC samples for Bayesian/Laplace teacher')

    parser.add_argument('--students', type=str, default='', help='comma-separated student nets (e.g., "resnet50,resnet34,resnet18")')
    parser.add_argument('--grid', type=str, default='', help='semicolon-separated "T:S:A" triples; must match number of students')
    parser.add_argument('--return_val_loss', action='store_true', help='save validation loss in metrics')

    parser.add_argument('--fewshot_percent', type=int, required=True, choices=[5,10,15,25,35,50],
                        help='percentage per class for the few-shot train set')
    parser.add_argument('--fewshot_root', type=str, default='../data/fewshot',
                        help='root where make_cifar100_fewshot.py saved the packs/manifests')

    opt = parser.parse_args()

    if hostname.startswith('visiongpu'):
        opt.model_path = '/path/to/my/model'
        opt.tb_path = '/path/to/my/tensorboard'
    else:
        opt.model_path = '../save/fewshot_experiment'
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

    opt.save_folder = os.path.join(
        opt.model_path, str(opt.fewshot_percent), opt.dist_type, opt.teacher_type, opt.student_name)
    opt.tb_folder = os.path.join(
        opt.tb_path, str(opt.fewshot_percent), opt.dist_type, opt.teacher_type, opt.student_name)

    os.makedirs(opt.tb_folder, exist_ok=True)
    os.makedirs(opt.save_folder, exist_ok=True)

    return opt

def parse_grid(grid_str, fallback_T, fallback_S, fallback_alpha):
    if not grid_str:
        return [(fallback_T, fallback_S, fallback_alpha)]
    triples = []
    for trip in grid_str.split(';'):
        T, S, A = trip.split(':')
        triples.append((float(T), float(S), float(A)))
    return triples


def parse_students(students_str, default_model):
    if not students_str:
        return [default_model]
    return [s.strip() for s in students_str.split(',') if s.strip()]


def reset_cache(module):
    if hasattr(module, "reset_cache"):
        module.reset_cache()

def laplace_probs_in_chunks(la, x, *, pred_type, link_approx, n_samples, chunk=64, T=1.0):
    outs = []
    for xb in x.split(chunk, dim=0):
        with torch.no_grad():
            pb = la(xb, pred_type=pred_type, link_approx=link_approx, n_samples=n_samples)  # [B', C]
            if T != 1.0:
                eps = 1e-12
                tb = (pb.clamp(min=eps)) ** (1.0 / T)
                pb = tb / tb.sum(dim=1, keepdim=True)
            outs.append(pb)
    return torch.cat(outs, dim=0)

class Experiment:
    def __init__(self, base_opt, student_arch, kd_T, kd_S, alpha, device):
        self.opt = copy.deepcopy(base_opt)
        self.opt.model = student_arch
        self.opt.kd_T = kd_T
        self.opt.kd_S = kd_S
        self.opt.alpha = alpha

        self.opt.dist_type = f"T:{self.opt.teacher_arch}_S:{self.opt.model}"
        temp_student_name = 'S:{}_{}_{}_{}_{}_{}_{}_{}'.format(
            self.opt.epochs, self.opt.learning_rate, self.opt.lr_decay_rate,
            self.opt.lr_decay_epochs, self.opt.alpha, self.opt.kd_T, self.opt.kd_S, self.opt.bnn_samples)
        self.opt.student_name = f"T:{self.opt.trial_part}_{temp_student_name}"
        self.opt.save_folder = os.path.join(
            self.opt.model_path, str(self.opt.fewshot_percent), self.opt.dist_type, self.opt.teacher_type, self.opt.student_name)
        self.opt.tb_folder = os.path.join(
            self.opt.tb_path, str(self.opt.fewshot_percent), self.opt.dist_type, self.opt.teacher_type, self.opt.student_name)
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

    train_loader, test_loader = load_fewshot_loaders(
        percent=opt.fewshot_percent,
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        pin_memory=True,
        fewshot_root=opt.fewshot_root,
    )

    pred_type = None
    link_approx = None

    if opt.teacher_type in ('Deterministic', 'MSE', 'MCMI'):
        teacher_model = bnn.nn.nets.make_network(opt.teacher_arch, kernel_size=3, remove_maxpool=True, out_features=100)
        teacher_snapshot = os.path.join(opt.teacher_path, "snapshot_sd.pt")
        state_dict = torch.load(teacher_snapshot, map_location=device)
        teacher_model.load_state_dict(state_dict, strict=False)
        teacher_model.to(device)
        teacher_model.eval()

    elif opt.teacher_type == 'Bayesian':
        teacher_model = bnn.nn.nets.make_network(opt.teacher_arch, kernel_size=3, remove_maxpool=True, out_features=100)
        with open('../configs/ffg_u_cifar100.json') as f:
            cfg = json.load(f)
        bnn.bayesianize_(teacher_model, **cfg)
        teacher_snapshot = os.path.join(opt.teacher_path, "snapshot_sd.pt")
        state_dict = torch.load(teacher_snapshot, map_location=device)
        teacher_model.load_state_dict(state_dict, strict=False)
        teacher_model.to(device)
        teacher_model.eval()

    elif opt.teacher_type == 'Laplace':
        tokens = opt.trial_part.split("_")
        trial_idx = None
        for i, tok in enumerate(tokens):
            if tok.startswith("trial"):
                trial_idx = i
        trial_part_back = "_".join(tokens[:trial_idx + 2])
        deterministic_path = os.path.join(
            "../save/teachers", "Deterministic", opt.teacher_arch, trial_part_back, "snapshot_sd.pt")

        base_model = bnn.nn.nets.make_network(opt.teacher_arch, kernel_size=3, remove_maxpool=True, out_features=100)
        state_dict = torch.load(deterministic_path, map_location=device)
        base_model.load_state_dict(state_dict, strict=False)
        base_model.to(device)

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

        teacher_model = Laplace(base_model, "classification", subset_of_weights=subs_weights, hessian_structure=hess_struct)
        laplace_state_dict = torch.load(os.path.join(opt.teacher_path, "snapshot_sd.pt"), map_location=device)
        teacher_model.load_state_dict(laplace_state_dict)
        base_model.eval()
        teacher_model.model.eval()

    students = parse_students(opt.students, opt.model)
    triples = parse_grid(opt.grid, opt.kd_T, opt.kd_S, opt.alpha)

    experiments = [Experiment(opt, s, T, S, A, device) for s, (T, S, A) in zip(students, triples)]

    unique_kd_T = sorted({exp.opt.kd_T for exp in experiments})

    start_epoch = min((len(exp.metrics.get("acc", [])) for exp in experiments), default=0)

    epoch_iter = trange(start_epoch, opt.epochs, desc="Epochs") if opt.progress_bar else range(start_epoch, opt.epochs)
    for epoch in epoch_iter:
        for exp in experiments:
            exp.net.train()

        batch_iter = tqdm(iter(train_loader), desc="Batches") if opt.progress_bar else iter(train_loader)
        for j, (x, y) in enumerate(batch_iter):
            x = x.to(device)
            y = y.to(device)

            teacher_probs_cache = {}
            if opt.teacher_type in ('Deterministic', 'MSE', 'MCMI'):
                with torch.no_grad():
                    logits = teacher_model(x)
                for T in unique_kd_T:
                    teacher_probs_cache[T] = torch.softmax(logits / T, dim=1)

            elif opt.teacher_type == 'Bayesian':
                with torch.no_grad():
                    logits_samples = []
                    for _ in range(opt.bnn_samples):
                        logits_samples.append(teacher_model(x))
                    logits_mc = torch.stack(logits_samples, dim=0)
                for T in unique_kd_T:
                    probs_T = torch.softmax(logits_mc / T, dim=2)
                    teacher_probs_cache[T] = probs_T.mean(dim=0)
                del logits_mc

            elif opt.teacher_type == 'Laplace':
                teacher_model.enable_backprop = False
                base_probs = laplace_probs_in_chunks(
                    teacher_model, x, pred_type=pred_type, link_approx=link_approx,
                    n_samples=opt.bnn_samples, chunk=opt.num_chunk, T=1.0)
                eps = 1e-12
                base_probs = base_probs.clamp(min=eps)
                for T in unique_kd_T:
                    if T == 1.0:
                        teacher_probs_cache[T] = base_probs
                    else:
                        tb = base_probs ** (1.0 / T)
                        teacher_probs_cache[T] = tb / tb.sum(dim=1, keepdim=True)

            for exp in experiments:
                exp.optim.zero_grad(set_to_none=True)

                student_logits = exp.net(x)

                kd_loss = torch.nn.functional.kl_div(
                    torch.nn.functional.log_softmax(student_logits / exp.opt.kd_S, dim=1),
                    teacher_probs_cache[exp.opt.kd_T],
                    reduction='batchmean'
                ) * (exp.opt.kd_T * exp.opt.kd_S)

                ce_loss = torch.nn.functional.cross_entropy(student_logits, y)
                loss = (1 - exp.opt.alpha) * ce_loss + exp.opt.alpha * kd_loss

                loss.backward()
                exp.optim.step()

                exp.metrics["loss"].append(loss.item())

            del teacher_probs_cache

        for exp in experiments:
            if exp.scheduler is not None:
                exp.scheduler.step()

        for exp in experiments:
            exp.net.eval()
            with torch.no_grad():
                if getattr(opt, 'return_val_loss', False):
                    top1_acc, top5_acc, curr_val_loss = validate_deterministic(test_loader, exp.net, device, return_loss=True)
                else:
                    top1_acc, top5_acc = validate_deterministic(test_loader, exp.net, device)
            exp.metrics["acc"].append(top1_acc)
            if getattr(opt, 'return_val_loss', False):
                exp.metrics["val_loss"].append(curr_val_loss)

            best_acc1 = max(best_acc1, top1_acc)
            best_acc5 = max(best_acc5, top5_acc)

            print(f"[{exp.opt.model} | T={exp.opt.kd_T} S={exp.opt.kd_S} α={exp.opt.alpha}] "
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
