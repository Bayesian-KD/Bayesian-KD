import argparse
from collections import defaultdict
import json
import os
import pickle
from datetime import datetime
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

from laplace import Laplace
import bnn
from bnn.calibration import calibration_curve, expected_calibration_error as ece
import socket
from helper_functions import validate_deterministic

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
                                 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19',
                                 'MobileNetV2', 'ShuffleV1', 'ShuffleV2', 'vgg_c8', 'vgg_c13'])
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100'], help='dataset')

    parser.add_argument('--alpha', type=float, default=1.0, help='weight balance for KD')
    parser.add_argument('--kd_T', type=float, default=1, help='temperature for KD distillation')
    parser.add_argument('--kd_S', type=float, default=1, help='temperature for KD distillation student')
    parser.add_argument('--num_chunk', type=int, default=100, help='chunks for laplace teacher')

    parser.add_argument("--teacher_path", type=str, required=True)
    parser.add_argument('--bnn_samples', type=int, default=10,
                        help='number of test samples for variational inference')

    opt = parser.parse_args()

    if opt.model in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
        opt.learning_rate = 0.01


    if hostname.startswith('visiongpu'):
        opt.model_path = '/path/to/my/model'
        opt.tb_path = '/path/to/my/tensorboard'
    else:
        opt.model_path = '../save/students'
        opt.tb_path = '../save/tensorboard'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    parts = opt.teacher_path.split(os.sep)
    opt.teacher_type = parts[3]
    opt.teacher_arch = parts[4]
    opt.trial_part = parts[5]

    opt.dist_type = f"T:{opt.teacher_arch}_S:{opt.model}"

    temp_student_name = 'S:{}_{}_{}_{}_{}_{}_{}_{}'.format(opt.epochs, opt.learning_rate, opt.lr_decay_rate,
                                                            opt.lr_decay_epochs, opt.alpha, opt.kd_T, opt.kd_S, opt.bnn_samples)

    opt.student_name = f"T:{opt.trial_part}_{temp_student_name}"

    opt.lamb_temps = 'lambd_{}tempt_{}temps_{}'.format(opt.alpha, opt.kd_T, opt.kd_S)

    opt.save_folder = os.path.join(
        opt.model_path,
        opt.dist_type,
        opt.teacher_type,
        opt.lamb_temps,
        opt.student_name
    )

    opt.tb_folder = os.path.join(
        opt.tb_path,
        opt.dist_type,
        opt.teacher_type,
        opt.lamb_temps,
        opt.student_name
    )

    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder, exist_ok=True)

    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder, exist_ok=True)

    return opt


def reset_cache(module):
    if hasattr(module, "reset_cache"):
        module.reset_cache()

def laplace_probs_in_chunks(la, x, *, pred_type, link_approx, n_samples, chunk=64, T=1.0):
    outs = []
    for xb in x.split(chunk, dim=0):
        with torch.no_grad():
            pb = la(xb, pred_type=pred_type, link_approx=link_approx, n_samples=n_samples)
            if T != 1.0:
                eps = 1e-12
                tb = (pb.clamp(min=eps)) ** (1.0 / T)
                pb = tb / tb.sum(dim=1, keepdim=True)
            outs.append(pb)
    return torch.cat(outs, dim=0)

def main():
    best_acc1 = 0
    best_acc5 = 0

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
    print(net)
    net.to(device)

    if opt.teacher_type == 'Deterministic' or opt.teacher_type == 'MSE' or opt.teacher_type == 'MCMI':
        teacher_model = bnn.nn.nets.make_network(opt.teacher_arch, kernel_size=3, remove_maxpool=True, out_features=100)

        teacher_snapshot = os.path.join(opt.teacher_path, "snapshot_sd.pt")
        state_dict = torch.load(teacher_snapshot, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        teacher_model.load_state_dict(state_dict, strict=False)
        teacher_model.to(device)
        teacher_model.eval()

    if opt.teacher_type == 'Bayesian':

        teacher_model = bnn.nn.nets.make_network(opt.teacher_arch, kernel_size=3, remove_maxpool=True, out_features=100)

        with open('../configs/ffg_u_cifar100.json') as f:
            cfg = json.load(f)
        bnn.bayesianize_(teacher_model, **cfg)

        teacher_snapshot = os.path.join(opt.teacher_path, "snapshot_sd.pt")
        state_dict = torch.load(teacher_snapshot,
                                map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        teacher_model.load_state_dict(state_dict, strict=False)
        teacher_model.to(device)
        teacher_model.eval()

    if opt.teacher_type == 'Laplace':

        tokens = opt.trial_part.split("_")

        trial_idx = None
        for i, tok in enumerate(tokens):
            if tok.startswith("trial"):
                trial_idx = i

        trial_part_back = "_".join(tokens[:trial_idx + 2])

        deterministic_path = os.path.join(
            "../save/teachers",
            "Deterministic",
            opt.teacher_arch,
            trial_part_back,
            "snapshot_sd.pt"
        )

        temp_model = bnn.nn.nets.make_network(opt.teacher_arch, kernel_size=3, remove_maxpool=True, out_features=100)

        state_dict = torch.load(deterministic_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        temp_model.load_state_dict(state_dict, strict=False)
        temp_model.to(device)

        tokens = opt.trial_part.split("_")

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

        teacher_model = Laplace(
            temp_model,
            "classification",
            subset_of_weights=subs_weights,
            hessian_structure=hess_struct
        )

        teacher_snapshot = os.path.join(opt.teacher_path, "snapshot_sd.pt")
        laplace_state_dict = torch.load(teacher_snapshot, map_location=device)

        teacher_model.load_state_dict(laplace_state_dict)

        temp_model.eval()
        teacher_model.model.eval()



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

    epoch_iter = trange(last_epoch + 1, opt.epochs, desc="Epochs") if opt.progress_bar else range(last_epoch + 1,
                                                                                              opt.epochs)
    for i in epoch_iter:
        net.train()
        batch_iter = tqdm(iter(train_loader), desc="Batches") if opt.progress_bar else iter(train_loader)
        for j, (x, y) in enumerate(batch_iter):
            x = x.to(device)
            y = y.to(device)

            optim.zero_grad()
            student_logits = net(x)

            if opt.teacher_type == 'Deterministic' or opt.teacher_type == 'MSE' or opt.teacher_type == 'MCMI':
                with torch.no_grad():
                    teacher_probs = torch.nn.functional.softmax(teacher_model(x) / opt.kd_T, dim=1)
            if opt.teacher_type == "Bayesian":
                with torch.no_grad():
                    teacher_probs = None
                    for _ in range(opt.bnn_samples):
                        p = torch.softmax(teacher_model(x) / opt.kd_T, dim=1)
                        teacher_probs = p if teacher_probs is None else teacher_probs + p
                    teacher_probs = teacher_probs / opt.bnn_samples
            if opt.teacher_type == "Laplace":
                teacher_model.enable_backprop = False
                teacher_probs = laplace_probs_in_chunks(
                    teacher_model,
                    x,
                    pred_type=pred_type,
                    link_approx=link_approx,
                    n_samples=opt.bnn_samples,
                    chunk=opt.num_chunk,
                    T=opt.kd_T
                )

            kd_loss = torch.nn.functional.kl_div(
                torch.nn.functional.log_softmax(student_logits / opt.kd_S, dim=1),
                teacher_probs,
                reduction='batchmean'
            ) * (opt.kd_T * opt.kd_S)

            ce_loss = torch.nn.functional.cross_entropy(student_logits, y)

            loss = (1 - opt.alpha) * ce_loss + opt.alpha * kd_loss
            loss.backward()
            optim.step()

            metrics["loss"].append(loss.item())

        if scheduler is not None:
            scheduler.step()

        net.eval()
        with torch.no_grad():
            top1_acc, top5_acc = validate_deterministic(test_loader, net, device)
        metrics["acc"].append(top1_acc)
        if top1_acc > best_acc1:
            best_acc1 = top1_acc
        if top5_acc > best_acc5:
            best_acc5 = top5_acc
        print(f"Epoch {i}: Top-1 Acc = {top1_acc:.2f}%, Top-5 Acc = {top5_acc:.2f}%")

        if opt.save_folder is not None:
            torch.save(net.state_dict(), snapshot_sd_path)
            torch.save(optim.state_dict(), snapshot_optim_path)
            with open(metrics_path, "wb") as fn:
                pickle.dump(metrics, fn)

    print(f"Final test accuracy: Top-1 Acc = {top1_acc:.2f}%, Top-5 Acc = {top5_acc:.2f}%")
    print(f"Best test accuracy: Top-1 Acc = {best_acc1:.2f}%, Top-5 Acc = {best_acc5:.2f}%")


    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    save_path1 = os.path.join(opt.save_folder, "metrics.txt")

    lines = [
        f"Timestamp:        {ts}",
        f"Top-1 Final Accuracy: {top1_acc:.3f}%",
        f"Top-5 Final Accuracy: {top5_acc:.3f}%",
        f"Top-1 Best Accuracy:  {best_acc1:.3f}%",
        f"Top-5 Best Accuracy:  {best_acc5:.3f}%",
    ]

    with open(save_path1, "w") as f:
        f.write("\n".join(lines) + "\n")

if __name__ == '__main__':
    main()