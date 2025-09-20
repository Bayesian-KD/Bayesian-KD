import sys
import time
import torch
import math

from util import AverageMeter, accuracy, accuracy_from_probs
import torch.nn.functional as F


def validate_deterministic(val_loader, model, device, return_loss=False):
    batch_time = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    if return_loss==True:
        losses = AverageMeter()

    model.eval()

    with torch.no_grad():
        end = time.time()
        for idx, (input, target) in enumerate(val_loader):

            input = input.float()
            if torch.cuda.is_available():
                input = input.to(device)
                target = target.to(device)

            output = model(input)

            if return_loss==True:
                loss = F.cross_entropy(output, target, reduction='mean')
                losses.update(loss.item(), input.size(0))

            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

        if return_loss==True:
            print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} Loss {loss.avg:.4f}'
              .format(top1=top1, top5=top5, loss=losses))
        else:
            print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    if return_loss==True:
        return top1.avg, top5.avg, losses.avg
    return top1.avg, top5.avg

def validate_bayesian(val_loader, model, device, bnn_samples=5, total_samples=20):

    repetitions = math.ceil(total_samples / bnn_samples)

    acc1_list, acc5_list = [], []

    model.eval()
    with torch.no_grad():
        for rep in range(repetitions):
            top1 = AverageMeter()
            top5 = AverageMeter()

            start = time.time()
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                logits_list = []
                for _ in range(bnn_samples):
                    logits_list.append(model(x))
                logits = torch.stack(logits_list, dim=0)

                probs = torch.softmax(logits, dim=-1).mean(dim=0)

                acc1, acc5 = accuracy_from_probs(probs, y, topk=(1, 5))
                top1.update(acc1[0], y.size(0))
                top5.update(acc5[0], y.size(0))

                start = time.time()

            acc1_list.append(top1.avg)
            acc5_list.append(top5.avg)

    mean_acc1 = sum(acc1_list) / len(acc1_list)
    mean_acc5 = sum(acc5_list) / len(acc5_list)

    print(f" * Bayesian Acc@1 {mean_acc1:.3f}  Acc@5 {mean_acc5:.3f} "
          f"(reps={repetitions}, bnn_samples={bnn_samples})")

    return mean_acc1, mean_acc5


def validate_laplace(
    val_loader,
    la,
    device,
    bnn_samples=5,
    total_samples=20,
    pred_type=None,
    link_approx=None,
):

    repetitions = math.ceil(total_samples / max(1, bnn_samples))
    acc1_runs, acc5_runs = [], []

    la.model.eval() if hasattr(la, "model") else None

    with torch.no_grad():
        for _ in range(repetitions):
            top1 = AverageMeter()
            top5 = AverageMeter()

            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                pred_mean = la(x, pred_type=pred_type, link_approx=link_approx, n_samples=bnn_samples)

                if pred_mean.dim() == 3:
                    pred_mean = pred_mean.mean(dim=0)

                acc1, acc5 = accuracy_from_probs(pred_mean, y, topk=(1, 5))
                top1.update(acc1[0], y.size(0))
                top5.update(acc5[0], y.size(0))

            acc1_runs.append(top1.avg)
            acc5_runs.append(top5.avg)

    mean_acc1 = float(sum(acc1_runs) / len(acc1_runs))
    mean_acc5 = float(sum(acc5_runs) / len(acc5_runs))

    print(f" * Laplace Acc@1 {mean_acc1:.3f}  Acc@5 {mean_acc5:.3f} "
          f"(reps={repetitions}, bnn_samples={bnn_samples})")

    return mean_acc1, mean_acc5