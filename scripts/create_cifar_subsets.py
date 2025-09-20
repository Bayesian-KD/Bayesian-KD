import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets

CIFAR100_STATS = {
    "mean": (0.50707516, 0.48654887, 0.44091784),
    "std":  (0.26733429, 0.25643846, 0.27615047),
}

def _random_crop_reflect(x: torch.Tensor, padding: int = 4) -> torch.Tensor:
    if padding > 0:
        x = torch.nn.functional.pad(x.unsqueeze(0).float(), (padding, padding, padding, padding), mode="reflect").squeeze(0).to(x.dtype)
    _, H, W = x.shape
    top = torch.randint(0, H - 32 + 1, (1,)).item()
    left = torch.randint(0, W - 32 + 1, (1,)).item()
    return x[:, top:top+32, left:left+32]

def _random_hflip(x: torch.Tensor, p: float = 0.5) -> torch.Tensor:
    if torch.rand(()) < p:
        return torch.flip(x, dims=(2,))
    return x

def _to_float_and_normalize(x: torch.Tensor) -> torch.Tensor:
    x = x.to(torch.float32) / 255.0
    mean = torch.tensor(CIFAR100_STATS["mean"]).view(3, 1, 1)
    std = torch.tensor(CIFAR100_STATS["std"]).view(3, 1, 1)
    return (x - mean) / std


class PackedCIFAR100FewShot(Dataset):
    def __init__(self, pack_path: Path, train: bool = True):
        obj = torch.load(pack_path, map_location="cpu")
        self.images: torch.Tensor = obj["images"]
        self.labels: torch.Tensor = obj["labels"].to(torch.long)
        self.train = train

    def __len__(self):
        return self.labels.numel()

    def __getitem__(self, idx: int):
        x = self.images[idx]
        y = self.labels[idx].item()
        if self.train:
            x = _random_crop_reflect(x, padding=4)
            x = _random_hflip(x, p=0.5)
        x = _to_float_and_normalize(x)
        return x, y


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def _class_buckets(targets: List[int]) -> Dict[int, List[int]]:
    buckets: Dict[int, List[int]] = {i: [] for i in range(100)}
    for i, t in enumerate(targets):
        buckets[int(t)].append(i)
    return buckets


def _select_indices_perc(buckets: Dict[int, List[int]], percent: int, rng: torch.Generator) -> Dict[int, List[int]]:

    k = max(1, int(500 * (percent / 100.0) + 0.5))
    selected: Dict[int, List[int]] = {}
    for c, idxs in buckets.items():
        idxs_t = torch.tensor(idxs, dtype=torch.long)
        perm = torch.randperm(len(idxs_t), generator=rng)
        chosen = idxs_t[perm][:k].tolist()
        selected[c] = chosen
    return selected


def _flatten_selected(selected: Dict[int, List[int]]) -> List[int]:
    order: List[int] = []
    for c in range(100):
        order.extend(selected[c])
    return order


def _pack_and_save(trainset: datasets.CIFAR100, indices: List[int], out_dir: Path, percent: int, seed: int, selected: Dict[int, List[int]]):
    _ensure_dir(out_dir)

    imgs = []
    labels = []
    for c in range(100):
        for i in selected[c]:
            arr = trainset.data[i]
            tens = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
            imgs.append(tens)
            labels.append(c)
    images = torch.stack(imgs, dim=0).to(torch.uint8)
    labels = torch.tensor(labels, dtype=torch.int64)

    torch.save({
        "images": images,
        "labels": labels,
        "percent": percent,
        "seed": seed,
        "total": images.shape[0],
        "per_class": images.shape[0] // 100,
    }, out_dir / "train.pt")

    with open(out_dir / "manifest.json", "w") as f:
        json.dump({
            "dataset": "CIFAR100",
            "split": "train",
            "percent": percent,
            "seed": seed,
            "class_to_indices": {str(k): v for k, v in selected.items()}
        }, f, indent=2)


def get_fewshot_loaders(
    percent: int,
    *,
    data_root: str = "./data",
    out_root: str = "../data/fewshot",
    batch_size: int = 100,
    num_workers: int = 8,
    pin_memory: bool = True,
    shuffle: bool = True,
):

    pack_path = Path(out_root) / f"cifar100_{percent}" / "train.pt"

    train_ds = PackedCIFAR100FewShot(pack_path, train=True)

    test_base = datasets.CIFAR100(root=os.path.join(data_root, "cifar100"), train=False, download=True, transform=None)

    class _TestWrap(Dataset):
        def __init__(self, base):
            self.base = base
        def __len__(self):
            return len(self.base)
        def __getitem__(self, idx):
            img, label = self.base.data[idx], self.base.targets[idx]
            x = torch.from_numpy(img).permute(2,0,1).to(torch.uint8)
            x = _to_float_and_normalize(x)
            return x, int(label)

    test_ds = _TestWrap(test_base)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle,
                              num_workers=num_workers, pin_memory=pin_memory)
    test_loader  = DataLoader(test_ds, batch_size=1000, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, test_loader



def main():
    ap = argparse.ArgumentParser("Make and/or use few-shot CIFAR-100 packs")
    ap.add_argument("--data-root", type=str, default="./data", help="Torchvision root where CIFAR downloads")
    ap.add_argument("--out-root", type=str, default="../data/fewshot", help="Output root for packs")
    ap.add_argument("--percents", type=int, nargs="*", default=[5,10,15,25,35,50], help="Percents per class")
    ap.add_argument("--seed", type=int, default=1234, help="Sampling seed")
    ap.add_argument("--make-packs", action="store_true", help="Create/update packed subsets")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing packs")
    args = ap.parse_args()

    if args.make_packs:
        rng = torch.Generator().manual_seed(args.seed)
        out_root = Path(args.out_root)
        _ensure_dir(out_root)

        trainset = datasets.CIFAR100(root=os.path.join(args.data_root, "cifar100"), train=True, transform=None, download=True)
        buckets = _class_buckets(trainset.targets)


        for p in args.percents:
            subset_dir = out_root / f"cifar100_{p}"

            selected = _select_indices_perc(buckets, p, rng)
            indices = _flatten_selected(selected)
            _pack_and_save(trainset, indices, subset_dir, p, args.seed, selected)

if __name__ == "__main__":
    main()
