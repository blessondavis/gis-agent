"""Fine-tune a U-Net for road extraction on the Massachusetts Roads tiles.

Zero-shot SAM 3 finds roads well in suburbia and badly in a dense city grid,
because a concept model reads narrow shadowed streets as texture between
rooftops. A small supervised model trained on this dataset does not have that
problem -- the labels teach it exactly what a road looks like at 1 m/px.

Deliberately modest: a ResNet-34 U-Net at 512 px trains in minutes on 8 GB and
is enough to show the difference. The point is the comparison, not a
state-of-the-art number.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class TrainConfig:
    encoder: str = "resnet34"
    crop: int = 512
    batch_size: int = 8
    epochs: int = 12
    steps_per_epoch: int = 250
    lr: float = 3e-4
    weight_decay: float = 1e-4
    val_split: float = 0.15
    seed: int = 0
    amp: bool = True
    num_workers: int = 0          # Windows: worker startup costs more than it saves
    # fine-tuning
    init_checkpoint: str = ""     # start from these weights instead of ImageNet
    checkpoint_name: str = "unet_roads.pt"
    extra_share: float = 0.5      # share of training crops drawn from the extra source


@dataclass
class EpochRecord:
    epoch: int
    train_loss: float
    val_iou: float
    val_f1: float
    val_precision: float
    val_recall: float
    seconds: float
    best: bool = False
    val_iou_extra: float | None = None     # the fine-tuning domain, when there is one
    val_f1_extra: float | None = None


@dataclass
class TrainReport:
    config: dict
    n_train_tiles: int
    n_val_tiles: int
    epochs: list[dict] = field(default_factory=list)
    best_iou: float = 0.0
    best_epoch: int = -1
    checkpoint: str = ""
    total_seconds: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def _metrics(prob: "np.ndarray", truth: "np.ndarray", thr: float = 0.5) -> dict:
    pred = prob > thr
    t = truth > 0.5
    tp = float((pred & t).sum())
    fp = float((pred & ~t).sum())
    fn = float((~pred & t).sum())
    prec = tp / max(tp + fp, 1e-9)
    rec = tp / max(tp + fn, 1e-9)
    return {
        "iou": tp / max(tp + fp + fn, 1e-9),
        "f1": 2 * prec * rec / max(prec + rec, 1e-9),
        "precision": prec,
        "recall": rec,
    }


def split_pairs(pairs: list, val_split: float = 0.15, seed: int = 0) -> tuple[list, list]:
    """The deterministic train/val split. Shared with evaluation, so a model is
    only ever scored on tiles it did not train on."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(pairs))
    n_val = max(1, int(len(pairs) * val_split))
    return [pairs[i] for i in order[n_val:]], [pairs[i] for i in order[:n_val]]


def _validate(model, dl, device: str, use_amp: bool) -> dict:
    import torch

    model.eval()
    tp = fp = fn = 0.0
    with torch.no_grad():
        for x, y in dl:
            x = x.to(device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                prob = torch.sigmoid(model(x)).float().cpu()
            pred = prob > 0.5
            t = y > 0.5
            tp += float((pred & t).sum())
            fp += float((pred & ~t).sum())
            fn += float((~pred & t).sum())
    prec = tp / max(tp + fp, 1e-9)
    rec = tp / max(tp + fn, 1e-9)
    return {
        "iou": tp / max(tp + fp + fn, 1e-9),
        "f1": 2 * prec * rec / max(prec + rec, 1e-9),
        "precision": prec,
        "recall": rec,
    }


def train(
    sat_dir: Path,
    map_dir: Path,
    out_dir: Path,
    config: TrainConfig | None = None,
    *,
    device: str | None = None,
    progress=None,
    extra_train: list | None = None,
    extra_val: list | None = None,
) -> TrainReport:
    """Train (or, with ``init_checkpoint``, fine-tune) the road U-Net.

    ``extra_train`` / ``extra_val`` are TilePairs from a second dataset. When
    given, ``extra_share`` of the training crops come from it and the rest
    from the original tiles, and the best epoch is chosen on the *mean* of the
    two validation IoUs -- so a gain on the new domain cannot hide a loss on
    the old one.
    """
    import segmentation_models_pytorch as smp
    import torch
    from torch.utils.data import DataLoader

    from gisagent.config import get_settings
    from gisagent.train.data import MassRoadsCrops, TilePair, TileWindows

    cfg = config or TrainConfig()
    device = device or get_settings().device
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = TilePair.discover(Path(sat_dir), Path(map_dir))
    if len(pairs) < 4:
        raise RuntimeError(
            f"need at least 4 tile pairs to train, found {len(pairs)} in {sat_dir}"
        )

    train_pairs, val_pairs = split_pairs(pairs, cfg.val_split, cfg.seed)

    weights = None
    all_train = list(train_pairs)
    if extra_train:
        share = min(max(cfg.extra_share, 0.0), 1.0)
        weights = ([(1 - share) / len(train_pairs)] * len(train_pairs)
                   + [share / len(extra_train)] * len(extra_train))
        all_train += list(extra_train)

    train_ds = MassRoadsCrops(
        all_train, crop=cfg.crop,
        length=cfg.steps_per_epoch * cfg.batch_size, seed=cfg.seed,
        weights=weights,
    )
    val_ds = TileWindows(val_pairs, crop=cfg.crop)

    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=False,
                          num_workers=cfg.num_workers, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=cfg.num_workers)
    extra_dl = None
    if extra_val:
        extra_dl = DataLoader(TileWindows(list(extra_val), crop=cfg.crop),
                              batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers)

    encoder = cfg.encoder
    init = None
    if cfg.init_checkpoint:
        init = torch.load(cfg.init_checkpoint, map_location="cpu", weights_only=False)
        encoder = init.get("encoder", encoder)
    model = smp.Unet(
        encoder_name=encoder,
        encoder_weights=None if init else "imagenet",
        in_channels=3,
        classes=1,
    )
    if init:
        model.load_state_dict(init["state_dict"])
    model = model.to(device)

    # Roads are a few percent of pixels. Plain BCE happily predicts all
    # background; Dice pushes back on that directly.
    bce = smp.losses.SoftBCEWithLogitsLoss()
    dice = smp.losses.DiceLoss(mode="binary", from_logits=True)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    use_amp = cfg.amp and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    report = TrainReport(
        config=asdict(cfg),
        n_train_tiles=len(all_train),
        n_val_tiles=len(val_pairs) + len(extra_val or []),
    )
    ckpt = out_dir / cfg.checkpoint_name
    t_start = time.perf_counter()

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.perf_counter()
        model.train()
        total = 0.0
        for x, y in train_dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(x)
                loss = 0.5 * bce(logits, y) + 0.5 * dice(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += float(loss.detach())
        sched.step()
        train_loss = total / max(len(train_dl), 1)

        rec_ = _validate(model, val_dl, device, use_amp)
        ext = _validate(model, extra_dl, device, use_amp) if extra_dl else None
        # with two domains, select on their mean so neither can be sacrificed
        score = (rec_["iou"] + ext["iou"]) / 2 if ext else rec_["iou"]

        best = score > report.best_iou
        if best:
            report.best_iou = score
            report.best_epoch = epoch
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "encoder": encoder,
                    "crop": cfg.crop,
                    "val_iou": rec_["iou"],
                    "val_iou_extra": ext["iou"] if ext else None,
                    "epoch": epoch,
                    "init_checkpoint": cfg.init_checkpoint,
                },
                ckpt,
            )

        row = EpochRecord(
            epoch=epoch, train_loss=train_loss, val_iou=rec_["iou"],
            val_f1=rec_["f1"], val_precision=rec_["precision"],
            val_recall=rec_["recall"], seconds=time.perf_counter() - t0,
            best=best,
            val_iou_extra=ext["iou"] if ext else None,
            val_f1_extra=ext["f1"] if ext else None,
        )
        report.epochs.append(asdict(row))
        if progress:
            progress(row)

    report.total_seconds = time.perf_counter() - t_start
    report.checkpoint = str(ckpt)
    name = ("train_report.json" if cfg.checkpoint_name == "unet_roads.pt"
            else f"{Path(cfg.checkpoint_name).stem}_report.json")
    (out_dir / name).write_text(json.dumps(report.to_dict(), indent=2))
    return report
