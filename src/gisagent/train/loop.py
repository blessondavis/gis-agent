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


def train(
    sat_dir: Path,
    map_dir: Path,
    out_dir: Path,
    config: TrainConfig | None = None,
    *,
    device: str | None = None,
    progress=None,
) -> TrainReport:
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

    rng = np.random.default_rng(cfg.seed)
    order = rng.permutation(len(pairs))
    n_val = max(1, int(len(pairs) * cfg.val_split))
    val_pairs = [pairs[i] for i in order[:n_val]]
    train_pairs = [pairs[i] for i in order[n_val:]]

    train_ds = MassRoadsCrops(
        train_pairs, crop=cfg.crop,
        length=cfg.steps_per_epoch * cfg.batch_size, seed=cfg.seed,
    )
    val_ds = TileWindows(val_pairs, crop=cfg.crop)

    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=False,
                          num_workers=cfg.num_workers, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=cfg.num_workers)

    model = smp.Unet(
        encoder_name=cfg.encoder,
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    ).to(device)

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
        n_train_tiles=len(train_pairs),
        n_val_tiles=len(val_pairs),
    )
    ckpt = out_dir / "unet_roads.pt"
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

        model.eval()
        tp = fp = fn = 0.0
        with torch.no_grad():
            for x, y in val_dl:
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
        rec_ = {
            "iou": tp / max(tp + fp + fn, 1e-9),
            "f1": 2 * prec * rec / max(prec + rec, 1e-9),
            "precision": prec,
            "recall": rec,
        }

        best = rec_["iou"] > report.best_iou
        if best:
            report.best_iou = rec_["iou"]
            report.best_epoch = epoch
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "encoder": cfg.encoder,
                    "crop": cfg.crop,
                    "val_iou": rec_["iou"],
                    "epoch": epoch,
                },
                ckpt,
            )

        row = EpochRecord(
            epoch=epoch, train_loss=train_loss, val_iou=rec_["iou"],
            val_f1=rec_["f1"], val_precision=rec_["precision"],
            val_recall=rec_["recall"], seconds=time.perf_counter() - t0,
            best=best,
        )
        report.epochs.append(asdict(row))
        if progress:
            progress(row)

    report.total_seconds = time.perf_counter() - t_start
    report.checkpoint = str(ckpt)
    (out_dir / "train_report.json").write_text(json.dumps(report.to_dict(), indent=2))
    return report
