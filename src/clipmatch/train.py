"""Fine-tune the match classifier.

Example:
    uv run python -m clipmatch.train --csv data/sample.csv --out checkpoints/run1 \
        --head cosine --epochs 5 --batch-size 32

    # vision tower の最後の 2 ブロックも学習する場合
    uv run python -m clipmatch.train --csv data/train.csv --unfreeze-last 2 --lr 1e-5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import PairDataset, load_pairs, make_collate
from .model import ClipMatchClassifier, load_processor, save_checkpoint


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--val-csv", default=None, help="省略時は --csv を 8:2 に分割")
    ap.add_argument("--out", default="checkpoints/run")
    ap.add_argument("--model", default="openai/clip-vit-base-patch32")
    ap.add_argument("--head", choices=["cosine", "mlp"], default="cosine")
    ap.add_argument("--unfreeze-last", type=int, default=0, help="vision tower の末尾 n ブロックを学習")
    ap.add_argument("--neg-ratio", type=float, default=1.0, help="label 列が無い場合の負例比率")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3, help="head の学習率")
    ap.add_argument("--backbone-lr", type=float, default=1e-5)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    for batch in loader:
        labels = batch.pop("labels")
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch)
        ps.append(F.softmax(logits, -1)[:, 1].cpu())
        ys.append(labels)
    y = torch.cat(ys).numpy()
    p = torch.cat(ps).numpy()
    pred = (p >= 0.5).astype(int)
    out = {
        "acc": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred)),
    }
    if len(set(y.tolist())) == 2:
        out["auc"] = float(roc_auc_score(y, p))
    return out


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = load_pairs(args.csv, neg_ratio=args.neg_ratio, seed=args.seed)
    if args.val_csv:
        train_pairs, val_pairs = pairs, load_pairs(args.val_csv, args.neg_ratio, args.seed)
    else:
        labels = [p.label for p in pairs]
        n_val = max(int(round(len(pairs) * 0.2)), 2)
        stratify = labels if n_val >= 2 and min(labels.count(0), labels.count(1)) >= 2 else None
        train_pairs, val_pairs = train_test_split(
            pairs, test_size=n_val, random_state=args.seed, stratify=stratify
        )
    print(f"train={len(train_pairs)}  val={len(val_pairs)}")

    model = ClipMatchClassifier(args.model, head=args.head, freeze_backbone=True)
    processor = load_processor(args.model)
    collate = make_collate(processor, max_length=model.info["text_max_length"])
    train_loader = DataLoader(
        PairDataset(train_pairs), batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate,
    )
    val_loader = DataLoader(
        PairDataset(val_pairs), batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate,
    )

    if args.unfreeze_last > 0:
        model.unfreeze_vision_last_n(args.unfreeze_last)
    model.to(args.device)

    head_params = [p for n, p in model.named_parameters() if not n.startswith("clip.") and p.requires_grad]
    backbone_params = [p for n, p in model.named_parameters() if n.startswith("clip.") and p.requires_grad]
    groups = [{"params": head_params, "lr": args.lr}]
    if backbone_params:
        groups.append({"params": backbone_params, "lr": args.backbone_lr})
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    print(f"trainable params: head={sum(p.numel() for p in head_params):,} "
          f"backbone={sum(p.numel() for p in backbone_params):,}")

    best = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, n = 0.0, 0
        for batch in tqdm(train_loader, desc=f"epoch {epoch}"):
            labels = batch.pop("labels").to(args.device)
            batch = {k: v.to(args.device) for k, v in batch.items()}
            logits = model(**batch)
            loss = F.cross_entropy(logits, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(labels)
            n += len(labels)
        metrics = evaluate(model, val_loader, args.device)
        metrics["train_loss"] = total / max(n, 1)
        metrics["epoch"] = epoch
        history.append(metrics)
        print(json.dumps(metrics))
        score = metrics.get("auc", metrics["f1"])
        if score > best:
            best = score
            save_checkpoint(model, out_dir / "best.pt", args.model)
    save_checkpoint(model, out_dir / "last.pt", args.model)
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
