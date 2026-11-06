"""Grad-CAM heatmaps for the image-text match classifier.

「この文に対して，画像のどこを見て一致／不一致と判定したか」を可視化する．

Example:
    # 学習済みチェックポイントを使う
    uv run python -m clipmatch.visualize --ckpt checkpoints/run1/best.pt \
        --image data/images/001.jpg --text "a cat sitting on a sofa" --out outputs/

    # 学習なし（zero-shot）: CLIP の類似度そのものを可視化．判定は cos の閾値
    uv run python -m clipmatch.visualize --zeroshot --threshold 0.25 \
        --image img.jpg --text "..." --out outputs/

    # CSV をまとめて処理
    uv run python -m clipmatch.visualize --ckpt ... --csv data/val.csv --out outputs/

ヒートマップの意味:
    --direction match    : 類似度 cos(img, txt) を「上げている」領域（既定）
    --direction mismatch : 類似度を「下げている」領域（負の勾配）
    どちらも CAM のターゲットは ±(logit[match] - logit[mismatch])．
    cosine head ではこれは ±(exp(s)·cos + b) なので，head の学習有無に関わらず
    「この文と結びついている領域」を見ていることになる．

手法: ``--method`` で gradcam / gradcam++ / scorecam / eigencam / layercam を選択．
ターゲット層はデフォルトで vision encoder 最終ブロックの layer_norm1
（ViT 向け pytorch-grad-cam の標準的な選び方）．``--layer -2`` 等で変更可．
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from pytorch_grad_cam import (
    EigenCAM,
    GradCAM,
    GradCAMPlusPlus,
    LayerCAM,
    ScoreCAM,
)
from pytorch_grad_cam.utils.image import show_cam_on_image

from .model import ClipMatchClassifier, load_checkpoint, load_processor


class LogitDiffTarget:
    """CAM target = logit[match] - logit[mismatch] (sign flipped for 'mismatch').

    cosine head では mismatch 側の logit が定数 0 なので，これは
    ±(exp(s)·cos + b) に等しく，「類似度を上げる／下げる領域」になる．
    mlp head でも両クラスの差を取るので，head によらず同じ意味で使える．
    """

    def __init__(self, direction: str):
        self.sign = 1.0 if direction == "match" else -1.0

    def __call__(self, model_output):
        if model_output.dim() == 1:
            model_output = model_output[None]
        return self.sign * (model_output[:, 1] - model_output[:, 0])

METHODS = {
    "gradcam": GradCAM,
    "gradcam++": GradCAMPlusPlus,
    "scorecam": ScoreCAM,
    "eigencam": EigenCAM,
    "layercam": LayerCAM,
}


# --------------------------------------------------------------------------- #
class ImageOnlyWrapper(nn.Module):
    """pytorch-grad-cam は 1 テンソル入力のモデルを要求するので，
    テキスト側を固定した画像→logits の関数にラップする．"""

    def __init__(self, model: ClipMatchClassifier, input_ids, attention_mask):
        super().__init__()
        self.model = model
        self.input_ids = input_ids
        self.attention_mask = attention_mask

    def forward(self, pixel_values):
        b = pixel_values.shape[0]
        return self.model(
            pixel_values=pixel_values,
            input_ids=self.input_ids.expand(b, -1),
            attention_mask=self.attention_mask.expand(b, -1),
        )


def make_reshape_transform(grid: int):
    """ViT の (B, 1+N, C) トークン列を (B, C, H, W) に戻す．CLS は捨てる．"""

    def _reshape(tensor):
        x = tensor[:, 1:, :]  # drop CLS
        x = x.reshape(x.size(0), grid, grid, x.size(2))
        return x.permute(0, 3, 1, 2)

    return _reshape


def vit_grid_size(model: ClipMatchClassifier) -> int:
    cfg = model.clip.config.vision_config
    return cfg.image_size // cfg.patch_size


def build_model(args, device):
    if args.zeroshot:
        model = ClipMatchClassifier(args.model, head="cosine", freeze_backbone=False).to(device).eval()
        model_name = args.model
    else:
        model, model_name = load_checkpoint(args.ckpt, device)
    for p in model.parameters():  # CAM needs grads through the vision tower
        p.requires_grad_(True)
    return model, model_name


# --------------------------------------------------------------------------- #
def explain_one(
    model: ClipMatchClassifier,
    processor,
    image: Image.Image,
    text: str,
    device,
    method: str = "gradcam",
    layer: int = -1,
    direction: str = "match",
):
    """Returns (scores dict, cam[H,W] in [0,1], overlay RGB uint8).

    scores = {"cos": raw cosine similarity, "prob_match": softmax prob (学習済み head 経由)}
    """
    enc = processor(text=[text], images=image, return_tensors="pt", padding="max_length",
                    truncation=True, max_length=77)
    pixel_values = enc["pixel_values"].to(device)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    wrapper = ImageOnlyWrapper(model, input_ids, attention_mask)
    target_layer = model.clip.vision_model.encoder.layers[layer].layer_norm1
    cam_cls = METHODS[method]
    cam = cam_cls(
        model=wrapper,
        target_layers=[target_layer],
        reshape_transform=make_reshape_transform(vit_grid_size(model)),
    )
    grayscale = cam(input_tensor=pixel_values, targets=[LogitDiffTarget(direction)])[0]

    with torch.no_grad():
        cos = model.similarity(pixel_values, input_ids, attention_mask)[0].item()
        prob = F.softmax(wrapper(pixel_values), -1)[0, 1].item()
    scores = {"cos": cos, "prob_match": prob}

    # overlay on the *original* image (resized to model input size for alignment)
    size = model.clip.config.vision_config.image_size
    rgb = np.asarray(image.resize((size, size))).astype(np.float32) / 255.0
    overlay = show_cam_on_image(rgb, grayscale, use_rgb=True)
    return scores, grayscale, overlay


def decide(scores: dict, zeroshot: bool, threshold: float) -> int:
    """1 = match, 0 = mismatch.  zeroshot は cos の閾値，学習済みは p(match) >= 0.5．"""
    if zeroshot:
        return int(scores["cos"] >= threshold)
    return int(scores["prob_match"] >= 0.5)


def save_result(out_dir: Path, stem: str, image: Image.Image, overlay, cam, scores, pred,
                text, label=None, zeroshot=False, threshold=None, direction="match"):
    out_dir.mkdir(parents=True, exist_ok=True)
    size = overlay.shape[0]
    orig = np.asarray(image.resize((size, size)))
    # [original | heatmap] side by side, with caption
    panel = np.concatenate([orig, overlay], axis=1)
    panel = cv2.copyMakeBorder(panel, 0, 40, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    if zeroshot:
        caption = f"cos={scores['cos']:.3f} (thr={threshold:.2f}) pred={pred}"
    else:
        caption = f"p(match)={scores['prob_match']:.3f} cos={scores['cos']:.3f} pred={pred}"
    if label is not None:
        caption += f" label={label}"
    caption += f" [{direction}]"
    cv2.putText(panel, caption, (8, size + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    Image.fromarray(panel).save(out_dir / f"{stem}.png")
    np.save(out_dir / f"{stem}_cam.npy", cam)
    meta = {"text": text, "label": label, "pred": pred, "direction": direction, **scores}
    if zeroshot:
        meta["threshold"] = threshold
        meta.pop("prob_match")  # zeroshot では head 未学習なので意味を持たない
    (out_dir / f"{stem}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))


# --------------------------------------------------------------------------- #
def parse_args():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--ckpt", help="train.py で保存したチェックポイント")
    g.add_argument("--zeroshot", action="store_true", help="学習なしで CLIP 類似度を可視化")
    ap.add_argument("--model", default="openai/clip-vit-base-patch32", help="--zeroshot 時のモデル名")
    ap.add_argument("--image")
    ap.add_argument("--text")
    ap.add_argument("--csv", help="image_path,text[,label] を一括処理")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--method", choices=list(METHODS), default="gradcam")
    ap.add_argument("--layer", type=int, default=-1, help="vision encoder のブロック index")
    ap.add_argument("--direction", choices=["match", "mismatch"], default="match",
                    help="match=類似度を上げている領域, mismatch=下げている領域（負勾配）")
    ap.add_argument("--threshold", type=float, default=0.25,
                    help="--zeroshot 時の一致判定の cos 閾値（表示用．ヒートマップには影響しない）")
    ap.add_argument("--limit", type=int, default=None, help="CSV の先頭 N 件だけ")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    model, model_name = build_model(args, device)
    processor = load_processor(model_name)
    out_dir = Path(args.out)

    if args.csv:
        df = pd.read_csv(args.csv)
        if args.limit:
            df = df.head(args.limit)
        base = Path(args.csv).parent
        rows = []
        for i, r in enumerate(df.itertuples(index=False)):
            p = Path(r.image_path)
            if not p.is_absolute():
                p = base / p
            label = int(r.label) if "label" in df.columns else None
            img = Image.open(p).convert("RGB")
            scores, cam, overlay = explain_one(model, processor, img, str(r.text), device,
                                               args.method, args.layer, args.direction)
            pred = decide(scores, args.zeroshot, args.threshold)
            stem = f"{i:05d}_{p.stem}"
            save_result(out_dir, stem, img, overlay, cam, scores, pred, str(r.text), label,
                        args.zeroshot, args.threshold, args.direction)
            row = {"idx": i, "image_path": str(p), "text": r.text, "label": label,
                   "cos": scores["cos"], "pred": pred, "png": f"{stem}.png"}
            if not args.zeroshot:
                row["prob_match"] = scores["prob_match"]
            rows.append(row)
            print(f"[{i}] cos={scores['cos']:.3f} pred={pred} label={label} {p.name}")
        df_out = pd.DataFrame(rows)
        df_out.to_csv(out_dir / "summary.csv", index=False)
        if "label" in df.columns:
            acc = (df_out["pred"] == df_out["label"]).mean()
            print(f"accuracy={acc:.3f}  (n={len(df_out)})")
    else:
        if not (args.image and args.text):
            raise SystemExit("--image と --text，または --csv を指定してください")
        img = Image.open(args.image).convert("RGB")
        scores, cam, overlay = explain_one(model, processor, img, args.text, device,
                                           args.method, args.layer, args.direction)
        pred = decide(scores, args.zeroshot, args.threshold)
        stem = Path(args.image).stem
        save_result(out_dir, stem, img, overlay, cam, scores, pred, args.text, None,
                    args.zeroshot, args.threshold, args.direction)
        print(f"cos={scores['cos']:.3f} pred={pred}  ->  {out_dir / (stem + '.png')}")


if __name__ == "__main__":
    main()
