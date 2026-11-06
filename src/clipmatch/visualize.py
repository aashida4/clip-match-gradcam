"""Grad-CAM heatmaps for the image-text match classifier.

「この文に対して，画像のどこを見て一致／不一致と判定したか」を可視化する．

Example:
    # 学習済みチェックポイントを使う
    uv run python -m clipmatch.visualize --ckpt checkpoints/run1/best.pt \
        --image data/images/001.jpg --text "a cat sitting on a sofa" --out outputs/

    # 学習なし（zero-shot）: CLIP の類似度そのものを可視化
    uv run python -m clipmatch.visualize --zeroshot --image img.jpg --text "..." --out outputs/

    # CSV をまとめて処理
    uv run python -m clipmatch.visualize --ckpt ... --csv data/val.csv --out outputs/

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
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from .model import ClipMatchClassifier, load_checkpoint, load_processor

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
    target_class: int = 1,
):
    """Returns (prob_match, cam[H,W] in [0,1], overlay RGB uint8)."""
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
    grayscale = cam(input_tensor=pixel_values, targets=[ClassifierOutputTarget(target_class)])[0]

    with torch.no_grad():
        prob = F.softmax(wrapper(pixel_values), -1)[0, 1].item()

    # overlay on the *original* image (resized to model input size for alignment)
    size = model.clip.config.vision_config.image_size
    rgb = np.asarray(image.resize((size, size))).astype(np.float32) / 255.0
    overlay = show_cam_on_image(rgb, grayscale, use_rgb=True)
    return prob, grayscale, overlay


def save_result(out_dir: Path, stem: str, image: Image.Image, overlay, cam, prob, text, label=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    size = overlay.shape[0]
    orig = np.asarray(image.resize((size, size)))
    # [original | heatmap] side by side, with caption
    panel = np.concatenate([orig, overlay], axis=1)
    panel = cv2.copyMakeBorder(panel, 0, 40, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    caption = f"p(match)={prob:.3f}"
    if label is not None:
        caption += f"  label={label}"
    cv2.putText(panel, caption, (8, size + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    Image.fromarray(panel).save(out_dir / f"{stem}.png")
    np.save(out_dir / f"{stem}_cam.npy", cam)
    (out_dir / f"{stem}.json").write_text(
        json.dumps({"text": text, "prob_match": prob, "label": label}, ensure_ascii=False, indent=2)
    )


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
    ap.add_argument("--target", type=int, default=1, choices=[0, 1],
                    help="1=match の根拠, 0=mismatch の根拠")
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
            prob, cam, overlay = explain_one(model, processor, img, str(r.text), device,
                                             args.method, args.layer, args.target)
            stem = f"{i:05d}_{p.stem}"
            save_result(out_dir, stem, img, overlay, cam, prob, str(r.text), label)
            rows.append({"idx": i, "image_path": str(p), "text": r.text, "label": label,
                         "prob_match": prob, "png": f"{stem}.png"})
            print(f"[{i}] p(match)={prob:.3f} label={label} {p.name}")
        pd.DataFrame(rows).to_csv(out_dir / "summary.csv", index=False)
    else:
        if not (args.image and args.text):
            raise SystemExit("--image と --text，または --csv を指定してください")
        img = Image.open(args.image).convert("RGB")
        prob, cam, overlay = explain_one(model, processor, img, args.text, device,
                                         args.method, args.layer, args.target)
        stem = Path(args.image).stem
        save_result(out_dir, stem, img, overlay, cam, prob, args.text)
        print(f"p(match)={prob:.3f}  ->  {out_dir / (stem + '.png')}")


if __name__ == "__main__":
    main()
