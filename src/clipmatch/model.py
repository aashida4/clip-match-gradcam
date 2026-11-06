"""CLIP backbone + 2-class match head.

Two head types:

* ``cosine``: logit_match = s * cos(img, txt) + b  (2 learnable scalars).
  Zero-shot に近く，Grad-CAM のターゲットが「類似度そのもの」になるので
  ヒートマップの解釈が一番素直．
* ``mlp``: [img, txt, img*txt, |img-txt|] を MLP に通して 2 クラス分類．
  表現力は高いが，ヒートマップは MLP 経由の勾配になる．

どちらも ``forward`` は shape (B, 2) の logits を返す（index 1 = match）．
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor


class ClipMatchClassifier(nn.Module):
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        head: str = "cosine",
        freeze_backbone: bool = True,
        hidden: int = 512,
    ):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(model_name)
        self.head_type = head
        d = self.clip.config.projection_dim

        if head == "cosine":
            # init scale from CLIP's own logit_scale (exp(4.6) ≈ 100)
            self.scale = nn.Parameter(self.clip.logit_scale.detach().clone())
            self.bias = nn.Parameter(torch.zeros(()))
        elif head == "mlp":
            self.mlp = nn.Sequential(
                nn.Linear(4 * d, hidden),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden, 2),
            )
        else:
            raise ValueError(f"unknown head: {head}")

        if freeze_backbone:
            for p in self.clip.parameters():
                p.requires_grad_(False)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _as_tensor(out):
        # transformers 4.x returns a Tensor, 5.x returns BaseModelOutputWithPooling
        if isinstance(out, torch.Tensor):
            return out
        return out.pooler_output

    def encode(self, pixel_values, input_ids, attention_mask):
        img = self._as_tensor(self.clip.get_image_features(pixel_values=pixel_values))
        txt = self._as_tensor(
            self.clip.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
        )
        img = F.normalize(img, dim=-1)
        txt = F.normalize(txt, dim=-1)
        return img, txt

    def similarity(self, pixel_values, input_ids, attention_mask):
        """Raw cosine similarity between image and text embeddings (B,)."""
        img, txt = self.encode(pixel_values, input_ids, attention_mask)
        return (img * txt).sum(-1)

    def forward(self, pixel_values, input_ids, attention_mask, **_):
        img, txt = self.encode(pixel_values, input_ids, attention_mask)
        if self.head_type == "cosine":
            cos = (img * txt).sum(-1)
            logit = self.scale.exp() * cos + self.bias
            # 2-class logits: [mismatch, match]; mismatch fixed at 0
            return torch.stack([torch.zeros_like(logit), logit], dim=-1)
        feats = torch.cat([img, txt, img * txt, (img - txt).abs()], dim=-1)
        return self.mlp(feats)

    # ------------------------------------------------------------------ #
    def unfreeze_vision_last_n(self, n: int):
        """Optionally fine-tune the last *n* transformer blocks of the vision tower."""
        layers = self.clip.vision_model.encoder.layers
        for blk in layers[-n:]:
            for p in blk.parameters():
                p.requires_grad_(True)
        for p in self.clip.vision_model.post_layernorm.parameters():
            p.requires_grad_(True)
        for p in self.clip.visual_projection.parameters():
            p.requires_grad_(True)


def load_processor(model_name: str) -> CLIPProcessor:
    return CLIPProcessor.from_pretrained(model_name)


def save_checkpoint(model: ClipMatchClassifier, path, model_name: str):
    torch.save(
        {
            "model_name": model_name,
            "head": model.head_type,
            "state_dict": model.state_dict(),
        },
        path,
    )


def load_checkpoint(path, device="cpu") -> tuple[ClipMatchClassifier, str]:
    ckpt = torch.load(path, map_location=device)
    model = ClipMatchClassifier(ckpt["model_name"], head=ckpt["head"], freeze_backbone=False)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt["model_name"]
