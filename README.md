# clip-match-gradcam

自然言語の文と画像を入力し，**両者が同じ意味を指しているか（一致／不一致）** を CLIP で 2 クラス分類し，
判定の根拠となった画像領域を **Grad-CAM 系ヒートマップ** で可視化する．

## セットアップ（uv）

```bash
uv sync
```

CUDA 版 PyTorch を使う場合は `pyproject.toml` の `[tool.uv]` 以下のコメントを外して
index を `cu121` / `cu124` 等に合わせてから `uv sync` する．

## データ形式

`data/sample.csv` 参照．

```csv
image_path,text,label
images/001.jpg,"a cat sitting on a sofa",1
images/001.jpg,"a dog running in a park",0
```

- `image_path`: CSV からの相対パス（絶対パスも可）
- `label`: 1 = 一致，0 = 不一致
- `label` 列が無い場合は全行を一致ペアとみなし，他行のテキストをランダムに組み合わせて不一致ペアを自動生成する（`--neg-ratio` で比率指定）

## 学習

```bash
# head（類似度のスケール・バイアス）のみ学習．バックボーンは凍結
uv run python -m clipmatch.train --csv data/train.csv --out checkpoints/run1

# vision tower の末尾 2 ブロックも微調整
uv run python -m clipmatch.train --csv data/train.csv --out checkpoints/run2 \
    --unfreeze-last 2 --backbone-lr 1e-5

# MLP head（表現力高め）
uv run python -m clipmatch.train --csv data/train.csv --head mlp
```

`--val-csv` を省略すると 8:2 に stratified split する．`best.pt`（val AUC 最良）と `last.pt` を保存．

## 可視化

```bash
# 1 件
uv run python -m clipmatch.visualize --ckpt checkpoints/run1/best.pt \
    --image path/to/img.jpg --text "a cat sitting on a sofa" --out outputs/

# CSV 一括（summary.csv も出力）
uv run python -m clipmatch.visualize --ckpt checkpoints/run1/best.pt --csv data/val.csv --out outputs/val

# 学習せずに CLIP の類似度をそのまま可視化
uv run python -m clipmatch.visualize --zeroshot --image img.jpg --text "..." --out outputs/
```

出力: `<stem>.png`（左: 元画像，右: ヒートマップ重畳，下: p(match)），`<stem>_cam.npy`，`<stem>.json`．

オプション:

| フラグ | 内容 |
|---|---|
| `--method` | `gradcam`（既定） / `gradcam++` / `scorecam` / `eigencam` / `layercam` |
| `--layer` | ターゲットとする vision encoder ブロック（既定 `-1` = 最終ブロック） |
| `--target` | `1` = 「一致」と判定した根拠（既定），`0` = 「不一致」の根拠 |

## 設計メモ

- **head = cosine**（既定）: `logit = exp(s)·cos(img, txt) + b`．Grad-CAM のターゲットが
  実質「画像–テキスト類似度」になるので，ヒートマップは「この文に対応する領域」として素直に読める．
- **head = mlp**: `[img, txt, img⊙txt, |img−txt|]` → MLP．表現力は高いが，ヒートマップは MLP 経由の勾配になる．
- ターゲット層は `vision_model.encoder.layers[-1].layer_norm1`，reshape は CLS を除いて
  `(B, 1+N, C) → (B, C, √N, √N)`（pytorch-grad-cam の ViT 向け標準手順）．
- ViT-B/32 では 7×7 グリッドなので粗い．細かく見たい場合は `--model openai/clip-vit-base-patch16`（14×14）
  や `clip-vit-large-patch14`（16×16）を使う．

## 最小動作確認

```bash
bash scripts/run_demo.sh
```

`data/images/` に `001.jpg`, `002.jpg` を置いてから実行．
