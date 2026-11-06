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

## 基本的な使い方（学習なし）

学習済み CLIP をそのまま使う．分類は「cos 類似度 ≥ 閾値」で行い，ヒートマップはその類似度に対する Grad-CAM．

```bash
# 1 件
uv run python -m clipmatch.visualize --zeroshot --threshold 0.25 \
    --image img.jpg --text "a cat sitting on a sofa" --out outputs/

# CSV 一括（label 列があれば accuracy も表示）
uv run python -m clipmatch.visualize --zeroshot --threshold 0.25 --csv data/val.csv --out outputs/zs
```

閾値は CLIP のモデルやドメインで変わる（ViT-B/32 では 0.2〜0.3 あたりが目安）．
手持ちのラベル付きペアで `summary.csv` の `cos` 列を見て決めるか，下の train.py で自動的に決める．

## 閾値のキャリブレーション／微調整（train.py）

train.py は必須ではない．用途は 2 つ：

1. **閾値キャリブレーション**（既定）: `cosine` head の scale と bias（2 パラメータ）だけをラベル付きペアで学習する．
   CLIP 本体は凍結．数十〜数百ペアで足りる．これで `p(match)` が校正された確率になる
2. **本当の微調整**: `--unfreeze-last N` や `--head mlp` を使う．CLIP の表現がドメインに合っていない場合のみ

```bash
# 閾値キャリブレーションのみ（CLIP は凍結，学習パラメータは 2 個）
uv run python -m clipmatch.train --csv data/train.csv --out checkpoints/run1

# vision tower の末尾 2 ブロックも微調整
uv run python -m clipmatch.train --csv data/train.csv --out checkpoints/run2 \
    --unfreeze-last 2 --backbone-lr 1e-5

# MLP head（表現力高め）
uv run python -m clipmatch.train --csv data/train.csv --head mlp
```

`--val-csv` を省略すると 8:2 に stratified split する．`best.pt`（val AUC 最良）と `last.pt` を保存．

## 学習済みチェックポイントでの可視化

```bash
uv run python -m clipmatch.visualize --ckpt checkpoints/run1/best.pt \
    --image path/to/img.jpg --text "a cat sitting on a sofa" --out outputs/
uv run python -m clipmatch.visualize --ckpt checkpoints/run1/best.pt --csv data/val.csv --out outputs/val
```

出力: `<stem>.png`（左: 元画像，右: ヒートマップ重畳，下: スコアと判定），`<stem>_cam.npy`，`<stem>.json`，一括時は `summary.csv`．

## ヒートマップの意味

- `--direction match`（既定）: **cos(画像, 文) を上げている領域**．「この文と結びついている場所」
- `--direction mismatch`: cos を**下げている**領域（負の勾配）．「この文に反している場所」

CAM のターゲットは ±(logit[match] − logit[mismatch])．cosine head ではこれが ±(exp(s)·cos + b) に等しいので，
zero-shot でも学習後でもヒートマップの意味は同じ（学習でスケール・バイアスが変わっても形は変わらない）．
`--head mlp` の場合だけ MLP 経由の勾配になる．

オプション:

| フラグ | 内容 |
|---|---|
| `--method` | `gradcam`（既定） / `gradcam++` / `scorecam` / `eigencam` / `layercam` |
| `--layer` | ターゲットとする vision encoder ブロック（既定 `-1` = 最終ブロック） |
| `--direction` | `match`（既定）/ `mismatch`．上記参照 |
| `--threshold` | `--zeroshot` 時の判定閾値（cos）．ヒートマップには影響しない |

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

## 既知の注意点

- ViT の最終ブロックは self-attention で全パッチが混ざるため，CNN の Grad-CAM ほど局所的でない場合がある．
  ノイズが気になる場合は `--layer -2` や `--method gradcam++` / `scorecam` を試す
- CLIP に素の Grad-CAM をかけると局所化精度が落ちるという報告がある（gScoreCAM, ACCV 2022）
