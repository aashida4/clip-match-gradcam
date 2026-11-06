# CLAUDE.md

画像と自然言語文の意味一致を CLIP / SigLIP で 2 クラス判定し，判定根拠を Grad-CAM 系ヒートマップで可視化するリポジトリ．
研究用途（学習分析・行動コーディング）．詳細な使い方は README.md を参照．

## 環境

- Python は **uv** で管理する．conda は使わない．`uv sync` で `.venv` が作られる
- 実行は常に `uv run python -m clipmatch.<module> ...`．`pip install` や `python` 直叩きはしない
- パッケージ追加は `pyproject.toml` の `dependencies` に書いて `uv sync`（`uv add <pkg>` でもよい）
- `uv.lock` は **コミットしない**（`.gitignore` 済み）．PyTorch の index が環境依存のため
- PyTorch は既定で **cu128** の index から入る（`[tool.uv.sources]`）．CPU 機ではそのブロックをコメントアウト
- 主要な実行環境は研究室の GPU サーバー（2× Quadro RTX 8000 48GB, sm_75, スケジューラ無し）．
  他のジョブと同居するので `CUDA_VISIBLE_DEVICES` を明示し，両 GPU を占有しない

## レイアウト

```
src/clipmatch/
  data.py       CSV 読み込み（image_path,text[,label]），label 無しなら負例を自動生成
  model.py      ClipMatchClassifier: AutoModel バックボーン + cosine / mlp head，backbone_info()
  train.py      閾値キャリブレーション（既定: head の 2 パラメータのみ）/ 部分微調整
  visualize.py  Grad-CAM（pytorch-grad-cam）と --signed（ReLU なし自前実装），描画は matplotlib
data/sample.csv CSV の形式例（画像は含まない）
scripts/run_demo.sh
```

## 設計上の決まり

- **zero-shot が基本経路**．train.py は任意（閾値校正または微調整）．train を前提にした変更をしない
- バックボーンは `model_type in ("clip", "siglip", "siglip2")` のみ．SigLIP 2 は FixRes 版のみ（NaFlex は拒否）．
  構造差（CLS の有無，グリッド，テキスト max_length）は `backbone_info()` に集約する．他の場所に分岐を散らさない
- `forward` は常に `(B, 2)` logits（index 1 = match）を返す．CAM のターゲットは `LogitDiffTarget`（±(logit[1]−logit[0])）
- `--signed` は符号を保つため ReLU を通さない．pytorch-grad-cam の内部実装に依存しない
- ヒートマップの値は「同じモデル・同じ層・同じ direction」でだけ比較可能．これを崩す正規化を入れない
- transformers 4.x / 5.x 両対応（`get_image_features` の返り値差は `_as_tensor` で吸収）．片方しか動かない変更をしない

## 動作確認

Hugging Face に届かない環境でも通る経路確認を用意してある．小さなランダム初期化モデルを作って全経路を回す：

```bash
# 1) 極小 CLIP / SigLIP を /tmp に作る（README の「最小動作確認」とは別物）
uv run python - <<'EOF'
from transformers import CLIPConfig, CLIPModel, CLIPImageProcessor, CLIPProcessor, PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, pre_tokenizers
cfg = CLIPConfig(text_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2, vocab_size=100),
                 vision_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2, image_size=224, patch_size=32),
                 projection_dim=16)
CLIPModel(cfg).save_pretrained("/tmp/tinyclip")
vocab = {w: i for i, w in enumerate(["[UNK]", "[PAD]", "a", "cat", "dog", "sitting", "on", "sofa"])}
tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]")); tok.pre_tokenizer = pre_tokenizers.Whitespace()
hf = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]")
CLIPProcessor(image_processor=CLIPImageProcessor(), tokenizer=hf).save_pretrained("/tmp/tinyclip")
EOF
# 2) ノイズ画像を data/images/001.jpg, 002.jpg に置いて
uv run python -m clipmatch.visualize --zeroshot --model /tmp/tinyclip --csv data/sample.csv --signed --shared-scale --out /tmp/out --device cpu
uv run python -m clipmatch.train --csv data/sample.csv --model /tmp/tinyclip --out /tmp/ck --epochs 1 --batch-size 2 --num-workers 0 --device cpu
uv run python -m clipmatch.visualize --ckpt /tmp/ck/best.pt --image data/images/001.jpg --text "a cat" --out /tmp/out2 --device cpu
```

実重みでの確認は GPU サーバーで `openai/clip-vit-base-patch32` → `google/siglip2-base-patch16-224` の順に行う．
変更後は少なくとも zero-shot（signed / 非 signed）・学習・学習済み可視化の 3 経路を回してから commit する．

## コミット

- 1 コミット 1 目的．メッセージは英語，本文に何をなぜ変えたかを書く
- `data/images/`，`outputs/`，`checkpoints/`，`*.pt` はコミットしない
- README の記述と挙動がずれる変更をしたら，同じコミットで README も直す

## 書き方

- コメント・docstring・README は日本語でよい（句読点は全角「，．」）．コード識別子とコミットメッセージは英語
- 推測で書かない．未確認の挙動（実重みで試していない等）は「未確認」と明記する
