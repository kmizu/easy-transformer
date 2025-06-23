# 用語集

## A

### Attention（アテンション、注意機構）
入力系列の各要素に対して、他の要素との関連性を計算し、重み付けを行う機構。Transformerの中核技術。

```python
# アテンションの基本計算
attention_weights = softmax(Q @ K.T / sqrt(d_k))
output = attention_weights @ V
```

### Autoregressive（自己回帰）
過去の出力を入力として使用し、次の出力を予測するモデル。GPTなどの言語モデルで使用。

## B

### BERT (Bidirectional Encoder Representations from Transformers)
Googleが開発した双方向Transformer。文脈の両方向から情報を取得。

### Batch Normalization（バッチ正規化）
バッチ次元で正規化を行う手法。Transformerでは主にLayer Normalizationが使用される。

### Beam Search（ビームサーチ）
複数の候補を並列に探索し、最も良い系列を見つける探索アルゴリズム。

```python
# ビームサーチの概念
beams = [(initial_sequence, 0.0)]  # (系列, スコア)
for step in range(max_length):
    new_beams = []
    for seq, score in beams:
        # 各ビームから次の候補を生成
        candidates = generate_next_tokens(seq)
        new_beams.extend([(seq + [token], score + log_prob) 
                         for token, log_prob in candidates])
    # 上位k個を選択
    beams = sorted(new_beams, key=lambda x: x[1])[:beam_size]
```

### BPE (Byte Pair Encoding)
頻出する文字ペアを繰り返しマージすることで語彙を構築するトークン化手法。

## C

### Causal Mask（因果マスク）
未来の情報を見ないようにするマスク。デコーダやGPTで使用。

```python
# 因果マスクの作成
causal_mask = torch.tril(torch.ones(seq_len, seq_len))
```

### Cross-Attention（クロスアテンション）
エンコーダの出力をキー・バリューとし、デコーダの状態をクエリとするアテンション。

### Cross-Entropy Loss（クロスエントロピー損失）
分類タスクで使用される損失関数。言語モデルの学習で標準的に使用。

## D

### Decoder（デコーダ）
エンコーダの出力を受け取り、目的の出力を生成する部分。自己回帰的に動作。

### Dropout（ドロップアウト）
訓練時にランダムにニューロンを無効化する正則化手法。

```python
dropout = nn.Dropout(p=0.1)  # 10%の確率でドロップ
```

## E

### Embedding（埋め込み）
離散的なトークンを連続的なベクトル空間に写像する処理。

```python
embedding = nn.Embedding(vocab_size, d_model)
token_vectors = embedding(token_ids)
```

### Encoder（エンコーダ）
入力を処理し、文脈を考慮した表現を生成する部分。

## F

### Feed-Forward Network (FFN)
Transformer内の位置ごとに適用される2層のニューラルネットワーク。

```python
class FFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))
```

### Fine-tuning（ファインチューニング）
事前学習済みモデルを特定のタスクに適応させる学習プロセス。

### Flash Attention
メモリ効率的なアテンション計算手法。IO複雑度を削減。

## G

### GPT (Generative Pre-trained Transformer)
OpenAIが開発した自己回帰型の言語モデル。デコーダのみのアーキテクチャ。

### Gradient Accumulation（勾配累積）
複数のミニバッチの勾配を累積してから更新する手法。メモリ制約下で大きなバッチサイズを実現。

### Gradient Clipping（勾配クリッピング）
勾配の大きさを制限して学習を安定化させる手法。

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

## H

### Head（ヘッド）
Multi-Head Attentionにおける個別のアテンション計算単位。

### Hugging Face
Transformerモデルのライブラリとコミュニティを提供する企業・プラットフォーム。

## I

### Inference（推論）
学習済みモデルを使用して予測を行うプロセス。

## K

### Key（キー）
アテンション機構において、各位置の情報を表現するベクトル。

```python
K = W_k @ X  # キーの計算
```

## L

### Layer Normalization（層正規化）
各サンプルの特徴次元で正規化を行う手法。Transformerで標準的に使用。

```python
layer_norm = nn.LayerNorm(d_model)
normalized = layer_norm(x)
```

### Learning Rate Schedule（学習率スケジュール）
訓練中に学習率を調整する戦略。Transformerではwarmup + 減衰が一般的。

### LoRA (Low-Rank Adaptation)
効率的なファインチューニング手法。低ランク行列でパラメータ更新を近似。

```python
# LoRAの基本的なアイデア
delta_W = A @ B  # A: d×r, B: r×d, r << d
W_new = W_original + alpha * delta_W
```

## M

### Masked Language Model (MLM)
BERTで使用される事前学習タスク。ランダムにマスクしたトークンを予測。

### Multi-Head Attention（マルチヘッドアテンション）
複数のアテンションを並列に計算し、異なる関係性を捉える機構。

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
```

## N

### Next Token Prediction（次トークン予測）
言語モデルの基本タスク。現在までのトークンから次のトークンを予測。

### Nucleus Sampling（Top-p Sampling）
累積確率がpを超えるまでの上位トークンからサンプリングする手法。

## O

### Optimizer（オプティマイザ）
モデルのパラメータを更新するアルゴリズム。AdamWが標準的。

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
```

## P

### Padding（パディング）
系列を同じ長さに揃えるための処理。

### Perplexity（パープレキシティ）
言語モデルの性能指標。低いほど良い。

```python
perplexity = torch.exp(cross_entropy_loss)
```

### Position Encoding（位置エンコーディング）
系列内の位置情報をモデルに伝える仕組み。

### Pre-training（事前学習）
大規模データで汎用的な表現を学習するプロセス。

## Q

### Query（クエリ）
アテンション機構において、情報を検索する側のベクトル。

```python
Q = W_q @ X  # クエリの計算
```

### Quantization（量子化）
モデルの重みや活性化を低精度で表現する圧縮技術。

## R

### Residual Connection（残差接続）
入力を出力に直接加算する接続。勾配消失を防ぐ。

```python
output = layer(x) + x  # 残差接続
```

### RoPE (Rotary Position Embedding)
回転行列を使用した位置エンコーディング手法。

## S

### Self-Attention（自己注意機構）
系列内の要素間の関係を計算するアテンション。

### SentencePiece
言語に依存しないトークナイザー。サブワード単位で分割。

### Softmax
確率分布を生成する活性化関数。アテンション重みの計算で使用。

```python
softmax = torch.nn.functional.softmax(x, dim=-1)
```

## T

### Temperature（温度）
生成時の確率分布の鋭さを制御するパラメータ。

```python
logits = logits / temperature  # 温度でスケーリング
probs = softmax(logits)
```

### Token（トークン）
テキストを処理可能な単位に分割したもの。

### Tokenizer（トークナイザー）
テキストをトークンに分割するツール。

### Top-k Sampling
上位k個のトークンからサンプリングする生成手法。

### Training（訓練/学習）
データからモデルのパラメータを最適化するプロセス。

### Transformer
自己注意機構を基盤とするニューラルネットワークアーキテクチャ。

## V

### Value（バリュー）
アテンション機構において、実際に集約される情報を含むベクトル。

```python
V = W_v @ X  # バリューの計算
```

### Vocabulary（語彙）
モデルが扱えるトークンの集合。

## W

### Warmup
学習初期に学習率を徐々に上げる手法。

```python
def warmup_schedule(step, warmup_steps):
    if step < warmup_steps:
        return step / warmup_steps
    return 1.0
```

### Weight Decay（重み減衰）
正則化の一種。パラメータの大きさにペナルティを課す。

### WordPiece
BERTで使用されるサブワードトークン化手法。

## 数式記号

### d_model
モデルの隠れ層の次元数（例：512, 768, 1024）

### d_k, d_v
キーとバリューの次元数。通常 d_model / n_heads

### d_ff
Feed-Forward Networkの中間層の次元数。通常 4 * d_model

### n_heads
Multi-Head Attentionのヘッド数（例：8, 12, 16）

### n_layers
Transformerブロックの層数（例：6, 12, 24）

## コンパイラとの対応関係

| Transformer用語 | コンパイラ用語 | 説明 |
|---------------|-------------|------|
| Tokenization | 字句解析 | テキストを基本単位に分割 |
| Embedding | シンボルテーブル | トークンを内部表現に変換 |
| Self-Attention | 文脈解析 | 要素間の依存関係を解析 |
| Layer | 最適化パス | 段階的な変換処理 |
| Position Encoding | 行番号/位置情報 | 要素の位置を記録 |
| Feed-Forward | 変換規則 | 個別要素の変換 |
| Decoder | コード生成 | 最終的な出力を生成 |