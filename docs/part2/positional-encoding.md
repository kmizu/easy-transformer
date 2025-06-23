# 位置エンコーディング

## はじめに：順序の問題

プログラミング言語では、トークンの順序が決定的に重要です。`a = b` と `b = a` は全く異なる意味を持ちます。パーサーは各トークンの位置を常に追跡し、文法規則に従って構文木を構築します。

しかし、前章で学んだ注意機構には重大な問題があります。それは**位置不変性**です。Self-Attentionは本質的に集合演算であり、入力の順序を考慮しません。これは「並列処理可能」という利点の裏返しでもあります。

この章では、Transformerがどのようにして位置情報を扱い、順序を理解するのかを詳しく見ていきます。

## 7.1 なぜ位置情報が必要か

### 位置不変性の実証

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Optional
import math
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

class PositionalProblemDemo:
    """位置情報の必要性を実証"""
    
    def __init__(self):
        self.d_model = 64
        self.attention = self._create_simple_attention()
    
    def _create_simple_attention(self):
        """シンプルな自己注意層"""
        class SimpleAttention(nn.Module):
            def __init__(self, d_model):
                super().__init__()
                self.d_model = d_model
                self.W_q = nn.Linear(d_model, d_model)
                self.W_k = nn.Linear(d_model, d_model)
                self.W_v = nn.Linear(d_model, d_model)
            
            def forward(self, x):
                Q = self.W_q(x)
                K = self.W_k(x)
                V = self.W_v(x)
                
                scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_model)
                weights = torch.softmax(scores, dim=-1)
                output = torch.matmul(weights, V)
                
                return output, weights
        
        return SimpleAttention(self.d_model)
    
    def demonstrate_position_blindness(self):
        """位置の無視を実証"""
        print("=== 位置不変性の問題 ===\n")
        
        # 同じ単語、異なる順序
        sentences = [
            ["The", "cat", "ate", "the", "fish"],
            ["The", "fish", "ate", "the", "cat"],
            ["ate", "The", "cat", "the", "fish"]  # 文法的に変だが、デモのため
        ]
        
        # 単語の埋め込み（仮想的）
        word_embeddings = {
            "The": torch.randn(1, self.d_model),
            "the": torch.randn(1, self.d_model),  # 大文字小文字は区別
            "cat": torch.randn(1, self.d_model),
            "fish": torch.randn(1, self.d_model),
            "ate": torch.randn(1, self.d_model)
        }
        
        outputs = []
        
        for sent_idx, sentence in enumerate(sentences):
            # 文の埋め込みを作成
            embeddings = []
            for word in sentence:
                embeddings.append(word_embeddings[word])
            
            x = torch.cat(embeddings, dim=0).unsqueeze(0)  # [1, seq_len, d_model]
            
            # 自己注意を適用
            with torch.no_grad():
                output, weights = self.attention(x)
            
            outputs.append(output)
            
            print(f"文{sent_idx + 1}: {' '.join(sentence)}")
            print(f"出力の平均: {output.mean().item():.4f}")
            print(f"出力の標準偏差: {output.std().item():.4f}\n")
        
        # 出力の類似性を計算
        print("出力間の類似性:")
        for i in range(len(outputs)):
            for j in range(i + 1, len(outputs)):
                similarity = torch.cosine_similarity(
                    outputs[i].flatten(), 
                    outputs[j].flatten(), 
                    dim=0
                ).item()
                print(f"文{i+1} vs 文{j+1}: {similarity:.4f}")
        
        print("\n→ 順序が異なっても出力が似ている！")
        
        self._visualize_position_blindness(sentences, word_embeddings)
    
    def _visualize_position_blindness(self, sentences, word_embeddings):
        """位置の無視を可視化"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for idx, (ax, sentence) in enumerate(zip(axes, sentences)):
            # 各単語の埋め込みを2次元に投影（PCA的な処理）
            embeddings = []
            for word in sentence:
                embeddings.append(word_embeddings[word].squeeze().numpy()[:2])
            
            embeddings = np.array(embeddings)
            
            # 散布図
            colors = plt.cm.tab10(np.arange(len(sentence)))
            
            for i, (word, emb, color) in enumerate(zip(sentence, embeddings, colors)):
                ax.scatter(emb[0], emb[1], c=[color], s=200, alpha=0.7)
                ax.annotate(f"{i}: {word}", 
                           xy=(emb[0], emb[1]), 
                           xytext=(5, 5), 
                           textcoords='offset points',
                           fontsize=10)
            
            ax.set_title(f"文{idx + 1}: {' '.join(sentence)}")
            ax.set_xlabel("Dimension 1")
            ax.set_ylabel("Dimension 2")
            ax.grid(True, alpha=0.3)
        
        plt.suptitle("位置情報なし：単語の順序が失われる", fontsize=14)
        plt.tight_layout()
        plt.show()
```

### 順序が重要な例

```python
class OrderImportanceExamples:
    """順序の重要性を示す様々な例"""
    
    def programming_language_examples(self):
        """プログラミング言語での順序の重要性"""
        print("=== プログラミング言語での順序 ===\n")
        
        examples = [
            {
                "correct": "result = function(arg1, arg2)",
                "reordered": "function = result(arg1, arg2)",
                "effect": "代入の方向が逆転"
            },
            {
                "correct": "if (condition) { action(); }",
                "reordered": "{ action(); } if (condition)",
                "effect": "構文エラー"
            },
            {
                "correct": "array[index] = value",
                "reordered": "value = array[index]",
                "effect": "読み取りと書き込みが逆"
            }
        ]
        
        for ex in examples:
            print(f"正しい順序: {ex['correct']}")
            print(f"順序変更後: {ex['reordered']}")
            print(f"影響: {ex['effect']}\n")
    
    def natural_language_examples(self):
        """自然言語での順序の重要性"""
        print("=== 自然言語での順序 ===\n")
        
        examples = [
            {
                "sentences": [
                    "Dog bites man",
                    "Man bites dog"
                ],
                "difference": "主語と目的語が入れ替わり、意味が逆転"
            },
            {
                "sentences": [
                    "I saw the man with the telescope",
                    "With the telescope, I saw the man"
                ],
                "difference": "修飾関係が変わり、望遠鏡を持っているのが誰か不明確に"
            },
            {
                "sentences": [
                    "Only I love you",
                    "I only love you",
                    "I love only you"
                ],
                "difference": "'only'の位置で意味が大きく変化"
            }
        ]
        
        for ex in examples:
            print("文のバリエーション:")
            for sent in ex["sentences"]:
                print(f"  - {sent}")
            print(f"違い: {ex['difference']}\n")
    
    def visualize_order_impact(self):
        """順序の影響を可視化"""
        # 依存関係の木構造で表現
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # 文1: "The cat chased the mouse"
        self._draw_dependency_tree(ax1, 
                                 ["The", "cat", "chased", "the", "mouse"],
                                 [(0, 1), (1, 2), (3, 4), (4, 2)],
                                 "The cat chased the mouse")
        
        # 文2: "The mouse chased the cat"
        self._draw_dependency_tree(ax2,
                                 ["The", "mouse", "chased", "the", "cat"],
                                 [(0, 1), (1, 2), (3, 4), (4, 2)],
                                 "The mouse chased the cat")
        
        plt.suptitle("同じ単語、異なる順序 → 異なる意味", fontsize=14)
        plt.tight_layout()
        plt.show()
    
    def _draw_dependency_tree(self, ax, words, edges, title):
        """依存関係木を描画"""
        positions = [(i, 0) for i in range(len(words))]
        
        # ノード（単語）を描画
        for i, (x, y) in enumerate(positions):
            circle = plt.Circle((x, y), 0.3, color='lightblue', ec='black')
            ax.add_patch(circle)
            ax.text(x, y, words[i], ha='center', va='center', fontsize=10)
        
        # エッジ（依存関係）を描画
        for start, end in edges:
            x1, y1 = positions[start]
            x2, y2 = positions[end]
            
            # 曲線矢印
            ax.annotate('', xy=(x2, y2 + 0.3), xytext=(x1, y1 + 0.3),
                       arrowprops=dict(arrowstyle='->', 
                                     connectionstyle="arc3,rad=0.3",
                                     color='red', lw=2))
        
        ax.set_xlim(-0.5, len(words) - 0.5)
        ax.set_ylim(-1, 2)
        ax.set_title(title)
        ax.axis('off')
```

## 7.2 絶対位置エンコーディング

### サイン・コサイン位置エンコーディング

```python
class SinusoidalPositionalEncoding:
    """オリジナルTransformerの位置エンコーディング"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        self.d_model = d_model
        self.max_len = max_len
        self.pe = self._create_positional_encoding()
    
    def _create_positional_encoding(self) -> torch.Tensor:
        """位置エンコーディングの作成"""
        pe = torch.zeros(self.max_len, self.d_model)
        position = torch.arange(0, self.max_len).unsqueeze(1).float()
        
        # 波長を計算
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * 
                            -(math.log(10000.0) / self.d_model))
        
        # サインとコサインを交互に適用
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe
    
    def explain_formula(self):
        """数式の説明"""
        print("=== サイン・コサイン位置エンコーディング ===\n")
        print("位置posの次元iにおけるエンコーディング:")
        print("PE(pos, 2i) = sin(pos / 10000^(2i/d_model))")
        print("PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))\n")
        
        print("なぜこの数式？")
        print("1. 周期性: 異なる次元で異なる周期を持つ")
        print("2. 相対位置: PE(pos+k)をPE(pos)とPE(k)の線形変換で表現可能")
        print("3. 外挿性: 学習時より長い系列にも対応可能")
        
        self._demonstrate_properties()
    
    def _demonstrate_properties(self):
        """位置エンコーディングの性質を実証"""
        # 小さな例で性質を示す
        d_model = 8
        positions = [0, 1, 2, 5, 10, 20]
        
        pe_small = torch.zeros(30, d_model)
        position = torch.arange(0, 30).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            -(math.log(10000.0) / d_model))
        
        pe_small[:, 0::2] = torch.sin(position * div_term)
        pe_small[:, 1::2] = torch.cos(position * div_term)
        
        print("\n位置エンコーディングの値（最初の4次元）:")
        for pos in positions:
            print(f"Position {pos:2d}: {pe_small[pos, :4].numpy()}")
    
    def visualize_encoding(self):
        """位置エンコーディングの可視化"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. ヒートマップ
        ax = axes[0, 0]
        im = ax.imshow(self.pe[:100, :64].T, cmap='RdBu', aspect='auto')
        ax.set_xlabel('Position')
        ax.set_ylabel('Dimension')
        ax.set_title('位置エンコーディングのヒートマップ')
        plt.colorbar(im, ax=ax)
        
        # 2. 個別次元の波形
        ax = axes[0, 1]
        positions = np.arange(100)
        dimensions = [0, 1, 4, 5, 10, 11]  # サイン・コサインのペア
        
        for i, dim in enumerate(dimensions):
            ax.plot(positions, self.pe[:100, dim], 
                   label=f'dim {dim} ({"sin" if dim % 2 == 0 else "cos"})',
                   alpha=0.8)
        
        ax.set_xlabel('Position')
        ax.set_ylabel('Value')
        ax.set_title('異なる次元の波形')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. 周波数スペクトル
        ax = axes[1, 0]
        for dim in range(0, min(10, self.d_model), 2):
            wavelength = 10000 ** (dim / self.d_model)
            frequency = 1 / wavelength
            ax.scatter(dim, frequency, s=100)
        
        ax.set_xlabel('Dimension')
        ax.set_ylabel('Frequency')
        ax.set_yscale('log')
        ax.set_title('各次元の周波数')
        ax.grid(True, alpha=0.3)
        
        # 4. 位置間の類似度
        ax = axes[1, 1]
        positions_to_compare = [0, 10, 20, 30, 40]
        similarity_matrix = np.zeros((len(positions_to_compare), len(positions_to_compare)))
        
        for i, pos1 in enumerate(positions_to_compare):
            for j, pos2 in enumerate(positions_to_compare):
                similarity = torch.cosine_similarity(
                    self.pe[pos1], self.pe[pos2], dim=0
                ).item()
                similarity_matrix[i, j] = similarity
        
        im = ax.imshow(similarity_matrix, cmap='Blues', aspect='auto')
        ax.set_xticks(range(len(positions_to_compare)))
        ax.set_yticks(range(len(positions_to_compare)))
        ax.set_xticklabels(positions_to_compare)
        ax.set_yticklabels(positions_to_compare)
        ax.set_title('位置間のコサイン類似度')
        
        # 値を表示
        for i in range(len(positions_to_compare)):
            for j in range(len(positions_to_compare)):
                ax.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                       ha='center', va='center')
        
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.show()
    
    def demonstrate_relative_position(self):
        """相対位置の性質を実証"""
        print("\n=== 相対位置の性質 ===")
        
        # PE(pos+k) ≈ f(PE(pos), PE(k)) を確認
        pos = 10
        k = 5
        
        pe_pos = self.pe[pos]
        pe_k = self.pe[k]
        pe_pos_plus_k = self.pe[pos + k]
        
        # 線形変換で近似できることを示す
        # （実際には回転行列による変換）
        
        print(f"PE({pos}) の最初の4要素: {pe_pos[:4].numpy()}")
        print(f"PE({k}) の最初の4要素: {pe_k[:4].numpy()}")
        print(f"PE({pos+k}) の最初の4要素: {pe_pos_plus_k[:4].numpy()}")
        
        # 加法定理を使った理論値
        # sin(a+b) = sin(a)cos(b) + cos(a)sin(b)
        # cos(a+b) = cos(a)cos(b) - sin(a)sin(b)
        
        theoretical = torch.zeros_like(pe_pos_plus_k)
        theoretical[0::2] = pe_pos[0::2] * pe_k[1::2] + pe_pos[1::2] * pe_k[0::2]
        theoretical[1::2] = pe_pos[1::2] * pe_k[1::2] - pe_pos[0::2] * pe_k[0::2]
        
        error = torch.abs(pe_pos_plus_k - theoretical).mean().item()
        print(f"\n理論値との平均絶対誤差: {error:.6f}")
        print("→ 相対位置が線形変換で表現可能！")
```

### 学習可能な位置埋め込み

```python
class LearnablePositionalEmbedding:
    """学習可能な位置埋め込み（BERT方式）"""
    
    def __init__(self, max_len: int, d_model: int):
        self.max_len = max_len
        self.d_model = d_model
        self.position_embeddings = nn.Embedding(max_len, d_model)
        
        # 初期化
        nn.init.normal_(self.position_embeddings.weight, std=0.02)
    
    def compare_with_sinusoidal(self):
        """サイン・コサイン方式との比較"""
        print("=== 学習可能 vs サイン・コサイン ===\n")
        
        comparison = {
            "学習可能": {
                "長所": [
                    "タスク特化の位置表現を学習",
                    "実装がシンプル",
                    "短い系列では性能が良い"
                ],
                "短所": [
                    "学習時の最大長を超えられない",
                    "パラメータ数が増える",
                    "相対位置の関係が不明瞭"
                ]
            },
            "サイン・コサイン": {
                "長所": [
                    "任意の長さに対応",
                    "パラメータ不要",
                    "相対位置を自然に表現"
                ],
                "短所": [
                    "固定パターンのみ",
                    "タスク特化の最適化不可",
                    "理論的に複雑"
                ]
            }
        }
        
        for method, props in comparison.items():
            print(f"{method}:")
            print("  長所:")
            for pro in props["長所"]:
                print(f"    - {pro}")
            print("  短所:")
            for con in props["短所"]:
                print(f"    - {con}")
            print()
    
    def visualize_learned_patterns(self, num_positions: int = 50):
        """学習されたパターンを可視化"""
        # 仮想的な学習済み埋め込み
        torch.manual_seed(42)
        
        # 位置による段階的な変化をシミュレート
        learned_embeddings = torch.zeros(num_positions, self.d_model)
        
        for pos in range(num_positions):
            # 基本パターン + ノイズ
            base_pattern = torch.sin(torch.arange(self.d_model).float() * pos / 10)
            noise = torch.randn(self.d_model) * 0.1
            learned_embeddings[pos] = base_pattern + noise
            
            # 正規化
            learned_embeddings[pos] = learned_embeddings[pos] / learned_embeddings[pos].norm() * math.sqrt(self.d_model)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # ヒートマップ
        im = ax1.imshow(learned_embeddings[:30].T, cmap='RdBu', aspect='auto')
        ax1.set_xlabel('Position')
        ax1.set_ylabel('Dimension')
        ax1.set_title('学習された位置埋め込み')
        plt.colorbar(im, ax=ax1)
        
        # 位置間の類似度
        similarity_matrix = torch.matmul(learned_embeddings, learned_embeddings.T)
        im = ax2.imshow(similarity_matrix[:30, :30], cmap='Blues', aspect='auto')
        ax2.set_xlabel('Position')
        ax2.set_ylabel('Position')
        ax2.set_title('位置間の類似度')
        plt.colorbar(im, ax=ax2)
        
        plt.tight_layout()
        plt.show()
```

## 7.3 相対位置エンコーディング

### 相対位置の概念

```python
class RelativePositionalEncoding:
    """相対位置エンコーディングの実装"""
    
    def __init__(self, d_model: int, max_relative_position: int = 128):
        self.d_model = d_model
        self.max_relative_position = max_relative_position
        
        # 相対位置埋め込み
        self.relative_positions_embeddings = nn.Embedding(
            2 * max_relative_position + 1, d_model
        )
    
    def explain_concept(self):
        """相対位置の概念を説明"""
        print("=== 相対位置エンコーディング ===\n")
        
        print("絶対位置 vs 相対位置:")
        print("- 絶対位置: 'The' は位置0、'cat' は位置1")
        print("- 相対位置: 'cat' は 'The' から見て+1の位置\n")
        
        print("利点:")
        print("1. 文の長さに依存しない")
        print("2. 同じパターンが異なる位置で再利用可能")
        print("3. より自然な帰納バイアス")
        
        self._demonstrate_relative_positions()
    
    def _demonstrate_relative_positions(self):
        """相対位置の例を示す"""
        sentence = ["The", "quick", "brown", "fox", "jumps"]
        
        print("\n相対位置行列:")
        print("    ", "  ".join(f"{w:>6}" for w in sentence))
        
        for i, word_i in enumerate(sentence):
            row = []
            for j, word_j in enumerate(sentence):
                relative_pos = j - i
                row.append(f"{relative_pos:6d}")
            print(f"{word_i:>6}", " ".join(row))
    
    def compute_relative_positions(self, seq_len: int) -> torch.Tensor:
        """相対位置行列を計算"""
        # 各位置ペアの相対距離
        positions = torch.arange(seq_len)
        relative_positions = positions.unsqueeze(0) - positions.unsqueeze(1)
        
        # クリッピング
        relative_positions = torch.clamp(
            relative_positions, 
            -self.max_relative_position, 
            self.max_relative_position
        )
        
        # インデックスに変換（負の値に対応）
        relative_positions = relative_positions + self.max_relative_position
        
        return relative_positions
    
    def visualize_relative_attention(self):
        """相対位置を考慮した注意を可視化"""
        seq_len = 10
        relative_positions = self.compute_relative_positions(seq_len)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 1. 相対位置行列
        ax = axes[0]
        im = ax.imshow(relative_positions - self.max_relative_position, 
                      cmap='RdBu', aspect='auto')
        ax.set_xlabel('Position j')
        ax.set_ylabel('Position i')
        ax.set_title('相対位置 (j - i)')
        plt.colorbar(im, ax=ax)
        
        # 2. 距離による減衰
        ax = axes[1]
        distances = torch.abs(torch.arange(seq_len).unsqueeze(0) - torch.arange(seq_len).unsqueeze(1))
        decay = 1.0 / (1.0 + distances.float())
        
        im = ax.imshow(decay, cmap='Blues', aspect='auto')
        ax.set_xlabel('Position j')
        ax.set_ylabel('Position i')
        ax.set_title('距離による減衰')
        plt.colorbar(im, ax=ax)
        
        # 3. 注意パターンの例
        ax = axes[2]
        # 仮想的な注意重み（相対位置を考慮）
        attention = torch.softmax(-distances.float() / 2.0, dim=-1)
        
        im = ax.imshow(attention, cmap='Blues', aspect='auto')
        ax.set_xlabel('Attended to')
        ax.set_ylabel('Attending from')
        ax.set_title('相対位置ベースの注意')
        plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        plt.show()
```

### Transformer-XLとRelative Position Encodingの実装

```python
class TransformerXLPositioning:
    """Transformer-XL方式の相対位置エンコーディング"""
    
    def __init__(self, d_model: int, n_heads: int):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        # 相対位置バイアス
        self.r_w_bias = nn.Parameter(torch.randn(n_heads, self.d_head))
        self.r_r_bias = nn.Parameter(torch.randn(n_heads, self.d_head))
    
    def explain_mechanism(self):
        """Transformer-XLの仕組みを説明"""
        print("=== Transformer-XLの相対位置 ===\n")
        
        print("標準のAttention:")
        print("Attention(Q, K, V) = softmax(QK^T / √d_k)V\n")
        
        print("Transformer-XLのAttention:")
        print("各ヘッドで以下を計算:")
        print("1. コンテンツベースの注意: q_i · k_j")
        print("2. コンテンツ-位置の注意: q_i · r_{i-j}")
        print("3. グローバルコンテンツバイアス: u · k_j")
        print("4. グローバル位置バイアス: v · r_{i-j}")
        
        print("\nここで:")
        print("- r_{i-j}: 相対位置i-jのエンコーディング")
        print("- u, v: 学習可能なバイアスパラメータ")
    
    def compute_relative_attention(self, 
                                 query: torch.Tensor,
                                 key: torch.Tensor,
                                 value: torch.Tensor,
                                 relative_embeddings: torch.Tensor) -> torch.Tensor:
        """相対位置を考慮した注意の計算"""
        batch_size, seq_len, _ = query.shape
        
        # Multi-headに分割
        query = query.view(batch_size, seq_len, self.n_heads, self.d_head)
        key = key.view(batch_size, seq_len, self.n_heads, self.d_head)
        
        # コンテンツベースの注意
        content_score = torch.einsum('bihd,bjhd->bhij', query, key)
        
        # 位置ベースの注意（簡略化版）
        # 実際の実装はより複雑
        position_score = self._compute_position_scores(query, relative_embeddings)
        
        # 合計スコア
        scores = content_score + position_score
        
        # Softmaxと値の集約
        attention_weights = torch.softmax(scores / math.sqrt(self.d_head), dim=-1)
        
        return attention_weights
    
    def _compute_position_scores(self, query, relative_embeddings):
        """位置スコアの計算（簡略化）"""
        # 実際の実装では、相対位置埋め込みとの複雑な計算
        # ここでは概念を示すための簡略版
        batch_size, seq_len = query.shape[:2]
        position_scores = torch.zeros(batch_size, self.n_heads, seq_len, seq_len)
        
        return position_scores
```

## 7.4 最新の位置エンコーディング手法

### Rotary Position Embedding (RoPE)

```python
class RotaryPositionalEmbedding:
    """Rotary Position Embedding（RoPE）の実装"""
    
    def __init__(self, d_model: int, max_seq_len: int = 2048, base: int = 10000):
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.base = base
        
        # 回転行列の周波数を事前計算
        self.inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
    
    def explain_rope(self):
        """RoPEの概念を説明"""
        print("=== Rotary Position Embedding (RoPE) ===\n")
        
        print("基本アイデア:")
        print("- 位置情報を回転として表現")
        print("- Query/Keyベクトルを位置に応じて回転")
        print("- 内積が相対位置のみに依存\n")
        
        print("利点:")
        print("1. 相対位置を自然に表現")
        print("2. 任意の長さに外挿可能")
        print("3. 計算効率が良い")
        print("4. 理論的に美しい")
        
        self._visualize_rotation_concept()
    
    def _visualize_rotation_concept(self):
        """回転の概念を可視化"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 1. 2次元での回転
        ax = axes[0]
        angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
        
        for i, angle in enumerate(angles):
            x = np.cos(angle)
            y = np.sin(angle)
            ax.arrow(0, 0, x*0.8, y*0.8, head_width=0.1, head_length=0.1,
                    fc=plt.cm.viridis(i/len(angles)), 
                    ec=plt.cm.viridis(i/len(angles)))
            ax.text(x*1.1, y*1.1, f'pos={i}', ha='center', va='center')
        
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title('2D回転による位置表現')
        
        # 2. 周波数の違い
        ax = axes[1]
        positions = np.arange(32)
        
        for i, freq_idx in enumerate([0, 2, 4]):
            freq = self.inv_freq[freq_idx].item()
            angles = positions * freq
            ax.plot(positions, np.sin(angles), 
                   label=f'dim {2*freq_idx}', alpha=0.8)
        
        ax.set_xlabel('Position')
        ax.set_ylabel('Sin(position * freq)')
        ax.set_title('異なる次元の回転周波数')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. 相対位置の保存
        ax = axes[2]
        pos1, pos2 = 5, 8
        relative_pos = pos2 - pos1
        
        # 各次元での角度差
        dims = np.arange(0, 16, 2)
        freqs = self.inv_freq[:len(dims)].numpy()
        angle_diffs = relative_pos * freqs
        
        ax.stem(dims, angle_diffs, basefmt=' ')
        ax.set_xlabel('Dimension')
        ax.set_ylabel('Angle difference')
        ax.set_title(f'相対位置 {relative_pos} の角度差')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def apply_rope(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """RoPEを適用"""
        batch_size, seq_len, d_model = x.shape
        
        # xを実部と虚部に分割
        x_complex = x.view(batch_size, seq_len, -1, 2)
        x_complex = torch.view_as_complex(x_complex)
        
        # 位置に応じた回転角
        position_ids = position_ids.view(-1)
        sinusoid = torch.einsum('i,j->ij', position_ids, self.inv_freq)
        
        # 複素数表現での回転
        cos = sinusoid.cos()
        sin = sinusoid.sin()
        
        # 回転を適用（簡略化版）
        # 実際の実装ではより効率的な方法を使用
        
        return x  # 簡略化のため元のテンソルを返す
    
    def demonstrate_rope_properties(self):
        """RoPEの性質を実証"""
        print("\n=== RoPEの性質 ===")
        
        # 小さな例で性質を確認
        d_model = 4
        q = torch.randn(1, 1, d_model)  # Query at position m
        k = torch.randn(1, 1, d_model)  # Key at position n
        
        # 異なる位置でのRoPE適用をシミュレート
        positions = [0, 5, 10]
        
        print("\n内積の変化:")
        for m in positions:
            for n in positions:
                # 実際の計算は複雑なので、概念的な結果を表示
                relative = n - m
                print(f"Q(pos={m}) · K(pos={n}) = f(relative_pos={relative})")
```

### ALiBi (Attention with Linear Biases)

```python
class ALiBiPositioning:
    """ALiBi（Attention with Linear Biases）の実装"""
    
    def __init__(self, n_heads: int):
        self.n_heads = n_heads
        self.slopes = self._get_slopes()
    
    def _get_slopes(self) -> torch.Tensor:
        """各ヘッドのスロープを計算"""
        def get_slopes_power_of_2(n):
            start = 2 ** (-2 ** -(math.log2(n) - 3))
            ratio = start
            return [start * (ratio ** i) for i in range(n)]
        
        if math.log2(self.n_heads).is_integer():
            slopes = get_slopes_power_of_2(self.n_heads)
        else:
            # 最も近い2のべき乗から補間
            closest_power_of_2 = 2 ** math.floor(math.log2(self.n_heads))
            slopes = get_slopes_power_of_2(closest_power_of_2)
            # 追加のスロープを補間
            extra_slopes = get_slopes_power_of_2(2 * closest_power_of_2)[0::2][:self.n_heads - closest_power_of_2]
            slopes.extend(extra_slopes)
        
        return torch.tensor(slopes)
    
    def explain_alibi(self):
        """ALiBiの概念を説明"""
        print("=== ALiBi (Attention with Linear Biases) ===\n")
        
        print("基本アイデア:")
        print("- 位置エンコーディングを追加せず、注意スコアに直接バイアスを加える")
        print("- バイアス = -m * |i - j| （mはヘッドごとに異なる）")
        print("- シンプルで効果的\n")
        
        print("利点:")
        print("1. 実装が非常にシンプル")
        print("2. 学習時より長い系列に自然に外挿")
        print("3. メモリ効率が良い")
        print("4. 多くのタスクで良好な性能")
    
    def create_alibi_bias(self, seq_len: int) -> torch.Tensor:
        """ALiBiバイアスを作成"""
        # 相対位置行列
        positions = torch.arange(seq_len)
        relative_positions = positions.unsqueeze(0) - positions.unsqueeze(1)
        relative_positions = -torch.abs(relative_positions).float()
        
        # 各ヘッドに異なるスロープを適用
        alibi_biases = relative_positions.unsqueeze(0) * self.slopes.view(-1, 1, 1)
        
        return alibi_biases
    
    def visualize_alibi(self):
        """ALiBiの効果を可視化"""
        seq_len = 20
        alibi_biases = self.create_alibi_bias(seq_len)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()
        
        # 各ヘッドのバイアスパターンを表示
        for head_idx in range(min(4, self.n_heads)):
            ax = axes[head_idx]
            im = ax.imshow(alibi_biases[head_idx], cmap='Blues_r', aspect='auto')
            ax.set_xlabel('Key Position')
            ax.set_ylabel('Query Position')
            ax.set_title(f'Head {head_idx} (slope={self.slopes[head_idx]:.4f})')
            plt.colorbar(im, ax=ax)
        
        plt.suptitle('ALiBi: 各ヘッドの距離ペナルティ')
        plt.tight_layout()
        plt.show()
    
    def compare_extrapolation(self):
        """外挿能力の比較"""
        print("\n=== 外挿能力の比較 ===")
        
        train_length = 512
        test_lengths = [512, 1024, 2048, 4096]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 各手法の性能（仮想的）
        methods = {
            'Sinusoidal': [100, 95, 85, 70],
            'Learned': [100, 90, 60, 30],
            'RoPE': [100, 98, 95, 90],
            'ALiBi': [100, 99, 98, 95]
        }
        
        for method, scores in methods.items():
            ax.plot(test_lengths, scores, marker='o', label=method, linewidth=2)
        
        ax.axvline(x=train_length, color='red', linestyle='--', alpha=0.5)
        ax.text(train_length + 50, 50, 'Training\nLength', color='red')
        
        ax.set_xlabel('Sequence Length')
        ax.set_ylabel('Performance (%)')
        ax.set_title('位置エンコーディング手法の外挿性能')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log', base=2)
        
        plt.tight_layout()
        plt.show()
```

## 7.5 位置エンコーディングの実装と統合

### 完全な位置エンコーディングモジュール

```python
class PositionalEncodingModule(nn.Module):
    """実用的な位置エンコーディングモジュール"""
    
    def __init__(self, 
                 d_model: int,
                 max_len: int = 5000,
                 encoding_type: str = 'sinusoidal',
                 dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.encoding_type = encoding_type
        self.dropout = nn.Dropout(dropout)
        
        if encoding_type == 'sinusoidal':
            self.pos_encoding = self._create_sinusoidal_encoding(max_len, d_model)
        elif encoding_type == 'learned':
            self.pos_encoding = nn.Embedding(max_len, d_model)
        elif encoding_type == 'rope':
            self.rope = RotaryPositionalEmbedding(d_model, max_len)
        elif encoding_type == 'alibi':
            # ALiBiは別途処理
            pass
        else:
            raise ValueError(f"Unknown encoding type: {encoding_type}")
    
    def _create_sinusoidal_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """サイン・コサイン位置エンコーディングを作成"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                            -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # バッチ次元を追加
        return pe
    
    def forward(self, x: torch.Tensor, position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
            position_ids: [batch_size, seq_len] (optional)
        Returns:
            x with positional encoding: [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape
        
        if self.encoding_type == 'sinusoidal':
            # 必要な長さだけ取得
            pos_encoding = self.pos_encoding[:, :seq_len, :].to(x.device)
            x = x + pos_encoding
            
        elif self.encoding_type == 'learned':
            if position_ids is None:
                position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
            
            pos_embeddings = self.pos_encoding(position_ids)
            x = x + pos_embeddings
            
        elif self.encoding_type == 'rope':
            # RoPEは別の方法で適用（Q, Kに対して）
            pass
        
        return self.dropout(x)
    
    def visualize_encoding_effect(self, sentence: str):
        """エンコーディングの効果を可視化"""
        words = sentence.split()
        seq_len = len(words)
        
        # ダミーの単語埋め込み
        word_embeddings = torch.randn(1, seq_len, self.d_model)
        
        # 位置エンコーディングを適用
        with torch.no_grad():
            encoded = self.forward(word_embeddings)
        
        # 元の埋め込みと比較
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 元の埋め込み
        ax = axes[0]
        im = ax.imshow(word_embeddings[0, :, :32].T, cmap='RdBu', aspect='auto')
        ax.set_xlabel('Position')
        ax.set_ylabel('Dimension')
        ax.set_title('元の単語埋め込み')
        plt.colorbar(im, ax=ax)
        
        # 位置エンコーディング
        ax = axes[1]
        if self.encoding_type == 'sinusoidal':
            pe = self.pos_encoding[0, :seq_len, :32].T
        else:
            pe = encoded[0, :, :32].T - word_embeddings[0, :, :32].T
        
        im = ax.imshow(pe, cmap='RdBu', aspect='auto')
        ax.set_xlabel('Position')
        ax.set_ylabel('Dimension')
        ax.set_title('位置エンコーディング')
        plt.colorbar(im, ax=ax)
        
        # 合計
        ax = axes[2]
        im = ax.imshow(encoded[0, :, :32].T, cmap='RdBu', aspect='auto')
        ax.set_xlabel('Position')
        ax.set_ylabel('Dimension')
        ax.set_title('位置エンコーディング適用後')
        plt.colorbar(im, ax=ax)
        
        # 単語を表示
        for ax in axes:
            ax.set_xticks(range(seq_len))
            ax.set_xticklabels(words, rotation=45, ha='right')
        
        plt.suptitle(f'{self.encoding_type.capitalize()} Encoding の効果')
        plt.tight_layout()
        plt.show()
```

## まとめ：位置情報の扱い方

この章で学んだ位置エンコーディングの要点：

1. **必要性**：Self-Attentionの位置不変性を補完
2. **手法の進化**：
   - Sinusoidal：理論的に美しく、外挿可能
   - Learned：タスク特化だが長さ制限あり
   - RoPE：相対位置を自然に表現
   - ALiBi：シンプルで効果的
3. **選択基準**：
   - タスクの性質（固定長 vs 可変長）
   - 外挿の必要性
   - 計算効率
   - 実装の複雑さ

位置エンコーディングは、Transformerが「順序」を理解するための重要な要素です。次章では、これらの要素を組み合わせて、深層学習の力を引き出す方法を学びます。

## 演習問題

1. **実装課題**：2次元位置エンコーディング（画像用）を実装してください。

2. **分析課題**：学習データの最大長が100の場合、各位置エンコーディング手法が長さ200の系列でどのように振る舞うか分析してください。

3. **比較課題**：同じタスクで異なる位置エンコーディングを使い、性能を比較してください。

4. **理論課題**：RoPEがなぜ相対位置のみに依存することを数学的に証明してください。

---

次章「層の概念と深層学習」へ続く。