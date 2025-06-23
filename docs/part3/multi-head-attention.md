# Multi-Head Attention

## はじめに：なぜ「マルチヘッド」なのか

コンパイラの最適化を考えてみましょう。単一の最適化パスですべての最適化を行うより、特定の目的に特化した複数のパス（定数畳み込み、デッドコード削除、ループ最適化など）を組み合わせる方が効果的です。各パスは異なる観点からコードを分析し、それぞれの強みを活かします。

Multi-Head Attentionも同じ発想です。単一の注意機構ではなく、複数の「ヘッド」が異なる観点から入力を分析します。あるヘッドは文法的な関係に注目し、別のヘッドは意味的な関連性を、さらに別のヘッドは長距離の依存関係を捉えるかもしれません。

この章では、なぜMulti-Head Attentionが強力なのか、そしてどのように実装されるのかを詳しく見ていきます。

## 9.1 Single-Head から Multi-Head へ

### Single-Head Attention の限界

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Optional, Union
import math
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle
from matplotlib.patches import ConnectionPatch
import matplotlib.patches as mpatches

class SingleHeadLimitations:
    """Single-Head Attentionの限界を実証"""
    
    def __init__(self, d_model: int = 512, seq_len: int = 10):
        self.d_model = d_model
        self.seq_len = seq_len
        self.single_head_attention = self._create_single_head_attention()
    
    def _create_single_head_attention(self):
        """Single-Head Attentionの実装"""
        class SingleHeadAttention(nn.Module):
            def __init__(self, d_model):
                super().__init__()
                self.d_model = d_model
                self.W_q = nn.Linear(d_model, d_model)
                self.W_k = nn.Linear(d_model, d_model)
                self.W_v = nn.Linear(d_model, d_model)
                
            def forward(self, query, key, value, mask=None):
                batch_size = query.size(0)
                
                # Q, K, V の計算
                Q = self.W_q(query)
                K = self.W_k(key)
                V = self.W_v(value)
                
                # 注意スコアの計算
                scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_model)
                
                if mask is not None:
                    scores = scores.masked_fill(mask == 0, -1e9)
                
                # 注意の重み
                attention_weights = F.softmax(scores, dim=-1)
                
                # コンテキストベクトル
                context = torch.matmul(attention_weights, V)
                
                return context, attention_weights
        
        return SingleHeadAttention(self.d_model)
    
    def demonstrate_limitations(self):
        """Single-Headの限界を実証"""
        print("=== Single-Head Attention の限界 ===\n")
        
        # テスト文
        sentence = "The cat sat on the mat while the dog played"
        words = sentence.split()
        
        # ダミーの埋め込み
        torch.manual_seed(42)
        embeddings = torch.randn(1, len(words), self.d_model)
        
        # Single-Head Attentionを適用
        with torch.no_grad():
            output, attention_weights = self.single_head_attention(
                embeddings, embeddings, embeddings
            )
        
        # 注意パターンを可視化
        self._visualize_attention_pattern(words, attention_weights[0])
        
        print("\n問題点:")
        print("1. 単一の表現しか学習できない")
        print("2. 異なる種類の関係を同時に捉えられない")
        print("3. 表現力が限定的")
    
    def _visualize_attention_pattern(self, words, attention_weights):
        """注意パターンの可視化"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # ヒートマップ
        im = ax.imshow(attention_weights.detach().numpy(), 
                      cmap='Blues', aspect='auto')
        
        # 軸の設定
        ax.set_xticks(range(len(words)))
        ax.set_yticks(range(len(words)))
        ax.set_xticklabels(words, rotation=45, ha='right')
        ax.set_yticklabels(words)
        
        # カラーバー
        plt.colorbar(im, ax=ax)
        
        ax.set_title('Single-Head Attention Pattern')
        ax.set_xlabel('Keys')
        ax.set_ylabel('Queries')
        
        plt.tight_layout()
        plt.show()
    
    def compare_attention_types(self):
        """異なるタイプの注意の必要性"""
        print("\n=== 必要な異なるタイプの注意 ===\n")
        
        # 例文での異なる関係
        sentence = "The bank by the river has a beautiful view of the old bridge"
        words = sentence.split()
        
        # 異なるタイプの関係
        relationships = {
            "文法的関係": [
                ("The", "bank", "冠詞-名詞"),
                ("has", "view", "動詞-目的語"),
                ("beautiful", "view", "形容詞-名詞")
            ],
            "意味的関係": [
                ("bank", "river", "場所の関連"),
                ("view", "bridge", "視覚的対象"),
                ("old", "bridge", "属性")
            ],
            "長距離依存": [
                ("bank", "view", "主語-目的語"),
                ("The", "bridge", "冠詞の対応")
            ]
        }
        
        # 可視化
        self._visualize_relationship_types(words, relationships)
    
    def _visualize_relationship_types(self, words, relationships):
        """異なる関係タイプの可視化"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for idx, (rel_type, relations) in enumerate(relationships.items()):
            ax = axes[idx]
            
            # 単語を配置
            positions = {}
            for i, word in enumerate(words):
                x = i % 4
                y = 3 - i // 4
                positions[word] = (x, y)
                
                # 単語を表示
                circle = Circle((x, y), 0.3, color='lightblue', ec='black')
                ax.add_patch(circle)
                ax.text(x, y, word, ha='center', va='center', fontsize=8)
            
            # 関係を矢印で表示
            colors = plt.cm.tab10(np.arange(len(relations)))
            for i, (w1, w2, label) in enumerate(relations):
                if w1 in positions and w2 in positions:
                    x1, y1 = positions[w1]
                    x2, y2 = positions[w2]
                    
                    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                               arrowprops=dict(arrowstyle='->', 
                                             color=colors[i],
                                             linewidth=2,
                                             connectionstyle="arc3,rad=0.3"))
                    
                    # ラベル
                    mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                    ax.text(mid_x, mid_y + 0.3, label, 
                           fontsize=7, ha='center',
                           bbox=dict(boxstyle="round,pad=0.3", 
                                   facecolor='white', alpha=0.8))
            
            ax.set_xlim(-0.5, 3.5)
            ax.set_ylim(-0.5, 3.5)
            ax.set_title(rel_type, fontsize=14)
            ax.axis('off')
        
        plt.suptitle('単一の注意機構では捉えきれない多様な関係', fontsize=16)
        plt.tight_layout()
        plt.show()
```

### Multi-Head の基本概念

```python
class MultiHeadConcept:
    """Multi-Head Attentionの概念説明"""
    
    def explain_multi_head_idea(self):
        """Multi-Headのアイデアを説明"""
        print("=== Multi-Head Attention の基本概念 ===\n")
        
        print("アナロジー：会議での意思決定")
        print("- Single-Head: 1人の専門家が全てを判断")
        print("- Multi-Head: 複数の専門家が異なる観点から分析\n")
        
        print("各ヘッドの特化例:")
        specialists = [
            ("Head 1", "文法専門家", "主語-動詞の一致、修飾関係"),
            ("Head 2", "意味専門家", "単語の意味的関連性"),
            ("Head 3", "文脈専門家", "長距離の文脈依存"),
            ("Head 4", "位置専門家", "単語の相対位置関係")
        ]
        
        for head, role, focus in specialists:
            print(f"{head} ({role}): {focus}")
        
        # 図解
        self._visualize_multi_head_concept()
    
    def _visualize_multi_head_concept(self):
        """Multi-Headの概念を図解"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 入力
        input_rect = FancyBboxPatch((0.1, 0.4), 0.15, 0.2,
                                   boxstyle="round,pad=0.02",
                                   facecolor='lightblue',
                                   edgecolor='black')
        ax.add_patch(input_rect)
        ax.text(0.175, 0.5, 'Input\nSequence', ha='center', va='center')
        
        # 各ヘッド
        head_colors = ['lightgreen', 'lightcoral', 'lightyellow', 'lightpink']
        head_names = ['Head 1\n(Grammar)', 'Head 2\n(Semantic)', 
                     'Head 3\n(Context)', 'Head 4\n(Position)']
        
        for i, (color, name) in enumerate(zip(head_colors, head_names)):
            y_pos = 0.7 - i * 0.15
            
            # ヘッドの矩形
            head_rect = FancyBboxPatch((0.4, y_pos), 0.2, 0.1,
                                      boxstyle="round,pad=0.02",
                                      facecolor=color,
                                      edgecolor='black')
            ax.add_patch(head_rect)
            ax.text(0.5, y_pos + 0.05, name, ha='center', va='center', fontsize=10)
            
            # 入力からの矢印
            ax.arrow(0.25, 0.5, 0.14, y_pos + 0.05 - 0.5,
                    head_width=0.02, head_length=0.01,
                    fc='gray', ec='gray')
        
        # 結合
        concat_rect = FancyBboxPatch((0.7, 0.4), 0.15, 0.2,
                                    boxstyle="round,pad=0.02",
                                    facecolor='lavender',
                                    edgecolor='black')
        ax.add_patch(concat_rect)
        ax.text(0.775, 0.5, 'Concat', ha='center', va='center')
        
        # 各ヘッドから結合への矢印
        for i in range(4):
            y_pos = 0.75 - i * 0.15
            ax.arrow(0.6, y_pos, 0.09, 0.5 - y_pos,
                    head_width=0.02, head_length=0.01,
                    fc='gray', ec='gray')
        
        # 出力
        output_rect = FancyBboxPatch((0.9, 0.4), 0.15, 0.2,
                                    boxstyle="round,pad=0.02",
                                    facecolor='lightsteelblue',
                                    edgecolor='black')
        ax.add_patch(output_rect)
        ax.text(0.975, 0.5, 'Output', ha='center', va='center')
        
        # 最終矢印
        ax.arrow(0.85, 0.5, 0.04, 0,
                head_width=0.02, head_length=0.01,
                fc='black', ec='black')
        
        ax.set_xlim(0, 1.1)
        ax.set_ylim(0.2, 0.8)
        ax.set_title('Multi-Head Attention: 複数の専門家による分析', fontsize=16)
        ax.axis('off')
        
        plt.tight_layout()
        plt.show()
```

## 9.2 Multi-Head Attention の数学的定義

### 計算の詳細

```python
class MultiHeadMathematics:
    """Multi-Head Attentionの数学的詳細"""
    
    def __init__(self, d_model: int = 512, n_heads: int = 8):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads
    
    def explain_dimensions(self):
        """次元の分割を説明"""
        print("=== Multi-Head Attention の次元 ===\n")
        
        print(f"モデル次元 (d_model): {self.d_model}")
        print(f"ヘッド数 (n_heads): {self.n_heads}")
        print(f"各ヘッドのKey/Query次元 (d_k): {self.d_k}")
        print(f"各ヘッドのValue次元 (d_v): {self.d_v}")
        
        print(f"\n重要な関係:")
        print(f"d_model = n_heads × d_k = {self.n_heads} × {self.d_k} = {self.d_model}")
        
        # 次元分割の可視化
        self._visualize_dimension_split()
    
    def _visualize_dimension_split(self):
        """次元分割の可視化"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Single-Head
        ax1.set_title('Single-Head: 全次元を使用', fontsize=14)
        single_rect = Rectangle((0, 0), self.d_model, 1,
                              facecolor='lightblue', edgecolor='black')
        ax1.add_patch(single_rect)
        ax1.text(self.d_model/2, 0.5, f'd_model = {self.d_model}',
                ha='center', va='center', fontsize=12)
        ax1.set_xlim(-10, self.d_model + 10)
        ax1.set_ylim(-0.5, 1.5)
        ax1.set_xlabel('Dimension')
        ax1.axis('off')
        
        # Multi-Head
        ax2.set_title(f'Multi-Head: {self.n_heads}個のヘッドに分割', fontsize=14)
        colors = plt.cm.tab10(np.arange(self.n_heads))
        
        for i in range(self.n_heads):
            x_start = i * self.d_k
            rect = Rectangle((x_start, 0), self.d_k, 1,
                           facecolor=colors[i], edgecolor='black',
                           alpha=0.7)
            ax2.add_patch(rect)
            ax2.text(x_start + self.d_k/2, 0.5, f'Head {i+1}\n{self.d_k}',
                    ha='center', va='center', fontsize=10)
        
        ax2.set_xlim(-10, self.d_model + 10)
        ax2.set_ylim(-0.5, 1.5)
        ax2.set_xlabel('Dimension')
        ax2.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def explain_projection_matrices(self):
        """投影行列の説明"""
        print("\n=== 投影行列の役割 ===\n")
        
        print("各ヘッドごとに独立した投影行列:")
        print(f"- W_q^i: [d_model × d_k] = [{self.d_model} × {self.d_k}]")
        print(f"- W_k^i: [d_model × d_k] = [{self.d_model} × {self.d_k}]")
        print(f"- W_v^i: [d_model × d_v] = [{self.d_model} × {self.d_v}]")
        
        print(f"\n全ヘッド合計のパラメータ数:")
        total_params = self.n_heads * (3 * self.d_model * self.d_k)
        print(f"3 × n_heads × d_model × d_k = 3 × {self.n_heads} × {self.d_model} × {self.d_k}")
        print(f"= {total_params:,} パラメータ")
        
        # 実装の効率化
        print("\n実装の効率化:")
        print("個別の行列ではなく、大きな行列として実装")
        print(f"W_Q: [{self.d_model} × {self.d_model}]")
        print(f"W_K: [{self.d_model} × {self.d_model}]")
        print(f"W_V: [{self.d_model} × {self.d_model}]")
```

### 並列計算の仕組み

```python
class ParallelComputation:
    """Multi-Head Attentionの並列計算"""
    
    def __init__(self, d_model: int = 512, n_heads: int = 8):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
    
    def demonstrate_parallel_computation(self):
        """並列計算の実演"""
        print("=== Multi-Head の並列計算 ===\n")
        
        # サンプルデータ
        batch_size = 2
        seq_len = 4
        
        # 入力
        X = torch.randn(batch_size, seq_len, self.d_model)
        
        # 投影行列
        W_Q = torch.randn(self.d_model, self.d_model)
        W_K = torch.randn(self.d_model, self.d_model)
        W_V = torch.randn(self.d_model, self.d_model)
        
        # 方法1: ループによる計算（非効率）
        print("方法1: ループによる計算")
        heads_loop = []
        for i in range(self.n_heads):
            start_idx = i * self.d_k
            end_idx = (i + 1) * self.d_k
            
            # 各ヘッドの投影
            Q_i = torch.matmul(X, W_Q[:, start_idx:end_idx])
            K_i = torch.matmul(X, W_K[:, start_idx:end_idx])
            V_i = torch.matmul(X, W_V[:, start_idx:end_idx])
            
            # 注意の計算
            scores_i = torch.matmul(Q_i, K_i.transpose(-2, -1)) / math.sqrt(self.d_k)
            weights_i = F.softmax(scores_i, dim=-1)
            head_i = torch.matmul(weights_i, V_i)
            
            heads_loop.append(head_i)
        
        output_loop = torch.cat(heads_loop, dim=-1)
        print(f"出力形状: {output_loop.shape}")
        
        # 方法2: 並列計算（効率的）
        print("\n方法2: 並列計算")
        
        # 一度に全ヘッドの投影を計算
        Q = torch.matmul(X, W_Q)  # [batch, seq, d_model]
        K = torch.matmul(X, W_K)
        V = torch.matmul(X, W_V)
        
        # 形状を変更してヘッドに分割
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        # 形状: [batch, n_heads, seq_len, d_k]
        
        # 全ヘッドで同時に注意を計算
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        weights = F.softmax(scores, dim=-1)
        heads = torch.matmul(weights, V)
        
        # 形状を戻して結合
        heads = heads.transpose(1, 2).contiguous()
        output_parallel = heads.view(batch_size, seq_len, self.d_model)
        print(f"出力形状: {output_parallel.shape}")
        
        # 計算時間の比較を可視化
        self._visualize_computation_efficiency()
    
    def _visualize_computation_efficiency(self):
        """計算効率の可視化"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # ループ方式
        ax1.set_title('ループによる逐次計算', fontsize=14)
        
        for i in range(self.n_heads):
            # 各ヘッドの計算時間
            rect = Rectangle((i * 1.2, 0), 1, 1,
                           facecolor=plt.cm.Reds((i+1)/self.n_heads),
                           edgecolor='black')
            ax1.add_patch(rect)
            ax1.text(i * 1.2 + 0.5, 0.5, f'Head {i+1}',
                    ha='center', va='center', fontsize=10)
            
            # 矢印で順序を示す
            if i < self.n_heads - 1:
                ax1.arrow(i * 1.2 + 1, 0.5, 0.15, 0,
                         head_width=0.1, head_length=0.05,
                         fc='black', ec='black')
        
        ax1.set_xlim(-0.5, self.n_heads * 1.2)
        ax1.set_ylim(-0.5, 1.5)
        ax1.set_xlabel('Time →')
        ax1.axis('off')
        
        # 並列方式
        ax2.set_title('並列計算', fontsize=14)
        
        # 全ヘッドを同時に表示
        for i in range(self.n_heads):
            rect = Rectangle((0, i * 0.15), 1, 0.12,
                           facecolor=plt.cm.Greens((i+1)/self.n_heads),
                           edgecolor='black')
            ax2.add_patch(rect)
            ax2.text(0.5, i * 0.15 + 0.06, f'Head {i+1}',
                    ha='center', va='center', fontsize=10)
        
        ax2.set_xlim(-0.5, 1.5)
        ax2.set_ylim(-0.1, self.n_heads * 0.15 + 0.1)
        ax2.set_xlabel('Time →')
        ax2.axis('off')
        
        # 時間の比較
        ax1.text(self.n_heads * 0.6, -0.3, f'総時間: {self.n_heads}T',
                ha='center', fontsize=12, color='red')
        ax2.text(0.5, -0.05, '総時間: T',
                ha='center', fontsize=12, color='green')
        
        plt.suptitle('Multi-Head Attention の計算効率', fontsize=16)
        plt.tight_layout()
        plt.show()
```

## 9.3 Multi-Head Attention の実装

### 完全な実装

```python
class MultiHeadAttentionImplementation:
    """Multi-Head Attentionの完全な実装"""
    
    def __init__(self, d_model: int = 512, n_heads: int = 8, dropout: float = 0.1):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads
        self.dropout = dropout
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
    
    def create_multi_head_attention(self) -> nn.Module:
        """Multi-Head Attentionモジュールを作成"""
        
        class MultiHeadAttention(nn.Module):
            def __init__(self, d_model, n_heads, dropout=0.1):
                super().__init__()
                self.d_model = d_model
                self.n_heads = n_heads
                self.d_k = d_model // n_heads
                self.d_v = d_model // n_heads
                
                # 線形投影層
                self.W_q = nn.Linear(d_model, d_model, bias=False)
                self.W_k = nn.Linear(d_model, d_model, bias=False)
                self.W_v = nn.Linear(d_model, d_model, bias=False)
                self.W_o = nn.Linear(d_model, d_model)
                
                # Dropout
                self.dropout = nn.Dropout(dropout)
                
                # 初期化
                self._init_weights()
            
            def _init_weights(self):
                # Xavier初期化
                for module in [self.W_q, self.W_k, self.W_v, self.W_o]:
                    nn.init.xavier_uniform_(module.weight)
                
                # 出力層のバイアスは0に
                if hasattr(self.W_o, 'bias') and self.W_o.bias is not None:
                    nn.init.constant_(self.W_o.bias, 0.)
            
            def forward(self, query, key, value, mask=None):
                """
                Args:
                    query: [batch_size, seq_len_q, d_model]
                    key: [batch_size, seq_len_k, d_model]
                    value: [batch_size, seq_len_v, d_model]
                    mask: [batch_size, n_heads, seq_len_q, seq_len_k]
                """
                batch_size = query.size(0)
                seq_len_q = query.size(1)
                seq_len_k = key.size(1)
                
                # 1. 線形投影
                Q = self.W_q(query)  # [batch_size, seq_len_q, d_model]
                K = self.W_k(key)    # [batch_size, seq_len_k, d_model]
                V = self.W_v(value)  # [batch_size, seq_len_v, d_model]
                
                # 2. ヘッドに分割
                Q = Q.view(batch_size, seq_len_q, self.n_heads, self.d_k)
                K = K.view(batch_size, seq_len_k, self.n_heads, self.d_k)
                V = V.view(batch_size, seq_len_k, self.n_heads, self.d_v)
                
                # 3. 転置してヘッドを前に
                Q = Q.transpose(1, 2)  # [batch_size, n_heads, seq_len_q, d_k]
                K = K.transpose(1, 2)  # [batch_size, n_heads, seq_len_k, d_k]
                V = V.transpose(1, 2)  # [batch_size, n_heads, seq_len_v, d_v]
                
                # 4. スケールドドット積注意
                attention_output, attention_weights = self.scaled_dot_product_attention(
                    Q, K, V, mask
                )
                
                # 5. ヘッドを結合
                attention_output = attention_output.transpose(1, 2).contiguous()
                attention_output = attention_output.view(
                    batch_size, seq_len_q, self.d_model
                )
                
                # 6. 最終線形層
                output = self.W_o(attention_output)
                
                return output, attention_weights
            
            def scaled_dot_product_attention(self, Q, K, V, mask=None):
                """スケールドドット積注意の計算"""
                # スコアの計算
                scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
                
                # マスクの適用
                if mask is not None:
                    scores = scores.masked_fill(mask == 0, -1e9)
                
                # Softmax
                attention_weights = F.softmax(scores, dim=-1)
                attention_weights = self.dropout(attention_weights)
                
                # 値の重み付き和
                context = torch.matmul(attention_weights, V)
                
                return context, attention_weights
        
        return MultiHeadAttention(self.d_model, self.n_heads, self.dropout)
    
    def test_implementation(self):
        """実装のテスト"""
        print("=== Multi-Head Attention の実装テスト ===\n")
        
        # モジュールの作成
        mha = self.create_multi_head_attention()
        
        # テストデータ
        batch_size = 2
        seq_len = 5
        x = torch.randn(batch_size, seq_len, self.d_model)
        
        # Forward pass
        output, attention_weights = mha(x, x, x)
        
        print(f"入力形状: {x.shape}")
        print(f"出力形状: {output.shape}")
        print(f"注意の重み形状: {attention_weights.shape}")
        
        # 各ヘッドの注意パターンを可視化
        self._visualize_head_patterns(attention_weights[0])
    
    def _visualize_head_patterns(self, attention_weights):
        """各ヘッドの注意パターンを可視化"""
        n_heads = attention_weights.shape[0]
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.ravel()
        
        for head in range(n_heads):
            ax = axes[head]
            im = ax.imshow(attention_weights[head].detach().numpy(),
                          cmap='Blues', aspect='auto')
            ax.set_title(f'Head {head + 1}')
            ax.set_xlabel('Key positions')
            ax.set_ylabel('Query positions')
            
            # カラーバーは最初と最後のヘッドのみ
            if head == 0 or head == n_heads - 1:
                plt.colorbar(im, ax=ax, fraction=0.046)
        
        plt.suptitle('各ヘッドの注意パターン', fontsize=16)
        plt.tight_layout()
        plt.show()
```

### マスクの実装

```python
class AttentionMasking:
    """注意マスクの実装と理解"""
    
    def explain_mask_types(self):
        """マスクの種類を説明"""
        print("=== 注意マスクの種類 ===\n")
        
        mask_types = {
            "Padding Mask": {
                "目的": "パディングトークンへの注意を防ぐ",
                "使用場所": "エンコーダ、デコーダ両方",
                "形状": "[batch_size, seq_len]"
            },
            "Look-ahead Mask": {
                "目的": "未来の情報への注意を防ぐ（自己回帰）",
                "使用場所": "デコーダの自己注意",
                "形状": "[seq_len, seq_len]（下三角行列）"
            },
            "Cross-attention Mask": {
                "目的": "エンコーダ出力の特定部分への注意を制御",
                "使用場所": "デコーダのクロス注意",
                "形状": "[tgt_len, src_len]"
            }
        }
        
        for mask_name, properties in mask_types.items():
            print(f"{mask_name}:")
            for key, value in properties.items():
                print(f"  {key}: {value}")
            print()
    
    def create_masks(self, seq_len: int = 6):
        """各種マスクを作成"""
        
        # 1. Padding Mask
        # 実際のトークン長を仮定
        actual_lengths = [4, 6]  # バッチ内の各系列の実際の長さ
        padding_mask = self._create_padding_mask(seq_len, actual_lengths)
        
        # 2. Look-ahead Mask
        look_ahead_mask = self._create_look_ahead_mask(seq_len)
        
        # 3. Combined Mask（デコーダ用）
        combined_mask = self._combine_masks(padding_mask[0], look_ahead_mask)
        
        # 可視化
        self._visualize_masks(padding_mask[0], look_ahead_mask, combined_mask)
        
        return padding_mask, look_ahead_mask, combined_mask
    
    def _create_padding_mask(self, seq_len: int, actual_lengths: List[int]):
        """パディングマスクを作成"""
        batch_size = len(actual_lengths)
        mask = torch.zeros(batch_size, seq_len)
        
        for i, length in enumerate(actual_lengths):
            if length < seq_len:
                mask[i, length:] = 1  # パディング位置を1に
        
        return mask
    
    def _create_look_ahead_mask(self, seq_len: int):
        """Look-aheadマスクを作成"""
        # 下三角行列（対角線含む）が1
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        return mask
    
    def _combine_masks(self, padding_mask, look_ahead_mask):
        """マスクを結合"""
        # padding_maskをブロードキャスト
        padding_mask_expanded = padding_mask.unsqueeze(0).expand(
            look_ahead_mask.shape[0], -1
        )
        
        # 論理和（どちらかが1なら1）
        combined = torch.maximum(padding_mask_expanded, look_ahead_mask)
        return combined
    
    def _visualize_masks(self, padding_mask, look_ahead_mask, combined_mask):
        """マスクを可視化"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        masks = [
            (padding_mask, "Padding Mask"),
            (look_ahead_mask, "Look-ahead Mask"),
            (combined_mask, "Combined Mask")
        ]
        
        for ax, (mask, title) in zip(axes, masks):
            # マスクを可視化（1が黒、0が白）
            im = ax.imshow(mask, cmap='binary', aspect='auto')
            ax.set_title(title, fontsize=14)
            ax.set_xlabel('Position')
            ax.set_ylabel('Position')
            
            # グリッド
            ax.set_xticks(np.arange(mask.shape[-1]))
            ax.set_yticks(np.arange(mask.shape[0]))
            ax.grid(True, alpha=0.3)
            
            # 値を表示
            for i in range(mask.shape[0]):
                for j in range(mask.shape[1]):
                    ax.text(j, i, int(mask[i, j].item()),
                           ha='center', va='center',
                           color='white' if mask[i, j] > 0.5 else 'black')
        
        plt.suptitle('注意マスクの種類（黒=マスク、白=注意可能）', fontsize=16)
        plt.tight_layout()
        plt.show()
```

## 9.4 各ヘッドが学習する表現

### ヘッドの特化を分析

```python
class HeadSpecialization:
    """各ヘッドの特化を分析"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        if model_path:
            # 実際のモデルをロード
            pass
        else:
            # デモ用の仮想的なパターン
            self.create_demo_patterns()
    
    def create_demo_patterns(self):
        """デモ用の注意パターンを作成"""
        self.demo_patterns = {
            "positional": self._create_positional_pattern,
            "syntactic": self._create_syntactic_pattern,
            "semantic": self._create_semantic_pattern,
            "broad": self._create_broad_pattern
        }
    
    def _create_positional_pattern(self, seq_len: int = 10):
        """位置的なパターン（隣接する単語に注目）"""
        pattern = torch.zeros(seq_len, seq_len)
        for i in range(seq_len):
            for j in range(max(0, i-1), min(seq_len, i+2)):
                pattern[i, j] = 1.0 if i != j else 0.5
        
        # 正規化
        pattern = pattern / pattern.sum(dim=-1, keepdim=True)
        return pattern
    
    def _create_syntactic_pattern(self, seq_len: int = 10):
        """文法的パターン（特定の構造に注目）"""
        pattern = torch.eye(seq_len) * 0.3
        
        # 動詞位置（仮定）から主語・目的語への注目
        verb_positions = [2, 6]
        for verb_pos in verb_positions:
            if verb_pos < seq_len:
                # 主語（前方）への注目
                if verb_pos > 0:
                    pattern[verb_pos, verb_pos-1] = 0.4
                # 目的語（後方）への注目
                if verb_pos < seq_len - 1:
                    pattern[verb_pos, verb_pos+1] = 0.3
        
        # 正規化
        pattern = pattern / pattern.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        return pattern
    
    def _create_semantic_pattern(self, seq_len: int = 10):
        """意味的パターン（関連する単語に注目）"""
        pattern = torch.rand(seq_len, seq_len) * 0.3
        pattern = pattern + pattern.T  # 対称性
        pattern = pattern / pattern.sum(dim=-1, keepdim=True)
        return pattern
    
    def _create_broad_pattern(self, seq_len: int = 10):
        """広範なパターン（全体的な文脈）"""
        pattern = torch.ones(seq_len, seq_len) / seq_len
        # 自己への注目を少し強める
        pattern = pattern + torch.eye(seq_len) * 0.1
        pattern = pattern / pattern.sum(dim=-1, keepdim=True)
        return pattern
    
    def analyze_head_patterns(self):
        """ヘッドパターンの分析"""
        print("=== 各ヘッドの特化パターン ===\n")
        
        seq_len = 10
        words = ["The", "cat", "sat", "on", "the", "mat", "and", "looked", "around", "."]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.ravel()
        
        patterns = [
            ("位置的注意", self._create_positional_pattern(seq_len)),
            ("文法的注意", self._create_syntactic_pattern(seq_len)),
            ("意味的注意", self._create_semantic_pattern(seq_len)),
            ("広範な注意", self._create_broad_pattern(seq_len))
        ]
        
        for idx, (name, pattern) in enumerate(patterns):
            ax = axes[idx]
            
            im = ax.imshow(pattern.numpy(), cmap='Blues', aspect='auto')
            ax.set_title(name, fontsize=14)
            ax.set_xlabel('Attended to')
            ax.set_ylabel('Attending from')
            
            # 単語を表示
            ax.set_xticks(range(seq_len))
            ax.set_yticks(range(seq_len))
            ax.set_xticklabels(words, rotation=45, ha='right')
            ax.set_yticklabels(words)
            
            plt.colorbar(im, ax=ax)
        
        plt.suptitle('Multi-Head Attention: 各ヘッドの特化したパターン', fontsize=16)
        plt.tight_layout()
        plt.show()
        
        # パターンの説明
        self._explain_patterns()
    
    def _explain_patterns(self):
        """各パターンの説明"""
        print("\n各パターンの特徴:")
        print("\n1. 位置的注意:")
        print("   - 近隣の単語に注目")
        print("   - 局所的な文脈を捉える")
        print("   - n-gramのような特徴")
        
        print("\n2. 文法的注意:")
        print("   - 文法的な依存関係に注目")
        print("   - 主語-動詞、動詞-目的語など")
        print("   - 構文解析的な情報")
        
        print("\n3. 意味的注意:")
        print("   - 意味的に関連する単語に注目")
        print("   - 離れた位置でも関連があれば注目")
        print("   - 文脈理解に重要")
        
        print("\n4. 広範な注意:")
        print("   - 文全体を均等に見る")
        print("   - グローバルな文脈")
        print("   - 文の要約的な情報")
```

### 実際の学習パターン

```python
class LearnedPatterns:
    """実際に学習されるパターンの分析"""
    
    def demonstrate_layer_wise_patterns(self):
        """層ごとのパターンの変化"""
        print("=== 層による注意パターンの変化 ===\n")
        
        layers = ["Layer 1", "Layer 6", "Layer 12"]
        layer_characteristics = [
            "表層的・位置的パターン",
            "文法的・構造的パターン",
            "意味的・抽象的パターン"
        ]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for idx, (layer, characteristic) in enumerate(zip(layers, layer_characteristics)):
            ax = axes[idx]
            
            # 層が深くなるにつれてパターンが変化
            if idx == 0:  # 浅い層
                pattern = self._create_shallow_pattern()
            elif idx == 1:  # 中間層
                pattern = self._create_middle_pattern()
            else:  # 深い層
                pattern = self._create_deep_pattern()
            
            im = ax.imshow(pattern, cmap='Purples', aspect='auto')
            ax.set_title(f'{layer}\n{characteristic}', fontsize=12)
            plt.colorbar(im, ax=ax)
        
        plt.suptitle('Transformerの層による注意パターンの進化', fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def _create_shallow_pattern(self):
        """浅い層のパターン（位置的）"""
        size = 8
        pattern = torch.zeros(size, size)
        
        # 対角線付近に強い注意
        for i in range(size):
            for j in range(size):
                distance = abs(i - j)
                pattern[i, j] = np.exp(-distance * 0.5)
        
        return pattern / pattern.sum(dim=-1, keepdim=True)
    
    def _create_middle_pattern(self):
        """中間層のパターン（構造的）"""
        size = 8
        pattern = torch.zeros(size, size)
        
        # ブロック的なパターン
        pattern[0:3, 0:3] = 0.3  # 句単位
        pattern[3:6, 3:6] = 0.3
        pattern[6:8, 6:8] = 0.3
        
        # 長距離の接続
        pattern[1, 5] = 0.5  # 文法的依存
        pattern[5, 1] = 0.5
        
        return pattern / pattern.sum(dim=-1, keepdim=True).clamp(min=1e-9)
    
    def _create_deep_pattern(self):
        """深い層のパターン（意味的）"""
        size = 8
        
        # より複雑で抽象的なパターン
        pattern = torch.rand(size, size) * 0.5 + 0.1
        
        # 特定の意味的関連を強調
        semantic_pairs = [(0, 4), (1, 6), (2, 7)]
        for i, j in semantic_pairs:
            pattern[i, j] = 0.8
            pattern[j, i] = 0.8
        
        return pattern / pattern.sum(dim=-1, keepdim=True)
```

## 9.5 Multi-Head Attention の最適化

### 効率的な実装

```python
class EfficientMultiHeadAttention:
    """効率的なMulti-Head Attentionの実装"""
    
    def explain_optimizations(self):
        """最適化手法の説明"""
        print("=== Multi-Head Attention の最適化 ===\n")
        
        optimizations = {
            "Fused Operations": {
                "説明": "複数の演算を1つのカーネルで実行",
                "効果": "メモリアクセスの削減",
                "例": "QKV投影を1つの行列演算に統合"
            },
            "Flash Attention": {
                "説明": "タイル化とメモリ効率的なアルゴリズム",
                "効果": "O(n²)→O(n)のメモリ使用量",
                "例": "長い系列でも効率的"
            },
            "Sparse Attention": {
                "説明": "注意行列をスパースに",
                "効果": "計算量をO(n²)→O(n log n)に削減",
                "例": "局所的+ストライドパターン"
            },
            "Low-rank Approximation": {
                "説明": "注意行列を低ランク近似",
                "効果": "パラメータ数と計算量の削減",
                "例": "Linformer, Performer"
            }
        }
        
        for name, details in optimizations.items():
            print(f"{name}:")
            for key, value in details.items():
                print(f"  {key}: {value}")
            print()
    
    def implement_grouped_query_attention(self):
        """Grouped Query Attention (GQA) の実装"""
        print("\n=== Grouped Query Attention ===")
        print("Key/Valueのヘッド数を削減して効率化\n")
        
        class GroupedQueryAttention(nn.Module):
            def __init__(self, d_model: int, n_heads: int, n_kv_heads: int):
                super().__init__()
                assert n_heads % n_kv_heads == 0
                
                self.d_model = d_model
                self.n_heads = n_heads
                self.n_kv_heads = n_kv_heads
                self.n_groups = n_heads // n_kv_heads
                self.d_k = d_model // n_heads
                
                # Q は全ヘッド分、K/V は削減されたヘッド数
                self.W_q = nn.Linear(d_model, d_model)
                self.W_k = nn.Linear(d_model, n_kv_heads * self.d_k)
                self.W_v = nn.Linear(d_model, n_kv_heads * self.d_k)
                self.W_o = nn.Linear(d_model, d_model)
                
                print(f"通常のMHA: {3 * d_model * d_model} パラメータ")
                print(f"GQA: {d_model * d_model + 2 * d_model * n_kv_heads * self.d_k} パラメータ")
                print(f"削減率: {1 - (1 + 2 * n_kv_heads / n_heads) / 3:.1%}")
            
            def forward(self, x):
                batch_size, seq_len = x.shape[:2]
                
                # Query は全ヘッド
                Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k)
                Q = Q.transpose(1, 2)
                
                # Key/Value は削減されたヘッド数
                K = self.W_k(x).view(batch_size, seq_len, self.n_kv_heads, self.d_k)
                K = K.transpose(1, 2)
                V = self.W_v(x).view(batch_size, seq_len, self.n_kv_heads, self.d_k)
                V = V.transpose(1, 2)
                
                # K/V を繰り返してQのヘッド数に合わせる
                K = K.repeat_interleave(self.n_groups, dim=1)
                V = V.repeat_interleave(self.n_groups, dim=1)
                
                # 通常の注意計算
                scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
                weights = F.softmax(scores, dim=-1)
                context = torch.matmul(weights, V)
                
                # 出力
                context = context.transpose(1, 2).contiguous()
                context = context.view(batch_size, seq_len, self.d_model)
                output = self.W_o(context)
                
                return output
        
        # 例
        gqa = GroupedQueryAttention(d_model=512, n_heads=8, n_kv_heads=2)
        return gqa
```

## まとめ：Multi-Head Attention の威力

この章で学んだMulti-Head Attentionの重要なポイント：

1. **複数視点の統合**：
   - 各ヘッドが異なる表現を学習
   - 位置的、文法的、意味的な情報を並列に処理
   - 単一ヘッドでは不可能な豊かな表現

2. **効率的な並列計算**：
   - すべてのヘッドを同時に計算
   - 行列演算による高速化
   - GPUに最適化された実装

3. **柔軟な注意パターン**：
   - 層が深くなるにつれて抽象度が上昇
   - タスクに応じた特化
   - マスクによる制御

4. **最新の最適化**：
   - Flash Attention による省メモリ化
   - Grouped Query Attention によるパラメータ削減
   - スパース化による高速化

Multi-Head Attentionは、Transformerの表現力の源泉です。次章では、この豊かな表現をさらに変換するFeed-Forward Networkについて学びます。

## 演習問題

1. **実装課題**：8ヘッドのMulti-Head Attentionを実装し、各ヘッドが異なるパターンを学習することを確認してください。

2. **分析課題**：事前学習済みモデルから注意の重みを抽出し、各層・各ヘッドがどのような情報に注目しているか分析してください。

3. **最適化課題**：Sparse Attentionパターンを実装し、通常のAttentionと性能を比較してください。

4. **理論課題**：なぜd_modelをn_headsで割り切れる必要があるのか、数学的に説明してください。

---

次章「Feed Forward Network」へ続く。