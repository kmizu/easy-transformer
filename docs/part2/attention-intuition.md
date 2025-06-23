# 注意機構の直感的理解

## はじめに：「注意」とは何か

人間が文章を読むとき、すべての単語に同じ注意を払うわけではありません。重要な単語に焦点を当て、文脈に応じて関連する部分を参照しながら理解を深めていきます。

コンパイラでも似たことが起きています。変数の参照を解決するとき、スコープ内のすべての宣言を調べますが、名前が一致するものに「注目」します。型推論では、関連する制約に「注意」を向けて、矛盾のない型を導き出します。

Transformerの「Attention（注意機構）」は、この人間的な「注意」のプロセスを数学的にモデル化したものです。驚くべきことに、この単純なアイデアが、現代のAI革命の中核となっています。

## 6.1 注意機構の必要性：なぜRNNでは不十分だったのか

### 長距離依存の問題を可視化

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Optional
import math
from IPython.display import HTML
import matplotlib.animation as animation

class AttentionMotivation:
    """注意機構が必要な理由を実例で示す"""
    
    def __init__(self):
        self.examples = self._create_examples()
    
    def _create_examples(self) -> List[Dict[str, str]]:
        """長距離依存の例を作成"""
        return [
            {
                "text": "The animal didn't cross the street because it was too tired.",
                "question": "What does 'it' refer to?",
                "answer": "animal",
                "distance": 7  # "it"から"animal"までの距離
            },
            {
                "text": "The animal didn't cross the street because it was too wide.",
                "question": "What does 'it' refer to?",
                "answer": "street",
                "distance": 4  # "it"から"street"までの距離
            },
            {
                "text": "In 1969, Neil Armstrong became the first person to walk on the moon, which was a giant leap for mankind.",
                "question": "What was a giant leap?",
                "answer": "walk on the moon",
                "distance": 12
            }
        ]
    
    def visualize_dependency_problem(self):
        """依存関係の問題を可視化"""
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # 例1: RNNの情報伝播
        ax = axes[0]
        self._plot_rnn_propagation(ax)
        
        # 例2: 必要な接続
        ax = axes[1]
        self._plot_ideal_connections(ax)
        
        plt.tight_layout()
        plt.show()
    
    def _plot_rnn_propagation(self, ax):
        """RNNでの情報伝播を図示"""
        words = ["The", "animal", "didn't", "cross", "the", "street", "because", "it", "was", "too", "tired"]
        positions = list(range(len(words)))
        
        # RNNの逐次的な接続
        for i in range(len(words) - 1):
            ax.arrow(i, 0, 0.8, 0, head_width=0.1, head_length=0.1, 
                    fc='blue', ec='blue', alpha=0.5)
        
        # 単語を表示
        for i, word in enumerate(words):
            ax.text(i, -0.5, word, ha='center', va='top', fontsize=10, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        
        # "it"から"animal"への必要な接続（薄く表示）
        ax.arrow(7, 0.2, -5.5, 0, head_width=0.1, head_length=0.1,
                fc='red', ec='red', alpha=0.3, linestyle='--')
        ax.text(4.5, 0.4, "7 steps", ha='center', color='red')
        
        ax.set_xlim(-0.5, len(words) - 0.5)
        ax.set_ylim(-1, 1)
        ax.set_title("RNN: 情報は逐次的にしか伝播しない", fontsize=14)
        ax.axis('off')
    
    def _plot_ideal_connections(self, ax):
        """理想的な接続を図示"""
        words = ["The", "animal", "didn't", "cross", "the", "street", "because", "it", "was", "too", "tired"]
        positions = list(range(len(words)))
        
        # すべての単語を配置
        for i, word in enumerate(words):
            ax.text(i, 0, word, ha='center', va='center', fontsize=10,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
        
        # 重要な接続を直接描画
        connections = [
            (7, 1, "refers to"),    # it -> animal
            (7, 5, "could refer"),  # it -> street
            (10, 1, "describes"),   # tired -> animal
        ]
        
        for start, end, label in connections:
            # 曲線矢印を描画
            ax.annotate('', xy=(end, 0), xytext=(start, 0),
                       arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=0.3",
                                     color='red' if start == 7 and end == 1 else 'gray',
                                     linewidth=2 if start == 7 and end == 1 else 1))
            
            # ラベルを追加
            mid_x = (start + end) / 2
            ax.text(mid_x, 0.3, label, ha='center', fontsize=8,
                   color='red' if start == 7 and end == 1 else 'gray')
        
        ax.set_xlim(-0.5, len(words) - 0.5)
        ax.set_ylim(-1, 1)
        ax.set_title("理想: 任意の単語間で直接接続が可能", fontsize=14)
        ax.axis('off')
```

### 計算効率の比較

```python
class ComputationalEfficiency:
    """RNNとAttentionの計算効率を比較"""
    
    def compare_complexity(self):
        """計算量の比較"""
        sequence_lengths = [10, 50, 100, 500, 1000]
        
        # 理論的な計算量
        rnn_sequential = sequence_lengths  # O(n) 逐次
        rnn_parallel = [1] * len(sequence_lengths)  # 並列化不可
        
        attention_sequential = [n**2 for n in sequence_lengths]  # O(n²) 逐次
        attention_parallel = [1] * len(sequence_lengths)  # O(1) 並列時
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # 逐次計算時間
        ax1.plot(sequence_lengths, rnn_sequential, 'b-o', label='RNN', linewidth=2)
        ax1.plot(sequence_lengths, attention_sequential, 'r-s', label='Attention', linewidth=2)
        ax1.set_xlabel('Sequence Length')
        ax1.set_ylabel('Sequential Steps')
        ax1.set_title('逐次計算ステップ数')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # 並列計算の可能性
        ax2.bar(['RNN', 'Attention'], [0, 100], color=['blue', 'red'], alpha=0.7)
        ax2.set_ylabel('並列化可能率 (%)')
        ax2.set_title('並列計算の可能性')
        ax2.set_ylim(0, 120)
        
        # 注釈を追加
        ax2.text(0, 10, 'Cannot\nparallelize', ha='center', va='bottom', fontsize=12)
        ax2.text(1, 105, 'Fully\nparallelizable', ha='center', va='bottom', fontsize=12)
        
        plt.tight_layout()
        plt.show()
```

## 6.2 注意機構の直感的理解

### 人間の注意メカニズムとの類似

```python
class HumanAttentionAnalogy:
    """人間の注意メカニズムとの類似性を示す"""
    
    def __init__(self):
        self.sentence = "The quick brown fox jumps over the lazy dog"
        self.words = self.sentence.split()
    
    def visualize_human_attention(self):
        """人間がどのように文を読むかを可視化"""
        # タスクごとの注意パターン
        attention_patterns = {
            "動物を探す": {
                "fox": 0.8, "dog": 0.7, "brown": 0.3, "lazy": 0.3,
                "The": 0.1, "quick": 0.2, "jumps": 0.2, "over": 0.1, "the": 0.1
            },
            "動作を探す": {
                "jumps": 0.9, "over": 0.4,
                "quick": 0.2, "lazy": 0.1,
                "The": 0.1, "brown": 0.1, "fox": 0.2, "the": 0.1, "dog": 0.2
            },
            "形容詞を探す": {
                "quick": 0.8, "brown": 0.8, "lazy": 0.8,
                "The": 0.1, "fox": 0.2, "jumps": 0.1, "over": 0.1, "the": 0.1, "dog": 0.2
            }
        }
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        for idx, (task, attention) in enumerate(attention_patterns.items()):
            ax = axes[idx]
            
            # 単語ごとの注意の重みを可視化
            x_pos = np.arange(len(self.words))
            weights = [attention.get(word, 0.1) for word in self.words]
            
            bars = ax.bar(x_pos, weights, alpha=0.7)
            
            # 色を重みに応じて変更
            for bar, weight in zip(bars, weights):
                bar.set_color(plt.cm.Reds(weight))
            
            ax.set_xticks(x_pos)
            ax.set_xticklabels(self.words, rotation=45, ha='right')
            ax.set_ylabel('Attention Weight')
            ax.set_title(f'タスク: {task}')
            ax.set_ylim(0, 1)
            
            # 重要な単語をハイライト
            for i, (word, weight) in enumerate(zip(self.words, weights)):
                if weight > 0.5:
                    ax.annotate(f'{weight:.1f}', xy=(i, weight), 
                              xytext=(i, weight + 0.05),
                              ha='center', fontsize=10, color='red')
        
        plt.tight_layout()
        plt.show()
    
    def attention_as_similarity(self):
        """注意を類似性として理解"""
        print("=== 注意機構の本質：類似性の計算 ===\n")
        
        # クエリ（質問）
        query = "What animal?"
        
        # 文中の各単語（キー）
        keys = self.words
        
        # 簡易的な類似度計算（実際はベクトルの内積）
        similarities = {
            "The": 0.1,
            "quick": 0.2,
            "brown": 0.3,
            "fox": 0.9,  # 高い類似度
            "jumps": 0.2,
            "over": 0.1,
            "the": 0.1,
            "lazy": 0.3,
            "dog": 0.8   # 高い類似度
        }
        
        print(f"Query (質問): '{query}'")
        print("\nKey (文中の単語) との類似度:")
        for word, score in similarities.items():
            bar = "█" * int(score * 20)
            print(f"  {word:10s}: {bar} {score:.2f}")
        
        print("\n→ 'fox'と'dog'に高い注意が向けられる")
```

### 注意機構の3つの要素：Query, Key, Value

```python
class QKVExplanation:
    """Query, Key, Valueの概念を説明"""
    
    def __init__(self, d_model: int = 64):
        self.d_model = d_model
        
    def explain_qkv_concept(self):
        """QKVの概念を具体例で説明"""
        print("=== Query, Key, Value の概念 ===\n")
        
        # データベースのアナロジー
        print("1. データベース検索のアナロジー:")
        print("   - Query: 検索クエリ（何を探しているか）")
        print("   - Key: インデックス（どこを見るべきか）")
        print("   - Value: 実際のデータ（取得したい情報）")
        
        # 具体例
        print("\n2. 辞書検索の例:")
        dictionary = {
            "apple": {"meaning": "りんご", "type": "noun", "color": "red"},
            "run": {"meaning": "走る", "type": "verb", "tense": "present"},
            "happy": {"meaning": "幸せな", "type": "adjective", "emotion": "positive"}
        }
        
        query = "fruit"
        print(f"\nQuery: '{query}'を探す")
        
        # 各単語（Key）との関連性をチェック
        relevance = {
            "apple": 0.9,   # フルーツなので高い関連性
            "run": 0.1,     # 動詞なので低い関連性
            "happy": 0.1    # 形容詞なので低い関連性
        }
        
        print("\nKey との関連性:")
        for key, score in relevance.items():
            print(f"  {key}: {score:.1f}")
        
        # 関連性に基づいてValueを取得
        print("\nValue の重み付き取得:")
        for key, score in relevance.items():
            if score > 0.5:
                print(f"  {key}: {dictionary[key]} (weight: {score})")
    
    def implement_simple_attention(self):
        """シンプルな注意機構の実装"""
        print("\n=== シンプルな注意機構の実装 ===")
        
        # 例：3つの単語ベクトル
        words = ["cat", "sat", "mat"]
        d_model = 4  # 小さな次元で例示
        
        # ランダムな埋め込み（実際は学習される）
        embeddings = {
            "cat": torch.tensor([0.7, 0.2, 0.1, 0.8]),
            "sat": torch.tensor([0.1, 0.9, 0.3, 0.2]),
            "mat": torch.tensor([0.6, 0.1, 0.2, 0.9])
        }
        
        # Query, Key, Value 変換行列（実際は学習される）
        W_q = torch.randn(d_model, d_model) * 0.1
        W_k = torch.randn(d_model, d_model) * 0.1
        W_v = torch.randn(d_model, d_model) * 0.1
        
        # "cat"に対する注意を計算
        query_word = "cat"
        query_vector = embeddings[query_word]
        
        # Query変換
        Q = torch.matmul(query_vector, W_q)
        
        print(f"'{query_word}'に対する注意の計算:")
        
        attention_scores = {}
        for word in words:
            # Key変換
            K = torch.matmul(embeddings[word], W_k)
            
            # 注意スコア = Query · Key
            score = torch.dot(Q, K).item()
            attention_scores[word] = score
            
            print(f"  {word}: score = {score:.3f}")
        
        # Softmaxで正規化
        scores_tensor = torch.tensor(list(attention_scores.values()))
        attention_weights = torch.softmax(scores_tensor, dim=0)
        
        print("\nSoftmax後の注意の重み:")
        for i, word in enumerate(words):
            print(f"  {word}: {attention_weights[i]:.3f}")
        
        # 重み付き和でコンテキストベクトルを計算
        context = torch.zeros(d_model)
        for i, word in enumerate(words):
            V = torch.matmul(embeddings[word], W_v)
            context += attention_weights[i] * V
        
        print(f"\n最終的なコンテキストベクトル: {context}")
        
        return attention_weights.numpy()
```

### スケールドドット積注意の仕組み

```python
class ScaledDotProductAttention:
    """スケールドドット積注意の詳細な実装と説明"""
    
    def __init__(self):
        self.d_k = 64  # Key/Queryの次元
        
    def why_scaling_matters(self):
        """なぜスケーリングが必要かを実証"""
        print("=== スケーリングの重要性 ===")
        
        # 異なる次元での内積の分散を比較
        dimensions = [8, 32, 64, 128, 256, 512]
        variances = []
        
        for d in dimensions:
            # ランダムなベクトルを生成
            q = torch.randn(1000, d)
            k = torch.randn(1000, d)
            
            # 内積を計算
            dots = torch.sum(q * k, dim=1)
            variances.append(dots.var().item())
        
        plt.figure(figsize=(10, 6))
        plt.plot(dimensions, variances, 'b-o', linewidth=2, markersize=8)
        plt.plot(dimensions, dimensions, 'r--', label='y = x (理論値)', alpha=0.7)
        plt.xlabel('Vector Dimension (d_k)')
        plt.ylabel('Variance of Dot Product')
        plt.title('内積の分散は次元に比例して増加')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        print("\n問題：次元が大きいと...")
        print("1. 内積の値が大きくなる")
        print("2. Softmaxが極端な値（0または1）になる")
        print("3. 勾配消失が起きやすい")
        print(f"\n解決策：√d_k = √{self.d_k} = {math.sqrt(self.d_k):.1f} で割る")
    
    def implement_scaled_attention(self, 
                                 query: torch.Tensor, 
                                 key: torch.Tensor, 
                                 value: torch.Tensor,
                                 mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """スケールドドット積注意の実装"""
        
        # 入力の形状
        batch_size, seq_len, d_k = query.shape
        
        print(f"=== スケールドドット積注意の計算 ===")
        print(f"入力形状: batch_size={batch_size}, seq_len={seq_len}, d_k={d_k}")
        
        # ステップ1: Q·K^T の計算
        scores = torch.matmul(query, key.transpose(-2, -1))
        print(f"\n1. 注意スコア (Q·K^T): shape = {scores.shape}")
        
        # スケーリング前の値を記録
        pre_scale_max = scores.max().item()
        pre_scale_std = scores.std().item()
        
        # ステップ2: スケーリング
        scores = scores / math.sqrt(d_k)
        print(f"\n2. スケーリング後:")
        print(f"   スケーリング前: max={pre_scale_max:.2f}, std={pre_scale_std:.2f}")
        print(f"   スケーリング後: max={scores.max().item():.2f}, std={scores.std().item():.2f}")
        
        # ステップ3: マスク適用（オプション）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            print(f"\n3. マスク適用済み")
        
        # ステップ4: Softmax
        attention_weights = torch.softmax(scores, dim=-1)
        print(f"\n4. Softmax後の注意の重み: shape = {attention_weights.shape}")
        
        # 注意の重みの統計
        print(f"   最大重み: {attention_weights.max().item():.4f}")
        print(f"   最小重み: {attention_weights.min().item():.4f}")
        print(f"   エントロピー: {-(attention_weights * attention_weights.log()).sum(-1).mean().item():.4f}")
        
        # ステップ5: 重み付き和
        context = torch.matmul(attention_weights, value)
        print(f"\n5. コンテキストベクトル: shape = {context.shape}")
        
        return context, attention_weights
    
    def visualize_attention_computation(self):
        """注意計算の各ステップを可視化"""
        # 小さな例で可視化
        seq_len = 8
        d_k = 4
        
        # ダミーデータ
        torch.manual_seed(42)
        Q = torch.randn(1, seq_len, d_k)
        K = torch.randn(1, seq_len, d_k)
        V = torch.randn(1, seq_len, d_k)
        
        # 計算
        scores = torch.matmul(Q, K.transpose(-2, -1))[0]
        scaled_scores = scores / math.sqrt(d_k)
        attention_weights = torch.softmax(scaled_scores, dim=-1)
        
        # 可視化
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. 生のスコア
        ax = axes[0, 0]
        im1 = ax.imshow(scores.detach(), cmap='RdBu', aspect='auto')
        ax.set_title('1. 注意スコア (Q·K^T)')
        ax.set_xlabel('Key positions')
        ax.set_ylabel('Query positions')
        plt.colorbar(im1, ax=ax)
        
        # 2. スケール後のスコア
        ax = axes[0, 1]
        im2 = ax.imshow(scaled_scores.detach(), cmap='RdBu', aspect='auto')
        ax.set_title(f'2. スケール後 (÷√{d_k})')
        ax.set_xlabel('Key positions')
        ax.set_ylabel('Query positions')
        plt.colorbar(im2, ax=ax)
        
        # 3. Softmax後
        ax = axes[1, 0]
        im3 = ax.imshow(attention_weights.detach(), cmap='Blues', aspect='auto')
        ax.set_title('3. Softmax後の注意の重み')
        ax.set_xlabel('Key positions')
        ax.set_ylabel('Query positions')
        plt.colorbar(im3, ax=ax)
        
        # 4. 行ごとの分布
        ax = axes[1, 1]
        for i in range(min(4, seq_len)):  # 最初の4行を表示
            ax.plot(attention_weights[i].detach(), label=f'Query pos {i}', marker='o')
        ax.set_title('4. 注意の重み分布（各クエリ位置）')
        ax.set_xlabel('Key positions')
        ax.set_ylabel('Attention weight')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
```

## 6.3 自己注意機構：文が自分自身に注意を向ける

### Self-Attentionの動作原理

```python
class SelfAttentionMechanism:
    """自己注意機構の詳細な実装と解説"""
    
    def __init__(self, d_model: int = 256, d_k: int = 64):
        self.d_model = d_model
        self.d_k = d_k
        
        # 線形変換層
        self.W_q = nn.Linear(d_model, d_k, bias=False)
        self.W_k = nn.Linear(d_model, d_k, bias=False)
        self.W_v = nn.Linear(d_model, d_k, bias=False)
    
    def explain_self_attention(self):
        """自己注意の概念を説明"""
        print("=== 自己注意機構（Self-Attention）===\n")
        
        print("通常の注意機構:")
        print("  - Query: 質問側の文")
        print("  - Key/Value: 参照側の文")
        print("  - 例: 翻訳での原文と訳文の対応")
        
        print("\n自己注意機構:")
        print("  - Query, Key, Value: すべて同じ文から生成")
        print("  - 文中の各単語が、同じ文の他の単語に注意を向ける")
        print("  - 例: 文中の代名詞が何を指すかを理解")
        
        # 具体例で示す
        self._demonstrate_self_attention()
    
    def _demonstrate_self_attention(self):
        """自己注意の具体例"""
        sentence = "The cat sat on the mat"
        words = sentence.split()
        
        # 仮の注意パターン（実際は学習で獲得）
        attention_matrix = torch.tensor([
            # The  cat  sat  on  the  mat
            [0.8, 0.1, 0.0, 0.0, 0.0, 0.1],  # The → 自分自身とmat
            [0.2, 0.5, 0.2, 0.0, 0.0, 0.1],  # cat → 自分自身とsat
            [0.1, 0.3, 0.4, 0.1, 0.0, 0.1],  # sat → catと自分自身
            [0.0, 0.0, 0.1, 0.6, 0.0, 0.3],  # on → 自分自身とmat
            [0.0, 0.0, 0.0, 0.0, 0.7, 0.3],  # the → 自分自身とmat
            [0.1, 0.2, 0.1, 0.2, 0.1, 0.3],  # mat → 他の単語を均等に参照
        ])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(attention_matrix, 
                    xticklabels=words, 
                    yticklabels=words,
                    cmap='Blues', 
                    cbar_kws={'label': 'Attention Weight'},
                    annot=True, 
                    fmt='.1f')
        
        plt.title('Self-Attention: 各単語が文中の他の単語に向ける注意')
        plt.xlabel('Attended to (Key)')
        plt.ylabel('Attending from (Query)')
        plt.tight_layout()
        plt.show()
    
    def implement_self_attention_layer(self):
        """自己注意層の完全な実装"""
        
        class SelfAttentionLayer(nn.Module):
            def __init__(self, d_model: int, d_k: int):
                super().__init__()
                self.d_k = d_k
                
                # Q, K, V の線形変換
                self.W_q = nn.Linear(d_model, d_k, bias=False)
                self.W_k = nn.Linear(d_model, d_k, bias=False)
                self.W_v = nn.Linear(d_model, d_k, bias=False)
                
                # 初期化
                self._init_weights()
            
            def _init_weights(self):
                # Xavier初期化
                for module in [self.W_q, self.W_k, self.W_v]:
                    nn.init.xavier_uniform_(module.weight)
            
            def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
                """
                Args:
                    x: [batch_size, seq_len, d_model]
                    mask: [batch_size, seq_len, seq_len]
                Returns:
                    output: [batch_size, seq_len, d_k]
                    attention_weights: [batch_size, seq_len, seq_len]
                """
                batch_size, seq_len, d_model = x.shape
                
                # 1. 線形変換でQ, K, Vを生成
                Q = self.W_q(x)  # [batch_size, seq_len, d_k]
                K = self.W_k(x)  # [batch_size, seq_len, d_k]
                V = self.W_v(x)  # [batch_size, seq_len, d_k]
                
                # 2. 注意スコアの計算
                scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
                
                # 3. マスクの適用（オプション）
                if mask is not None:
                    scores = scores.masked_fill(mask == 0, -1e9)
                
                # 4. Softmax
                attention_weights = torch.softmax(scores, dim=-1)
                
                # 5. 重み付き和
                output = torch.matmul(attention_weights, V)
                
                return output, attention_weights
        
        return SelfAttentionLayer(self.d_model, self.d_k)
    
    def analyze_attention_patterns(self, sentence: str):
        """実際の注意パターンを分析"""
        # トークン化（簡易版）
        words = sentence.lower().split()
        seq_len = len(words)
        
        # ダミーの埋め込み（実際は学習済み埋め込みを使用）
        torch.manual_seed(42)
        embeddings = torch.randn(1, seq_len, self.d_model)
        
        # 自己注意層を作成
        self_attention = self.implement_self_attention_layer()
        
        # 注意を計算
        with torch.no_grad():
            output, attention_weights = self_attention(embeddings)
        
        # 注意パターンを可視化
        attention_matrix = attention_weights[0].numpy()
        
        plt.figure(figsize=(10, 8))
        
        # ヒートマップ
        mask = np.zeros_like(attention_matrix)
        np.fill_diagonal(mask, True)
        
        sns.heatmap(attention_matrix, 
                    xticklabels=words,
                    yticklabels=words,
                    cmap='Blues',
                    cbar_kws={'label': 'Attention Weight'},
                    mask=mask,  # 対角線をマスク（見やすさのため）
                    linewidths=0.5)
        
        # 対角線は別の色で表示
        for i in range(seq_len):
            plt.scatter(i + 0.5, i + 0.5, s=attention_matrix[i, i] * 1000, 
                       c='red', marker='s', alpha=0.5)
        
        plt.title(f'Self-Attention Pattern: "{sentence}"')
        plt.xlabel('Attended to')
        plt.ylabel('Attending from')
        
        # 重要な接続をハイライト
        threshold = attention_matrix.mean() + attention_matrix.std()
        for i in range(seq_len):
            for j in range(seq_len):
                if i != j and attention_matrix[i, j] > threshold:
                    plt.annotate('', xy=(j + 0.5, i + 0.5), 
                               xytext=(i + 0.5, i + 0.5),
                               arrowprops=dict(arrowstyle='->', 
                                             color='green', 
                                             alpha=0.5,
                                             linewidth=2))
        
        plt.tight_layout()
        plt.show()
```

### 位置情報の重要性

```python
class PositionalInformation:
    """自己注意における位置情報の重要性"""
    
    def why_position_matters(self):
        """なぜ位置情報が必要かを実証"""
        print("=== 位置情報の重要性 ===\n")
        
        # 同じ単語、異なる順序の文
        sentences = [
            "The cat chased the mouse",
            "The mouse chased the cat"
        ]
        
        print("問題：自己注意は順序を考慮しない")
        print(f"文1: {sentences[0]}")
        print(f"文2: {sentences[1]}")
        print("\n両文で 'cat' と 'mouse' の関係は同じように見える！")
        
        # バッグオブワーズ的な表現
        self._demonstrate_order_blindness()
    
    def _demonstrate_order_blindness(self):
        """順序の無視を実証"""
        # 2つの文の単語埋め込み（仮想的）
        words1 = ["The", "cat", "chased", "the", "mouse"]
        words2 = ["The", "mouse", "chased", "the", "cat"]
        
        # 単語の埋め込み（位置情報なし）
        word_embeddings = {
            "the": torch.tensor([0.1, 0.2]),
            "cat": torch.tensor([0.8, 0.3]),
            "mouse": torch.tensor([0.7, 0.4]),
            "chased": torch.tensor([0.2, 0.9])
        }
        
        # 自己注意での類似度計算
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        for ax, words, title in [(ax1, words1, "文1"), (ax2, words2, "文2")]:
            # 類似度行列
            similarity_matrix = np.zeros((len(words), len(words)))
            
            for i, word_i in enumerate(words):
                for j, word_j in enumerate(words):
                    emb_i = word_embeddings[word_i.lower()]
                    emb_j = word_embeddings[word_j.lower()]
                    similarity = torch.cosine_similarity(emb_i, emb_j, dim=0).item()
                    similarity_matrix[i, j] = similarity
            
            im = ax.imshow(similarity_matrix, cmap='Blues', aspect='auto')
            ax.set_xticks(range(len(words)))
            ax.set_yticks(range(len(words)))
            ax.set_xticklabels(words, rotation=45)
            ax.set_yticklabels(words)
            ax.set_title(f'{title}: 位置情報なしの類似度')
            
            # 類似度の値を表示
            for i in range(len(words)):
                for j in range(len(words)):
                    ax.text(j, i, f'{similarity_matrix[i, j]:.2f}', 
                           ha='center', va='center')
        
        plt.tight_layout()
        plt.show()
        
        print("\n→ 位置情報がないと、文の意味が正しく理解できない！")
```

## 6.4 注意機構の実例と応用

### 翻訳タスクでの注意

```python
class AttentionInTranslation:
    """翻訳タスクにおける注意機構の動作"""
    
    def __init__(self):
        self.examples = [
            {
                "source": "I love cats",
                "target": "私は猫が大好きです",
                "alignment": {
                    "I": ["私は"],
                    "love": ["大好きです"],
                    "cats": ["猫が"]
                }
            },
            {
                "source": "The weather is beautiful today",
                "target": "今日の天気は素晴らしいです",
                "alignment": {
                    "The weather": ["天気は"],
                    "is beautiful": ["素晴らしいです"],
                    "today": ["今日の"]
                }
            }
        ]
    
    def visualize_translation_attention(self):
        """翻訳での注意パターンを可視化"""
        for example in self.examples:
            source_words = example["source"].split()
            target_words = example["target"]
            
            # 仮想的な注意行列（実際は学習で獲得）
            attention_matrix = self._create_attention_matrix(example)
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(attention_matrix,
                       xticklabels=source_words,
                       yticklabels=list(target_words),
                       cmap='Blues',
                       cbar_kws={'label': 'Attention Weight'})
            
            plt.title(f'翻訳の注意: "{example["source"]}" → "{example["target"]}"')
            plt.xlabel('Source (English)')
            plt.ylabel('Target (Japanese)')
            plt.tight_layout()
            plt.show()
    
    def _create_attention_matrix(self, example):
        """アラインメント情報から注意行列を作成"""
        source_words = example["source"].split()
        target_chars = list(example["target"])
        
        matrix = np.zeros((len(target_chars), len(source_words)))
        
        # アラインメント情報に基づいて注意の重みを設定
        for src_word, tgt_phrases in example["alignment"].items():
            src_idx = source_words.index(src_word) if src_word in source_words else -1
            if src_idx >= 0:
                for tgt_phrase in tgt_phrases:
                    tgt_start = example["target"].find(tgt_phrase)
                    if tgt_start >= 0:
                        for i in range(tgt_start, tgt_start + len(tgt_phrase)):
                            if i < len(target_chars):
                                matrix[i, src_idx] = 1.0
        
        # 正規化
        matrix = matrix / (matrix.sum(axis=1, keepdims=True) + 1e-9)
        
        return matrix
```

### 文書要約での注意

```python
class AttentionInSummarization:
    """文書要約における注意機構の役割"""
    
    def demonstrate_extractive_attention(self):
        """抽出型要約での注意パターン"""
        document = """
        Artificial intelligence has made remarkable progress in recent years.
        Deep learning models have achieved human-level performance in many tasks.
        The transformer architecture revolutionized natural language processing.
        Attention mechanisms allow models to focus on relevant information.
        These advances have enabled applications like ChatGPT and GPT-4.
        """
        
        sentences = [s.strip() for s in document.strip().split('.') if s.strip()]
        
        # 重要度スコア（仮想的）
        importance_scores = [0.7, 0.8, 0.9, 0.8, 0.6]
        
        # 文間の注意行列（どの文が他の文を参照するか）
        cross_sentence_attention = np.array([
            [1.0, 0.3, 0.2, 0.3, 0.2],
            [0.3, 1.0, 0.4, 0.5, 0.3],
            [0.2, 0.4, 1.0, 0.6, 0.4],
            [0.3, 0.5, 0.6, 1.0, 0.3],
            [0.2, 0.3, 0.4, 0.3, 1.0]
        ])
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # 重要度スコア
        ax1.barh(range(len(sentences)), importance_scores, color='skyblue')
        ax1.set_yticks(range(len(sentences)))
        ax1.set_yticklabels([f"Sent {i+1}" for i in range(len(sentences))])
        ax1.set_xlabel('Importance Score')
        ax1.set_title('文の重要度スコア')
        ax1.set_xlim(0, 1)
        
        # 重要な文をハイライト
        for i, score in enumerate(importance_scores):
            if score > 0.75:
                ax1.barh(i, score, color='orange')
        
        # 文間の注意
        im = ax2.imshow(cross_sentence_attention, cmap='Blues', aspect='auto')
        ax2.set_xticks(range(len(sentences)))
        ax2.set_yticks(range(len(sentences)))
        ax2.set_xticklabels([f"S{i+1}" for i in range(len(sentences))])
        ax2.set_yticklabels([f"S{i+1}" for i in range(len(sentences))])
        ax2.set_title('文間の注意パターン')
        plt.colorbar(im, ax=ax2)
        
        plt.tight_layout()
        plt.show()
        
        # 要約の生成
        print("=== 抽出型要約 ===")
        print("\n元の文書:")
        for i, sent in enumerate(sentences):
            print(f"{i+1}. {sent}.")
        
        print("\n重要な文（スコア > 0.75）:")
        for i, (sent, score) in enumerate(zip(sentences, importance_scores)):
            if score > 0.75:
                print(f"- {sent}. (score: {score})")
```

## 6.5 注意機構のインタラクティブなデモ

```python
class InteractiveAttentionDemo:
    """インタラクティブな注意機構のデモンストレーション"""
    
    def __init__(self):
        self.d_model = 64
        self.attention_layer = self._create_attention_layer()
    
    def _create_attention_layer(self):
        """簡単な注意層を作成"""
        return ScaledDotProductAttention()
    
    def interactive_attention_explorer(self, sentence: str):
        """文に対する注意パターンを探索"""
        words = sentence.split()
        seq_len = len(words)
        
        # シンプルな埋め込み
        embeddings = self._get_simple_embeddings(words)
        
        # Q, K, Vを生成
        Q = embeddings
        K = embeddings
        V = embeddings
        
        # 注意を計算
        context, attention_weights = self.attention_layer.implement_scaled_attention(
            Q.unsqueeze(0), K.unsqueeze(0), V.unsqueeze(0)
        )
        
        # インタラクティブな可視化
        self._create_interactive_visualization(words, attention_weights[0])
    
    def _get_simple_embeddings(self, words: List[str]) -> torch.Tensor:
        """単語の簡易埋め込みを生成"""
        # 単語タイプに基づく簡易埋め込み
        word_types = {
            "noun": torch.tensor([1.0, 0.0, 0.0, 0.0]),
            "verb": torch.tensor([0.0, 1.0, 0.0, 0.0]),
            "adj": torch.tensor([0.0, 0.0, 1.0, 0.0]),
            "other": torch.tensor([0.0, 0.0, 0.0, 1.0])
        }
        
        # 簡易的な品詞推定
        embeddings = []
        for word in words:
            if word.lower() in ["cat", "dog", "man", "woman", "car"]:
                embeddings.append(word_types["noun"])
            elif word.lower() in ["run", "jump", "eat", "sleep", "is", "was"]:
                embeddings.append(word_types["verb"])
            elif word.lower() in ["big", "small", "red", "blue", "happy"]:
                embeddings.append(word_types["adj"])
            else:
                embeddings.append(word_types["other"])
        
        return torch.stack(embeddings) + torch.randn(len(words), 4) * 0.1
    
    def _create_interactive_visualization(self, words: List[str], attention_weights: torch.Tensor):
        """インタラクティブな注意の可視化"""
        attention_matrix = attention_weights.detach().numpy()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # ベースのヒートマップ
        im = ax.imshow(attention_matrix, cmap='Blues', aspect='auto', vmin=0, vmax=1)
        
        # 軸の設定
        ax.set_xticks(range(len(words)))
        ax.set_yticks(range(len(words)))
        ax.set_xticklabels(words, rotation=45, ha='right')
        ax.set_yticklabels(words)
        
        # グリッドライン
        ax.set_xticks(np.arange(len(words) + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(words) + 1) - 0.5, minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
        
        # 値を表示
        for i in range(len(words)):
            for j in range(len(words)):
                text = ax.text(j, i, f'{attention_matrix[i, j]:.2f}',
                             ha='center', va='center', color='black' if attention_matrix[i, j] < 0.5 else 'white')
        
        plt.colorbar(im, ax=ax, label='Attention Weight')
        plt.title('Interactive Attention Explorer')
        plt.xlabel('Attended to (Key)')
        plt.ylabel('Attending from (Query)')
        
        # 各行の最大値をハイライト
        for i in range(len(words)):
            max_idx = np.argmax(attention_matrix[i])
            rect = plt.Rectangle((max_idx - 0.45, i - 0.45), 0.9, 0.9, 
                               fill=False, edgecolor='red', linewidth=3)
            ax.add_patch(rect)
        
        plt.tight_layout()
        plt.show()
```

## まとめ：注意機構の本質

この章で学んだ注意機構の重要なポイント：

1. **動機**：長距離依存と並列計算の必要性
2. **仕組み**：Query-Key-Valueによる類似度ベースの情報集約
3. **自己注意**：文が自分自身の異なる部分に注意を向ける
4. **応用**：翻訳、要約、質問応答など幅広いタスクで有効

注意機構は、まさに「関連する情報を見つけて重み付けする」という、人間の認知プロセスを模倣したメカニズムです。次章では、この注意機構に「位置」の概念を加える位置エンコーディングについて学びます。

## 演習問題

1. **実装課題**：マスク付き自己注意を実装し、文の後半が前半しか見えないようにしてください。

2. **分析課題**：与えられた文に対して、どの単語ペアが高い注意スコアを持つか予測し、実際に計算して確認してください。

3. **応用課題**：注意機構を使って、文中の重要な単語を抽出するアルゴリズムを設計してください。

4. **理論課題**：なぜ注意スコアを√d_kで割る必要があるのか、数学的に証明してください。

---

次章「位置エンコーディング」へ続く。