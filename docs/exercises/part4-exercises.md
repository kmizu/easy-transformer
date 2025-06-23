# 第4部 演習問題

## 演習 4.1: 最小限のTransformer実装

### 問題 1
位置エンコーディングを実装し、異なる次元での周期性を可視化してください。

??? 解答
    ```python
    import torch
    import torch.nn as nn
    import matplotlib.pyplot as plt
    import numpy as np
    
    class PositionalEncoding(nn.Module):
        def __init__(self, d_model=512, max_len=5000):
            super().__init__()
            
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            
            # 周波数の計算
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                               (-math.log(10000.0) / d_model))
            
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            
            self.register_buffer('pe', pe.unsqueeze(0).transpose(0, 1))
            
        def forward(self, x):
            return x + self.pe[:x.size(0), :]
    
    # 可視化
    pos_enc = PositionalEncoding(d_model=128, max_len=100)
    pe_data = pos_enc.pe.squeeze(1).numpy()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # 異なる次元での周期性
    dimensions = [0, 1, 10, 50]
    for i, dim in enumerate(dimensions):
        ax = axes[i//2, i%2]
        ax.plot(pe_data[:100, dim])
        ax.set_title(f'Dimension {dim} ({"sin" if dim%2==0 else "cos"})')
        ax.set_xlabel('Position')
        ax.set_ylabel('Value')
        ax.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # ヒートマップで全体像
    plt.figure(figsize=(15, 8))
    plt.imshow(pe_data[:100, :64].T, cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.title('Positional Encoding Heatmap')
    plt.xlabel('Position')
    plt.ylabel('Dimension')
    plt.show()
    ```

### 問題 2
シンプルなTransformerエンコーダを実装し、段階的に複雑にしてください。

??? 解答
    ```python
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import math
    
    class MinimalTransformerEncoder(nn.Module):
        """最小限のTransformerエンコーダ"""
        
        def __init__(self, vocab_size=1000, d_model=256, n_heads=8, 
                     n_layers=4, d_ff=1024, max_len=512, dropout=0.1):
            super().__init__()
            
            self.d_model = d_model
            
            # 埋め込み層
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.pos_encoding = PositionalEncoding(d_model, max_len)
            
            # Transformerブロック
            self.transformer_blocks = nn.ModuleList([
                TransformerBlock(d_model, n_heads, d_ff, dropout)
                for _ in range(n_layers)
            ])
            
            self.dropout = nn.Dropout(dropout)
            self.layer_norm = nn.LayerNorm(d_model)
            
        def forward(self, x, mask=None):
            # 埋め込み + 位置エンコーディング
            x = self.embedding(x) * math.sqrt(self.d_model)
            x = self.pos_encoding(x)
            x = self.dropout(x)
            
            # Transformerブロックを順次適用
            for block in self.transformer_blocks:
                x = block(x, mask)
            
            return self.layer_norm(x)
    
    class TransformerBlock(nn.Module):
        """単一のTransformerブロック"""
        
        def __init__(self, d_model, n_heads, d_ff, dropout):
            super().__init__()
            
            self.attention = MultiHeadAttention(d_model, n_heads, dropout)
            self.feed_forward = FeedForward(d_model, d_ff, dropout)
            
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.dropout = nn.Dropout(dropout)
            
        def forward(self, x, mask=None):
            # Self-Attention + 残差接続
            attn_out = self.attention(x, x, x, mask)
            x = self.norm1(x + self.dropout(attn_out))
            
            # Feed Forward + 残差接続
            ff_out = self.feed_forward(x)
            x = self.norm2(x + self.dropout(ff_out))
            
            return x
    
    class MultiHeadAttention(nn.Module):
        def __init__(self, d_model, n_heads, dropout=0.1):
            super().__init__()
            assert d_model % n_heads == 0
            
            self.d_model = d_model
            self.n_heads = n_heads
            self.d_k = d_model // n_heads
            
            self.W_q = nn.Linear(d_model, d_model)
            self.W_k = nn.Linear(d_model, d_model)
            self.W_v = nn.Linear(d_model, d_model)
            self.W_o = nn.Linear(d_model, d_model)
            
            self.dropout = nn.Dropout(dropout)
            
        def forward(self, query, key, value, mask=None):
            batch_size = query.size(0)
            
            # Q, K, V の計算
            Q = self.W_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
            K = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
            V = self.W_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
            
            # Scaled Dot-Product Attention
            attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
            
            # ヘッドを結合
            attn_output = attn_output.transpose(1, 2).contiguous().view(
                batch_size, -1, self.d_model)
            
            return self.W_o(attn_output)
        
        def scaled_dot_product_attention(self, Q, K, V, mask=None):
            d_k = Q.size(-1)
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
            
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
            
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            return torch.matmul(attn_weights, V)
    
    class FeedForward(nn.Module):
        def __init__(self, d_model, d_ff, dropout=0.1):
            super().__init__()
            self.linear1 = nn.Linear(d_model, d_ff)
            self.linear2 = nn.Linear(d_ff, d_model)
            self.dropout = nn.Dropout(dropout)
            
        def forward(self, x):
            return self.linear2(self.dropout(F.relu(self.linear1(x))))
    
    # テスト
    model = MinimalTransformerEncoder(vocab_size=1000, d_model=256)
    x = torch.randint(0, 1000, (2, 50))  # バッチサイズ2, 系列長50
    output = model(x)
    print(f"Output shape: {output.shape}")  # [2, 50, 256]
    ```

## 演習 4.2: コンポーネント実装の詳細

### 問題 3
異なる注意機構（additive attention, scaled dot-product attention）を実装し、性能を比較してください。

??? 解答
    ```python
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import time
    import matplotlib.pyplot as plt
    
    class AdditiveAttention(nn.Module):
        """Additive Attention (Bahdanau Attention)"""
        
        def __init__(self, hidden_size):
            super().__init__()
            self.hidden_size = hidden_size
            self.W_q = nn.Linear(hidden_size, hidden_size, bias=False)
            self.W_k = nn.Linear(hidden_size, hidden_size, bias=False)
            self.v = nn.Linear(hidden_size, 1, bias=False)
            
        def forward(self, query, key, value, mask=None):
            # query: [batch, seq_len_q, hidden]
            # key, value: [batch, seq_len_k, hidden]
            
            batch_size, seq_len_q, _ = query.shape
            seq_len_k = key.shape[1]
            
            # クエリとキーを結合するため次元を拡張
            q_transformed = self.W_q(query).unsqueeze(2)  # [batch, seq_len_q, 1, hidden]
            k_transformed = self.W_k(key).unsqueeze(1)    # [batch, 1, seq_len_k, hidden]
            
            # ブロードキャストで結合
            combined = torch.tanh(q_transformed + k_transformed)  # [batch, seq_len_q, seq_len_k, hidden]
            
            # スコア計算
            scores = self.v(combined).squeeze(-1)  # [batch, seq_len_q, seq_len_k]
            
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
            
            attn_weights = F.softmax(scores, dim=-1)
            
            # 重み付け和
            context = torch.bmm(attn_weights, value)  # [batch, seq_len_q, hidden]
            
            return context, attn_weights
    
    class ScaledDotProductAttention(nn.Module):
        """Scaled Dot-Product Attention"""
        
        def __init__(self, hidden_size, dropout=0.1):
            super().__init__()
            self.hidden_size = hidden_size
            self.dropout = nn.Dropout(dropout)
            
        def forward(self, query, key, value, mask=None):
            d_k = query.size(-1)
            
            # 内積計算
            scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
            
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
            
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            context = torch.matmul(attn_weights, value)
            
            return context, attn_weights
    
    def benchmark_attention_mechanisms():
        """注意機構の性能比較"""
        
        hidden_size = 256
        batch_size = 32
        seq_lengths = [64, 128, 256, 512]
        
        additive_attn = AdditiveAttention(hidden_size)
        scaled_attn = ScaledDotProductAttention(hidden_size)
        
        results = {'additive': [], 'scaled': []}
        
        for seq_len in seq_lengths:
            # ダミーデータ生成
            query = torch.randn(batch_size, seq_len, hidden_size)
            key = torch.randn(batch_size, seq_len, hidden_size)
            value = torch.randn(batch_size, seq_len, hidden_size)
            
            # Additive Attention
            start_time = time.time()
            for _ in range(10):
                _, _ = additive_attn(query, key, value)
            additive_time = (time.time() - start_time) / 10
            results['additive'].append(additive_time)
            
            # Scaled Dot-Product Attention
            start_time = time.time()
            for _ in range(10):
                _, _ = scaled_attn(query, key, value)
            scaled_time = (time.time() - start_time) / 10
            results['scaled'].append(scaled_time)
            
            print(f"Seq Length {seq_len}:")
            print(f"  Additive: {additive_time:.4f}s")
            print(f"  Scaled: {scaled_time:.4f}s")
            print(f"  Speedup: {additive_time/scaled_time:.2f}x")
            print()
        
        # 可視化
        plt.figure(figsize=(10, 6))
        plt.plot(seq_lengths, results['additive'], 'ro-', label='Additive Attention')
        plt.plot(seq_lengths, results['scaled'], 'bo-', label='Scaled Dot-Product')
        plt.xlabel('Sequence Length')
        plt.ylabel('Time (seconds)')
        plt.title('Attention Mechanism Performance Comparison')
        plt.legend()
        plt.grid(True)
        plt.yscale('log')
        plt.show()
        
        return results
    
    # ベンチマーク実行
    results = benchmark_attention_mechanisms()
    ```

### 問題 4
層正規化とバッチ正規化の違いを実装を通して理解してください。

??? 解答
    ```python
    import torch
    import torch.nn as nn
    import numpy as np
    import matplotlib.pyplot as plt
    
    class LayerNorm(nn.Module):
        """Layer Normalization の手動実装"""
        
        def __init__(self, features, eps=1e-6):
            super().__init__()
            self.gamma = nn.Parameter(torch.ones(features))
            self.beta = nn.Parameter(torch.zeros(features))
            self.eps = eps
            
        def forward(self, x):
            # 最後の次元で正規化
            mean = x.mean(dim=-1, keepdim=True)
            std = x.std(dim=-1, keepdim=True, unbiased=False)
            
            return self.gamma * (x - mean) / (std + self.eps) + self.beta
    
    class BatchNorm1D(nn.Module):
        """Batch Normalization の手動実装"""
        
        def __init__(self, features, eps=1e-5, momentum=0.1):
            super().__init__()
            self.gamma = nn.Parameter(torch.ones(features))
            self.beta = nn.Parameter(torch.zeros(features))
            self.eps = eps
            self.momentum = momentum
            
            # 実行時統計
            self.register_buffer('running_mean', torch.zeros(features))
            self.register_buffer('running_var', torch.ones(features))
            
        def forward(self, x):
            if self.training:
                # バッチ次元で正規化
                mean = x.mean(dim=0)
                var = x.var(dim=0, unbiased=False)
                
                # 実行時統計を更新
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
            else:
                mean = self.running_mean
                var = self.running_var
            
            return self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta
    
    def compare_normalizations():
        """正規化手法の比較"""
        
        # ダミーデータ生成
        batch_size, seq_len, features = 32, 50, 128
        x = torch.randn(batch_size, seq_len, features) * 2 + 1
        
        # 正規化手法
        layer_norm = LayerNorm(features)
        batch_norm = BatchNorm1D(features)
        
        # Layer Normalization適用
        x_ln = layer_norm(x)
        
        # Batch Normalization適用（2D入力に変換）
        x_2d = x.view(-1, features)
        x_bn_2d = batch_norm(x_2d)
        x_bn = x_bn_2d.view(batch_size, seq_len, features)
        
        # 統計情報を計算
        print("=== 正規化前 ===")
        print(f"Mean: {x.mean():.4f}, Std: {x.std():.4f}")
        print(f"Shape: {x.shape}")
        
        print("\\n=== Layer Normalization後 ===")
        print(f"Mean: {x_ln.mean():.4f}, Std: {x_ln.std():.4f}")
        # 各サンプルの各時刻での統計
        print(f"Per-sample mean: {x_ln.mean(dim=-1).mean():.4f}")
        print(f"Per-sample std: {x_ln.std(dim=-1).mean():.4f}")
        
        print("\\n=== Batch Normalization後 ===")
        print(f"Mean: {x_bn.mean():.4f}, Std: {x_bn.std():.4f}")
        # 各特徴次元での統計
        print(f"Per-feature mean: {x_bn.mean(dim=(0,1)).mean():.4f}")
        print(f"Per-feature std: {x_bn.mean(dim=(0,1)).std():.4f}")
        
        # 可視化
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # オリジナル
        axes[0, 0].hist(x.flatten().numpy(), bins=50, alpha=0.7, color='red')
        axes[0, 0].set_title('Original Distribution')
        axes[0, 0].set_ylabel('Frequency')
        
        # Layer Norm
        axes[0, 1].hist(x_ln.flatten().numpy(), bins=50, alpha=0.7, color='blue')
        axes[0, 1].set_title('After Layer Normalization')
        
        # Batch Norm
        axes[0, 2].hist(x_bn.flatten().numpy(), bins=50, alpha=0.7, color='green')
        axes[0, 2].set_title('After Batch Normalization')
        
        # 特徴量ごとの分布（最初の数個）
        for i in range(3):
            axes[1, i].plot(x[0, :, i].numpy(), 'r-', alpha=0.7, label='Original')
            axes[1, i].plot(x_ln[0, :, i].numpy(), 'b-', alpha=0.7, label='LayerNorm')
            axes[1, i].plot(x_bn[0, :, i].numpy(), 'g-', alpha=0.7, label='BatchNorm')
            axes[1, i].set_title(f'Feature {i} - First Sample')
            axes[1, i].legend()
            axes[1, i].set_xlabel('Sequence Position')
        
        plt.tight_layout()
        plt.show()
        
        return x, x_ln, x_bn
    
    # 比較実行
    original, layer_normed, batch_normed = compare_normalizations()
    ```

## 演習 4.3: デバッグとビジュアライゼーション

### 問題 5
アテンション重みを可視化するツールを作成してください。

??? 解答
    ```python
    import torch
    import torch.nn as nn
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    from typing import List, Optional
    
    class AttentionVisualizer:
        """アテンション重みの可視化ツール"""
        
        def __init__(self):
            self.attention_weights = []
            self.layer_names = []
        
        def add_attention_weights(self, weights: torch.Tensor, layer_name: str):
            """アテンション重みを追加"""
            self.attention_weights.append(weights.detach().cpu())
            self.layer_names.append(layer_name)
        
        def visualize_head_patterns(self, layer_idx: int = 0, sample_idx: int = 0, 
                                  tokens: Optional[List[str]] = None):
            """各ヘッドのアテンションパターンを可視化"""
            
            if layer_idx >= len(self.attention_weights):
                print(f"Layer {layer_idx} not found")
                return
            
            weights = self.attention_weights[layer_idx]  # [batch, heads, seq, seq]
            sample_weights = weights[sample_idx]  # [heads, seq, seq]
            
            n_heads = sample_weights.shape[0]
            seq_len = sample_weights.shape[1]
            
            # トークンラベル
            if tokens is None:
                tokens = [f"T{i}" for i in range(seq_len)]
            
            # ヘッド数に応じてサブプロット配置を決定
            cols = min(4, n_heads)
            rows = (n_heads + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
            if rows == 1 and cols == 1:
                axes = [axes]
            elif rows == 1 or cols == 1:
                axes = axes.flatten()
            else:
                axes = axes.flatten()
            
            for head in range(n_heads):
                ax = axes[head] if n_heads > 1 else axes[0]
                
                # ヒートマップ
                sns.heatmap(sample_weights[head].numpy(), 
                           xticklabels=tokens, yticklabels=tokens,
                           cmap='Blues', ax=ax, cbar=True,
                           square=True, annot=True if seq_len <= 10 else False,
                           fmt='.2f')
                
                ax.set_title(f'Head {head + 1}')
                ax.set_xlabel('Key Position')
                ax.set_ylabel('Query Position')
            
            # 未使用のサブプロットを非表示
            for i in range(n_heads, len(axes)):
                axes[i].set_visible(False)
            
            plt.suptitle(f'{self.layer_names[layer_idx]} - Sample {sample_idx}')
            plt.tight_layout()
            plt.show()
        
        def analyze_attention_patterns(self, layer_idx: int = 0):
            """アテンションパターンの分析"""
            
            weights = self.attention_weights[layer_idx]  # [batch, heads, seq, seq]
            batch_size, n_heads, seq_len, _ = weights.shape
            
            # 各ヘッドの特性分析
            head_stats = []
            
            for head in range(n_heads):
                head_weights = weights[:, head, :, :]  # [batch, seq, seq]
                
                # エントロピー（注意の分散度）
                entropy = -(head_weights * torch.log(head_weights + 1e-9)).sum(dim=-1).mean()
                
                # 対角線の強さ（self-attention の度合い）
                diag_strength = torch.diagonal(head_weights, dim1=-2, dim2=-1).mean()
                
                # 局所性（近い位置への注意の強さ）
                positions = torch.arange(seq_len).float()
                pos_diff = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs()
                locality = (head_weights * (-pos_diff).exp()).sum(dim=-1).mean()
                
                head_stats.append({
                    'head': head,
                    'entropy': entropy.item(),
                    'diag_strength': diag_strength.item(),
                    'locality': locality.item()
                })
            
            # 結果表示
            print(f"=== {self.layer_names[layer_idx]} Analysis ===")
            print(f"{'Head':<6} {'Entropy':<10} {'Self-Attn':<10} {'Locality':<10}")
            print("-" * 40)
            
            for stats in head_stats:
                print(f"{stats['head']:<6} {stats['entropy']:<10.4f} "
                      f"{stats['diag_strength']:<10.4f} {stats['locality']:<10.4f}")
            
            # 可視化
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            
            metrics = ['entropy', 'diag_strength', 'locality']
            titles = ['Attention Entropy', 'Self-Attention Strength', 'Locality']
            
            for i, (metric, title) in enumerate(zip(metrics, titles)):
                values = [stats[metric] for stats in head_stats]
                axes[i].bar(range(n_heads), values, color=f'C{i}')
                axes[i].set_title(title)
                axes[i].set_xlabel('Head')
                axes[i].set_xticks(range(n_heads))
                axes[i].set_xticklabels([f'H{i}' for i in range(n_heads)])
                axes[i].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
            return head_stats
        
        def compare_layers(self):
            """層間でのアテンションパターン比較"""
            
            if len(self.attention_weights) < 2:
                print("比較には少なくとも2層のデータが必要です")
                return
            
            fig, axes = plt.subplots(len(self.attention_weights), 1, 
                                   figsize=(10, 3*len(self.attention_weights)))
            
            if len(self.attention_weights) == 1:
                axes = [axes]
            
            for layer_idx, (weights, name) in enumerate(zip(self.attention_weights, self.layer_names)):
                # 最初のサンプルの最初のヘッドを使用
                sample_weights = weights[0, 0].numpy()  # [seq, seq]
                
                sns.heatmap(sample_weights, ax=axes[layer_idx], 
                           cmap='Blues', cbar=True, square=True)
                axes[layer_idx].set_title(f'{name} (Head 0)')
                axes[layer_idx].set_xlabel('Key Position')
                axes[layer_idx].set_ylabel('Query Position')
            
            plt.tight_layout()
            plt.show()
    
    # 使用例
    def test_attention_visualizer():
        """ビジュアライザーのテスト"""
        
        # ダミーのアテンション重み生成
        torch.manual_seed(42)
        batch_size, n_heads, seq_len = 2, 8, 12
        
        # 異なるパターンのアテンション重み
        patterns = []
        
        # パターン1: 局所的なアテンション
        local_attn = torch.zeros(batch_size, n_heads, seq_len, seq_len)
        for i in range(seq_len):
            for j in range(max(0, i-2), min(seq_len, i+3)):
                local_attn[:, :4, i, j] = torch.exp(-abs(i-j))
        local_attn[:, :4] = torch.softmax(local_attn[:, :4], dim=-1)
        
        # パターン2: グローバルなアテンション
        global_attn = torch.randn(batch_size, n_heads, seq_len, seq_len)
        global_attn[:, 4:] = torch.softmax(global_attn[:, 4:], dim=-1)
        
        combined_attn = local_attn + global_attn
        combined_attn = torch.softmax(combined_attn, dim=-1)
        
        # ビジュアライザーのテスト
        visualizer = AttentionVisualizer()
        visualizer.add_attention_weights(combined_attn, "Layer 1")
        
        # トークンリスト
        tokens = ["the", "cat", "sat", "on", "the", "mat", "and", "looked", "at", "the", "dog", "."]
        
        # 可視化
        visualizer.visualize_head_patterns(0, 0, tokens)
        
        # 分析
        stats = visualizer.analyze_attention_patterns(0)
        
        return visualizer
    
    # テスト実行
    visualizer = test_attention_visualizer()
    ```

## 演習 4.4: 動作確認とテスト

### 問題 6
Transformerモデルの各コンポーネントに対する単体テストを作成してください。

??? 解答
    ```python
    import torch
    import torch.nn as nn
    import unittest
    import math
    
    class TestTransformerComponents(unittest.TestCase):
        """Transformerコンポーネントの単体テスト"""
        
        def setUp(self):
            """テスト前の準備"""
            self.batch_size = 2
            self.seq_len = 10
            self.d_model = 64
            self.n_heads = 8
            self.vocab_size = 100
            
            torch.manual_seed(42)
        
        def test_positional_encoding_shape(self):
            """位置エンコーディングの形状テスト"""
            from part4.minimal_transformer import PositionalEncoding
            
            pe = PositionalEncoding(self.d_model, max_len=100)
            x = torch.randn(self.batch_size, self.seq_len, self.d_model)
            
            output = pe(x)
            
            self.assertEqual(output.shape, x.shape)
            print("✓ Positional Encoding shape test passed")
        
        def test_positional_encoding_properties(self):
            """位置エンコーディングの数学的性質テスト"""
            from part4.minimal_transformer import PositionalEncoding
            
            pe = PositionalEncoding(self.d_model)
            
            # sin/cos の周期性をテスト
            pe_matrix = pe.pe.squeeze(1)  # [max_len, d_model]
            
            # 偶数インデックスはsin、奇数インデックスはcos
            for i in range(0, min(self.d_model, 10), 2):
                pos_vals = pe_matrix[:, i]
                # sin の値域は [-1, 1]
                self.assertTrue(torch.all(pos_vals >= -1.1))
                self.assertTrue(torch.all(pos_vals <= 1.1))
            
            print("✓ Positional Encoding properties test passed")
        
        def test_multihead_attention_shape(self):
            """Multi-Head Attentionの形状テスト"""
            from part4.minimal_transformer import MultiHeadAttention
            
            mha = MultiHeadAttention(self.d_model, self.n_heads)
            x = torch.randn(self.batch_size, self.seq_len, self.d_model)
            
            output = mha(x, x, x)
            
            self.assertEqual(output.shape, x.shape)
            print("✓ Multi-Head Attention shape test passed")
        
        def test_attention_mask(self):
            """アテンションマスクのテスト"""
            from part4.minimal_transformer import MultiHeadAttention
            
            mha = MultiHeadAttention(self.d_model, self.n_heads)
            x = torch.randn(self.batch_size, self.seq_len, self.d_model)
            
            # 因果マスク（デコーダ用）
            mask = torch.tril(torch.ones(self.seq_len, self.seq_len))
            mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
            
            output = mha(x, x, x, mask)
            
            self.assertEqual(output.shape, x.shape)
            print("✓ Attention mask test passed")
        
        def test_feed_forward_shape(self):
            """Feed Forward Networkの形状テスト"""
            from part4.minimal_transformer import FeedForward
            
            ff = FeedForward(self.d_model, self.d_model * 4)
            x = torch.randn(self.batch_size, self.seq_len, self.d_model)
            
            output = ff(x)
            
            self.assertEqual(output.shape, x.shape)
            print("✓ Feed Forward shape test passed")
        
        def test_layer_norm_properties(self):
            """Layer Normalizationの性質テスト"""
            layer_norm = nn.LayerNorm(self.d_model)
            x = torch.randn(self.batch_size, self.seq_len, self.d_model) * 5 + 10
            
            output = layer_norm(x)
            
            # 各サンプルの各位置で平均≈0、分散≈1
            mean = output.mean(dim=-1)
            var = output.var(dim=-1, unbiased=False)
            
            self.assertTrue(torch.allclose(mean, torch.zeros_like(mean), atol=1e-5))
            self.assertTrue(torch.allclose(var, torch.ones_like(var), atol=1e-5))
            print("✓ Layer Normalization properties test passed")
        
        def test_transformer_block_residual(self):
            """Transformerブロックの残差接続テスト"""
            from part4.minimal_transformer import TransformerBlock
            
            block = TransformerBlock(self.d_model, self.n_heads, self.d_model * 4, 0.0)
            x = torch.randn(self.batch_size, self.seq_len, self.d_model)
            
            # dropout=0なので、残差接続の効果を確認できる
            output = block(x)
            
            # 出力が入力と大きく異なることを確認（学習が起こっている）
            diff = torch.norm(output - x, dim=-1).mean()
            self.assertTrue(diff > 0.1)  # 何らかの変換が起こっている
            print("✓ Transformer block residual test passed")
        
        def test_full_model_forward(self):
            """完全なモデルの順伝播テスト"""
            from part4.minimal_transformer import MinimalTransformerEncoder
            
            model = MinimalTransformerEncoder(
                vocab_size=self.vocab_size,
                d_model=self.d_model,
                n_heads=self.n_heads,
                n_layers=2
            )
            
            x = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
            
            output = model(x)
            
            expected_shape = (self.batch_size, self.seq_len, self.d_model)
            self.assertEqual(output.shape, expected_shape)
            print("✓ Full model forward test passed")
        
        def test_gradient_flow(self):
            """勾配の流れのテスト"""
            from part4.minimal_transformer import MinimalTransformerEncoder
            
            model = MinimalTransformerEncoder(
                vocab_size=self.vocab_size,
                d_model=self.d_model,
                n_heads=self.n_heads,
                n_layers=2
            )
            
            x = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
            
            # 順伝播
            output = model(x)
            loss = output.sum()
            
            # 逆伝播
            loss.backward()
            
            # 全パラメータに勾配が流れていることを確認
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.assertIsNotNone(param.grad, f"No gradient for {name}")
                    self.assertFalse(torch.allclose(param.grad, torch.zeros_like(param.grad)), 
                                   f"Zero gradient for {name}")
            
            print("✓ Gradient flow test passed")
        
        def test_attention_weights_sum(self):
            """アテンション重みの和が1になることをテスト"""
            from part4.minimal_transformer import MultiHeadAttention
            
            class AttentionWithWeights(MultiHeadAttention):
                def forward(self, query, key, value, mask=None):
                    batch_size = query.size(0)
                    
                    Q = self.W_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
                    K = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
                    V = self.W_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
                    
                    d_k = Q.size(-1)
                    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
                    
                    if mask is not None:
                        scores = scores.masked_fill(mask == 0, -1e9)
                    
                    attn_weights = torch.softmax(scores, dim=-1)
                    context = torch.matmul(attn_weights, V)
                    context = context.transpose(1, 2).contiguous().view(
                        batch_size, -1, self.d_model)
                    
                    return self.W_o(context), attn_weights
            
            mha = AttentionWithWeights(self.d_model, self.n_heads)
            x = torch.randn(self.batch_size, self.seq_len, self.d_model)
            
            output, weights = mha(x, x, x)
            
            # 最後の次元での和が1になることを確認
            weight_sums = weights.sum(dim=-1)
            expected_sums = torch.ones_like(weight_sums)
            
            self.assertTrue(torch.allclose(weight_sums, expected_sums, atol=1e-5))
            print("✓ Attention weights sum test passed")
    
    def run_all_tests():
        """全テストを実行"""
        print("=== Transformer Components Unit Tests ===\\n")
        
        # テストスイートを作成
        suite = unittest.TestLoader().loadTestsFromTestCase(TestTransformerComponents)
        
        # テストを実行
        runner = unittest.TextTestRunner(verbosity=0)
        result = runner.run(suite)
        
        print(f"\\n=== Test Results ===")
        print(f"Tests run: {result.testsRun}")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
        
        if result.failures:
            print("\\nFailures:")
            for test, traceback in result.failures:
                print(f"- {test}: {traceback}")
        
        if result.errors:
            print("\\nErrors:")
            for test, traceback in result.errors:
                print(f"- {test}: {traceback}")
        
        return result.wasSuccessful()
    
    # テスト実行
    if __name__ == "__main__":
        success = run_all_tests()
        if success:
            print("\\n🎉 All tests passed!")
        else:
            print("\\n❌ Some tests failed!")
    ```

### 問題 7
モデルのメモリ使用量と計算量を分析するプロファイラーを作成してください。

??? 解答
    ```python
    import torch
    import torch.nn as nn
    import time
    import psutil
    import os
    from typing import Dict, List, Tuple
    import matplotlib.pyplot as plt
    
    class TransformerProfiler:
        """Transformerモデルのプロファイラー"""
        
        def __init__(self):
            self.profile_data = {}
        
        def profile_memory_usage(self, model: nn.Module, input_sizes: List[Tuple[int, int]]) -> Dict:
            """メモリ使用量のプロファイリング"""
            
            results = {
                'input_sizes': [],
                'model_memory': [],
                'forward_memory': [],
                'backward_memory': [],
                'total_memory': []
            }
            
            for batch_size, seq_len in input_sizes:
                # メモリをクリア
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
                # モデルのメモリ使用量
                model_params = sum(p.numel() * p.element_size() for p in model.parameters())
                model_buffers = sum(b.numel() * b.element_size() for b in model.buffers())
                model_memory = (model_params + model_buffers) / 1024**2  # MB
                
                # 入力データ作成
                device = next(model.parameters()).device
                x = torch.randint(0, 1000, (batch_size, seq_len), device=device)
                
                # Forward pass メモリ
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                memory_before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
                
                output = model(x)
                loss = output.sum()
                
                memory_after_forward = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
                forward_memory = (memory_after_forward - memory_before) / 1024**2
                
                # Backward pass メモリ
                loss.backward()
                
                memory_after_backward = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
                backward_memory = (memory_after_backward - memory_after_forward) / 1024**2
                
                total_memory = memory_after_backward / 1024**2
                
                results['input_sizes'].append(f"{batch_size}x{seq_len}")
                results['model_memory'].append(model_memory)
                results['forward_memory'].append(forward_memory)
                results['backward_memory'].append(backward_memory)
                results['total_memory'].append(total_memory)
                
                print(f"Size {batch_size}x{seq_len}: "
                      f"Model: {model_memory:.1f}MB, "
                      f"Forward: {forward_memory:.1f}MB, "
                      f"Backward: {backward_memory:.1f}MB, "
                      f"Total: {total_memory:.1f}MB")
                
                # メモリリーク防止
                del x, output, loss
                model.zero_grad()
            
            return results
        
        def profile_computation_time(self, model: nn.Module, input_sizes: List[Tuple[int, int]], 
                                   num_runs: int = 10) -> Dict:
            """計算時間のプロファイリング"""
            
            results = {
                'input_sizes': [],
                'forward_time': [],
                'backward_time': [],
                'total_time': []
            }
            
            model.eval()  # 安定した測定のため
            
            for batch_size, seq_len in input_sizes:
                device = next(model.parameters()).device
                
                forward_times = []
                backward_times = []
                
                for _ in range(num_runs):
                    x = torch.randint(0, 1000, (batch_size, seq_len), device=device)
                    
                    # Forward時間測定
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    
                    start_time = time.time()
                    output = model(x)
                    
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    
                    forward_time = time.time() - start_time
                    forward_times.append(forward_time)
                    
                    # Backward時間測定
                    loss = output.sum()
                    
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    
                    start_time = time.time()
                    loss.backward()
                    
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    
                    backward_time = time.time() - start_time
                    backward_times.append(backward_time)
                    
                    # クリーンアップ
                    del x, output, loss
                    model.zero_grad()
                
                avg_forward = sum(forward_times) / len(forward_times)
                avg_backward = sum(backward_times) / len(backward_times)
                avg_total = avg_forward + avg_backward
                
                results['input_sizes'].append(f"{batch_size}x{seq_len}")
                results['forward_time'].append(avg_forward * 1000)  # ms
                results['backward_time'].append(avg_backward * 1000)  # ms
                results['total_time'].append(avg_total * 1000)  # ms
                
                print(f"Size {batch_size}x{seq_len}: "
                      f"Forward: {avg_forward*1000:.2f}ms, "
                      f"Backward: {avg_backward*1000:.2f}ms, "
                      f"Total: {avg_total*1000:.2f}ms")
            
            return results
        
        def analyze_complexity(self, model: nn.Module, max_seq_len: int = 512) -> Dict:
            """計算複雑度の分析"""
            
            # モデル情報の取得
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            print(f"=== Model Analysis ===")
            print(f"Total parameters: {total_params:,}")
            print(f"Trainable parameters: {trainable_params:,}")
            print(f"Model size: {total_params * 4 / 1024**2:.2f} MB (float32)")
            
            # 層別パラメータ数
            layer_params = {}
            for name, param in model.named_parameters():
                layer_name = name.split('.')[0]  # 最上位レイヤー名
                if layer_name not in layer_params:
                    layer_params[layer_name] = 0
                layer_params[layer_name] += param.numel()
            
            print(f"\\n=== Parameters by Layer ===")
            for layer_name, params in sorted(layer_params.items(), key=lambda x: x[1], reverse=True):
                percentage = params / total_params * 100
                print(f"{layer_name}: {params:,} ({percentage:.1f}%)")
            
            # 理論的複雑度分析
            if hasattr(model, 'd_model') and hasattr(model, 'n_heads'):
                d_model = model.d_model
                n_heads = model.n_heads
                n_layers = len(model.transformer_blocks) if hasattr(model, 'transformer_blocks') else 1
                
                print(f"\\n=== Theoretical Complexity Analysis ===")
                print(f"d_model: {d_model}, n_heads: {n_heads}, n_layers: {n_layers}")
                
                # Self-attention complexity: O(n^2 * d)
                attention_ops = lambda n: n * n * d_model * n_layers
                
                # Feed-forward complexity: O(n * d^2)
                ff_ops = lambda n: n * d_model * d_model * 4 * n_layers  # assuming d_ff = 4 * d_model
                
                seq_lengths = [64, 128, 256, 512]
                
                print(f"\\n{'Seq Len':<8} {'Attention':<12} {'FF':<12} {'Total':<12} {'Memory (MB)':<12}")
                print("-" * 60)
                
                for seq_len in seq_lengths:
                    attn_ops = attention_ops(seq_len)
                    ff_ops_val = ff_ops(seq_len)
                    total_ops = attn_ops + ff_ops_val
                    
                    # メモリ推定 (activations)
                    memory_mb = seq_len * d_model * 4 / 1024**2  # float32
                    
                    print(f"{seq_len:<8} {attn_ops/1e6:.1f}M{'':<6} {ff_ops_val/1e6:.1f}M{'':<6} "
                          f"{total_ops/1e6:.1f}M{'':<6} {memory_mb:.2f}")
            
            return {
                'total_params': total_params,
                'trainable_params': trainable_params,
                'layer_params': layer_params
            }
        
        def visualize_profiles(self, memory_results: Dict, time_results: Dict):
            """プロファイル結果の可視化"""
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # メモリ使用量
            ax = axes[0, 0]
            x_pos = range(len(memory_results['input_sizes']))
            
            ax.bar([i-0.2 for i in x_pos], memory_results['model_memory'], 
                   width=0.4, label='Model', alpha=0.7)
            ax.bar([i+0.2 for i in x_pos], memory_results['forward_memory'], 
                   width=0.4, label='Forward', alpha=0.7)
            
            ax.set_title('Memory Usage by Component')
            ax.set_xlabel('Input Size')
            ax.set_ylabel('Memory (MB)')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(memory_results['input_sizes'], rotation=45)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 総メモリ使用量
            ax = axes[0, 1]
            ax.plot(memory_results['input_sizes'], memory_results['total_memory'], 
                    'ro-', linewidth=2, markersize=6)
            ax.set_title('Total Memory Usage')
            ax.set_xlabel('Input Size')
            ax.set_ylabel('Memory (MB)')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
            
            # 計算時間
            ax = axes[1, 0]
            x_pos = range(len(time_results['input_sizes']))
            
            ax.bar([i-0.2 for i in x_pos], time_results['forward_time'], 
                   width=0.4, label='Forward', alpha=0.7)
            ax.bar([i+0.2 for i in x_pos], time_results['backward_time'], 
                   width=0.4, label='Backward', alpha=0.7)
            
            ax.set_title('Computation Time by Phase')
            ax.set_xlabel('Input Size')
            ax.set_ylabel('Time (ms)')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(time_results['input_sizes'], rotation=45)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 総計算時間
            ax = axes[1, 1]
            ax.plot(time_results['input_sizes'], time_results['total_time'], 
                    'bo-', linewidth=2, markersize=6)
            ax.set_title('Total Computation Time')
            ax.set_xlabel('Input Size')
            ax.set_ylabel('Time (ms)')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
    
    # 使用例
    def run_profiling():
        """プロファイリングの実行例"""
        from part4.minimal_transformer import MinimalTransformerEncoder
        
        # モデル作成
        model = MinimalTransformerEncoder(
            vocab_size=1000,
            d_model=256,
            n_heads=8,
            n_layers=4
        )
        
        if torch.cuda.is_available():
            model = model.cuda()
        
        # プロファイラー初期化
        profiler = TransformerProfiler()
        
        # テスト用の入力サイズ
        input_sizes = [(4, 64), (4, 128), (4, 256), (2, 512)]
        
        print("=== Memory Profiling ===")
        memory_results = profiler.profile_memory_usage(model, input_sizes)
        
        print("\\n=== Time Profiling ===")
        time_results = profiler.profile_computation_time(model, input_sizes)
        
        print("\\n=== Complexity Analysis ===")
        complexity_results = profiler.analyze_complexity(model)
        
        # 可視化
        profiler.visualize_profiles(memory_results, time_results)
        
        return profiler, memory_results, time_results, complexity_results
    
    # プロファイリング実行
    if __name__ == "__main__":
        profiler, mem_results, time_results, complexity = run_profiling()
    ```

## チャレンジ問題

### 問題 8 🌟
効率的なTransformer実装の最適化技術を実装してください：
- Flash Attention
- Gradient Checkpointing
- Mixed Precision Training

??? 解答
    ```python
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.cuda.amp import autocast, GradScaler
    import math
    from typing import Optional
    
    class OptimizedTransformer(nn.Module):
        """最適化技術を含むTransformer実装"""
        
        def __init__(self, vocab_size=1000, d_model=512, n_heads=8, n_layers=6,
                     d_ff=2048, max_len=2048, dropout=0.1, 
                     use_flash_attention=True, use_gradient_checkpointing=True):
            super().__init__()
            
            self.d_model = d_model
            self.use_gradient_checkpointing = use_gradient_checkpointing
            
            # 埋め込み層
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.pos_encoding = OptimizedPositionalEncoding(d_model, max_len)
            
            # Transformerブロック
            self.transformer_blocks = nn.ModuleList([
                OptimizedTransformerBlock(
                    d_model, n_heads, d_ff, dropout, use_flash_attention
                ) for _ in range(n_layers)
            ])
            
            self.dropout = nn.Dropout(dropout)
            self.layer_norm = nn.LayerNorm(d_model)
            
        def forward(self, x, mask=None):
            with autocast():  # Mixed precision
                x = self.embedding(x) * math.sqrt(self.d_model)
                x = self.pos_encoding(x)
                x = self.dropout(x)
                
                # Gradient checkpointing
                if self.use_gradient_checkpointing and self.training:
                    for block in self.transformer_blocks:
                        x = torch.utils.checkpoint.checkpoint(block, x, mask)
                else:
                    for block in self.transformer_blocks:
                        x = block(x, mask)
                
                return self.layer_norm(x)
    
    class OptimizedPositionalEncoding(nn.Module):
        """メモリ効率的な位置エンコーディング"""
        
        def __init__(self, d_model, max_len=5000):
            super().__init__()
            self.d_model = d_model
            
            # 事前計算せず、必要時に計算
            self.register_buffer('inv_freq', 
                               torch.exp(torch.arange(0, d_model, 2).float() * 
                                       (-math.log(10000.0) / d_model)))
            
        def forward(self, x):
            seq_len = x.size(1)
            device = x.device
            
            # 動的に位置エンコーディングを計算
            position = torch.arange(seq_len, device=device).float().unsqueeze(1)
            
            pe = torch.zeros(seq_len, self.d_model, device=device)
            pe[:, 0::2] = torch.sin(position * self.inv_freq)
            pe[:, 1::2] = torch.cos(position * self.inv_freq)
            
            return x + pe.unsqueeze(0)
    
    class OptimizedTransformerBlock(nn.Module):
        """最適化されたTransformerブロック"""
        
        def __init__(self, d_model, n_heads, d_ff, dropout, use_flash_attention=True):
            super().__init__()
            
            if use_flash_attention:
                self.attention = FlashMultiHeadAttention(d_model, n_heads, dropout)
            else:
                self.attention = StandardMultiHeadAttention(d_model, n_heads, dropout)
            
            self.feed_forward = OptimizedFeedForward(d_model, d_ff, dropout)
            
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.dropout = nn.Dropout(dropout)
            
        def forward(self, x, mask=None):
            # Pre-norm (Post-normより安定)
            normed_x = self.norm1(x)
            attn_out = self.attention(normed_x, normed_x, normed_x, mask)
            x = x + self.dropout(attn_out)
            
            normed_x = self.norm2(x)
            ff_out = self.feed_forward(normed_x)
            x = x + self.dropout(ff_out)
            
            return x
    
    class FlashMultiHeadAttention(nn.Module):
        """Flash Attentionを模したメモリ効率的な実装"""
        
        def __init__(self, d_model, n_heads, dropout=0.1):
            super().__init__()
            assert d_model % n_heads == 0
            
            self.d_model = d_model
            self.n_heads = n_heads
            self.d_k = d_model // n_heads
            self.scale = 1.0 / math.sqrt(self.d_k)
            
            self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
            self.out_proj = nn.Linear(d_model, d_model)
            self.dropout = nn.Dropout(dropout)
            
        def forward(self, query, key, value, mask=None):
            batch_size, seq_len, _ = query.shape
            
            # QKVを一度に計算（メモリ帯域幅の効率化）
            qkv = self.qkv(query).chunk(3, dim=-1)
            q, k, v = [x.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2) 
                      for x in qkv]
            
            # メモリ効率的なアテンション計算
            if torch.cuda.is_available() and hasattr(F, 'scaled_dot_product_attention'):
                # PyTorch 2.0+ のFlash Attention
                attn_output = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=mask, dropout_p=self.dropout.p if self.training else 0.0
                )
            else:
                # フォールバック実装
                attn_output = self._flash_attention_fallback(q, k, v, mask)
            
            # 出力整形
            attn_output = attn_output.transpose(1, 2).contiguous().view(
                batch_size, seq_len, self.d_model
            )
            
            return self.out_proj(attn_output)
        
        def _flash_attention_fallback(self, q, k, v, mask=None):
            """Flash Attentionのフォールバック実装"""
            # ブロック化してメモリ使用量を削減
            batch_size, n_heads, seq_len, d_k = q.shape
            block_size = min(64, seq_len)  # ブロックサイズ
            
            output = torch.zeros_like(q)
            
            for i in range(0, seq_len, block_size):
                end_i = min(i + block_size, seq_len)
                q_block = q[:, :, i:end_i, :]
                
                # 現在のクエリブロックに対するアテンション計算
                scores = torch.matmul(q_block, k.transpose(-2, -1)) * self.scale
                
                if mask is not None:
                    scores = scores.masked_fill(mask[:, :, i:end_i, :] == 0, -1e9)
                
                attn_weights = F.softmax(scores, dim=-1)
                attn_weights = self.dropout(attn_weights)
                
                output[:, :, i:end_i, :] = torch.matmul(attn_weights, v)
            
            return output
    
    class StandardMultiHeadAttention(nn.Module):
        """標準的なMulti-Head Attention（比較用）"""
        
        def __init__(self, d_model, n_heads, dropout=0.1):
            super().__init__()
            assert d_model % n_heads == 0
            
            self.d_model = d_model
            self.n_heads = n_heads
            self.d_k = d_model // n_heads
            
            self.W_q = nn.Linear(d_model, d_model)
            self.W_k = nn.Linear(d_model, d_model)
            self.W_v = nn.Linear(d_model, d_model)
            self.W_o = nn.Linear(d_model, d_model)
            
            self.dropout = nn.Dropout(dropout)
            
        def forward(self, query, key, value, mask=None):
            batch_size = query.size(0)
            
            Q = self.W_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
            K = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
            V = self.W_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
            
            d_k = Q.size(-1)
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
            
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
            
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            context = torch.matmul(attn_weights, V)
            context = context.transpose(1, 2).contiguous().view(
                batch_size, -1, self.d_model
            )
            
            return self.W_o(context)
    
    class OptimizedFeedForward(nn.Module):
        """最適化されたFeed Forward Network"""
        
        def __init__(self, d_model, d_ff, dropout=0.1):
            super().__init__()
            
            # SwiGLU activation (GPTなどで使用)
            self.w1 = nn.Linear(d_model, d_ff, bias=False)
            self.w2 = nn.Linear(d_ff, d_model, bias=False)
            self.w3 = nn.Linear(d_model, d_ff, bias=False)
            self.dropout = nn.Dropout(dropout)
            
        def forward(self, x):
            # SwiGLU: x -> SiLU(W1(x)) * W3(x) -> W2
            return self.w2(self.dropout(F.silu(self.w1(x)) * self.w3(x)))
    
    class MixedPrecisionTrainer:
        """Mixed Precision Training のヘルパークラス"""
        
        def __init__(self, model, optimizer, enabled=True):
            self.model = model
            self.optimizer = optimizer
            self.scaler = GradScaler(enabled=enabled)
            self.enabled = enabled
        
        def train_step(self, x, targets, criterion):
            """1回の訓練ステップ"""
            self.optimizer.zero_grad()
            
            with autocast(enabled=self.enabled):
                outputs = self.model(x)
                loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            
            # スケールされた勾配での逆伝播
            self.scaler.scale(loss).backward()
            
            # 勾配クリッピング
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # パラメータ更新
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            return loss.item()
    
    # 使用例とベンチマーク
    def benchmark_optimizations():
        """最適化技術のベンチマーク"""
        
        if not torch.cuda.is_available():
            print("CUDA not available, skipping benchmark")
            return
        
        device = torch.device('cuda')
        
        # モデル設定
        configs = [
            ("Standard", False, False),
            ("Flash Attention", True, False),
            ("Flash + Gradient Checkpointing", True, True),
        ]
        
        results = {}
        
        for name, use_flash, use_checkpoint in configs:
            print(f"\\n=== Testing {name} ===")
            
            model = OptimizedTransformer(
                vocab_size=32000,
                d_model=512,
                n_heads=8,
                n_layers=6,
                use_flash_attention=use_flash,
                use_gradient_checkpointing=use_checkpoint
            ).to(device)
            
            # テストデータ
            batch_size, seq_len = 4, 1024
            x = torch.randint(0, 32000, (batch_size, seq_len), device=device)
            
            # メモリ使用量測定
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            # Forward pass
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            
            start_time.record()
            with autocast():
                output = model(x)
                loss = output.sum()
            
            # Backward pass
            loss.backward()
            end_time.record()
            
            torch.cuda.synchronize()
            
            time_ms = start_time.elapsed_time(end_time)
            peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
            
            results[name] = {
                'time_ms': time_ms,
                'memory_mb': peak_memory
            }
            
            print(f"Time: {time_ms:.1f} ms")
            print(f"Peak Memory: {peak_memory:.1f} MB")
            
            del model, x, output, loss
            torch.cuda.empty_cache()
        
        # 結果比較
        print(f"\\n=== Benchmark Results ===")
        baseline = results["Standard"]
        
        for name, metrics in results.items():
            time_ratio = baseline['time_ms'] / metrics['time_ms']
            memory_ratio = baseline['memory_mb'] / metrics['memory_mb']
            
            print(f"{name}:")
            print(f"  Speedup: {time_ratio:.2f}x")
            print(f"  Memory Efficiency: {memory_ratio:.2f}x")
        
        return results
    
    # ベンチマーク実行
    if __name__ == "__main__":
        results = benchmark_optimizations()
    ```

## 次のステップ

これらの演習を完了したら、第5部に進んで実際のLLMアーキテクチャとその応用を学びましょう！

💡 **学習のコツ**: 
- 各実装を段階的に理解し、必要に応じて簡略化したバージョンから始める
- プロファイリングツールを使って性能特性を理解する
- 実際の学習データで小規模な実験を行ってみる
- 最適化技術は必要に応じて段階的に導入する