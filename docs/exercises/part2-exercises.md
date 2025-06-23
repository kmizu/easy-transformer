# 第2部 演習問題

## 演習 2.1: トークン化と埋め込み

### 問題 1
簡単なBPE（Byte Pair Encoding）アルゴリズムを実装してください。
以下のコーパスに対して、3回のマージ操作を行います。

```python
corpus = ["low", "lower", "newest", "widest"]
```

??? 解答
    ```python
    from collections import defaultdict, Counter
    
    def get_vocab(corpus):
        """単語を文字単位に分解し、頻度をカウント"""
        vocab = defaultdict(int)
        for word in corpus:
            word_tokens = ' '.join(list(word)) + ' </w>'
            vocab[word_tokens] += 1
        return vocab
    
    def get_stats(vocab):
        """隣接するトークンペアの頻度を計算"""
        pairs = defaultdict(int)
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[symbols[i], symbols[i + 1]] += freq
        return pairs
    
    def merge_vocab(pair, vocab):
        """最頻出ペアをマージ"""
        out = {}
        bigram = ' '.join(pair)
        replacement = ''.join(pair)
        
        for word in vocab:
            new_word = word.replace(bigram, replacement)
            out[new_word] = vocab[word]
        return out
    
    # BPE実行
    corpus = ["low", "lower", "newest", "widest"]
    vocab = get_vocab(corpus)
    print("初期語彙:", vocab)
    
    for i in range(3):
        pairs = get_stats(vocab)
        if not pairs:
            break
            
        best = max(pairs, key=pairs.get)
        vocab = merge_vocab(best, vocab)
        print(f"\nマージ {i+1}: {best} -> {''.join(best)}")
        print("更新後の語彙:", vocab)
    
    # 出力例:
    # マージ 1: ('e', 's') -> 'es'
    # マージ 2: ('es', 't') -> 'est'
    # マージ 3: ('l', 'o') -> 'lo'
    ```

### 問題 2
単語埋め込みの類似度を計算する関数を実装してください。
コサイン類似度を使用します。

??? 解答
    ```python
    import numpy as np
    
    def cosine_similarity(vec1, vec2):
        """2つのベクトル間のコサイン類似度を計算"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        # ゼロ除算を避ける
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return dot_product / (norm1 * norm2)
    
    # テスト
    # 仮想的な単語埋め込み
    embeddings = {
        "cat": np.array([0.2, 0.8, 0.1]),
        "dog": np.array([0.3, 0.7, 0.2]),
        "car": np.array([0.9, 0.1, 0.3]),
        "truck": np.array([0.8, 0.2, 0.4])
    }
    
    # 類似度計算
    print(f"cat vs dog: {cosine_similarity(embeddings['cat'], embeddings['dog']):.3f}")
    print(f"cat vs car: {cosine_similarity(embeddings['cat'], embeddings['car']):.3f}")
    print(f"car vs truck: {cosine_similarity(embeddings['car'], embeddings['truck']):.3f}")
    
    # 最も類似した単語を見つける
    def find_most_similar(word, embeddings, top_k=2):
        target_vec = embeddings[word]
        similarities = []
        
        for other_word, other_vec in embeddings.items():
            if other_word != word:
                sim = cosine_similarity(target_vec, other_vec)
                similarities.append((other_word, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    print(f"\n'cat'に最も類似: {find_most_similar('cat', embeddings)}")
    ```

## 演習 2.2: 注意機構

### 問題 3
スケールドドット積注意を実装してください。
マスクのサポートも含めます。

??? 解答
    ```python
    import torch
    import torch.nn.functional as F
    import math
    
    def scaled_dot_product_attention(Q, K, V, mask=None):
        """
        スケールドドット積注意の実装
        
        Args:
            Q: クエリ [batch_size, seq_len, d_k]
            K: キー [batch_size, seq_len, d_k]
            V: バリュー [batch_size, seq_len, d_v]
            mask: マスク [batch_size, seq_len, seq_len] or None
        
        Returns:
            output: 注意の出力 [batch_size, seq_len, d_v]
            attention_weights: 注意の重み [batch_size, seq_len, seq_len]
        """
        d_k = Q.size(-1)
        
        # スコアの計算: QK^T / √d_k
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        
        # マスクの適用
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # ソフトマックスで注意の重みを計算
        attention_weights = F.softmax(scores, dim=-1)
        
        # 重み付き和を計算
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights
    
    # テスト
    batch_size, seq_len, d_k = 2, 4, 8
    Q = torch.randn(batch_size, seq_len, d_k)
    K = torch.randn(batch_size, seq_len, d_k)
    V = torch.randn(batch_size, seq_len, d_k)
    
    # マスクなし
    output, weights = scaled_dot_product_attention(Q, K, V)
    print(f"出力形状: {output.shape}")
    print(f"注意重み形状: {weights.shape}")
    
    # 因果的マスク（下三角行列）
    mask = torch.tril(torch.ones(seq_len, seq_len))
    mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
    
    output_masked, weights_masked = scaled_dot_product_attention(Q, K, V, mask)
    print(f"\nマスク適用後の注意重み（最初のサンプル）:")
    print(weights_masked[0])
    ```

### 問題 4
セルフアテンション層を実装し、位置による注意パターンの違いを可視化してください。

??? 解答
    ```python
    import torch
    import torch.nn as nn
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    class SelfAttention(nn.Module):
        def __init__(self, d_model, d_k=None):
            super().__init__()
            if d_k is None:
                d_k = d_model
                
            self.d_k = d_k
            self.W_q = nn.Linear(d_model, d_k, bias=False)
            self.W_k = nn.Linear(d_model, d_k, bias=False)
            self.W_v = nn.Linear(d_model, d_k, bias=False)
            self.W_o = nn.Linear(d_k, d_model, bias=False)
            
        def forward(self, x, mask=None):
            Q = self.W_q(x)
            K = self.W_k(x)
            V = self.W_v(x)
            
            # スケールドドット積注意
            d_k = self.d_k
            scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
            
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
                
            attn_weights = F.softmax(scores, dim=-1)
            context = torch.matmul(attn_weights, V)
            
            output = self.W_o(context)
            
            return output, attn_weights
    
    # テストと可視化
    torch.manual_seed(42)
    d_model = 64
    seq_len = 8
    
    # ダミーの入力（位置によって異なるパターン）
    x = torch.randn(1, seq_len, d_model)
    
    # 位置情報を追加（簡易版）
    for i in range(seq_len):
        x[0, i, :] += torch.sin(torch.arange(d_model) * i / 10)
    
    # セルフアテンション適用
    self_attn = SelfAttention(d_model)
    output, attn_weights = self_attn(x)
    
    # 注意パターンの可視化
    plt.figure(figsize=(8, 6))
    sns.heatmap(attn_weights[0].detach().numpy(), 
                annot=True, fmt='.2f', cmap='Blues',
                xticklabels=[f'Pos{i}' for i in range(seq_len)],
                yticklabels=[f'Pos{i}' for i in range(seq_len)])
    plt.title('Self-Attention Pattern')
    plt.xlabel('Keys')
    plt.ylabel('Queries')
    plt.tight_layout()
    plt.show()
    
    # 各位置がどこに注目しているか分析
    for i in range(min(4, seq_len)):
        top_3 = torch.topk(attn_weights[0, i], 3)
        print(f"位置{i}が最も注目している位置: {top_3.indices.tolist()}")
    ```

## 演習 2.3: 位置エンコーディング

### 問題 5
正弦波位置エンコーディングを実装し、異なる次元での周期性を可視化してください。

??? 解答
    ```python
    import numpy as np
    import matplotlib.pyplot as plt
    
    def positional_encoding(max_len, d_model):
        """正弦波位置エンコーディングの生成"""
        pe = np.zeros((max_len, d_model))
        position = np.arange(0, max_len).reshape(-1, 1)
        
        # 周波数項の計算
        div_term = np.exp(np.arange(0, d_model, 2) * 
                          -(np.log(10000.0) / d_model))
        
        # sin と cos を適用
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        return pe
    
    # 生成と可視化
    max_len = 100
    d_model = 64
    pe = positional_encoding(max_len, d_model)
    
    # 異なる次元での周期性を可視化
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # 最初の4次元
    for i, ax in enumerate(axes.flat):
        ax.plot(pe[:, i], label=f'dim={i}')
        ax.set_xlabel('Position')
        ax.set_ylabel('Value')
        ax.set_title(f'Positional Encoding - Dimension {i}')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    plt.show()
    
    # ヒートマップで全体像を表示
    plt.figure(figsize=(12, 4))
    plt.imshow(pe.T, aspect='auto', cmap='RdBu', 
               extent=[0, max_len, d_model, 0])
    plt.colorbar()
    plt.xlabel('Position')
    plt.ylabel('Dimension')
    plt.title('Positional Encoding Heatmap')
    plt.show()
    
    # 相対位置の性質を確認
    def check_relative_position_property(pe, pos1, pos2, k):
        """相対位置k離れた位置エンコーディングの内積"""
        vec1 = pe[pos1]
        vec2 = pe[pos2]
        vec1_k = pe[pos1 + k] if pos1 + k < len(pe) else None
        vec2_k = pe[pos2 + k] if pos2 + k < len(pe) else None
        
        if vec1_k is not None and vec2_k is not None:
            dot1 = np.dot(vec1, vec2)
            dot2 = np.dot(vec1_k, vec2_k)
            print(f"位置{pos1}と{pos2}の内積: {dot1:.3f}")
            print(f"位置{pos1+k}と{pos2+k}の内積: {dot2:.3f}")
            print(f"差: {abs(dot1 - dot2):.3f}")
    
    print("\n相対位置の性質:")
    check_relative_position_property(pe, 10, 15, 5)
    ```

### 問題 6
学習可能な位置埋め込みと正弦波エンコーディングを比較する実験を設計してください。

??? 解答
    ```python
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    class LearnablePositionalEmbedding(nn.Module):
        """学習可能な位置埋め込み"""
        def __init__(self, max_len, d_model):
            super().__init__()
            self.pos_embedding = nn.Embedding(max_len, d_model)
            
        def forward(self, x):
            seq_len = x.size(1)
            positions = torch.arange(seq_len, device=x.device)
            return x + self.pos_embedding(positions)
    
    class SinusoidalPositionalEncoding(nn.Module):
        """正弦波位置エンコーディング"""
        def __init__(self, max_len, d_model):
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len).unsqueeze(1).float()
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                               -(np.log(10000.0) / d_model))
            
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            
            self.register_buffer('pe', pe.unsqueeze(0))
            
        def forward(self, x):
            return x + self.pe[:, :x.size(1)]
    
    # 簡単な系列タスクで比較
    class PositionAwareModel(nn.Module):
        def __init__(self, vocab_size, d_model, pos_encoding_type='learnable'):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, d_model)
            
            if pos_encoding_type == 'learnable':
                self.pos_encoding = LearnablePositionalEmbedding(100, d_model)
            else:
                self.pos_encoding = SinusoidalPositionalEncoding(100, d_model)
                
            self.lstm = nn.LSTM(d_model, d_model, batch_first=True)
            self.output = nn.Linear(d_model, vocab_size)
            
        def forward(self, x):
            x = self.embedding(x)
            x = self.pos_encoding(x)
            x, _ = self.lstm(x)
            return self.output(x)
    
    # 位置依存タスクの作成（位置を反転する）
    def create_position_task(seq_len=10, vocab_size=20, n_samples=100):
        data = []
        for _ in range(n_samples):
            # ランダムな系列を生成
            seq = torch.randint(3, vocab_size, (seq_len,))
            # 反転した系列が目標
            target = torch.flip(seq, [0])
            data.append((seq, target))
        return data
    
    # 訓練と評価
    def train_and_evaluate(model, data, epochs=50):
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        losses = []
        
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for seq, target in data:
                seq = seq.unsqueeze(0)
                target = target.unsqueeze(0)
                
                output = model(seq)
                loss = F.cross_entropy(output.reshape(-1, output.size(-1)), 
                                     target.reshape(-1))
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                # 精度計算
                _, predicted = output.max(-1)
                correct += (predicted == target).sum().item()
                total += target.numel()
            
            accuracy = correct / total
            losses.append(total_loss / len(data))
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss={losses[-1]:.3f}, Acc={accuracy:.3f}")
        
        return losses
    
    # 実験実行
    print("位置エンコーディングの比較実験")
    print("タスク: 入力系列を反転する\n")
    
    vocab_size = 20
    d_model = 32
    data = create_position_task()
    
    # 学習可能な位置埋め込み
    print("1. 学習可能な位置埋め込み:")
    model_learnable = PositionAwareModel(vocab_size, d_model, 'learnable')
    losses_learnable = train_and_evaluate(model_learnable, data)
    
    print("\n2. 正弦波位置エンコーディング:")
    model_sinusoidal = PositionAwareModel(vocab_size, d_model, 'sinusoidal')
    losses_sinusoidal = train_and_evaluate(model_sinusoidal, data)
    
    # 結果の可視化
    plt.figure(figsize=(8, 6))
    plt.plot(losses_learnable, label='Learnable')
    plt.plot(losses_sinusoidal, label='Sinusoidal')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Position Encoding Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    ```

## 演習 2.4: 深層学習の基礎

### 問題 7
残差接続を含む簡単なネットワークを実装し、勾配の流れを可視化してください。

??? 解答
    ```python
    import torch
    import torch.nn as nn
    import matplotlib.pyplot as plt
    
    class ResidualBlock(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.fc1 = nn.Linear(dim, dim)
            self.fc2 = nn.Linear(dim, dim)
            self.relu = nn.ReLU()
            
        def forward(self, x):
            residual = x
            out = self.fc1(x)
            out = self.relu(out)
            out = self.fc2(out)
            out = out + residual  # 残差接続
            return out
    
    class DeepNetwork(nn.Module):
        def __init__(self, input_dim, n_blocks, use_residual=True):
            super().__init__()
            self.use_residual = use_residual
            self.blocks = nn.ModuleList()
            
            for _ in range(n_blocks):
                if use_residual:
                    self.blocks.append(ResidualBlock(input_dim))
                else:
                    self.blocks.append(nn.Sequential(
                        nn.Linear(input_dim, input_dim),
                        nn.ReLU(),
                        nn.Linear(input_dim, input_dim)
                    ))
                    
        def forward(self, x):
            activations = [x]
            for block in self.blocks:
                x = block(x)
                activations.append(x)
            return x, activations
    
    # 勾配の流れを記録
    def record_gradients(model, input_data, target):
        gradients = []
        
        def hook_fn(module, grad_input, grad_output):
            gradients.append(grad_output[0].norm().item())
        
        # フックを登録
        handles = []
        for block in model.blocks:
            handle = block.register_backward_hook(hook_fn)
            handles.append(handle)
        
        # 順伝播と逆伝播
        output, _ = model(input_data)
        loss = nn.MSELoss()(output, target)
        loss.backward()
        
        # フックを削除
        for handle in handles:
            handle.remove()
            
        return gradients[::-1]  # 逆順にして入力側から並べる
    
    # 実験
    input_dim = 64
    n_blocks = 10
    batch_size = 32
    
    # データ
    x = torch.randn(batch_size, input_dim)
    target = torch.randn(batch_size, input_dim)
    
    # 残差接続ありとなしで比較
    model_with_residual = DeepNetwork(input_dim, n_blocks, use_residual=True)
    model_without_residual = DeepNetwork(input_dim, n_blocks, use_residual=False)
    
    grad_with = record_gradients(model_with_residual, x.clone(), target)
    grad_without = record_gradients(model_without_residual, x.clone(), target)
    
    # 可視化
    plt.figure(figsize=(10, 6))
    blocks = list(range(1, n_blocks + 1))
    
    plt.semilogy(blocks, grad_with, 'bo-', label='With Residual', linewidth=2)
    plt.semilogy(blocks, grad_without, 'ro-', label='Without Residual', linewidth=2)
    
    plt.xlabel('Block Number (from input)')
    plt.ylabel('Gradient Norm (log scale)')
    plt.title('Gradient Flow in Deep Networks')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print(f"最初の層の勾配ノルム:")
    print(f"  残差接続あり: {grad_with[0]:.6f}")
    print(f"  残差接続なし: {grad_without[0]:.6f}")
    print(f"  比率: {grad_with[0] / grad_without[0]:.2f}x")
    ```

## チャレンジ問題

### 問題 8 🌟
マルチヘッドアテンションを簡易実装し、各ヘッドが異なるパターンを学習することを確認してください。

??? 解答
    ```python
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    class MultiHeadAttention(nn.Module):
        def __init__(self, d_model, n_heads):
            super().__init__()
            assert d_model % n_heads == 0
            
            self.d_model = d_model
            self.n_heads = n_heads
            self.d_k = d_model // n_heads
            
            self.W_q = nn.Linear(d_model, d_model)
            self.W_k = nn.Linear(d_model, d_model)
            self.W_v = nn.Linear(d_model, d_model)
            self.W_o = nn.Linear(d_model, d_model)
            
        def forward(self, query, key, value, mask=None):
            batch_size, seq_len, _ = query.shape
            
            # 線形変換と形状変更
            Q = self.W_q(query).view(batch_size, seq_len, self.n_heads, self.d_k)
            K = self.W_k(key).view(batch_size, seq_len, self.n_heads, self.d_k)
            V = self.W_v(value).view(batch_size, seq_len, self.n_heads, self.d_k)
            
            # 転置してヘッドを別次元に
            Q = Q.transpose(1, 2)  # [batch, heads, seq_len, d_k]
            K = K.transpose(1, 2)
            V = V.transpose(1, 2)
            
            # 各ヘッドでアテンション計算
            scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
            
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
            
            attn_weights = F.softmax(scores, dim=-1)
            context = torch.matmul(attn_weights, V)
            
            # ヘッドを結合
            context = context.transpose(1, 2).contiguous()
            context = context.view(batch_size, seq_len, self.d_model)
            
            output = self.W_o(context)
            
            return output, attn_weights
    
    # 異なるパターンを学習させる実験
    def create_pattern_data():
        """異なる依存関係を持つデータを作成"""
        seq_len = 8
        d_model = 64
        batch_size = 100
        
        data = []
        
        for _ in range(batch_size):
            # パターン1: 隣接依存
            x1 = torch.randn(seq_len, d_model)
            for i in range(1, seq_len):
                x1[i] += 0.5 * x1[i-1]
            
            # パターン2: 長距離依存
            x2 = torch.randn(seq_len, d_model)
            x2[seq_len//2:] += x2[:seq_len//2]
            
            # 混合
            x = x1 + x2
            x = x / x.norm(dim=-1, keepdim=True)
            
            data.append(x)
            
        return torch.stack(data)
    
    # モデルの訓練
    def train_multihead_attention():
        d_model = 64
        n_heads = 4
        mha = MultiHeadAttention(d_model, n_heads)
        
        # データ作成
        data = create_pattern_data()
        
        # 自己教師あり学習（入力を再構成）
        optimizer = torch.optim.Adam(mha.parameters(), lr=0.001)
        
        for epoch in range(100):
            # ノイズを加えた入力
            noisy_input = data + 0.1 * torch.randn_like(data)
            
            # アテンション適用
            output, attn_weights = mha(noisy_input, noisy_input, noisy_input)
            
            # 再構成誤差
            loss = F.mse_loss(output, data)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
        
        return mha, data, attn_weights
    
    # 実行と可視化
    mha, data, attn_weights = train_multihead_attention()
    
    # 各ヘッドのアテンションパターンを可視化
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    sample_idx = 0  # 最初のサンプル
    
    for head in range(4):
        ax = axes[head]
        attn = attn_weights[sample_idx, head].detach().numpy()
        
        sns.heatmap(attn, ax=ax, cmap='Blues', cbar=True)
        ax.set_title(f'Head {head + 1} Attention Pattern')
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')
        
        # 主要なパターンを分析
        avg_attn = attn.mean(axis=0)
        main_focus = avg_attn.argmax()
        print(f"Head {head + 1} - 平均的な焦点位置: {main_focus}")
    
    plt.tight_layout()
    plt.show()
    
    # ヘッド間の多様性を定量化
    def attention_diversity(attn_weights):
        """ヘッド間のアテンションパターンの多様性を計算"""
        n_heads = attn_weights.shape[1]
        
        # 各ヘッドのアテンションを平坦化
        flattened = attn_weights.reshape(attn_weights.shape[0], n_heads, -1)
        
        # ヘッド間のコサイン類似度
        similarities = []
        for i in range(n_heads):
            for j in range(i + 1, n_heads):
                sim = F.cosine_similarity(
                    flattened[:, i], 
                    flattened[:, j], 
                    dim=-1
                ).mean().item()
                similarities.append(sim)
        
        # 多様性 = 1 - 平均類似度
        diversity = 1 - np.mean(similarities)
        return diversity
    
    diversity = attention_diversity(attn_weights)
    print(f"\nアテンションヘッドの多様性: {diversity:.3f}")
    print("(1に近いほど多様、0に近いほど類似)")
    ```

## 実践プロジェクト 🚀

### プロジェクト: ミニ言語モデルの実装
第2部で学んだすべての要素を組み合わせて、小さな言語モデルを実装してください。

要件：
- トークン化（簡単な空白区切り）
- 埋め込み層
- 位置エンコーディング
- セルフアテンション（1層）
- 出力層

??? 解答
    ```python
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    class MiniLanguageModel(nn.Module):
        def __init__(self, vocab_size, d_model=128, max_len=100):
            super().__init__()
            
            # 埋め込み層
            self.token_embedding = nn.Embedding(vocab_size, d_model)
            
            # 位置エンコーディング
            self.position_embedding = nn.Embedding(max_len, d_model)
            
            # セルフアテンション
            self.attention = nn.MultiheadAttention(d_model, num_heads=4, 
                                                  batch_first=True)
            
            # フィードフォワード
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.ReLU(),
                nn.Linear(d_model * 4, d_model)
            )
            
            # 層正規化
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            
            # 出力層
            self.output = nn.Linear(d_model, vocab_size)
            
        def forward(self, x, mask=None):
            seq_len = x.size(1)
            
            # 埋め込み
            token_emb = self.token_embedding(x)
            pos_ids = torch.arange(seq_len, device=x.device).unsqueeze(0)
            pos_emb = self.position_embedding(pos_ids)
            
            x = token_emb + pos_emb
            
            # セルフアテンション（残差接続付き）
            attn_output, _ = self.attention(x, x, x, attn_mask=mask)
            x = self.norm1(x + attn_output)
            
            # フィードフォワード（残差接続付き）
            ff_output = self.ffn(x)
            x = self.norm2(x + ff_output)
            
            # 出力
            return self.output(x)
        
        def generate(self, start_tokens, max_length=50):
            """テキスト生成"""
            self.eval()
            generated = start_tokens.clone()
            
            with torch.no_grad():
                for _ in range(max_length):
                    # 現在の系列で予測
                    outputs = self(generated)
                    
                    # 最後のトークンの予測を取得
                    next_token_logits = outputs[:, -1, :]
                    
                    # サンプリング
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, 1)
                    
                    # 追加
                    generated = torch.cat([generated, next_token], dim=1)
                    
                    # 終了トークンのチェック（実装による）
                    
            return generated
    
    # 簡単なトークナイザー
    class SimpleTokenizer:
        def __init__(self, texts):
            self.word_to_id = {'<pad>': 0, '<unk>': 1, '<sos>': 2, '<eos>': 3}
            self.id_to_word = {v: k for k, v in self.word_to_id.items()}
            
            # 語彙構築
            for text in texts:
                for word in text.lower().split():
                    if word not in self.word_to_id:
                        idx = len(self.word_to_id)
                        self.word_to_id[word] = idx
                        self.id_to_word[idx] = word
                        
        def encode(self, text):
            tokens = []
            for word in text.lower().split():
                tokens.append(self.word_to_id.get(word, 1))  # 1 = <unk>
            return tokens
        
        def decode(self, tokens):
            words = []
            for token in tokens:
                if token in self.id_to_word:
                    words.append(self.id_to_word[token])
            return ' '.join(words)
    
    # 使用例
    texts = [
        "the cat sat on the mat",
        "the dog played in the park",
        "cats and dogs are pets",
        "the sun shines bright",
        "birds fly in the sky"
    ]
    
    # トークナイザー作成
    tokenizer = SimpleTokenizer(texts)
    print(f"語彙サイズ: {len(tokenizer.word_to_id)}")
    
    # モデル作成
    model = MiniLanguageModel(len(tokenizer.word_to_id))
    
    # データ準備
    encoded_texts = [torch.tensor(tokenizer.encode(text)) for text in texts]
    
    # 簡単な訓練ループ
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(50):
        total_loss = 0
        
        for encoded in encoded_texts:
            # バッチ次元を追加
            x = encoded[:-1].unsqueeze(0)
            y = encoded[1:].unsqueeze(0)
            
            # 予測
            outputs = model(x)
            loss = F.cross_entropy(outputs.reshape(-1, outputs.size(-1)), 
                                 y.reshape(-1))
            
            # 最適化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {total_loss / len(texts):.4f}")
    
    # 生成テスト
    start = torch.tensor([[tokenizer.word_to_id['the']]])
    generated = model.generate(start, max_length=10)
    print(f"\n生成: {tokenizer.decode(generated[0].tolist())}")
    ```

これで第2部の演習は完了です！次は第3部でTransformerの詳細な構成要素を学びましょう。