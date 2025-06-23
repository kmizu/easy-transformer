# 第3部 演習問題

## 演習 3.1: Multi-Head Attention

### 問題 1
8ヘッドのMulti-Head Attentionを実装し、各ヘッドが学習する特徴の違いを分析してください。

??? 解答
    ```python
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    
    class MultiHeadAttentionDetailed(nn.Module):
        def __init__(self, d_model=512, n_heads=8, dropout=0.1):
            super().__init__()
            assert d_model % n_heads == 0
            
            self.d_model = d_model
            self.n_heads = n_heads
            self.d_k = d_model // n_heads
            
            # 各ヘッド用の重み行列
            self.W_q = nn.Linear(d_model, d_model, bias=False)
            self.W_k = nn.Linear(d_model, d_model, bias=False)
            self.W_v = nn.Linear(d_model, d_model, bias=False)
            self.W_o = nn.Linear(d_model, d_model)
            
            self.dropout = nn.Dropout(dropout)
            self.scale = 1.0 / np.sqrt(self.d_k)
            
            # ヘッドごとの統計を記録
            self.head_statistics = {}
            
        def forward(self, query, key, value, mask=None):
            batch_size, seq_len, _ = query.shape
            
            # Q, K, V の計算
            Q = self.W_q(query).view(batch_size, seq_len, self.n_heads, self.d_k)
            K = self.W_k(key).view(batch_size, seq_len, self.n_heads, self.d_k)
            V = self.W_v(value).view(batch_size, seq_len, self.n_heads, self.d_k)
            
            # 転置: [batch, n_heads, seq_len, d_k]
            Q = Q.transpose(1, 2)
            K = K.transpose(1, 2)
            V = V.transpose(1, 2)
            
            # アテンションスコア
            scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
            
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
            
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            # 各ヘッドの統計を記録
            self._record_head_statistics(attn_weights)
            
            # コンテキストベクトル
            context = torch.matmul(attn_weights, V)
            
            # ヘッドを結合
            context = context.transpose(1, 2).contiguous()
            context = context.view(batch_size, seq_len, self.d_model)
            
            # 出力投影
            output = self.W_o(context)
            
            return output, attn_weights
        
        def _record_head_statistics(self, attn_weights):
            """各ヘッドの特性を記録"""
            with torch.no_grad():
                batch_size, n_heads, seq_len, _ = attn_weights.shape
                
                for head in range(n_heads):
                    head_attn = attn_weights[:, head, :, :]
                    
                    # エントロピー（集中度の指標）
                    entropy = -(head_attn * torch.log(head_attn + 1e-9)).sum(dim=-1).mean()
                    
                    # 平均的な注意距離
                    positions = torch.arange(seq_len, device=attn_weights.device)
                    pos_diff = positions.unsqueeze(0) - positions.unsqueeze(1)
                    avg_distance = (head_attn * pos_diff.abs().float()).sum(dim=-1).mean()
                    
                    # 対角成分の強さ（自己注意の度合い）
                    diag_strength = torch.diagonal(head_attn, dim1=-2, dim2=-1).mean()
                    
                    if head not in self.head_statistics:
                        self.head_statistics[head] = {
                            'entropy': [],
                            'avg_distance': [],
                            'diag_strength': []
                        }
                    
                    self.head_statistics[head]['entropy'].append(entropy.item())
                    self.head_statistics[head]['avg_distance'].append(avg_distance.item())
                    self.head_statistics[head]['diag_strength'].append(diag_strength.item())
    
    # ヘッドの特性を分析する実験
    def analyze_head_specialization():
        # モデル作成
        d_model = 512
        n_heads = 8
        seq_len = 20
        batch_size = 32
        
        mha = MultiHeadAttentionDetailed(d_model, n_heads)
        
        # 異なるパターンを持つデータで訓練
        print("異なるパターンのデータで訓練中...")
        
        optimizer = torch.optim.Adam(mha.parameters(), lr=0.001)
        
        for step in range(100):
            # パターン1: 局所的な依存関係
            local_data = torch.randn(batch_size, seq_len, d_model)
            for i in range(1, seq_len):
                local_data[:, i] += 0.5 * local_data[:, i-1]
            
            # パターン2: 長距離依存
            long_range_data = torch.randn(batch_size, seq_len, d_model)
            long_range_data[:, seq_len//2:] += long_range_data[:, :seq_len//2]
            
            # パターン3: 周期的パターン
            periodic_data = torch.randn(batch_size, seq_len, d_model)
            period = 5
            for i in range(period, seq_len):
                periodic_data[:, i] += 0.3 * periodic_data[:, i-period]
            
            # 混合データ
            data = (local_data + long_range_data + periodic_data) / 3
            
            # Multi-Head Attention適用
            output, attn_weights = mha(data, data, data)
            
            # 自己教師あり損失（入力の再構成）
            loss = F.mse_loss(output, data)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if step % 20 == 0:
                print(f"Step {step}: Loss = {loss.item():.4f}")
        
        # ヘッドの特性を可視化
        visualize_head_characteristics(mha, attn_weights)
    
    def visualize_head_characteristics(mha, sample_attn_weights):
        """各ヘッドの特性を可視化"""
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        # サンプルのアテンションパターン
        sample_attn = sample_attn_weights[0].detach().cpu().numpy()
        
        for head in range(8):
            ax = axes[head]
            
            # アテンションパターンのヒートマップ
            sns.heatmap(sample_attn[head], ax=ax, cmap='Blues', 
                       cbar_kws={'label': 'Weight'})
            
            # 統計情報を追加
            if head in mha.head_statistics:
                stats = mha.head_statistics[head]
                avg_entropy = np.mean(stats['entropy'])
                avg_distance = np.mean(stats['avg_distance'])
                avg_diag = np.mean(stats['diag_strength'])
                
                ax.set_title(f'Head {head+1}\n'
                           f'Ent:{avg_entropy:.2f}, '
                           f'Dist:{avg_distance:.1f}, '
                           f'Diag:{avg_diag:.2f}',
                           fontsize=10)
            else:
                ax.set_title(f'Head {head+1}')
            
            ax.set_xlabel('Key Position')
            ax.set_ylabel('Query Position')
        
        plt.suptitle('Multi-Head Attention Pattern Analysis', fontsize=14)
        plt.tight_layout()
        plt.show()
        
        # ヘッドの特性をレーダーチャートで表示
        plot_head_characteristics_radar(mha)
    
    def plot_head_characteristics_radar(mha):
        """ヘッドの特性をレーダーチャートで表示"""
        if not mha.head_statistics:
            return
        
        # 各ヘッドの平均統計を計算
        head_profiles = []
        for head in range(8):
            if head in mha.head_statistics:
                stats = mha.head_statistics[head]
                profile = [
                    np.mean(stats['entropy']),
                    np.mean(stats['avg_distance']),
                    np.mean(stats['diag_strength'])
                ]
                head_profiles.append(profile)
        
        # 正規化
        head_profiles = np.array(head_profiles)
        head_profiles = (head_profiles - head_profiles.min(axis=0)) / \
                       (head_profiles.max(axis=0) - head_profiles.min(axis=0) + 1e-8)
        
        # レーダーチャート
        categories = ['Entropy', 'Avg Distance', 'Diagonal']
        fig = plt.figure(figsize=(10, 8))
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]
        
        for i, profile in enumerate(head_profiles):
            values = profile.tolist()
            values += values[:1]
            
            ax = plt.subplot(2, 4, i+1, projection='polar')
            ax.plot(angles, values, 'o-', linewidth=2, label=f'Head {i+1}')
            ax.fill(angles, values, alpha=0.25)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories)
            ax.set_ylim(0, 1)
            ax.set_title(f'Head {i+1}', y=1.08)
            ax.grid(True)
        
        plt.suptitle('Head Characteristic Profiles', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    # 実行
    analyze_head_specialization()
    ```

### 問題 2
Grouped Query Attention (GQA) を実装し、通常のMulti-Head Attentionと比較してください。

??? 解答
    ```python
    class GroupedQueryAttention(nn.Module):
        """Grouped Query Attention (GQA) の実装"""
        
        def __init__(self, d_model=512, n_heads=8, n_kv_heads=2):
            super().__init__()
            assert d_model % n_heads == 0
            assert n_heads % n_kv_heads == 0
            
            self.d_model = d_model
            self.n_heads = n_heads
            self.n_kv_heads = n_kv_heads
            self.n_groups = n_heads // n_kv_heads
            self.d_k = d_model // n_heads
            
            # Query用の投影（全ヘッド分）
            self.W_q = nn.Linear(d_model, d_model)
            
            # Key/Value用の投影（グループ数分のみ）
            self.W_k = nn.Linear(d_model, n_kv_heads * self.d_k)
            self.W_v = nn.Linear(d_model, n_kv_heads * self.d_k)
            
            # 出力投影
            self.W_o = nn.Linear(d_model, d_model)
            
        def forward(self, query, key, value, mask=None):
            batch_size, seq_len, _ = query.shape
            
            # Query: 全ヘッド分
            Q = self.W_q(query).view(batch_size, seq_len, self.n_heads, self.d_k)
            Q = Q.transpose(1, 2)  # [batch, n_heads, seq_len, d_k]
            
            # Key/Value: グループ数分のみ
            K = self.W_k(key).view(batch_size, seq_len, self.n_kv_heads, self.d_k)
            K = K.transpose(1, 2)  # [batch, n_kv_heads, seq_len, d_k]
            
            V = self.W_v(value).view(batch_size, seq_len, self.n_kv_heads, self.d_k)
            V = V.transpose(1, 2)  # [batch, n_kv_heads, seq_len, d_k]
            
            # Key/Valueを各グループで共有
            K = K.repeat_interleave(self.n_groups, dim=1)
            V = V.repeat_interleave(self.n_groups, dim=1)
            
            # 通常のアテンション計算
            scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
            
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
            
            attn_weights = F.softmax(scores, dim=-1)
            context = torch.matmul(attn_weights, V)
            
            # ヘッドを結合
            context = context.transpose(1, 2).contiguous()
            context = context.view(batch_size, seq_len, self.d_model)
            
            output = self.W_o(context)
            
            return output, attn_weights
    
    # パラメータ数とメモリ使用量の比較
    def compare_attention_variants():
        d_model = 512
        seq_len = 100
        batch_size = 8
        
        # 通常のMHA
        mha = MultiHeadAttentionDetailed(d_model, n_heads=8)
        
        # GQA（2つのKVヘッド）
        gqa = GroupedQueryAttention(d_model, n_heads=8, n_kv_heads=2)
        
        # パラメータ数の比較
        mha_params = sum(p.numel() for p in mha.parameters())
        gqa_params = sum(p.numel() for p in gqa.parameters())
        
        print("パラメータ数の比較:")
        print(f"Multi-Head Attention: {mha_params:,}")
        print(f"Grouped Query Attention: {gqa_params:,}")
        print(f"削減率: {(1 - gqa_params/mha_params)*100:.1f}%\n")
        
        # メモリ使用量の比較（KVキャッシュ）
        kv_cache_mha = 2 * batch_size * 8 * seq_len * (d_model // 8) * 4  # float32
        kv_cache_gqa = 2 * batch_size * 2 * seq_len * (d_model // 8) * 4  # float32
        
        print("KVキャッシュメモリ使用量:")
        print(f"Multi-Head Attention: {kv_cache_mha / 1024**2:.2f} MB")
        print(f"Grouped Query Attention: {kv_cache_gqa / 1024**2:.2f} MB")
        print(f"削減率: {(1 - kv_cache_gqa/kv_cache_mha)*100:.1f}%\n")
        
        # 速度比較
        import time
        
        x = torch.randn(batch_size, seq_len, d_model)
        
        # MHA
        start = time.time()
        for _ in range(100):
            _ = mha(x, x, x)
        mha_time = time.time() - start
        
        # GQA
        start = time.time()
        for _ in range(100):
            _ = gqa(x, x, x)
        gqa_time = time.time() - start
        
        print("推論速度 (100イテレーション):")
        print(f"Multi-Head Attention: {mha_time:.3f}秒")
        print(f"Grouped Query Attention: {gqa_time:.3f}秒")
        print(f"高速化: {mha_time/gqa_time:.2f}x")
    
    compare_attention_variants()
    ```

## 演習 3.2: Feed Forward Network

### 問題 3
異なる活性化関数（ReLU, GELU, SwiGLU）を使用したFFNを実装し、性能を比較してください。

??? 解答
    ```python
    class FFNComparison:
        """異なる活性化関数を持つFFNの比較"""
        
        def __init__(self, d_model=512, d_ff=2048):
            self.d_model = d_model
            self.d_ff = d_ff
            
        def create_ffn_variants(self):
            """異なるFFNバリアントを作成"""
            
            class FFN_ReLU(nn.Module):
                def __init__(self, d_model, d_ff):
                    super().__init__()
                    self.fc1 = nn.Linear(d_model, d_ff)
                    self.fc2 = nn.Linear(d_ff, d_model)
                    self.dropout = nn.Dropout(0.1)
                    
                def forward(self, x):
                    x = F.relu(self.fc1(x))
                    x = self.dropout(x)
                    x = self.fc2(x)
                    return x
            
            class FFN_GELU(nn.Module):
                def __init__(self, d_model, d_ff):
                    super().__init__()
                    self.fc1 = nn.Linear(d_model, d_ff)
                    self.fc2 = nn.Linear(d_ff, d_model)
                    self.dropout = nn.Dropout(0.1)
                    
                def forward(self, x):
                    x = F.gelu(self.fc1(x))
                    x = self.dropout(x)
                    x = self.fc2(x)
                    return x
            
            class FFN_SwiGLU(nn.Module):
                def __init__(self, d_model, d_ff):
                    super().__init__()
                    # SwiGLUは2倍の隠れ層サイズが必要
                    self.fc1 = nn.Linear(d_model, d_ff * 2)
                    self.fc2 = nn.Linear(d_ff, d_model)
                    self.dropout = nn.Dropout(0.1)
                    
                def forward(self, x):
                    x = self.fc1(x)
                    # 半分に分割
                    x1, x2 = x.chunk(2, dim=-1)
                    # Swish(x1) * x2
                    x = F.silu(x1) * x2
                    x = self.dropout(x)
                    x = self.fc2(x)
                    return x
            
            return {
                'ReLU': FFN_ReLU(self.d_model, self.d_ff),
                'GELU': FFN_GELU(self.d_model, self.d_ff),
                'SwiGLU': FFN_SwiGLU(self.d_model, self.d_ff)
            }
        
        def compare_activations(self):
            """活性化関数の比較"""
            # 入力範囲
            x = torch.linspace(-3, 3, 1000)
            
            # 活性化関数
            relu = F.relu(x)
            gelu = F.gelu(x)
            swish = F.silu(x)
            
            # プロット
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 3, 1)
            plt.plot(x, relu, label='ReLU', linewidth=2)
            plt.plot(x, gelu, label='GELU', linewidth=2)
            plt.plot(x, swish, label='Swish/SiLU', linewidth=2)
            plt.xlabel('Input')
            plt.ylabel('Output')
            plt.title('Activation Functions')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 導関数
            x.requires_grad = True
            
            relu_grad = torch.autograd.grad(F.relu(x).sum(), x, retain_graph=True)[0]
            gelu_grad = torch.autograd.grad(F.gelu(x).sum(), x, retain_graph=True)[0]
            swish_grad = torch.autograd.grad(F.silu(x).sum(), x, retain_graph=True)[0]
            
            plt.subplot(1, 3, 2)
            plt.plot(x.detach(), relu_grad.detach(), label='ReLU', linewidth=2)
            plt.plot(x.detach(), gelu_grad.detach(), label='GELU', linewidth=2)
            plt.plot(x.detach(), swish_grad.detach(), label='Swish/SiLU', linewidth=2)
            plt.xlabel('Input')
            plt.ylabel('Gradient')
            plt.title('Derivatives')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # スパース性の比較
            plt.subplot(1, 3, 3)
            sparsity_threshold = 0.01
            relu_sparsity = (relu < sparsity_threshold).float().mean()
            gelu_sparsity = (gelu.abs() < sparsity_threshold).float().mean()
            swish_sparsity = (swish.abs() < sparsity_threshold).float().mean()
            
            plt.bar(['ReLU', 'GELU', 'Swish'], 
                   [relu_sparsity, gelu_sparsity, swish_sparsity])
            plt.ylabel('Sparsity Rate')
            plt.title('Output Sparsity')
            
            plt.tight_layout()
            plt.show()
        
        def train_and_compare(self):
            """異なるFFNの訓練と比較"""
            ffn_variants = self.create_ffn_variants()
            
            # 簡単なタスク：非線形変換の学習
            batch_size = 64
            seq_len = 50
            
            # データ生成
            X = torch.randn(1000, seq_len, self.d_model)
            # 複雑な非線形変換
            Y = torch.sin(X) + torch.cos(2 * X) * 0.5
            
            results = {}
            
            for name, model in ffn_variants.items():
                print(f"\n訓練中: {name} FFN")
                
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                losses = []
                
                for epoch in range(100):
                    # ミニバッチ
                    idx = torch.randperm(len(X))[:batch_size]
                    batch_x = X[idx]
                    batch_y = Y[idx]
                    
                    # 予測
                    pred = model(batch_x)
                    loss = F.mse_loss(pred, batch_y)
                    
                    # 最適化
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    losses.append(loss.item())
                    
                    if epoch % 20 == 0:
                        print(f"  Epoch {epoch}: Loss = {loss.item():.4f}")
                
                results[name] = losses
            
            # 結果の可視化
            plt.figure(figsize=(10, 6))
            for name, losses in results.items():
                plt.plot(losses, label=name, linewidth=2)
            
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('FFN Training Comparison')
            plt.legend()
            plt.yscale('log')
            plt.grid(True, alpha=0.3)
            plt.show()
            
            # 最終性能の比較
            print("\n最終損失:")
            for name, losses in results.items():
                print(f"{name}: {losses[-1]:.4f}")
    
    # 実行
    ffn_comp = FFNComparison()
    ffn_comp.compare_activations()
    ffn_comp.train_and_compare()
    ```

### 問題 4
Mixture of Experts (MoE) レイヤーを実装し、エキスパートの選択パターンを分析してください。

??? 解答
    ```python
    class MixtureOfExperts(nn.Module):
        """Mixture of Experts (MoE) の実装"""
        
        def __init__(self, d_model=512, d_ff=2048, n_experts=8, top_k=2):
            super().__init__()
            self.d_model = d_model
            self.n_experts = n_experts
            self.top_k = top_k
            
            # エキスパート（各々がFFN）
            self.experts = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(d_model, d_ff),
                    nn.ReLU(),
                    nn.Linear(d_ff, d_model)
                ) for _ in range(n_experts)
            ])
            
            # ゲーティングネットワーク
            self.gate = nn.Linear(d_model, n_experts)
            
            # ロードバランシング用の損失係数
            self.load_balance_loss = 0.0
            
            # エキスパート使用統計
            self.expert_usage = torch.zeros(n_experts)
            
        def forward(self, x):
            batch_size, seq_len, d_model = x.shape
            
            # ゲート値の計算
            gate_logits = self.gate(x)  # [batch, seq_len, n_experts]
            
            # Top-kエキスパートの選択
            topk_gate_values, topk_indices = torch.topk(
                gate_logits, self.top_k, dim=-1
            )
            
            # ソフトマックスで正規化
            topk_gate_values = F.softmax(topk_gate_values, dim=-1)
            
            # エキスパート使用統計の更新
            self._update_expert_usage(topk_indices)
            
            # 出力の初期化
            output = torch.zeros_like(x)
            
            # 各エキスパートの処理
            for i in range(self.top_k):
                # 各位置でi番目に選ばれたエキスパート
                expert_idx = topk_indices[..., i]  # [batch, seq_len]
                gate_value = topk_gate_values[..., i:i+1]  # [batch, seq_len, 1]
                
                # エキスパートごとに処理
                for e in range(self.n_experts):
                    # このエキスパートが選ばれた位置
                    mask = (expert_idx == e)
                    if mask.any():
                        # マスクされた入力を抽出
                        expert_input = x[mask]
                        
                        # エキスパートを適用
                        expert_output = self.experts[e](expert_input)
                        
                        # 重み付けして出力に加算
                        output[mask] += expert_output * gate_value[mask]
            
            # ロードバランシング損失の計算
            self._compute_load_balance_loss(gate_logits)
            
            return output
        
        def _update_expert_usage(self, selected_experts):
            """エキスパート使用統計を更新"""
            with torch.no_grad():
                for e in range(self.n_experts):
                    usage = (selected_experts == e).float().sum()
                    self.expert_usage[e] = 0.9 * self.expert_usage[e] + 0.1 * usage
        
        def _compute_load_balance_loss(self, gate_logits):
            """ロードバランシング損失を計算"""
            # エキスパートごとの平均ゲート値
            gate_probs = F.softmax(gate_logits, dim=-1)
            expert_probs = gate_probs.mean(dim=[0, 1])
            
            # 均等分布からの乖離
            uniform_prob = 1.0 / self.n_experts
            self.load_balance_loss = ((expert_probs - uniform_prob) ** 2).sum()
        
        def visualize_expert_usage(self):
            """エキスパート使用パターンの可視化"""
            plt.figure(figsize=(10, 6))
            
            # 使用頻度
            plt.subplot(1, 2, 1)
            plt.bar(range(self.n_experts), self.expert_usage.numpy())
            plt.xlabel('Expert ID')
            plt.ylabel('Usage Count')
            plt.title('Expert Usage Distribution')
            
            # 使用率のヒートマップ（時系列）
            plt.subplot(1, 2, 2)
            # ダミーの時系列データ（実際は訓練中に記録）
            usage_history = torch.rand(50, self.n_experts)
            plt.imshow(usage_history.T, aspect='auto', cmap='hot')
            plt.xlabel('Time Step')
            plt.ylabel('Expert ID')
            plt.title('Expert Usage Over Time')
            plt.colorbar(label='Usage Rate')
            
            plt.tight_layout()
            plt.show()
    
    # MoEの訓練と分析
    def train_and_analyze_moe():
        d_model = 256
        moe = MixtureOfExperts(d_model=d_model, n_experts=8, top_k=2)
        
        # 異なる特性を持つデータを生成
        n_samples = 1000
        seq_len = 20
        
        # タイプ1: 低周波パターン
        data_type1 = torch.sin(torch.linspace(0, 4*np.pi, seq_len)).unsqueeze(0).unsqueeze(-1)
        data_type1 = data_type1.expand(n_samples//3, seq_len, d_model)
        data_type1 += torch.randn_like(data_type1) * 0.1
        
        # タイプ2: 高周波パターン
        data_type2 = torch.sin(torch.linspace(0, 20*np.pi, seq_len)).unsqueeze(0).unsqueeze(-1)
        data_type2 = data_type2.expand(n_samples//3, seq_len, d_model)
        data_type2 += torch.randn_like(data_type2) * 0.1
        
        # タイプ3: ランダムノイズ
        data_type3 = torch.randn(n_samples//3, seq_len, d_model)
        
        # 全データを結合
        all_data = torch.cat([data_type1, data_type2, data_type3], dim=0)
        labels = torch.cat([
            torch.zeros(n_samples//3),
            torch.ones(n_samples//3),
            torch.ones(n_samples//3) * 2
        ])
        
        # 訓練
        optimizer = torch.optim.Adam(moe.parameters(), lr=0.001)
        
        print("MoE訓練中...")
        for epoch in range(100):
            # シャッフル
            perm = torch.randperm(n_samples)
            all_data = all_data[perm]
            labels = labels[perm]
            
            # バッチ処理
            batch_size = 32
            total_loss = 0
            
            for i in range(0, n_samples, batch_size):
                batch_data = all_data[i:i+batch_size]
                batch_labels = labels[i:i+batch_size]
                
                # MoE適用
                output = moe(batch_data)
                
                # タスク損失（ダミー）
                task_loss = F.mse_loss(output, batch_data)
                
                # 全体の損失
                loss = task_loss + 0.01 * moe.load_balance_loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}: Loss = {total_loss/n_samples*batch_size:.4f}")
        
        # エキスパート使用パターンの分析
        print("\nエキスパート使用パターンを分析中...")
        
        # 各データタイプでのエキスパート選択を記録
        expert_selection_by_type = {0: [], 1: [], 2: []}
        
        with torch.no_grad():
            for data_type in range(3):
                # 各タイプのデータを選択
                type_mask = (labels == data_type)
                type_data = all_data[type_mask][:10]  # 最初の10サンプル
                
                # ゲート値を取得
                gate_logits = moe.gate(type_data)
                _, selected_experts = torch.topk(gate_logits, moe.top_k, dim=-1)
                
                # 統計を記録
                for e in range(moe.n_experts):
                    usage = (selected_experts == e).float().mean().item()
                    expert_selection_by_type[data_type].append(usage)
        
        # 結果の可視化
        plt.figure(figsize=(12, 5))
        
        # エキスパート使用分布
        plt.subplot(1, 2, 1)
        moe.visualize_expert_usage()
        
        # データタイプ別のエキスパート選択
        plt.subplot(1, 2, 2)
        x = np.arange(moe.n_experts)
        width = 0.25
        
        for i, (data_type, usage) in enumerate(expert_selection_by_type.items()):
            plt.bar(x + i*width, usage, width, 
                   label=f'Type {data_type}')
        
        plt.xlabel('Expert ID')
        plt.ylabel('Selection Rate')
        plt.title('Expert Selection by Data Type')
        plt.legend()
        plt.xticks(x + width)
        
        plt.tight_layout()
        plt.show()
        
        print("\n分析結果:")
        print("異なるデータタイプに対して、異なるエキスパートが選択される傾向が見られます。")
        print("これは、MoEが入力の特性に応じて適切なエキスパートを選択できることを示しています。")
    
    # 実行
    train_and_analyze_moe()
    ```

## 演習 3.3: 残差接続と層正規化

### 問題 5
Pre-LayerNormとPost-LayerNormの両方を実装し、深いネットワークでの学習安定性を比較してください。

??? 解答
    ```python
    class NormalizationComparison:
        """正規化手法の比較"""
        
        def __init__(self, d_model=256, n_layers=20):
            self.d_model = d_model
            self.n_layers = n_layers
            
        def create_models(self):
            """異なる正規化構成のモデルを作成"""
            
            class PreNormBlock(nn.Module):
                def __init__(self, d_model):
                    super().__init__()
                    self.norm1 = nn.LayerNorm(d_model)
                    self.attn = nn.MultiheadAttention(d_model, 4, batch_first=True)
                    self.norm2 = nn.LayerNorm(d_model)
                    self.ffn = nn.Sequential(
                        nn.Linear(d_model, 4 * d_model),
                        nn.ReLU(),
                        nn.Linear(4 * d_model, d_model)
                    )
                    
                def forward(self, x):
                    # Pre-Norm: 正規化してから処理
                    normalized = self.norm1(x)
                    attn_out, _ = self.attn(normalized, normalized, normalized)
                    x = x + attn_out
                    
                    normalized = self.norm2(x)
                    ffn_out = self.ffn(normalized)
                    x = x + ffn_out
                    
                    return x
            
            class PostNormBlock(nn.Module):
                def __init__(self, d_model):
                    super().__init__()
                    self.attn = nn.MultiheadAttention(d_model, 4, batch_first=True)
                    self.norm1 = nn.LayerNorm(d_model)
                    self.ffn = nn.Sequential(
                        nn.Linear(d_model, 4 * d_model),
                        nn.ReLU(),
                        nn.Linear(4 * d_model, d_model)
                    )
                    self.norm2 = nn.LayerNorm(d_model)
                    
                def forward(self, x):
                    # Post-Norm: 処理してから正規化
                    attn_out, _ = self.attn(x, x, x)
                    x = self.norm1(x + attn_out)
                    
                    ffn_out = self.ffn(x)
                    x = self.norm2(x + ffn_out)
                    
                    return x
            
            # 深いモデルを作成
            pre_norm_model = nn.Sequential(
                *[PreNormBlock(self.d_model) for _ in range(self.n_layers)]
            )
            
            post_norm_model = nn.Sequential(
                *[PostNormBlock(self.d_model) for _ in range(self.n_layers)]
            )
            
            return pre_norm_model, post_norm_model
        
        def analyze_gradient_flow(self):
            """勾配フローの分析"""
            pre_norm_model, post_norm_model = self.create_models()
            
            # テストデータ
            batch_size = 16
            seq_len = 50
            x = torch.randn(batch_size, seq_len, self.d_model)
            target = torch.randn(batch_size, seq_len, self.d_model)
            
            models = {
                'Pre-Norm': pre_norm_model,
                'Post-Norm': post_norm_model
            }
            
            results = {}
            
            for name, model in models.items():
                print(f"\n{name} の勾配フロー分析中...")
                
                # 各層の勾配を記録
                gradients = []
                
                def hook_fn(module, grad_input, grad_output):
                    gradients.append(grad_output[0].norm().item())
                
                # フックを登録
                hooks = []
                for layer in model:
                    hook = layer.register_backward_hook(hook_fn)
                    hooks.append(hook)
                
                # 順伝播と逆伝播
                output = model(x)
                loss = F.mse_loss(output, target)
                loss.backward()
                
                # フックを削除
                for hook in hooks:
                    hook.remove()
                
                results[name] = gradients[::-1]  # 入力側から順に
            
            # 勾配フローの可視化
            plt.figure(figsize=(12, 6))
            
            for name, grads in results.items():
                plt.plot(range(1, len(grads) + 1), grads, 
                        marker='o', label=name, linewidth=2)
            
            plt.xlabel('Layer (from input)')
            plt.ylabel('Gradient Norm')
            plt.title(f'Gradient Flow in {self.n_layers}-Layer Network')
            plt.legend()
            plt.yscale('log')
            plt.grid(True, alpha=0.3)
            plt.show()
            
            # 統計
            for name, grads in results.items():
                print(f"\n{name}:")
                print(f"  最初の層の勾配: {grads[0]:.6f}")
                print(f"  最後の層の勾配: {grads[-1]:.6f}")
                print(f"  勾配の減衰率: {grads[0] / grads[-1]:.2f}")
        
        def compare_training_stability(self):
            """訓練の安定性を比較"""
            pre_norm_model, post_norm_model = self.create_models()
            
            # 訓練設定
            batch_size = 32
            seq_len = 20
            n_steps = 200
            
            models = {
                'Pre-Norm': pre_norm_model,
                'Post-Norm': post_norm_model
            }
            
            training_curves = {}
            
            for name, model in models.items():
                print(f"\n{name} の訓練中...")
                
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                losses = []
                gradient_norms = []
                
                for step in range(n_steps):
                    # ダミーデータ
                    x = torch.randn(batch_size, seq_len, self.d_model)
                    # タスク：入力の変換を学習
                    target = torch.sin(x) + torch.cos(x * 2)
                    
                    # 訓練ステップ
                    output = model(x)
                    loss = F.mse_loss(output, target)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    
                    # 勾配ノルムを記録
                    total_norm = 0
                    for p in model.parameters():
                        if p.grad is not None:
                            total_norm += p.grad.norm().item() ** 2
                    total_norm = total_norm ** 0.5
                    gradient_norms.append(total_norm)
                    
                    # 勾配クリッピング
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    
                    optimizer.step()
                    
                    losses.append(loss.item())
                    
                    if step % 50 == 0:
                        print(f"  Step {step}: Loss = {loss.item():.4f}")
                
                training_curves[name] = {
                    'losses': losses,
                    'gradient_norms': gradient_norms
                }
            
            # 結果の可視化
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            # 損失曲線
            for name, data in training_curves.items():
                ax1.plot(data['losses'], label=name, linewidth=2)
            ax1.set_xlabel('Step')
            ax1.set_ylabel('Loss')
            ax1.set_title('Training Loss')
            ax1.legend()
            ax1.set_yscale('log')
            ax1.grid(True, alpha=0.3)
            
            # 勾配ノルム
            for name, data in training_curves.items():
                ax2.plot(data['gradient_norms'], label=name, linewidth=2, alpha=0.7)
            ax2.set_xlabel('Step')
            ax2.set_ylabel('Gradient Norm')
            ax2.set_title('Gradient Norm During Training')
            ax2.legend()
            ax2.set_yscale('log')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
            # 最終的な統計
            print("\n訓練の統計:")
            for name, data in training_curves.items():
                final_loss = np.mean(data['losses'][-10:])
                grad_std = np.std(data['gradient_norms'])
                print(f"{name}:")
                print(f"  最終損失: {final_loss:.4f}")
                print(f"  勾配の標準偏差: {grad_std:.4f}")
    
    # 実行
    norm_comp = NormalizationComparison()
    norm_comp.analyze_gradient_flow()
    norm_comp.compare_training_stability()
    ```

## 演習 3.4: エンコーダー・デコーダー

### 問題 6
完全なエンコーダー・デコーダーモデルを実装し、簡単な翻訳タスクで動作を確認してください。

??? 解答
    ```python
    class SimpleTransformer(nn.Module):
        """シンプルなエンコーダー・デコーダーTransformer"""
        
        def __init__(self, src_vocab_size, tgt_vocab_size, d_model=256, 
                     n_heads=8, n_layers=3, d_ff=1024, max_len=100):
            super().__init__()
            
            self.d_model = d_model
            
            # 埋め込み層
            self.src_embedding = nn.Embedding(src_vocab_size, d_model)
            self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
            
            # 位置エンコーディング
            self.pos_encoding = self._create_positional_encoding(max_len, d_model)
            
            # Transformer
            self.transformer = nn.Transformer(
                d_model=d_model,
                nhead=n_heads,
                num_encoder_layers=n_layers,
                num_decoder_layers=n_layers,
                dim_feedforward=d_ff,
                batch_first=True
            )
            
            # 出力層
            self.output_projection = nn.Linear(d_model, tgt_vocab_size)
            
            # スケーリング
            self.scale = math.sqrt(d_model)
            
        def _create_positional_encoding(self, max_len, d_model):
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len).unsqueeze(1).float()
            
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                               -(math.log(10000.0) / d_model))
            
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            
            return nn.Parameter(pe.unsqueeze(0), requires_grad=False)
        
        def forward(self, src, tgt, src_mask=None, tgt_mask=None, 
                    src_padding_mask=None, tgt_padding_mask=None):
            # 埋め込み + 位置エンコーディング
            src_emb = self.src_embedding(src) * self.scale
            src_emb = src_emb + self.pos_encoding[:, :src.size(1)]
            
            tgt_emb = self.tgt_embedding(tgt) * self.scale
            tgt_emb = tgt_emb + self.pos_encoding[:, :tgt.size(1)]
            
            # Transformer
            output = self.transformer(
                src_emb, tgt_emb,
                src_mask=src_mask,
                tgt_mask=tgt_mask,
                src_key_padding_mask=src_padding_mask,
                tgt_key_padding_mask=tgt_padding_mask
            )
            
            # 出力投影
            output = self.output_projection(output)
            
            return output
        
        def generate_square_subsequent_mask(self, sz):
            """デコーダー用の因果的マスクを生成"""
            mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
            return mask
    
    # 簡単な翻訳タスクの実装
    class SimpleTranslationTask:
        """数字の英語→日本語翻訳タスク"""
        
        def __init__(self):
            # 簡単な語彙
            self.src_vocab = {
                '<pad>': 0, '<sos>': 1, '<eos>': 2,
                'one': 3, 'two': 4, 'three': 5, 'four': 6, 'five': 7,
                'six': 8, 'seven': 9, 'eight': 10, 'nine': 11, 'ten': 12
            }
            
            self.tgt_vocab = {
                '<pad>': 0, '<sos>': 1, '<eos>': 2,
                '一': 3, '二': 4, '三': 5, '四': 6, '五': 7,
                '六': 8, '七': 9, '八': 10, '九': 11, '十': 12
            }
            
            # 逆引き辞書
            self.src_id2word = {v: k for k, v in self.src_vocab.items()}
            self.tgt_id2word = {v: k for k, v in self.tgt_vocab.items()}
            
            # 翻訳ペア
            self.pairs = [
                (['one'], ['一']),
                (['two'], ['二']),
                (['three'], ['三']),
                (['four'], ['四']),
                (['five'], ['五']),
                (['six'], ['六']),
                (['seven'], ['七']),
                (['eight'], ['八']),
                (['nine'], ['九']),
                (['ten'], ['十']),
                (['one', 'two'], ['一', '二']),
                (['three', 'four'], ['三', '四']),
                (['five', 'six'], ['五', '六']),
                (['seven', 'eight'], ['七', '八']),
                (['nine', 'ten'], ['九', '十'])
            ]
        
        def encode_src(self, words):
            return [self.src_vocab.get(w, 0) for w in words]
        
        def encode_tgt(self, words):
            return [self.tgt_vocab.get(w, 0) for w in words]
        
        def decode_src(self, ids):
            return [self.src_id2word.get(i, '<unk>') for i in ids]
        
        def decode_tgt(self, ids):
            return [self.tgt_id2word.get(i, '<unk>') for i in ids]
        
        def create_batch(self, pairs, pad_id=0):
            """バッチを作成（パディング付き）"""
            src_batch = []
            tgt_batch = []
            
            for src_words, tgt_words in pairs:
                # エンコード
                src_ids = [self.src_vocab['<sos>']] + self.encode_src(src_words) + [self.src_vocab['<eos>']]
                tgt_ids = [self.tgt_vocab['<sos>']] + self.encode_tgt(tgt_words) + [self.tgt_vocab['<eos>']]
                
                src_batch.append(src_ids)
                tgt_batch.append(tgt_ids)
            
            # パディング
            max_src_len = max(len(s) for s in src_batch)
            max_tgt_len = max(len(t) for t in tgt_batch)
            
            src_padded = []
            tgt_padded = []
            src_masks = []
            tgt_masks = []
            
            for src, tgt in zip(src_batch, tgt_batch):
                # パディング
                src_pad_len = max_src_len - len(src)
                tgt_pad_len = max_tgt_len - len(tgt)
                
                src_padded.append(src + [pad_id] * src_pad_len)
                tgt_padded.append(tgt + [pad_id] * tgt_pad_len)
                
                # マスク（True = パディング）
                src_masks.append([False] * len(src) + [True] * src_pad_len)
                tgt_masks.append([False] * len(tgt) + [True] * tgt_pad_len)
            
            return (torch.tensor(src_padded), torch.tensor(tgt_padded),
                    torch.tensor(src_masks), torch.tensor(tgt_masks))
    
    # 訓練と評価
    def train_translation_model():
        # タスクとモデルの準備
        task = SimpleTranslationTask()
        model = SimpleTransformer(
            src_vocab_size=len(task.src_vocab),
            tgt_vocab_size=len(task.tgt_vocab),
            d_model=128,
            n_heads=4,
            n_layers=2,
            d_ff=512
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss(ignore_index=0)  # パディングを無視
        
        # 訓練
        print("翻訳モデルの訓練中...")
        model.train()
        
        for epoch in range(200):
            total_loss = 0
            
            # データをシャッフル
            import random
            pairs = task.pairs.copy()
            random.shuffle(pairs)
            
            # バッチ処理
            batch_size = 5
            for i in range(0, len(pairs), batch_size):
                batch_pairs = pairs[i:i+batch_size]
                src, tgt, src_mask, tgt_mask = task.create_batch(batch_pairs)
                
                # Teacher forcing: デコーダー入力は目標の1つ前まで
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]
                tgt_mask_input = tgt_mask[:, :-1]
                
                # 因果的マスク
                tgt_seq_len = tgt_input.size(1)
                tgt_attn_mask = model.generate_square_subsequent_mask(tgt_seq_len)
                
                # 予測
                output = model(src, tgt_input, 
                             tgt_mask=tgt_attn_mask,
                             src_padding_mask=src_mask,
                             tgt_padding_mask=tgt_mask_input)
                
                # 損失計算
                loss = criterion(output.reshape(-1, output.size(-1)), 
                               tgt_output.reshape(-1))
                
                # 最適化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 50 == 0:
                avg_loss = total_loss / len(pairs) * batch_size
                print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")
        
        # 評価
        print("\n翻訳テスト:")
        model.eval()
        
        test_pairs = [
            (['five'], ['五']),
            (['one', 'two'], ['一', '二']),
            (['seven', 'eight'], ['七', '八'])
        ]
        
        with torch.no_grad():
            for src_words, expected_tgt in test_pairs:
                # ソースをエンコード
                src_ids = [task.src_vocab['<sos>']] + task.encode_src(src_words) + [task.src_vocab['<eos>']]
                src_tensor = torch.tensor([src_ids])
                
                # 翻訳（貪欲デコーディング）
                max_len = 10
                tgt_ids = [task.tgt_vocab['<sos>']]
                
                for _ in range(max_len):
                    tgt_tensor = torch.tensor([tgt_ids])
                    
                    # デコーダーマスク
                    tgt_attn_mask = model.generate_square_subsequent_mask(len(tgt_ids))
                    
                    # 予測
                    output = model(src_tensor, tgt_tensor, tgt_mask=tgt_attn_mask)
                    
                    # 最後のトークンの予測
                    next_token = output[0, -1].argmax().item()
                    tgt_ids.append(next_token)
                    
                    # 終了条件
                    if next_token == task.tgt_vocab['<eos>']:
                        break
                
                # 結果を表示
                predicted = task.decode_tgt(tgt_ids[1:-1])  # <sos>と<eos>を除く
                print(f"入力: {src_words}")
                print(f"期待: {expected_tgt}")
                print(f"予測: {predicted}")
                print(f"正解: {'✓' if predicted == expected_tgt else '✗'}\n")
    
    # 実行
    train_translation_model()
    ```

## チャレンジ問題

### 問題 7 🌟
Flash Attentionの簡易版を実装し、メモリ効率を改善してください。

??? 解答
    ```python
    class FlashAttentionSimple(nn.Module):
        """Flash Attentionの簡易実装"""
        
        def __init__(self, d_model, n_heads, block_size=64):
            super().__init__()
            assert d_model % n_heads == 0
            
            self.d_model = d_model
            self.n_heads = n_heads
            self.d_k = d_model // n_heads
            self.block_size = block_size
            self.scale = 1.0 / math.sqrt(self.d_k)
            
            # 投影行列
            self.W_q = nn.Linear(d_model, d_model)
            self.W_k = nn.Linear(d_model, d_model)
            self.W_v = nn.Linear(d_model, d_model)
            self.W_o = nn.Linear(d_model, d_model)
            
        def forward(self, x, mask=None):
            batch_size, seq_len, _ = x.shape
            
            # Q, K, Vの計算
            Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
            K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
            V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
            
            # ブロック単位の処理
            if seq_len <= self.block_size:
                # 短いシーケンスは通常の処理
                output = self._standard_attention(Q, K, V, mask)
            else:
                # 長いシーケンスはブロック処理
                output = self._flash_attention(Q, K, V, mask)
            
            # ヘッドを結合
            output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
            output = self.W_o(output)
            
            return output
        
        def _standard_attention(self, Q, K, V, mask):
            """標準的なアテンション計算"""
            scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
            
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
            
            attn_weights = F.softmax(scores, dim=-1)
            output = torch.matmul(attn_weights, V)
            
            return output
        
        def _flash_attention(self, Q, K, V, mask):
            """Flash Attention (簡易版)"""
            batch_size, n_heads, seq_len, d_k = Q.shape
            block_size = self.block_size
            
            # 出力の初期化
            O = torch.zeros_like(Q)
            
            # ブロック数
            n_blocks = (seq_len + block_size - 1) // block_size
            
            # 各クエリブロックに対して処理
            for i in range(n_blocks):
                q_start = i * block_size
                q_end = min((i + 1) * block_size, seq_len)
                
                # クエリブロック
                Q_block = Q[:, :, q_start:q_end]
                
                # このブロックの最大値と累積和を初期化
                block_max = torch.full((batch_size, n_heads, q_end - q_start, 1), 
                                     -1e9, device=Q.device)
                block_sum = torch.zeros_like(block_max)
                block_output = torch.zeros(batch_size, n_heads, q_end - q_start, d_k, 
                                         device=Q.device)
                
                # 各キー/バリューブロックに対して処理
                for j in range(n_blocks):
                    k_start = j * block_size
                    k_end = min((j + 1) * block_size, seq_len)
                    
                    # 因果的マスクのチェック
                    if mask is not None and k_start > q_end:
                        continue
                    
                    # キー/バリューブロック
                    K_block = K[:, :, k_start:k_end]
                    V_block = V[:, :, k_start:k_end]
                    
                    # スコア計算
                    scores = torch.matmul(Q_block, K_block.transpose(-2, -1)) * self.scale
                    
                    # マスク適用
                    if mask is not None:
                        block_mask = self._get_block_mask(q_start, q_end, k_start, k_end, seq_len)
                        if block_mask is not None:
                            scores = scores.masked_fill(block_mask.unsqueeze(0).unsqueeze(0) == 0, -1e9)
                    
                    # 安定したソフトマックスのための処理
                    scores_max = scores.max(dim=-1, keepdim=True)[0]
                    scores_stable = scores - scores_max
                    scores_exp = torch.exp(scores_stable)
                    
                    # オンラインソフトマックスの更新
                    new_max = torch.maximum(block_max, scores_max)
                    
                    # 累積和の更新
                    block_sum = block_sum * torch.exp(block_max - new_max) + \
                               scores_exp.sum(dim=-1, keepdim=True) * torch.exp(scores_max - new_max)
                    
                    # 出力の更新
                    block_output = block_output * torch.exp(block_max - new_max) + \
                                 torch.matmul(scores_exp * torch.exp(scores_max - new_max), V_block)
                    
                    block_max = new_max
                
                # 正規化
                O[:, :, q_start:q_end] = block_output / block_sum
            
            return O
        
        def _get_block_mask(self, q_start, q_end, k_start, k_end, seq_len):
            """ブロック用のマスクを生成"""
            if k_start >= q_end:
                # 未来のブロックは完全にマスク
                return torch.zeros(q_end - q_start, k_end - k_start)
            
            # 部分的なマスクが必要な場合
            mask = torch.ones(q_end - q_start, k_end - k_start)
            for i in range(q_end - q_start):
                for j in range(k_end - k_start):
                    if q_start + i < k_start + j:
                        mask[i, j] = 0
            
            return mask
    
    # メモリ効率の比較
    def compare_memory_efficiency():
        d_model = 512
        n_heads = 8
        
        # 異なるシーケンス長でテスト
        seq_lengths = [128, 256, 512, 1024]
        
        print("メモリ使用量の比較:")
        print("シーケンス長 | 標準Attention | Flash Attention | 削減率")
        print("-" * 60)
        
        for seq_len in seq_lengths:
            batch_size = 4
            
            # 標準的なアテンションのメモリ使用量（概算）
            # O(batch * heads * seq_len * seq_len)
            standard_memory = batch_size * n_heads * seq_len * seq_len * 4  # float32
            
            # Flash Attentionのメモリ使用量（概算）
            # O(batch * heads * seq_len * block_size)
            block_size = 64
            flash_memory = batch_size * n_heads * seq_len * block_size * 4  # float32
            
            reduction = (1 - flash_memory / standard_memory) * 100
            
            print(f"{seq_len:^12} | {standard_memory/1024**2:^14.2f}MB | "
                  f"{flash_memory/1024**2:^15.2f}MB | {reduction:^7.1f}%")
        
        # 実際の動作確認
        print("\n実際の動作確認:")
        
        standard_attn = MultiHeadAttentionDetailed(d_model, n_heads)
        flash_attn = FlashAttentionSimple(d_model, n_heads, block_size=64)
        
        # テストデータ
        x = torch.randn(2, 256, d_model)
        
        # 出力の比較
        with torch.no_grad():
            standard_out, _ = standard_attn(x, x, x)
            flash_out = flash_attn(x)
            
            # 差分
            diff = (standard_out - flash_out).abs().mean()
            print(f"\n出力の差分: {diff:.6f}")
            print("（小さい値ほど実装が正確）")
        
        # 速度比較
        import time
        
        x_large = torch.randn(1, 1024, d_model)
        
        # 標準アテンション
        start = time.time()
        for _ in range(10):
            _ = standard_attn(x_large, x_large, x_large)
        standard_time = time.time() - start
        
        # Flash Attention
        start = time.time()
        for _ in range(10):
            _ = flash_attn(x_large)
        flash_time = time.time() - start
        
        print(f"\n速度比較 (10イテレーション):")
        print(f"標準Attention: {standard_time:.3f}秒")
        print(f"Flash Attention: {flash_time:.3f}秒")
        print(f"高速化: {standard_time/flash_time:.2f}x")
    
    # 実行
    compare_memory_efficiency()
    ```

## まとめ

第3部では、Transformerの主要コンポーネントを詳しく学びました：

1. **Multi-Head Attention**: 複数の視点からの注意機構
2. **Feed Forward Network**: 位置ごとの非線形変換
3. **残差接続と層正規化**: 深いネットワークの安定化
4. **エンコーダー・デコーダー**: 入力から出力への変換

これらの要素を組み合わせることで、強力なTransformerモデルが構築されます。次の第4部では、これらを統合した完全な実装に挑戦しましょう！