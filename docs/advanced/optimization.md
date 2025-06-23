# 最適化技術とベストプラクティス

## はじめに：効率的なTransformerの実装

コンパイラの最適化を考えてみてください。`-O0`でコンパイルしたコードと`-O3`でコンパイルしたコードでは、パフォーマンスが数倍から数十倍違います。同様に、Transformerも適切な最適化により劇的に高速化できます。

この章では、実践的な最適化技術とベストプラクティスを学びます。

## 1. メモリ最適化

### 1.1 Gradient Checkpointing

メモリ使用量を削減する代わりに計算時間が増加するトレードオフ技術です。

```python
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

class OptimizedTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None, use_checkpoint=True):
        # Gradient checkpointingを使用
        if use_checkpoint and self.training:
            # アテンション部分
            def attention_block(x, mask):
                return self.dropout(self.attention(self.norm1(x), mask))
            
            attn_output = checkpoint(attention_block, x, mask)
            x = x + attn_output
            
            # Feed-forward部分
            def ff_block(x):
                return self.dropout(self.feed_forward(self.norm2(x)))
            
            ff_output = checkpoint(ff_block, x)
            x = x + ff_output
        else:
            # 通常の計算
            x = x + self.dropout(self.attention(self.norm1(x), mask))
            x = x + self.dropout(self.feed_forward(self.norm2(x)))
        
        return x

# メモリ使用量の比較
def compare_memory_usage():
    model = OptimizedTransformerBlock(512, 8, 2048)
    x = torch.randn(32, 100, 512, requires_grad=True)
    
    # 通常の計算
    torch.cuda.reset_peak_memory_stats()
    output1 = model(x, use_checkpoint=False)
    loss1 = output1.sum()
    loss1.backward()
    memory_without_checkpoint = torch.cuda.max_memory_allocated() / 1024**2
    
    # Gradient checkpointing
    model.zero_grad()
    x.grad = None
    torch.cuda.reset_peak_memory_stats()
    output2 = model(x, use_checkpoint=True)
    loss2 = output2.sum()
    loss2.backward()
    memory_with_checkpoint = torch.cuda.max_memory_allocated() / 1024**2
    
    print(f"Memory without checkpoint: {memory_without_checkpoint:.2f} MB")
    print(f"Memory with checkpoint: {memory_with_checkpoint:.2f} MB")
    print(f"Memory saved: {(1 - memory_with_checkpoint/memory_without_checkpoint)*100:.1f}%")
```

### 1.2 効率的なアテンション実装

Flash AttentionやMemory-Efficient Attentionの実装：

```python
class FlashAttention(nn.Module):
    """Flash Attentionの簡易実装（概念的）"""
    
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.d_k)
        
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        B, T, C = x.shape
        
        # QKVを一度に計算（メモリアクセスを削減）
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, T, D]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Flash Attentionの核心：ブロック単位で計算
        # 実際の実装はCUDAカーネルで行われる
        if torch.cuda.is_available() and hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            # PyTorch 2.0+のFlash Attention
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, 
                attn_mask=mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=True if mask is None else False
            )
        else:
            # フォールバック実装
            attn_output = self._manual_attention(q, k, v, mask)
        
        # 出力を整形
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(attn_output)
    
    def _manual_attention(self, q, k, v, mask=None):
        """手動実装（教育目的）"""
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        return torch.matmul(attn_weights, v)
```

## 2. 計算最適化

### 2.1 Mixed Precision Training

半精度浮動小数点を使用して計算を高速化：

```python
from torch.cuda.amp import autocast, GradScaler

class MixedPrecisionTrainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.scaler = GradScaler()
        
    def train_step(self, batch):
        self.optimizer.zero_grad()
        
        # 自動混合精度
        with autocast():
            outputs = self.model(batch['input_ids'])
            loss = self.compute_loss(outputs, batch['labels'])
        
        # スケールされた逆伝播
        self.scaler.scale(loss).backward()
        
        # 勾配のアンスケールとクリッピング
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # オプティマイザステップ
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return loss.item()
    
    def compute_loss(self, outputs, labels):
        return torch.nn.functional.cross_entropy(
            outputs.view(-1, outputs.size(-1)),
            labels.view(-1)
        )

# 速度比較
def benchmark_mixed_precision():
    import time
    
    model = TransformerModel(vocab_size=50000, d_model=512, n_heads=8, n_layers=6)
    model = model.cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # ダミーデータ
    batch = {
        'input_ids': torch.randint(0, 50000, (32, 128)).cuda(),
        'labels': torch.randint(0, 50000, (32, 128)).cuda()
    }
    
    # 通常の精度
    start_time = time.time()
    for _ in range(100):
        optimizer.zero_grad()
        outputs = model(batch['input_ids'])
        loss = torch.nn.functional.cross_entropy(
            outputs.view(-1, outputs.size(-1)),
            batch['labels'].view(-1)
        )
        loss.backward()
        optimizer.step()
    fp32_time = time.time() - start_time
    
    # Mixed precision
    trainer = MixedPrecisionTrainer(model, optimizer)
    start_time = time.time()
    for _ in range(100):
        trainer.train_step(batch)
    amp_time = time.time() - start_time
    
    print(f"FP32 time: {fp32_time:.2f}s")
    print(f"AMP time: {amp_time:.2f}s")
    print(f"Speedup: {fp32_time/amp_time:.2f}x")
```

### 2.2 効率的なバッチ処理

動的パディングとバケッティング：

```python
class EfficientDataLoader:
    """効率的なバッチ処理のためのデータローダー"""
    
    def __init__(self, dataset, batch_size, bucket_size=1000):
        self.dataset = dataset
        self.batch_size = batch_size
        self.bucket_size = bucket_size
        
    def __iter__(self):
        # 長さでソート
        indices = list(range(len(self.dataset)))
        indices.sort(key=lambda i: len(self.dataset[i]['input_ids']))
        
        # バケットに分割
        buckets = []
        for i in range(0, len(indices), self.bucket_size):
            bucket = indices[i:i + self.bucket_size]
            # バケット内でシャッフル（多様性を保つ）
            random.shuffle(bucket)
            buckets.append(bucket)
        
        # バケットをシャッフル
        random.shuffle(buckets)
        
        # バッチを生成
        for bucket in buckets:
            for i in range(0, len(bucket), self.batch_size):
                batch_indices = bucket[i:i + self.batch_size]
                yield self._collate_batch(batch_indices)
    
    def _collate_batch(self, indices):
        """動的パディングでバッチを作成"""
        batch = [self.dataset[i] for i in indices]
        
        # 最大長を見つける
        max_len = max(len(item['input_ids']) for item in batch)
        
        # パディング
        input_ids = []
        attention_mask = []
        
        for item in batch:
            ids = item['input_ids']
            pad_len = max_len - len(ids)
            
            input_ids.append(ids + [0] * pad_len)
            attention_mask.append([1] * len(ids) + [0] * pad_len)
        
        return {
            'input_ids': torch.tensor(input_ids),
            'attention_mask': torch.tensor(attention_mask)
        }
```

## 3. 分散学習

### 3.1 Data Parallel

複数GPUでの並列学習：

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

class DistributedTrainer:
    def __init__(self, model, rank, world_size):
        self.rank = rank
        self.world_size = world_size
        
        # 分散環境の初期化
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
        
        # モデルをGPUに配置
        torch.cuda.set_device(rank)
        model = model.cuda(rank)
        
        # DDPでラップ
        self.model = DDP(model, device_ids=[rank])
        
    def train(self, dataset, epochs=10):
        # 分散サンプラー
        sampler = DistributedSampler(
            dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True
        )
        
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=32,
            sampler=sampler,
            num_workers=4,
            pin_memory=True
        )
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        
        for epoch in range(epochs):
            # エポックごとにサンプラーを更新
            sampler.set_epoch(epoch)
            
            for batch in dataloader:
                loss = self.train_step(batch, optimizer)
                
                # すべてのプロセスで同期
                if self.rank == 0:
                    print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    def train_step(self, batch, optimizer):
        optimizer.zero_grad()
        
        outputs = self.model(batch['input_ids'].cuda(self.rank))
        loss = compute_loss(outputs, batch['labels'].cuda(self.rank))
        
        loss.backward()
        
        # 勾配を全プロセスで平均
        for param in self.model.parameters():
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= self.world_size
        
        optimizer.step()
        
        return loss.item()

# 使用例（マルチプロセスで実行）
def run_distributed_training(rank, world_size):
    model = TransformerModel(vocab_size=50000, d_model=512)
    trainer = DistributedTrainer(model, rank, world_size)
    trainer.train(dataset)
```

### 3.2 Model Parallel

モデルを複数GPUに分割：

```python
class ModelParallelTransformer(nn.Module):
    """シンプルなモデル並列の例"""
    
    def __init__(self, vocab_size, d_model, n_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model).cuda(0)
        
        # 層を2つのGPUに分割
        self.layers_gpu0 = nn.ModuleList([
            TransformerBlock(d_model) for _ in range(n_layers // 2)
        ]).cuda(0)
        
        self.layers_gpu1 = nn.ModuleList([
            TransformerBlock(d_model) for _ in range(n_layers // 2)
        ]).cuda(1)
        
        self.output_proj = nn.Linear(d_model, vocab_size).cuda(1)
        
    def forward(self, x):
        # GPU 0での計算
        x = self.embedding(x)
        for layer in self.layers_gpu0:
            x = layer(x)
        
        # GPU 1に転送
        x = x.cuda(1)
        
        # GPU 1での計算
        for layer in self.layers_gpu1:
            x = layer(x)
        
        return self.output_proj(x)
```

## 4. 推論最適化

### 4.1 KVキャッシュ

生成時の計算を削減：

```python
class CachedAttention(nn.Module):
    """KVキャッシュを使用したアテンション"""
    
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        # キャッシュ
        self.cache_k = None
        self.cache_v = None
        
    def forward(self, x, use_cache=False):
        B, T, C = x.shape
        
        # 新しいクエリ
        q = self.W_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        
        if use_cache and self.cache_k is not None:
            # キャッシュされたKVを使用
            k_new = self.W_k(x[:, -1:, :]).view(B, 1, self.n_heads, self.d_k).transpose(1, 2)
            v_new = self.W_v(x[:, -1:, :]).view(B, 1, self.n_heads, self.d_k).transpose(1, 2)
            
            k = torch.cat([self.cache_k, k_new], dim=2)
            v = torch.cat([self.cache_v, v_new], dim=2)
            
            # キャッシュを更新
            self.cache_k = k
            self.cache_v = v
            
            # 最後のトークンのクエリのみ使用
            q = q[:, :, -1:, :]
        else:
            # 通常の計算
            k = self.W_k(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
            v = self.W_v(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
            
            if use_cache:
                self.cache_k = k
                self.cache_v = v
        
        # アテンション計算
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, v)
        
        # 出力
        context = context.transpose(1, 2).contiguous().view(B, -1, C)
        return self.W_o(context)
    
    def clear_cache(self):
        """キャッシュをクリア"""
        self.cache_k = None
        self.cache_v = None
```

### 4.2 量子化

モデルサイズと計算量を削減：

```python
def quantize_model(model, bits=8):
    """シンプルな量子化の例"""
    
    class QuantizedLinear(nn.Module):
        def __init__(self, weight, bias, bits=8):
            super().__init__()
            self.bits = bits
            
            # 重みを量子化
            self.scale = weight.abs().max() / (2**(bits-1) - 1)
            self.zero_point = 0
            
            self.weight_int = torch.round(weight / self.scale).to(torch.int8)
            self.bias = bias
            
        def forward(self, x):
            # 逆量子化して計算
            weight = self.weight_int.float() * self.scale
            return torch.nn.functional.linear(x, weight, self.bias)
    
    # すべてのLinear層を量子化
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # 親モジュールを取得
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            parent = model
            
            if parent_name:
                for part in parent_name.split('.'):
                    parent = getattr(parent, part)
            
            # 量子化モジュールに置換
            quantized = QuantizedLinear(
                module.weight.data,
                module.bias.data if module.bias is not None else None,
                bits=bits
            )
            setattr(parent, child_name, quantized)
    
    return model

# 量子化の効果を測定
def measure_quantization_impact():
    model = TransformerModel(vocab_size=50000, d_model=512)
    
    # オリジナルモデルのサイズ
    original_size = sum(p.numel() * p.element_size() for p in model.parameters())
    
    # 量子化
    quantized_model = quantize_model(model, bits=8)
    
    # 量子化後のサイズ（概算）
    quantized_size = sum(
        p.numel() if p.dtype == torch.int8 else p.numel() * p.element_size()
        for p in quantized_model.parameters()
    )
    
    print(f"Original size: {original_size / 1024**2:.2f} MB")
    print(f"Quantized size: {quantized_size / 1024**2:.2f} MB")
    print(f"Compression ratio: {original_size / quantized_size:.2f}x")
```

## 5. プロファイリングとデバッグ

### 5.1 パフォーマンスプロファイリング

```python
import torch.profiler

def profile_model(model, input_data):
    """モデルのプロファイリング"""
    
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        with torch.profiler.record_function("model_inference"):
            for _ in range(10):
                output = model(input_data)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
    
    # 結果の分析
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    
    # TensorBoardに出力
    prof.export_chrome_trace("trace.json")
    
    # ボトルネックの特定
    for event in prof.key_averages():
        if event.cuda_time_total > 1000000:  # 1ms以上
            print(f"Bottleneck: {event.key} - {event.cuda_time_total/1000:.2f}ms")
```

### 5.2 メモリリークの検出

```python
class MemoryTracker:
    """メモリ使用量を追跡"""
    
    def __init__(self):
        self.snapshots = []
        
    def snapshot(self, label=""):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
        else:
            allocated = reserved = 0
            
        self.snapshots.append({
            'label': label,
            'allocated': allocated,
            'reserved': reserved
        })
        
    def report(self):
        print("=== Memory Usage Report ===")
        for i, snap in enumerate(self.snapshots):
            print(f"{i}: {snap['label']}")
            print(f"   Allocated: {snap['allocated']:.2f} MB")
            print(f"   Reserved: {snap['reserved']:.2f} MB")
            
            if i > 0:
                delta = snap['allocated'] - self.snapshots[i-1]['allocated']
                if delta > 0:
                    print(f"   Delta: +{delta:.2f} MB ⚠️")

# 使用例
tracker = MemoryTracker()
tracker.snapshot("Initial")

model = TransformerModel(vocab_size=50000, d_model=512)
tracker.snapshot("After model creation")

data = torch.randn(32, 100, 512)
tracker.snapshot("After data creation")

output = model(data)
tracker.snapshot("After forward pass")

loss = output.sum()
loss.backward()
tracker.snapshot("After backward pass")

tracker.report()
```

## まとめ

Transformerの最適化は、コンパイラの最適化と同様に、多層的なアプローチが必要です：

1. **アルゴリズムレベル**: Flash Attention、効率的なアーキテクチャ
2. **実装レベル**: Mixed Precision、Gradient Checkpointing
3. **システムレベル**: 分散学習、バッチ最適化
4. **ハードウェアレベル**: 量子化、カーネル融合

これらの技術を適切に組み合わせることで、Transformerの性能を大幅に向上させることができます。