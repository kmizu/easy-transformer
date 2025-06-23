# 第5部 演習問題

## 演習 5.1: GPTアーキテクチャ

### 問題 1
因果的（causal）マスキングを実装し、その効果を可視化してください。

??? 解答
    ```python
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    
    class CausalMasking:
        """因果的マスキングの実装と可視化"""
        
        @staticmethod
        def create_causal_mask(seq_len: int, device=None):
            """因果的マスクの作成"""
            # 下三角行列を作成（対角成分を含む）
            mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
            return mask
        
        @staticmethod
        def visualize_masks():
            """様々なマスキングパターンの可視化"""
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            seq_len = 10
            
            # 1. 因果的マスク
            causal_mask = CausalMasking.create_causal_mask(seq_len).numpy()
            sns.heatmap(causal_mask, ax=axes[0, 0], cmap='Blues', 
                       cbar=False, square=True, annot=True, fmt='.0f')
            axes[0, 0].set_title('Causal Mask (GPT-style)')
            
            # 2. 双方向マスク（BERT-style）
            bidirectional_mask = torch.ones(seq_len, seq_len).numpy()
            sns.heatmap(bidirectional_mask, ax=axes[0, 1], cmap='Blues', 
                       cbar=False, square=True, annot=True, fmt='.0f')
            axes[0, 1].set_title('Bidirectional Mask (BERT-style)')
            
            # 3. Prefix LMマスク
            prefix_len = 4
            prefix_mask = torch.ones(seq_len, seq_len)
            prefix_mask[prefix_len:, prefix_len:] = torch.tril(
                torch.ones(seq_len - prefix_len, seq_len - prefix_len)
            )
            sns.heatmap(prefix_mask.numpy(), ax=axes[0, 2], cmap='Blues', 
                       cbar=False, square=True, annot=True, fmt='.0f')
            axes[0, 2].set_title(f'Prefix LM Mask (prefix={prefix_len})')
            
            # 4. スライディングウィンドウマスク
            window_size = 3
            sliding_mask = torch.zeros(seq_len, seq_len)
            for i in range(seq_len):
                start = max(0, i - window_size + 1)
                end = min(seq_len, i + 1)
                sliding_mask[i, start:end] = 1
            sns.heatmap(sliding_mask.numpy(), ax=axes[1, 0], cmap='Blues', 
                       cbar=False, square=True, annot=True, fmt='.0f')
            axes[1, 0].set_title(f'Sliding Window (size={window_size})')
            
            # 5. ランダムマスク（ドロップアウト風）
            random_mask = (torch.rand(seq_len, seq_len) > 0.2).float()
            random_mask = torch.tril(random_mask)  # 因果性を保持
            sns.heatmap(random_mask.numpy(), ax=axes[1, 1], cmap='Blues', 
                       cbar=False, square=True, annot=True, fmt='.0f')
            axes[1, 1].set_title('Random Causal Mask (80% keep)')
            
            # 6. ブロック対角マスク
            block_size = 3
            block_mask = torch.zeros(seq_len, seq_len)
            for i in range(0, seq_len, block_size):
                end = min(i + block_size, seq_len)
                block_mask[i:end, i:end] = 1
            sns.heatmap(block_mask.numpy(), ax=axes[1, 2], cmap='Blues', 
                       cbar=False, square=True, annot=True, fmt='.0f')
            axes[1, 2].set_title(f'Block Diagonal (size={block_size})')
            
            plt.tight_layout()
            plt.show()
    
    class CausalGPTAttention(nn.Module):
        """因果的自己注意機構の実装"""
        
        def __init__(self, d_model, n_heads, max_len=1024, dropout=0.1):
            super().__init__()
            assert d_model % n_heads == 0
            
            self.d_model = d_model
            self.n_heads = n_heads
            self.d_k = d_model // n_heads
            self.scale = 1.0 / math.sqrt(self.d_k)
            
            self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
            self.out_proj = nn.Linear(d_model, d_model)
            self.dropout = nn.Dropout(dropout)
            
            # 因果的マスクをバッファとして登録
            self.register_buffer(
                'causal_mask',
                torch.tril(torch.ones(max_len, max_len)).unsqueeze(0).unsqueeze(0)
            )
        
        def forward(self, x, output_attentions=False):
            batch_size, seq_len, _ = x.shape
            
            # Q, K, V を計算
            qkv = self.qkv(x).chunk(3, dim=-1)
            q, k, v = [t.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2) 
                      for t in qkv]
            
            # アテンションスコア計算
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            
            # 因果的マスク適用
            causal_mask = self.causal_mask[:, :, :seq_len, :seq_len]
            scores = scores.masked_fill(causal_mask == 0, -1e9)
            
            # ソフトマックスとドロップアウト
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            # 値との積
            context = torch.matmul(attn_weights, v)
            context = context.transpose(1, 2).contiguous().view(
                batch_size, seq_len, self.d_model
            )
            
            output = self.out_proj(context)
            
            if output_attentions:
                return output, attn_weights
            return output
    
    def test_causal_masking():
        """因果的マスキングのテスト"""
        
        # マスクパターンの可視化
        CausalMasking.visualize_masks()
        
        # 実際のアテンション計算でのマスク効果
        d_model, n_heads = 64, 4
        model = CausalGPTAttention(d_model, n_heads)
        
        # テスト入力
        batch_size, seq_len = 2, 8
        x = torch.randn(batch_size, seq_len, d_model)
        
        # アテンション重みを取得
        output, attn_weights = model(x, output_attentions=True)
        
        # 最初のバッチ、最初のヘッドのアテンション重みを可視化
        plt.figure(figsize=(8, 6))
        sns.heatmap(attn_weights[0, 0].detach().numpy(), 
                   cmap='Blues', annot=True, fmt='.2f', square=True)
        plt.title('Causal Attention Weights (Batch 0, Head 0)')
        plt.xlabel('Key Position')
        plt.ylabel('Query Position')
        plt.show()
        
        # マスクの効果を確認（未来の情報が0になっているか）
        print("Attention weights upper triangle (should be ~0):")
        upper_triangle = torch.triu(attn_weights[0, 0], diagonal=1)
        print(upper_triangle)
        print(f"Max value in upper triangle: {upper_triangle.max().item():.6f}")
    
    # テスト実行
    test_causal_masking()
    ```

### 問題 2
GPTの位置エンコーディング（learned vs sinusoidal）を比較実装してください。

??? 解答
    ```python
    import torch
    import torch.nn as nn
    import numpy as np
    import matplotlib.pyplot as plt
    from typing import Dict, List
    
    class PositionalEncodingComparison:
        """位置エンコーディング手法の比較"""
        
        @staticmethod
        def create_sinusoidal_encoding(seq_len: int, d_model: int) -> torch.Tensor:
            """正弦波位置エンコーディング"""
            pe = torch.zeros(seq_len, d_model)
            position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
            
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                               (-math.log(10000.0) / d_model))
            
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            
            return pe
        
        @staticmethod
        def create_learned_encoding(seq_len: int, d_model: int) -> nn.Parameter:
            """学習可能な位置エンコーディング"""
            pe = nn.Parameter(torch.randn(seq_len, d_model) * 0.01)
            return pe
    
    class GPTWithPositionalEncoding(nn.Module):
        """異なる位置エンコーディングを持つGPTモデル"""
        
        def __init__(self, vocab_size, d_model, n_heads, n_layers, 
                     max_len=1024, encoding_type='sinusoidal'):
            super().__init__()
            
            self.d_model = d_model
            self.encoding_type = encoding_type
            
            # 埋め込み層
            self.token_embedding = nn.Embedding(vocab_size, d_model)
            
            # 位置エンコーディング
            if encoding_type == 'sinusoidal':
                pe = PositionalEncodingComparison.create_sinusoidal_encoding(max_len, d_model)
                self.register_buffer('position_encoding', pe)
            elif encoding_type == 'learned':
                self.position_encoding = PositionalEncodingComparison.create_learned_encoding(
                    max_len, d_model
                )
            elif encoding_type == 'rotary':
                self.rotary_emb = RotaryPositionalEmbedding(d_model // n_heads)
            
            # Transformerブロック
            self.blocks = nn.ModuleList([
                GPTBlock(d_model, n_heads) for _ in range(n_layers)
            ])
            
            self.ln_f = nn.LayerNorm(d_model)
            self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
            
        def forward(self, input_ids):
            batch_size, seq_len = input_ids.shape
            device = input_ids.device
            
            # トークン埋め込み
            x = self.token_embedding(input_ids)
            
            # 位置エンコーディングの追加
            if self.encoding_type in ['sinusoidal', 'learned']:
                positions = torch.arange(seq_len, device=device)
                x = x + self.position_encoding[positions]
            # rotaryの場合は各アテンション層で適用
            
            # Transformerブロック
            for block in self.blocks:
                if self.encoding_type == 'rotary':
                    x = block(x, rotary_emb=self.rotary_emb)
                else:
                    x = block(x)
            
            # 最終層正規化と出力
            x = self.ln_f(x)
            logits = self.lm_head(x)
            
            return logits
    
    class RotaryPositionalEmbedding(nn.Module):
        """Rotary Position Embedding (RoPE)"""
        
        def __init__(self, dim, max_position_embeddings=2048, base=10000):
            super().__init__()
            inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float() / dim))
            self.register_buffer('inv_freq', inv_freq)
            
            # キャッシュ
            self._cos_cached = None
            self._sin_cached = None
            self._seq_len_cached = None
            
        def forward(self, x, seq_len=None):
            if seq_len is None:
                seq_len = x.shape[1]
            
            if seq_len != self._seq_len_cached:
                self._seq_len_cached = seq_len
                t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
                freqs = torch.einsum('i,j->ij', t, self.inv_freq)
                emb = torch.cat((freqs, freqs), dim=-1)
                self._cos_cached = emb.cos()[None, :, None, :]
                self._sin_cached = emb.sin()[None, :, None, :]
            
            return self._cos_cached, self._sin_cached
        
        @staticmethod
        def rotate_half(x):
            x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
            return torch.cat((-x2, x1), dim=-1)
        
        def apply_rotary_pos_emb(self, q, k, cos, sin):
            q_embed = (q * cos) + (self.rotate_half(q) * sin)
            k_embed = (k * cos) + (self.rotate_half(k) * sin)
            return q_embed, k_embed
    
    class GPTBlock(nn.Module):
        """GPTのTransformerブロック"""
        
        def __init__(self, d_model, n_heads):
            super().__init__()
            self.ln_1 = nn.LayerNorm(d_model)
            self.attn = CausalGPTAttention(d_model, n_heads)
            self.ln_2 = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(
                nn.Linear(d_model, 4 * d_model),
                nn.GELU(),
                nn.Linear(4 * d_model, d_model),
                nn.Dropout(0.1)
            )
        
        def forward(self, x, rotary_emb=None):
            x = x + self.attn(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
            return x
    
    def compare_positional_encodings():
        """位置エンコーディングの比較実験"""
        
        # 設定
        vocab_size = 1000
        d_model = 128
        n_heads = 4
        n_layers = 2
        seq_len = 50
        
        # 各エンコーディングタイプのモデル作成
        models = {
            'sinusoidal': GPTWithPositionalEncoding(
                vocab_size, d_model, n_heads, n_layers, encoding_type='sinusoidal'
            ),
            'learned': GPTWithPositionalEncoding(
                vocab_size, d_model, n_heads, n_layers, encoding_type='learned'
            )
        }
        
        # 可視化
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Sinusoidal encoding
        sinusoidal_pe = PositionalEncodingComparison.create_sinusoidal_encoding(seq_len, d_model)
        im1 = axes[0, 0].imshow(sinusoidal_pe[:20, :64].T, cmap='viridis', aspect='auto')
        axes[0, 0].set_title('Sinusoidal Position Encoding')
        axes[0, 0].set_xlabel('Position')
        axes[0, 0].set_ylabel('Dimension')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # 2. Learned encoding (初期値)
        learned_pe = models['learned'].position_encoding.detach()[:20, :64].T
        im2 = axes[0, 1].imshow(learned_pe, cmap='viridis', aspect='auto')
        axes[0, 1].set_title('Learned Position Encoding (Initial)')
        axes[0, 1].set_xlabel('Position')
        axes[0, 1].set_ylabel('Dimension')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # 3. 位置による類似度（Sinusoidal）
        cos_sim_sin = torch.nn.functional.cosine_similarity(
            sinusoidal_pe.unsqueeze(1), sinusoidal_pe.unsqueeze(0), dim=2
        )
        im3 = axes[1, 0].imshow(cos_sim_sin[:20, :20], cmap='coolwarm', vmin=-1, vmax=1)
        axes[1, 0].set_title('Position Similarity (Sinusoidal)')
        axes[1, 0].set_xlabel('Position')
        axes[1, 0].set_ylabel('Position')
        plt.colorbar(im3, ax=axes[1, 0])
        
        # 4. 各エンコーディングの特性
        positions = torch.arange(seq_len)
        
        # 周期性の分析
        for i, dim in enumerate([0, 10, 20, 30]):
            if dim < d_model:
                axes[1, 1].plot(positions[:30], sinusoidal_pe[:30, dim], 
                              label=f'Dim {dim}', alpha=0.7)
        
        axes[1, 1].set_title('Sinusoidal Encoding Periodicity')
        axes[1, 1].set_xlabel('Position')
        axes[1, 1].set_ylabel('Value')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # 簡単な学習実験
        print("=== Training Comparison ===")
        
        # ダミーデータで簡単な訓練
        batch_size = 32
        criterion = nn.CrossEntropyLoss()
        
        for name, model in models.items():
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            
            losses = []
            for step in range(100):
                # ランダムな入力生成
                input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
                targets = torch.randint(0, vocab_size, (batch_size, seq_len))
                
                # 順伝播
                logits = model(input_ids)
                loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
                
                # 逆伝播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                losses.append(loss.item())
            
            print(f"{name} - Final loss: {losses[-1]:.4f}, "
                  f"Improvement: {(losses[0] - losses[-1]) / losses[0] * 100:.1f}%")
        
        return models
    
    # 比較実行
    models = compare_positional_encodings()
    ```

## 演習 5.2: 事前学習とファインチューニング

### 問題 3
簡単な言語モデルの事前学習ループを実装してください。

??? 解答
    ```python
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    import numpy as np
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    from typing import Dict, List, Optional
    import json
    
    class TextDataset(Dataset):
        """シンプルなテキストデータセット"""
        
        def __init__(self, texts: List[str], tokenizer, max_length: int = 128):
            self.tokenizer = tokenizer
            self.max_length = max_length
            self.texts = texts
            
        def __len__(self):
            return len(self.texts)
        
        def __getitem__(self, idx):
            text = self.texts[idx]
            tokens = self.tokenizer.encode(text)
            
            # パディングまたはトランケート
            if len(tokens) > self.max_length:
                tokens = tokens[:self.max_length]
            else:
                tokens = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
            
            tokens = torch.tensor(tokens, dtype=torch.long)
            
            # 言語モデリングでは入力と目標が1つずれる
            return {
                'input_ids': tokens[:-1],
                'labels': tokens[1:]
            }
    
    class SimpleTokenizer:
        """文字レベルの簡単なトークナイザー"""
        
        def __init__(self, vocab: Optional[Dict[str, int]] = None):
            if vocab is None:
                # 基本的な文字セット
                chars = list("abcdefghijklmnopqrstuvwxyz ")
                chars += list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
                chars += list("0123456789")
                chars += list(".,!?-'\"")
                
                self.vocab = {'<pad>': 0, '<unk>': 1, '<bos>': 2, '<eos>': 3}
                for i, char in enumerate(chars, start=4):
                    self.vocab[char] = i
            else:
                self.vocab = vocab
            
            self.inv_vocab = {v: k for k, v in self.vocab.items()}
            self.pad_token_id = self.vocab['<pad>']
            self.unk_token_id = self.vocab['<unk>']
            self.bos_token_id = self.vocab['<bos>']
            self.eos_token_id = self.vocab['<eos>']
            
        def encode(self, text: str) -> List[int]:
            tokens = [self.bos_token_id]
            for char in text:
                tokens.append(self.vocab.get(char, self.unk_token_id))
            tokens.append(self.eos_token_id)
            return tokens
        
        def decode(self, tokens: List[int]) -> str:
            chars = []
            for token in tokens:
                if token in self.inv_vocab:
                    char = self.inv_vocab[token]
                    if char not in ['<pad>', '<bos>', '<eos>', '<unk>']:
                        chars.append(char)
            return ''.join(chars)
    
    class PretrainingTrainer:
        """事前学習トレーナー"""
        
        def __init__(self, model, tokenizer, device='cuda' if torch.cuda.is_available() else 'cpu'):
            self.model = model.to(device)
            self.tokenizer = tokenizer
            self.device = device
            self.history = {'train_loss': [], 'val_loss': [], 'perplexity': []}
            
        def train(self, train_dataset, val_dataset=None, 
                 batch_size=32, num_epochs=10, learning_rate=1e-3,
                 warmup_steps=1000, gradient_clip=1.0):
            """事前学習の実行"""
            
            # データローダー
            train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                    shuffle=True, num_workers=2)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                                  shuffle=False) if val_dataset else None
            
            # オプティマイザとスケジューラ
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate,
                                        weight_decay=0.01)
            
            # 線形ウォームアップ + コサイン減衰
            total_steps = len(train_loader) * num_epochs
            scheduler = self.get_linear_schedule_with_warmup(
                optimizer, warmup_steps, total_steps
            )
            
            # 訓練ループ
            self.model.train()
            global_step = 0
            
            for epoch in range(num_epochs):
                epoch_loss = 0
                progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
                
                for batch in progress_bar:
                    input_ids = batch['input_ids'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    # 順伝播
                    logits = self.model(input_ids)
                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        labels.view(-1),
                        ignore_index=self.tokenizer.pad_token_id
                    )
                    
                    # 逆伝播
                    loss.backward()
                    
                    # 勾配クリッピング
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)
                    
                    # パラメータ更新
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    
                    # 統計情報
                    epoch_loss += loss.item()
                    global_step += 1
                    
                    # プログレスバー更新
                    progress_bar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'lr': f'{scheduler.get_last_lr()[0]:.2e}',
                        'ppl': f'{torch.exp(loss).item():.2f}'
                    })
                
                # エポック終了時の処理
                avg_train_loss = epoch_loss / len(train_loader)
                self.history['train_loss'].append(avg_train_loss)
                self.history['perplexity'].append(np.exp(avg_train_loss))
                
                # 検証
                if val_loader:
                    val_loss = self.evaluate(val_loader)
                    self.history['val_loss'].append(val_loss)
                    print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, "
                          f"Val Loss = {val_loss:.4f}, "
                          f"Val Perplexity = {np.exp(val_loss):.2f}")
                else:
                    print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, "
                          f"Perplexity = {np.exp(avg_train_loss):.2f}")
                
                # 生成サンプル
                if (epoch + 1) % 2 == 0:
                    self.generate_samples()
        
        def evaluate(self, dataloader):
            """モデルの評価"""
            self.model.eval()
            total_loss = 0
            
            with torch.no_grad():
                for batch in dataloader:
                    input_ids = batch['input_ids'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    logits = self.model(input_ids)
                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        labels.view(-1),
                        ignore_index=self.tokenizer.pad_token_id
                    )
                    
                    total_loss += loss.item()
            
            self.model.train()
            return total_loss / len(dataloader)
        
        def generate_samples(self, num_samples=3, max_length=50, temperature=0.8):
            """テキスト生成サンプル"""
            self.model.eval()
            
            print("\n=== Generated Samples ===")
            
            for i in range(num_samples):
                # BOS トークンから開始
                input_ids = torch.tensor([[self.tokenizer.bos_token_id]], 
                                        device=self.device)
                
                generated_tokens = [self.tokenizer.bos_token_id]
                
                for _ in range(max_length):
                    with torch.no_grad():
                        logits = self.model(input_ids)
                        next_token_logits = logits[0, -1, :] / temperature
                        
                        # サンプリング
                        probs = F.softmax(next_token_logits, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)
                        
                        generated_tokens.append(next_token.item())
                        
                        # EOS トークンで終了
                        if next_token.item() == self.tokenizer.eos_token_id:
                            break
                        
                        # 次の入力
                        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
                
                # デコード
                generated_text = self.tokenizer.decode(generated_tokens)
                print(f"Sample {i+1}: {generated_text}")
            
            print()
            self.model.train()
        
        @staticmethod
        def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
            """線形ウォームアップ + 線形減衰スケジューラ"""
            def lr_lambda(current_step):
                if current_step < num_warmup_steps:
                    return float(current_step) / float(max(1, num_warmup_steps))
                return max(0.0, float(num_training_steps - current_step) / 
                          float(max(1, num_training_steps - num_warmup_steps)))
            
            return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        def plot_training_history(self):
            """訓練履歴の可視化"""
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # 損失
            axes[0].plot(self.history['train_loss'], label='Train Loss')
            if self.history['val_loss']:
                axes[0].plot(self.history['val_loss'], label='Val Loss')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].set_title('Training and Validation Loss')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # パープレキシティ
            axes[1].plot(self.history['perplexity'], label='Train Perplexity')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Perplexity')
            axes[1].set_title('Model Perplexity')
            axes[1].set_yscale('log')
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
    
    # 使用例
    def run_pretraining_example():
        """事前学習の実行例"""
        
        # サンプルテキストデータ
        train_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is transforming the world.",
            "Natural language processing enables computers to understand human language.",
            "Deep learning models can learn complex patterns from data.",
            "Transformers have revolutionized NLP tasks.",
            "Attention is all you need for sequence modeling.",
            "Pre-training helps models learn general language understanding.",
            "Fine-tuning adapts models to specific tasks.",
        ] * 100  # データを増やす
        
        val_texts = [
            "AI systems are becoming more sophisticated.",
            "Language models can generate coherent text.",
        ] * 10
        
        # トークナイザーとデータセット
        tokenizer = SimpleTokenizer()
        train_dataset = TextDataset(train_texts, tokenizer, max_length=64)
        val_dataset = TextDataset(val_texts, tokenizer, max_length=64)
        
        # モデル作成
        vocab_size = len(tokenizer.vocab)
        model = GPTWithPositionalEncoding(
            vocab_size=vocab_size,
            d_model=128,
            n_heads=4,
            n_layers=2,
            encoding_type='learned'
        )
        
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Vocabulary size: {vocab_size}")
        
        # トレーナー作成と訓練
        trainer = PretrainingTrainer(model, tokenizer)
        
        trainer.train(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            batch_size=16,
            num_epochs=10,
            learning_rate=5e-4,
            warmup_steps=100
        )
        
        # 結果の可視化
        trainer.plot_training_history()
        
        return trainer, model
    
    # 実行
    trainer, pretrained_model = run_pretraining_example()
    ```

### 問題 4
ファインチューニングのための異なる戦略を実装してください（フル、最終層のみ、LoRA）。

??? 解答
    ```python
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    import numpy as np
    from typing import Dict, List, Optional, Tuple
    import matplotlib.pyplot as plt
    from copy import deepcopy
    
    class LoRALayer(nn.Module):
        """Low-Rank Adaptation (LoRA) レイヤー"""
        
        def __init__(self, in_features: int, out_features: int, 
                     rank: int = 16, alpha: float = 16.0):
            super().__init__()
            self.rank = rank
            self.alpha = alpha
            self.scaling = alpha / rank
            
            # LoRA パラメータ
            self.lora_A = nn.Parameter(torch.randn(in_features, rank) * 0.01)
            self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
            
            # 元の重みは凍結される（外部で管理）
            self.merged = False
            
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # LoRA の追加項を計算
            lora_output = x @ self.lora_A @ self.lora_B * self.scaling
            return lora_output
    
    class FineTuningStrategies:
        """ファインチューニング戦略の実装"""
        
        @staticmethod
        def freeze_all_but_last(model: nn.Module) -> nn.Module:
            """最終層以外をすべて凍結"""
            # すべてのパラメータを凍結
            for param in model.parameters():
                param.requires_grad = False
            
            # 最終層（言語モデルヘッド）のみ解凍
            if hasattr(model, 'lm_head'):
                for param in model.lm_head.parameters():
                    param.requires_grad = True
            
            return model
        
        @staticmethod
        def apply_lora(model: nn.Module, rank: int = 16, alpha: float = 16.0) -> nn.Module:
            """LoRAを適用"""
            # すべてのパラメータを凍結
            for param in model.parameters():
                param.requires_grad = False
            
            # Linear層にLoRAを追加
            lora_layers = {}
            
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear) and 'lm_head' not in name:
                    # LoRAレイヤーを作成
                    lora_layer = LoRALayer(
                        module.in_features, 
                        module.out_features,
                        rank=rank,
                        alpha=alpha
                    )
                    lora_layers[name] = lora_layer
            
            # LoRAレイヤーをモデルに追加
            for name, lora_layer in lora_layers.items():
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                parent = model
                
                if parent_name:
                    for part in parent_name.split('.'):
                        parent = getattr(parent, part)
                
                # 元のLinear層をラップ
                original_layer = getattr(parent, child_name)
                
                class LoRAWrapper(nn.Module):
                    def __init__(self, original, lora):
                        super().__init__()
                        self.original = original
                        self.lora = lora
                        
                    def forward(self, x):
                        return self.original(x) + self.lora(x)
                
                wrapped_layer = LoRAWrapper(original_layer, lora_layer)
                setattr(parent, child_name, wrapped_layer)
            
            return model
        
        @staticmethod
        def count_trainable_parameters(model: nn.Module) -> Tuple[int, int]:
            """訓練可能なパラメータ数をカウント"""
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            return trainable_params, total_params
    
    class ClassificationDataset(Dataset):
        """テキスト分類用データセット"""
        
        def __init__(self, texts: List[str], labels: List[int], 
                     tokenizer, max_length: int = 128):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_length = max_length
            
        def __len__(self):
            return len(self.texts)
        
        def __getitem__(self, idx):
            text = self.texts[idx]
            label = self.labels[idx]
            
            # トークン化
            tokens = self.tokenizer.encode(text)
            
            # パディング/トランケート
            if len(tokens) > self.max_length:
                tokens = tokens[:self.max_length]
            else:
                tokens = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
            
            return {
                'input_ids': torch.tensor(tokens, dtype=torch.long),
                'labels': torch.tensor(label, dtype=torch.long)
            }
    
    class FineTuningTrainer:
        """ファインチューニング用トレーナー"""
        
        def __init__(self, pretrained_model, num_classes: int, strategy: str = 'full'):
            self.strategy = strategy
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # 分類ヘッドを追加
            self.model = self._prepare_model_for_classification(
                pretrained_model, num_classes, strategy
            )
            self.model = self.model.to(self.device)
            
            # パラメータ数を表示
            trainable, total = FineTuningStrategies.count_trainable_parameters(self.model)
            print(f"Strategy: {strategy}")
            print(f"Trainable parameters: {trainable:,} / {total:,} "
                  f"({trainable/total*100:.2f}%)")
            
        def _prepare_model_for_classification(self, base_model, num_classes, strategy):
            """分類用にモデルを準備"""
            
            class ClassificationModel(nn.Module):
                def __init__(self, base_model, num_classes):
                    super().__init__()
                    self.base_model = base_model
                    self.dropout = nn.Dropout(0.1)
                    
                    # 分類ヘッド
                    hidden_size = base_model.d_model
                    self.classifier = nn.Linear(hidden_size, num_classes)
                    
                def forward(self, input_ids):
                    # ベースモデルの出力を取得
                    outputs = self.base_model(input_ids)  # [batch, seq_len, hidden]
                    
                    # 最初のトークン（[CLS]トークンの代わり）を使用
                    pooled_output = outputs[:, 0, :]
                    pooled_output = self.dropout(pooled_output)
                    
                    # 分類
                    logits = self.classifier(pooled_output)
                    return logits
            
            model = ClassificationModel(deepcopy(base_model), num_classes)
            
            # 戦略に応じてパラメータを調整
            if strategy == 'last_layer':
                model = FineTuningStrategies.freeze_all_but_last(model)
                # 分類ヘッドも訓練可能にする
                for param in model.classifier.parameters():
                    param.requires_grad = True
                    
            elif strategy == 'lora':
                model.base_model = FineTuningStrategies.apply_lora(
                    model.base_model, rank=8, alpha=16
                )
                # 分類ヘッドは訓練可能
                for param in model.classifier.parameters():
                    param.requires_grad = True
                    
            # strategy == 'full' の場合はすべて訓練可能
            
            return model
        
        def train(self, train_dataset, val_dataset, num_epochs=5, 
                 batch_size=16, learning_rate=2e-5):
            """ファインチューニングの実行"""
            
            # データローダー
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            
            # オプティマイザ（訓練可能なパラメータのみ）
            optimizer = torch.optim.AdamW(
                [p for p in self.model.parameters() if p.requires_grad],
                lr=learning_rate,
                weight_decay=0.01
            )
            
            # 損失関数
            criterion = nn.CrossEntropyLoss()
            
            # 訓練履歴
            history = {'train_loss': [], 'train_acc': [], 
                      'val_loss': [], 'val_acc': []}
            
            # 訓練ループ
            for epoch in range(num_epochs):
                # 訓練
                self.model.train()
                train_loss = 0
                train_correct = 0
                train_total = 0
                
                for batch in train_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    # 順伝播
                    logits = self.model(input_ids)
                    loss = criterion(logits, labels)
                    
                    # 逆伝播
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad], 
                        max_norm=1.0
                    )
                    optimizer.step()
                    
                    # 統計
                    train_loss += loss.item()
                    _, predicted = torch.max(logits, 1)
                    train_correct += (predicted == labels).sum().item()
                    train_total += labels.size(0)
                
                # 検証
                val_loss, val_acc = self.evaluate(val_loader, criterion)
                
                # 記録
                train_acc = train_correct / train_total
                avg_train_loss = train_loss / len(train_loader)
                
                history['train_loss'].append(avg_train_loss)
                history['train_acc'].append(train_acc)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                
                print(f"Epoch {epoch+1}/{num_epochs}:")
                print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}")
                print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            return history
        
        def evaluate(self, dataloader, criterion):
            """モデルの評価"""
            self.model.eval()
            total_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch in dataloader:
                    input_ids = batch['input_ids'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    logits = self.model(input_ids)
                    loss = criterion(logits, labels)
                    
                    total_loss += loss.item()
                    _, predicted = torch.max(logits, 1)
                    correct += (predicted == labels).sum().item()
                    total += labels.size(0)
            
            avg_loss = total_loss / len(dataloader)
            accuracy = correct / total
            
            return avg_loss, accuracy
    
    def compare_finetuning_strategies():
        """ファインチューニング戦略の比較"""
        
        # ダミーの分類データ
        positive_texts = [
            "This movie is fantastic! I loved every minute of it.",
            "Amazing performance by the actors. Highly recommended!",
            "Best film I've seen this year. Absolutely brilliant!",
        ] * 50
        
        negative_texts = [
            "Terrible movie. Complete waste of time.",
            "Boring plot and bad acting. Very disappointed.",
            "One of the worst films I've ever seen.",
        ] * 50
        
        # データセット作成
        texts = positive_texts + negative_texts
        labels = [1] * len(positive_texts) + [0] * len(negative_texts)
        
        # シャッフル
        import random
        data = list(zip(texts, labels))
        random.shuffle(data)
        texts, labels = zip(*data)
        
        # 訓練/検証分割
        split_idx = int(0.8 * len(texts))
        train_texts, val_texts = texts[:split_idx], texts[split_idx:]
        train_labels, val_labels = labels[:split_idx], labels[split_idx:]
        
        # トークナイザーとデータセット
        tokenizer = SimpleTokenizer()
        train_dataset = ClassificationDataset(train_texts, train_labels, tokenizer)
        val_dataset = ClassificationDataset(val_texts, val_labels, tokenizer)
        
        # 事前学習済みモデル（前の問題で作成したもの）
        pretrained_model = GPTWithPositionalEncoding(
            vocab_size=len(tokenizer.vocab),
            d_model=128,
            n_heads=4,
            n_layers=2
        )
        
        # 各戦略でファインチューニング
        strategies = ['full', 'last_layer', 'lora']
        results = {}
        
        for strategy in strategies:
            print(f"\n=== {strategy.upper()} Fine-tuning ===")
            
            trainer = FineTuningTrainer(pretrained_model, num_classes=2, strategy=strategy)
            history = trainer.train(
                train_dataset, val_dataset,
                num_epochs=5,
                batch_size=16,
                learning_rate=2e-5 if strategy == 'full' else 5e-4
            )
            
            results[strategy] = history
        
        # 結果の可視化
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        for strategy, history in results.items():
            ax1.plot(history['val_loss'], label=f'{strategy} - Val Loss')
            ax2.plot(history['val_acc'], label=f'{strategy} - Val Acc')
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Validation Loss Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Validation Accuracy Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return results
    
    # 比較実行
    results = compare_finetuning_strategies()
    ```

## 演習 5.3: トークナイザーの詳細

### 問題 5
BPE（Byte Pair Encoding）トークナイザーを完全実装してください。

??? 解答
    ```python
    import re
    from collections import defaultdict, Counter
    from typing import Dict, List, Tuple, Optional, Set
    import json
    import numpy as np
    
    class BPETokenizer:
        """Byte Pair Encoding トークナイザーの完全実装"""
        
        def __init__(self, vocab_size: int = 1000, min_frequency: int = 2):
            self.vocab_size = vocab_size
            self.min_frequency = min_frequency
            
            # 特殊トークン
            self.pad_token = '<pad>'
            self.unk_token = '<unk>'
            self.bos_token = '<bos>'
            self.eos_token = '<eos>'
            
            # 語彙
            self.word_tokenizer = re.compile(r'\w+|[^\w\s]')
            self.vocab = {}
            self.merges = []
            self.word_freq = defaultdict(int)
            
        def train(self, texts: List[str]):
            """BPEモデルの訓練"""
            print("Training BPE tokenizer...")
            
            # 1. 単語頻度をカウント
            for text in texts:
                words = self.word_tokenizer.findall(text.lower())
                for word in words:
                    self.word_freq[word] += 1
            
            # 2. 単語を文字に分割（終端記号付き）
            word_splits = {}
            for word, freq in self.word_freq.items():
                if freq >= self.min_frequency:
                    word_splits[word] = list(word) + ['</w>']
            
            # 3. 初期語彙を作成（個別文字）
            self.vocab = {
                self.pad_token: 0,
                self.unk_token: 1,
                self.bos_token: 2,
                self.eos_token: 3
            }
            
            char_freq = defaultdict(int)
            for word, freq in self.word_freq.items():
                if freq >= self.min_frequency:
                    for char in word:
                        char_freq[char] += freq
                    char_freq['</w>'] += freq
            
            # 文字を語彙に追加
            for char, freq in sorted(char_freq.items(), key=lambda x: -x[1]):
                if char not in self.vocab:
                    self.vocab[char] = len(self.vocab)
            
            # 4. BPEマージを学習
            while len(self.vocab) < self.vocab_size:
                # ペアの頻度を計算
                pair_freq = self._get_pair_frequencies(word_splits)
                
                if not pair_freq:
                    break
                
                # 最頻出ペアを選択
                best_pair = max(pair_freq, key=pair_freq.get)
                freq = pair_freq[best_pair]
                
                if freq < self.min_frequency:
                    break
                
                # マージを実行
                self.merges.append(best_pair)
                new_token = ''.join(best_pair)
                self.vocab[new_token] = len(self.vocab)
                
                # 単語分割を更新
                word_splits = self._merge_pair(word_splits, best_pair)
                
                if len(self.vocab) % 100 == 0:
                    print(f"Vocabulary size: {len(self.vocab)}")
            
            print(f"Training complete. Final vocabulary size: {len(self.vocab)}")
            
            # 逆引き辞書を作成
            self.id_to_token = {v: k for k, v in self.vocab.items()}
        
        def _get_pair_frequencies(self, word_splits: Dict[str, List[str]]) -> Dict[Tuple[str, str], int]:
            """隣接ペアの頻度を計算"""
            pair_freq = defaultdict(int)
            
            for word, split in word_splits.items():
                word_freq = self.word_freq[word]
                
                for i in range(len(split) - 1):
                    pair = (split[i], split[i + 1])
                    pair_freq[pair] += word_freq
            
            return pair_freq
        
        def _merge_pair(self, word_splits: Dict[str, List[str]], 
                       pair: Tuple[str, str]) -> Dict[str, List[str]]:
            """指定されたペアをマージ"""
            new_word_splits = {}
            merged = ''.join(pair)
            
            for word, split in word_splits.items():
                new_split = []
                i = 0
                
                while i < len(split):
                    if i < len(split) - 1 and (split[i], split[i + 1]) == pair:
                        new_split.append(merged)
                        i += 2
                    else:
                        new_split.append(split[i])
                        i += 1
                
                new_word_splits[word] = new_split
            
            return new_word_splits
        
        def _tokenize_word(self, word: str) -> List[str]:
            """単語をBPEトークンに分割"""
            word = word.lower()
            splits = list(word) + ['</w>']
            
            # 学習したマージを順番に適用
            for pair in self.merges:
                new_splits = []
                i = 0
                
                while i < len(splits):
                    if i < len(splits) - 1 and (splits[i], splits[i + 1]) == pair:
                        new_splits.append(''.join(pair))
                        i += 2
                    else:
                        new_splits.append(splits[i])
                        i += 1
                
                splits = new_splits
            
            return splits
        
        def encode(self, text: str) -> List[int]:
            """テキストをトークンIDに変換"""
            tokens = [self.bos_token]
            
            words = self.word_tokenizer.findall(text.lower())
            for word in words:
                word_tokens = self._tokenize_word(word)
                for token in word_tokens:
                    if token in self.vocab:
                        tokens.append(token)
                    else:
                        tokens.append(self.unk_token)
            
            tokens.append(self.eos_token)
            
            # トークンをIDに変換
            return [self.vocab.get(token, self.vocab[self.unk_token]) for token in tokens]
        
        def decode(self, ids: List[int]) -> str:
            """トークンIDをテキストに変換"""
            tokens = []
            
            for id in ids:
                if id in self.id_to_token:
                    token = self.id_to_token[id]
                    if token not in [self.pad_token, self.unk_token, 
                                   self.bos_token, self.eos_token]:
                        tokens.append(token)
            
            # トークンを結合
            text = ''.join(tokens)
            text = text.replace('</w>', ' ')
            return text.strip()
        
        def save(self, path: str):
            """トークナイザーを保存"""
            data = {
                'vocab': self.vocab,
                'merges': self.merges,
                'vocab_size': self.vocab_size,
                'min_frequency': self.min_frequency
            }
            
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        
        def load(self, path: str):
            """トークナイザーを読み込み"""
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.vocab = data['vocab']
            self.merges = [tuple(pair) for pair in data['merges']]
            self.vocab_size = data['vocab_size']
            self.min_frequency = data['min_frequency']
            self.id_to_token = {v: k for k, v in self.vocab.items()}
        
        def analyze_vocabulary(self):
            """語彙の分析"""
            print(f"=== Vocabulary Analysis ===")
            print(f"Total vocabulary size: {len(self.vocab)}")
            print(f"Number of merges: {len(self.merges)}")
            
            # トークン長の分布
            token_lengths = [len(token.replace('</w>', '')) for token in self.vocab.keys()
                           if token not in [self.pad_token, self.unk_token, 
                                          self.bos_token, self.eos_token]]
            
            print(f"\nToken length distribution:")
            length_counts = Counter(token_lengths)
            for length in sorted(length_counts.keys()):
                print(f"  Length {length}: {length_counts[length]} tokens")
            
            # 最も長いトークン
            longest_tokens = sorted(
                [(token, len(token.replace('</w>', ''))) for token in self.vocab.keys()
                 if token not in [self.pad_token, self.unk_token, 
                                self.bos_token, self.eos_token]],
                key=lambda x: -x[1]
            )[:10]
            
            print(f"\nLongest tokens:")
            for token, length in longest_tokens:
                print(f"  '{token}' (length: {length})")
            
            # 最初のマージ
            print(f"\nFirst 10 merges:")
            for i, (a, b) in enumerate(self.merges[:10]):
                print(f"  {i+1}. '{a}' + '{b}' -> '{a}{b}'")
    
    # テストと可視化
    def test_bpe_tokenizer():
        """BPEトークナイザーのテスト"""
        
        # 訓練データ
        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is transforming artificial intelligence.",
            "Natural language processing enables computers to understand text.",
            "Deep learning models can learn complex patterns.",
            "Transformers have revolutionized NLP tasks.",
            "Attention mechanism is the key innovation.",
            "Pre-training on large corpora improves performance.",
            "Fine-tuning adapts models to specific tasks.",
            "Tokenization is an important preprocessing step.",
            "Byte pair encoding is a subword tokenization method.",
        ] * 10  # データを増やす
        
        # BPEトークナイザーの訓練
        tokenizer = BPETokenizer(vocab_size=500, min_frequency=2)
        tokenizer.train(texts)
        
        # 語彙の分析
        tokenizer.analyze_vocabulary()
        
        # エンコード/デコードのテスト
        print("\n=== Encoding/Decoding Test ===")
        test_texts = [
            "Machine learning is amazing.",
            "Transformers revolutionized NLP.",
            "Unknown words like cryptocurrency.",
        ]
        
        for text in test_texts:
            print(f"\nOriginal: {text}")
            
            # エンコード
            ids = tokenizer.encode(text)
            tokens = [tokenizer.id_to_token.get(id, '?') for id in ids]
            print(f"Tokens: {tokens}")
            print(f"IDs: {ids}")
            
            # デコード
            decoded = tokenizer.decode(ids)
            print(f"Decoded: {decoded}")
        
        # 圧縮率の計算
        print("\n=== Compression Analysis ===")
        total_chars = 0
        total_tokens = 0
        
        for text in texts:
            chars = len(text)
            tokens = len(tokenizer.encode(text))
            total_chars += chars
            total_tokens += tokens
        
        compression_ratio = total_chars / total_tokens
        print(f"Average compression ratio: {compression_ratio:.2f} chars/token")
        
        return tokenizer
    
    # 実行
    bpe_tokenizer = test_bpe_tokenizer()
    ```

### 問題 6
異なるトークナイザー（BPE、WordPiece、SentencePiece）の性能を比較してください。

??? 解答
    ```python
    import time
    from typing import Dict, List, Tuple
    import matplotlib.pyplot as plt
    import numpy as np
    
    class WordPieceTokenizer:
        """WordPiece トークナイザーの簡易実装"""
        
        def __init__(self, vocab_size: int = 1000, min_frequency: int = 2):
            self.vocab_size = vocab_size
            self.min_frequency = min_frequency
            self.vocab = {}
            self.unk_token = '[UNK]'
            self.pad_token = '[PAD]'
            self.cls_token = '[CLS]'
            self.sep_token = '[SEP]'
            self.prefix = '##'  # サブワードプレフィックス
            
        def train(self, texts: List[str]):
            """WordPieceモデルの訓練"""
            # 簡略化のため、事前定義された語彙を使用
            # 実際のWordPieceはより複雑なアルゴリズムを使用
            
            # 特殊トークン
            self.vocab = {
                self.pad_token: 0,
                self.unk_token: 1,
                self.cls_token: 2,
                self.sep_token: 3,
            }
            
            # 文字と一般的なサブワードを追加
            chars = set()
            words = []
            
            for text in texts:
                for word in text.lower().split():
                    words.append(word)
                    chars.update(word)
            
            # 個別文字を追加
            for char in sorted(chars):
                if char not in self.vocab:
                    self.vocab[char] = len(self.vocab)
            
            # 頻出サブワードを追加（簡略化）
            from collections import Counter
            word_freq = Counter(words)
            
            # 単語の部分文字列を候補として生成
            subword_freq = Counter()
            for word, freq in word_freq.items():
                if freq >= self.min_frequency:
                    # 単語全体
                    if len(word) <= 5 and word not in self.vocab:
                        self.vocab[word] = len(self.vocab)
                    
                    # サブワード
                    for i in range(1, len(word)):
                        for j in range(i + 1, min(i + 6, len(word) + 1)):
                            subword = self.prefix + word[i:j]
                            subword_freq[subword] += freq
            
            # 頻出サブワードを語彙に追加
            for subword, freq in subword_freq.most_common():
                if len(self.vocab) >= self.vocab_size:
                    break
                if freq >= self.min_frequency and subword not in self.vocab:
                    self.vocab[subword] = len(self.vocab)
            
            self.id_to_token = {v: k for k, v in self.vocab.items()}
        
        def tokenize_word(self, word: str) -> List[str]:
            """単語をWordPieceトークンに分割"""
            tokens = []
            start = 0
            
            while start < len(word):
                end = len(word)
                cur_substr = None
                
                while start < end:
                    substr = word[start:end]
                    if start > 0:
                        substr = self.prefix + substr
                    
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    
                    end -= 1
                
                if cur_substr is None:
                    tokens.append(self.unk_token)
                    start += 1
                else:
                    tokens.append(cur_substr)
                    start = end
            
            return tokens
        
        def encode(self, text: str) -> List[int]:
            """テキストをトークンIDに変換"""
            tokens = [self.cls_token]
            
            for word in text.lower().split():
                word_tokens = self.tokenize_word(word)
                tokens.extend(word_tokens)
            
            tokens.append(self.sep_token)
            
            return [self.vocab.get(token, self.vocab[self.unk_token]) for token in tokens]
        
        def decode(self, ids: List[int]) -> str:
            """トークンIDをテキストに変換"""
            tokens = []
            
            for id in ids:
                if id in self.id_to_token:
                    token = self.id_to_token[id]
                    if token not in [self.pad_token, self.unk_token, 
                                   self.cls_token, self.sep_token]:
                        if token.startswith(self.prefix):
                            tokens.append(token[len(self.prefix):])
                        else:
                            if tokens:
                                tokens.append(' ')
                            tokens.append(token)
            
            return ''.join(tokens).strip()
    
    class SentencePieceTokenizer:
        """SentencePiece トークナイザーの簡易実装"""
        
        def __init__(self, vocab_size: int = 1000, character_coverage: float = 0.9995):
            self.vocab_size = vocab_size
            self.character_coverage = character_coverage
            self.vocab = {}
            self.unk_token = '<unk>'
            self.pad_token = '<pad>'
            self.bos_token = '<s>'
            self.eos_token = '</s>'
            self.space_symbol = '▁'  # スペースを表す記号
            
        def train(self, texts: List[str]):
            """SentencePieceモデルの訓練（簡略化版）"""
            # 実際のSentencePieceは unigram language model を使用
            
            self.vocab = {
                self.pad_token: 0,
                self.unk_token: 1,
                self.bos_token: 2,
                self.eos_token: 3,
            }
            
            # 文字頻度を計算
            char_freq = Counter()
            for text in texts:
                # スペースを特殊記号に置換
                normalized_text = self.space_symbol + text.replace(' ', self.space_symbol)
                char_freq.update(normalized_text)
            
            # 文字カバレッジに基づいて文字を選択
            total_chars = sum(char_freq.values())
            covered_chars = 0
            
            for char, freq in char_freq.most_common():
                if char not in self.vocab:
                    self.vocab[char] = len(self.vocab)
                    covered_chars += freq
                    
                    if covered_chars / total_chars >= self.character_coverage:
                        break
            
            # サブワード候補を生成（簡略化）
            subword_scores = {}
            
            for text in texts[:100]:  # 計算量削減のため一部のみ使用
                normalized = self.space_symbol + text.replace(' ', self.space_symbol)
                
                # 全てのサブストリングを候補として生成
                for i in range(len(normalized)):
                    for j in range(i + 1, min(i + 10, len(normalized) + 1)):
                        subword = normalized[i:j]
                        if len(subword) > 1:
                            # スコア計算（簡略化：頻度のみ）
                            subword_scores[subword] = subword_scores.get(subword, 0) + 1
            
            # スコアの高いサブワードを語彙に追加
            for subword, score in sorted(subword_scores.items(), 
                                       key=lambda x: -x[1]):
                if len(self.vocab) >= self.vocab_size:
                    break
                    
                if subword not in self.vocab and score > 1:
                    self.vocab[subword] = len(self.vocab)
            
            self.id_to_token = {v: k for k, v in self.vocab.items()}
        
        def encode(self, text: str) -> List[int]:
            """テキストをトークンIDに変換（簡略化版）"""
            # 実際のSentencePieceはViterbiアルゴリズムを使用
            
            normalized = self.space_symbol + text.replace(' ', self.space_symbol)
            tokens = [self.bos_token]
            
            i = 0
            while i < len(normalized):
                # 最長一致で貪欲に分割
                match_found = False
                
                for length in range(min(10, len(normalized) - i), 0, -1):
                    subword = normalized[i:i + length]
                    
                    if subword in self.vocab:
                        tokens.append(subword)
                        i += length
                        match_found = True
                        break
                
                if not match_found:
                    tokens.append(self.unk_token)
                    i += 1
            
            tokens.append(self.eos_token)
            
            return [self.vocab.get(token, self.vocab[self.unk_token]) for token in tokens]
        
        def decode(self, ids: List[int]) -> str:
            """トークンIDをテキストに変換"""
            tokens = []
            
            for id in ids:
                if id in self.id_to_token:
                    token = self.id_to_token[id]
                    if token not in [self.pad_token, self.unk_token, 
                                   self.bos_token, self.eos_token]:
                        tokens.append(token)
            
            text = ''.join(tokens)
            return text.replace(self.space_symbol, ' ').strip()
    
    def compare_tokenizers():
        """トークナイザーの比較"""
        
        # テストデータ
        train_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning transforms artificial intelligence.",
            "Natural language processing is fascinating.",
            "Deep neural networks learn complex patterns.",
            "Tokenization is crucial for text processing.",
        ] * 20
        
        test_texts = [
            "Machine learning is amazing!",
            "Tokenizers split text into tokens.",
            "Unknown words like cryptocurrency appear.",
            "The quick brown fox runs fast.",
            "Deep learning revolutionizes AI.",
        ]
        
        # 各トークナイザーを訓練
        tokenizers = {
            'BPE': BPETokenizer(vocab_size=500),
            'WordPiece': WordPieceTokenizer(vocab_size=500),
            'SentencePiece': SentencePieceTokenizer(vocab_size=500),
        }
        
        print("Training tokenizers...")
        for name, tokenizer in tokenizers.items():
            print(f"\nTraining {name}...")
            start_time = time.time()
            tokenizer.train(train_texts)
            train_time = time.time() - start_time
            print(f"{name} training time: {train_time:.2f}s")
        
        # 比較メトリクス
        results = {name: {
            'compression_ratio': [],
            'unk_rate': [],
            'encode_time': [],
            'decode_time': [],
            'vocab_size': len(tokenizer.vocab)
        } for name, tokenizer in tokenizers.items()}
        
        print("\n=== Tokenization Comparison ===")
        
        for text in test_texts:
            print(f"\nText: {text}")
            
            for name, tokenizer in tokenizers.items():
                # エンコード
                start_time = time.time()
                ids = tokenizer.encode(text)
                encode_time = time.time() - start_time
                
                # デコード
                start_time = time.time()
                decoded = tokenizer.decode(ids)
                decode_time = time.time() - start_time
                
                # メトリクス計算
                compression_ratio = len(text) / len(ids)
                
                if name == 'BPE':
                    unk_count = sum(1 for id in ids if id == tokenizer.vocab[tokenizer.unk_token])
                elif name == 'WordPiece':
                    unk_count = sum(1 for id in ids if id == tokenizer.vocab[tokenizer.unk_token])
                else:  # SentencePiece
                    unk_count = sum(1 for id in ids if id == tokenizer.vocab[tokenizer.unk_token])
                
                unk_rate = unk_count / len(ids) if len(ids) > 0 else 0
                
                # 結果保存
                results[name]['compression_ratio'].append(compression_ratio)
                results[name]['unk_rate'].append(unk_rate)
                results[name]['encode_time'].append(encode_time)
                results[name]['decode_time'].append(decode_time)
                
                # トークン表示
                if name == 'BPE':
                    tokens = [tokenizer.id_to_token.get(id, '?') for id in ids]
                elif name == 'WordPiece':
                    tokens = [tokenizer.id_to_token.get(id, '?') for id in ids]
                else:  # SentencePiece
                    tokens = [tokenizer.id_to_token.get(id, '?') for id in ids]
                
                print(f"\n{name}:")
                print(f"  Tokens: {tokens}")
                print(f"  Compression: {compression_ratio:.2f}")
                print(f"  UNK rate: {unk_rate:.2%}")
        
        # 結果の可視化
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. 圧縮率
        ax = axes[0, 0]
        for name in tokenizers.keys():
            avg_compression = np.mean(results[name]['compression_ratio'])
            ax.bar(name, avg_compression, alpha=0.7)
        ax.set_title('Average Compression Ratio')
        ax.set_ylabel('Characters per Token')
        ax.grid(True, alpha=0.3)
        
        # 2. UNKトークン率
        ax = axes[0, 1]
        for name in tokenizers.keys():
            avg_unk = np.mean(results[name]['unk_rate']) * 100
            ax.bar(name, avg_unk, alpha=0.7)
        ax.set_title('Average UNK Token Rate')
        ax.set_ylabel('UNK Rate (%)')
        ax.grid(True, alpha=0.3)
        
        # 3. エンコード時間
        ax = axes[1, 0]
        for name in tokenizers.keys():
            avg_encode = np.mean(results[name]['encode_time']) * 1000
            ax.bar(name, avg_encode, alpha=0.7)
        ax.set_title('Average Encoding Time')
        ax.set_ylabel('Time (ms)')
        ax.grid(True, alpha=0.3)
        
        # 4. 語彙サイズ
        ax = axes[1, 1]
        for name in tokenizers.keys():
            ax.bar(name, results[name]['vocab_size'], alpha=0.7)
        ax.set_title('Vocabulary Size')
        ax.set_ylabel('Number of Tokens')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # サマリー表示
        print("\n=== Summary ===")
        print(f"{'Tokenizer':<15} {'Compression':<12} {'UNK Rate':<10} {'Encode(ms)':<12} {'Vocab Size':<10}")
        print("-" * 65)
        
        for name in tokenizers.keys():
            avg_comp = np.mean(results[name]['compression_ratio'])
            avg_unk = np.mean(results[name]['unk_rate']) * 100
            avg_encode = np.mean(results[name]['encode_time']) * 1000
            vocab_size = results[name]['vocab_size']
            
            print(f"{name:<15} {avg_comp:<12.2f} {avg_unk:<10.1f} {avg_encode:<12.3f} {vocab_size:<10}")
        
        return tokenizers, results
    
    # 比較実行
    tokenizers, comparison_results = compare_tokenizers()
    ```

## 演習 5.4: 推論時の工夫

### 問題 7
ビームサーチとサンプリング手法（top-k, top-p）を実装し、生成品質を比較してください。

??? 解答
    ```python
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    from typing import List, Tuple, Optional, Dict
    import matplotlib.pyplot as plt
    from dataclasses import dataclass
    
    @dataclass
    class GenerationConfig:
        """生成設定"""
        max_length: int = 50
        temperature: float = 1.0
        top_k: int = 50
        top_p: float = 0.9
        repetition_penalty: float = 1.0
        num_beams: int = 4
        do_sample: bool = True
        
    class TextGenerator:
        """テキスト生成クラス"""
        
        def __init__(self, model, tokenizer, device='cuda' if torch.cuda.is_available() else 'cpu'):
            self.model = model.to(device)
            self.tokenizer = tokenizer
            self.device = device
            self.model.eval()
        
        def generate(self, prompt: str, config: GenerationConfig, 
                    method: str = 'sampling') -> Dict[str, any]:
            """指定された手法でテキストを生成"""
            
            # プロンプトをエンコード
            input_ids = torch.tensor([self.tokenizer.encode(prompt)], device=self.device)
            
            if method == 'greedy':
                output_ids, scores = self._greedy_search(input_ids, config)
            elif method == 'beam_search':
                output_ids, scores = self._beam_search(input_ids, config)
            elif method == 'sampling':
                output_ids, scores = self._sampling(input_ids, config)
            elif method == 'top_k':
                output_ids, scores = self._top_k_sampling(input_ids, config)
            elif method == 'top_p':
                output_ids, scores = self._top_p_sampling(input_ids, config)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            # デコード
            generated_text = self.tokenizer.decode(output_ids[0].tolist())
            
            return {
                'text': generated_text,
                'ids': output_ids,
                'scores': scores,
                'method': method
            }
        
        def _greedy_search(self, input_ids: torch.Tensor, 
                          config: GenerationConfig) -> Tuple[torch.Tensor, List[float]]:
            """貪欲法による生成"""
            generated = input_ids.clone()
            scores = []
            
            for _ in range(config.max_length - input_ids.shape[1]):
                with torch.no_grad():
                    outputs = self.model(generated)
                    next_token_logits = outputs[:, -1, :]
                    
                    # Repetition penalty
                    if config.repetition_penalty != 1.0:
                        self._apply_repetition_penalty(
                            next_token_logits, generated, config.repetition_penalty
                        )
                    
                    # 最も確率の高いトークンを選択
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                    score = F.softmax(next_token_logits, dim=-1).max().item()
                    scores.append(score)
                    
                    generated = torch.cat([generated, next_token], dim=1)
                    
                    # EOSトークンで終了
                    if next_token.item() == self.tokenizer.eos_token_id:
                        break
            
            return generated, scores
        
        def _beam_search(self, input_ids: torch.Tensor, 
                        config: GenerationConfig) -> Tuple[torch.Tensor, List[float]]:
            """ビームサーチによる生成"""
            batch_size = input_ids.shape[0]
            num_beams = config.num_beams
            
            # ビームの初期化
            beam_scores = torch.zeros((batch_size, num_beams), device=self.device)
            beam_scores[:, 1:] = -1e9  # 最初は1つのビームのみ有効
            
            # 各ビームの系列
            beam_sequences = input_ids.unsqueeze(1).repeat(1, num_beams, 1)
            beam_sequences = beam_sequences.view(batch_size * num_beams, -1)
            
            # 完了したビーム
            done = [False] * (batch_size * num_beams)
            scores_history = []
            
            for step in range(config.max_length - input_ids.shape[1]):
                with torch.no_grad():
                    outputs = self.model(beam_sequences)
                    next_token_logits = outputs[:, -1, :]
                    
                    # スコア計算
                    vocab_size = next_token_logits.shape[-1]
                    next_token_scores = F.log_softmax(next_token_logits, dim=-1)
                    
                    # 現在のビームスコアを加算
                    next_token_scores = next_token_scores.view(batch_size, num_beams, -1)
                    next_token_scores = next_token_scores + beam_scores.unsqueeze(-1)
                    
                    # 全候補から上位k個を選択
                    next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)
                    next_scores, next_tokens = torch.topk(
                        next_token_scores, 2 * num_beams, dim=-1, largest=True, sorted=True
                    )
                    
                    # 次のビームを構築
                    next_batch_beam = []
                    
                    for batch_idx in range(batch_size):
                        next_sent_beam = []
                        
                        for rank, (token_score, token_id) in enumerate(
                            zip(next_scores[batch_idx], next_tokens[batch_idx])
                        ):
                            beam_id = token_id // vocab_size
                            token_id = token_id % vocab_size
                            
                            effective_beam_id = batch_idx * num_beams + beam_id
                            
                            # EOSトークンの処理
                            if token_id.item() == self.tokenizer.eos_token_id:
                                done[effective_beam_id] = True
                            
                            next_sent_beam.append({
                                'score': token_score,
                                'token_id': token_id,
                                'beam_id': effective_beam_id
                            })
                            
                            if len(next_sent_beam) >= num_beams:
                                break
                        
                        next_batch_beam.extend(next_sent_beam)
                    
                    # ビームを更新
                    beam_scores = beam_scores.new_zeros((batch_size, num_beams))
                    beam_sequences = []
                    
                    for beam_idx, beam in enumerate(next_batch_beam):
                        batch_idx = beam_idx // num_beams
                        beam_scores[batch_idx, beam_idx % num_beams] = beam['score']
                        
                        prev_seq = beam_sequences[beam['beam_id']]
                        new_seq = torch.cat([prev_seq, beam['token_id'].unsqueeze(0)])
                        beam_sequences.append(new_seq)
                    
                    beam_sequences = torch.stack(beam_sequences, dim=0)
                    scores_history.append(beam_scores[0, 0].item())
                    
                    if all(done):
                        break
            
            # 最良のビームを返す
            best_beam_idx = beam_scores[0].argmax()
            best_sequence = beam_sequences[best_beam_idx].unsqueeze(0)
            
            return best_sequence, scores_history
        
        def _sampling(self, input_ids: torch.Tensor, 
                     config: GenerationConfig) -> Tuple[torch.Tensor, List[float]]:
            """温度付きサンプリング"""
            generated = input_ids.clone()
            scores = []
            
            for _ in range(config.max_length - input_ids.shape[1]):
                with torch.no_grad():
                    outputs = self.model(generated)
                    next_token_logits = outputs[:, -1, :] / config.temperature
                    
                    # Repetition penalty
                    if config.repetition_penalty != 1.0:
                        self._apply_repetition_penalty(
                            next_token_logits, generated, config.repetition_penalty
                        )
                    
                    # サンプリング
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    score = probs.gather(-1, next_token).item()
                    scores.append(score)
                    
                    generated = torch.cat([generated, next_token], dim=1)
                    
                    if next_token.item() == self.tokenizer.eos_token_id:
                        break
            
            return generated, scores
        
        def _top_k_sampling(self, input_ids: torch.Tensor, 
                           config: GenerationConfig) -> Tuple[torch.Tensor, List[float]]:
            """Top-kサンプリング"""
            generated = input_ids.clone()
            scores = []
            
            for _ in range(config.max_length - input_ids.shape[1]):
                with torch.no_grad():
                    outputs = self.model(generated)
                    next_token_logits = outputs[:, -1, :] / config.temperature
                    
                    # Top-kフィルタリング
                    top_k_logits, top_k_indices = torch.topk(
                        next_token_logits, config.top_k, dim=-1
                    )
                    
                    # 確率を再計算
                    probs = F.softmax(top_k_logits, dim=-1)
                    
                    # サンプリング
                    sample_idx = torch.multinomial(probs, num_samples=1)
                    next_token = top_k_indices.gather(-1, sample_idx)
                    score = probs.gather(-1, sample_idx).item()
                    scores.append(score)
                    
                    generated = torch.cat([generated, next_token], dim=1)
                    
                    if next_token.item() == self.tokenizer.eos_token_id:
                        break
            
            return generated, scores
        
        def _top_p_sampling(self, input_ids: torch.Tensor, 
                           config: GenerationConfig) -> Tuple[torch.Tensor, List[float]]:
            """Top-p (Nucleus) サンプリング"""
            generated = input_ids.clone()
            scores = []
            
            for _ in range(config.max_length - input_ids.shape[1]):
                with torch.no_grad():
                    outputs = self.model(generated)
                    next_token_logits = outputs[:, -1, :] / config.temperature
                    
                    # ソートして累積確率を計算
                    sorted_logits, sorted_indices = torch.sort(
                        next_token_logits, descending=True
                    )
                    cumulative_probs = torch.cumsum(
                        F.softmax(sorted_logits, dim=-1), dim=-1
                    )
                    
                    # Top-pを超える位置を見つける
                    sorted_indices_to_remove = cumulative_probs > config.top_p
                    # 少なくとも1つは残す
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    # フィルタリング
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    next_token_logits[indices_to_remove] = -float('inf')
                    
                    # サンプリング
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    score = probs.gather(-1, next_token).item()
                    scores.append(score)
                    
                    generated = torch.cat([generated, next_token], dim=1)
                    
                    if next_token.item() == self.tokenizer.eos_token_id:
                        break
            
            return generated, scores
        
        def _apply_repetition_penalty(self, logits: torch.Tensor, 
                                    generated: torch.Tensor, penalty: float):
            """繰り返しペナルティを適用"""
            for token_id in generated[0].unique():
                if logits[:, token_id] < 0:
                    logits[:, token_id] *= penalty
                else:
                    logits[:, token_id] /= penalty
    
    def compare_generation_methods():
        """生成手法の比較"""
        
        # ダミーモデルとトークナイザー（前の演習から）
        vocab_size = 100
        model = GPTWithPositionalEncoding(
            vocab_size=vocab_size,
            d_model=128,
            n_heads=4,
            n_layers=2
        )
        
        tokenizer = SimpleTokenizer()
        
        # テキストジェネレーター
        generator = TextGenerator(model, tokenizer)
        
        # プロンプト
        prompts = [
            "The weather today is",
            "Machine learning can",
            "In the future, we will",
        ]
        
        # 各手法で生成
        methods = ['greedy', 'beam_search', 'sampling', 'top_k', 'top_p']
        configs = {
            'greedy': GenerationConfig(temperature=1.0),
            'beam_search': GenerationConfig(num_beams=4),
            'sampling': GenerationConfig(temperature=0.8, do_sample=True),
            'top_k': GenerationConfig(temperature=0.8, top_k=40),
            'top_p': GenerationConfig(temperature=0.8, top_p=0.95),
        }
        
        results = {method: [] for method in methods}
        
        print("=== Generation Method Comparison ===\n")
        
        for prompt in prompts:
            print(f"Prompt: '{prompt}'")
            print("-" * 50)
            
            for method in methods:
                result = generator.generate(prompt, configs[method], method=method)
                results[method].append(result)
                
                print(f"\n{method.upper()}:")
                print(f"Generated: {result['text']}")
                print(f"Avg Score: {np.mean(result['scores']):.4f}")
        
        # 多様性と品質の分析
        analyze_generation_quality(results, prompts)
        
        return results
    
    def analyze_generation_quality(results: Dict[str, List[Dict]], prompts: List[str]):
        """生成品質の分析"""
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. 平均スコア（信頼度）
        ax = axes[0, 0]
        avg_scores = {}
        for method, method_results in results.items():
            scores = [np.mean(r['scores']) for r in method_results]
            avg_scores[method] = np.mean(scores)
            ax.bar(method, avg_scores[method], alpha=0.7)
        
        ax.set_title('Average Generation Confidence')
        ax.set_ylabel('Average Score')
        ax.set_xticklabels(list(results.keys()), rotation=45)
        ax.grid(True, alpha=0.3)
        
        # 2. 生成長の分布
        ax = axes[0, 1]
        for method, method_results in results.items():
            lengths = [len(r['ids'][0]) for r in method_results]
            ax.plot(lengths, label=method, marker='o')
        
        ax.set_title('Generation Length by Method')
        ax.set_xlabel('Example Index')
        ax.set_ylabel('Sequence Length')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. トークンの多様性（ユニークトークン数）
        ax = axes[1, 0]
        diversity_scores = {}
        for method, method_results in results.items():
            unique_ratios = []
            for r in method_results:
                tokens = r['ids'][0].tolist()
                unique_ratio = len(set(tokens)) / len(tokens)
                unique_ratios.append(unique_ratio)
            diversity_scores[method] = np.mean(unique_ratios)
            ax.bar(method, diversity_scores[method], alpha=0.7)
        
        ax.set_title('Token Diversity (Unique Token Ratio)')
        ax.set_ylabel('Unique Token Ratio')
        ax.set_xticklabels(list(results.keys()), rotation=45)
        ax.grid(True, alpha=0.3)
        
        # 4. スコアの安定性（標準偏差）
        ax = axes[1, 1]
        stability_scores = {}
        for method, method_results in results.items():
            score_stds = [np.std(r['scores']) if len(r['scores']) > 1 else 0 
                         for r in method_results]
            stability_scores[method] = np.mean(score_stds)
            ax.bar(method, stability_scores[method], alpha=0.7)
        
        ax.set_title('Score Stability (Lower is More Stable)')
        ax.set_ylabel('Average Score Std Dev')
        ax.set_xticklabels(list(results.keys()), rotation=45)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # サマリー統計
        print("\n=== Generation Quality Summary ===")
        print(f"{'Method':<15} {'Avg Score':<12} {'Diversity':<12} {'Stability':<12}")
        print("-" * 55)
        
        for method in results.keys():
            print(f"{method:<15} {avg_scores[method]:<12.4f} "
                  f"{diversity_scores[method]:<12.4f} "
                  f"{stability_scores[method]:<12.4f}")
    
    # 実行
    generation_results = compare_generation_methods()
    ```

## チャレンジ問題

### 問題 8 🌟
実用的なチャットボットシステムを構築してください。以下の機能を含む：
- コンテキスト管理（会話履歴）
- プロンプトエンジニアリング
- 安全性フィルター
- ストリーミング出力

??? 解答
    ```python
    import torch
    import torch.nn as nn
    from typing import List, Dict, Optional, Generator, Tuple
    import asyncio
    from dataclasses import dataclass, field
    from datetime import datetime
    import re
    from collections import deque
    
    @dataclass
    class Message:
        """チャットメッセージ"""
        role: str  # 'user' or 'assistant'
        content: str
        timestamp: datetime = field(default_factory=datetime.now)
        metadata: Dict = field(default_factory=dict)
    
    @dataclass
    class ChatConfig:
        """チャット設定"""
        max_context_length: int = 2048
        max_response_length: int = 512
        temperature: float = 0.7
        top_p: float = 0.9
        repetition_penalty: float = 1.1
        system_prompt: str = "You are a helpful assistant."
        safety_threshold: float = 0.8
        streaming: bool = True
        
    class SafetyFilter:
        """安全性フィルター"""
        
        def __init__(self, threshold: float = 0.8):
            self.threshold = threshold
            # 簡易的な禁止語リスト（実際はより洗練されたモデルを使用）
            self.banned_patterns = [
                r'\b(hate|violence|illegal)\b',
                r'\b(harmful|dangerous)\b',
            ]
            
        def is_safe(self, text: str) -> Tuple[bool, Optional[str]]:
            """テキストの安全性をチェック"""
            text_lower = text.lower()
            
            # 禁止パターンのチェック
            for pattern in self.banned_patterns:
                if re.search(pattern, text_lower):
                    return False, f"Content contains prohibited pattern: {pattern}"
            
            # その他の安全性チェック（簡略化）
            if len(text) > 10000:
                return False, "Content too long"
            
            return True, None
        
        def sanitize(self, text: str) -> str:
            """テキストをサニタイズ"""
            # 基本的なサニタイゼーション
            text = text.strip()
            # 連続する空白を1つに
            text = re.sub(r'\s+', ' ', text)
            return text
    
    class ConversationManager:
        """会話コンテキスト管理"""
        
        def __init__(self, max_context_length: int = 2048):
            self.max_context_length = max_context_length
            self.messages: deque[Message] = deque()
            self.token_counts: deque[int] = deque()
            
        def add_message(self, message: Message, token_count: int):
            """メッセージを追加"""
            self.messages.append(message)
            self.token_counts.append(token_count)
            
            # コンテキスト長を管理
            while sum(self.token_counts) > self.max_context_length and len(self.messages) > 2:
                self.messages.popleft()
                self.token_counts.popleft()
        
        def get_context(self) -> List[Message]:
            """現在のコンテキストを取得"""
            return list(self.messages)
        
        def clear(self):
            """会話履歴をクリア"""
            self.messages.clear()
            self.token_counts.clear()
        
        def format_for_model(self, tokenizer) -> str:
            """モデル入力用にフォーマット"""
            formatted_messages = []
            
            for msg in self.messages:
                if msg.role == 'user':
                    formatted_messages.append(f"User: {msg.content}")
                else:
                    formatted_messages.append(f"Assistant: {msg.content}")
            
            return "\n".join(formatted_messages) + "\nAssistant:"
    
    class PromptEngineering:
        """プロンプトエンジニアリング"""
        
        @staticmethod
        def create_system_prompt(config: ChatConfig) -> str:
            """システムプロンプトを作成"""
            return f"""<|system|>
{config.system_prompt}

Guidelines:
1. Be helpful, harmless, and honest
2. Provide clear and concise responses
3. Admit when you don't know something
4. Refuse inappropriate requests politely
<|endofsystem|>"""
        
        @staticmethod
        def format_chat_prompt(messages: List[Message], config: ChatConfig) -> str:
            """チャットプロンプトをフォーマット"""
            prompt_parts = [PromptEngineering.create_system_prompt(config)]
            
            for msg in messages:
                if msg.role == 'user':
                    prompt_parts.append(f"<|user|>\n{msg.content}\n<|endofuser|>")
                else:
                    prompt_parts.append(f"<|assistant|>\n{msg.content}\n<|endofassistant|>")
            
            # 最後のアシスタントプロンプトを追加
            prompt_parts.append("<|assistant|>")
            
            return "\n".join(prompt_parts)
    
    class StreamingChatbot:
        """ストリーミング対応チャットボット"""
        
        def __init__(self, model, tokenizer, config: ChatConfig):
            self.model = model
            self.tokenizer = tokenizer
            self.config = config
            self.device = next(model.parameters()).device
            
            self.conversation = ConversationManager(config.max_context_length)
            self.safety_filter = SafetyFilter(config.safety_threshold)
            self.prompt_engineering = PromptEngineering()
            
        def generate_streaming(self, prompt: str) -> Generator[str, None, None]:
            """ストリーミング生成"""
            input_ids = torch.tensor([self.tokenizer.encode(prompt)], device=self.device)
            generated_ids = input_ids.clone()
            
            past_key_values = None
            generated_text = ""
            
            for _ in range(self.config.max_response_length):
                with torch.no_grad():
                    if past_key_values is None:
                        outputs = self.model(generated_ids)
                    else:
                        # 効率化のため前回のキャッシュを使用
                        outputs = self.model(
                            generated_ids[:, -1:],
                            past_key_values=past_key_values
                        )
                    
                    next_token_logits = outputs[:, -1, :]
                    
                    # 温度スケーリング
                    next_token_logits = next_token_logits / self.config.temperature
                    
                    # Repetition penalty
                    if self.config.repetition_penalty != 1.0:
                        for token_id in generated_ids[0].unique():
                            if next_token_logits[:, token_id] < 0:
                                next_token_logits[:, token_id] *= self.config.repetition_penalty
                            else:
                                next_token_logits[:, token_id] /= self.config.repetition_penalty
                    
                    # Top-p sampling
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > self.config.top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    next_token_logits[indices_to_remove] = -float('inf')
                    
                    # サンプリング
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    
                    # トークンを追加
                    generated_ids = torch.cat([generated_ids, next_token], dim=1)
                    
                    # デコード
                    token_text = self.tokenizer.decode([next_token.item()])
                    generated_text += token_text
                    
                    # ストリーミング出力
                    yield token_text
                    
                    # 終了条件
                    if next_token.item() == self.tokenizer.eos_token_id:
                        break
                    
                    # 安全性チェック（定期的に）
                    if len(generated_text) % 50 == 0:
                        is_safe, _ = self.safety_filter.is_safe(generated_text)
                        if not is_safe:
                            yield "\n[Content filtered]"
                            break
        
        async def chat_async(self, user_input: str) -> AsyncGenerator[str, None]:
            """非同期チャット"""
            # 入力の安全性チェック
            is_safe, reason = self.safety_filter.is_safe(user_input)
            if not is_safe:
                yield f"I cannot process this request. Reason: {reason}"
                return
            
            # サニタイズ
            user_input = self.safety_filter.sanitize(user_input)
            
            # ユーザーメッセージを追加
            user_msg = Message(role='user', content=user_input)
            user_token_count = len(self.tokenizer.encode(user_input))
            self.conversation.add_message(user_msg, user_token_count)
            
            # プロンプトを構築
            prompt = self.prompt_engineering.format_chat_prompt(
                self.conversation.get_context(), 
                self.config
            )
            
            # ストリーミング生成
            response_text = ""
            async for token in self._async_generate(prompt):
                response_text += token
                yield token
            
            # アシスタントメッセージを追加
            assistant_msg = Message(role='assistant', content=response_text)
            assistant_token_count = len(self.tokenizer.encode(response_text))
            self.conversation.add_message(assistant_msg, assistant_token_count)
        
        async def _async_generate(self, prompt: str) -> AsyncGenerator[str, None]:
            """非同期生成（実際の実装では別スレッドで実行）"""
            for token in self.generate_streaming(prompt):
                await asyncio.sleep(0.01)  # シミュレートされた遅延
                yield token
        
        def chat(self, user_input: str) -> str:
            """同期的なチャット（ストリーミングなし）"""
            # 入力チェック
            is_safe, reason = self.safety_filter.is_safe(user_input)
            if not is_safe:
                return f"I cannot process this request. Reason: {reason}"
            
            user_input = self.safety_filter.sanitize(user_input)
            
            # メッセージ追加
            user_msg = Message(role='user', content=user_input)
            user_token_count = len(self.tokenizer.encode(user_input))
            self.conversation.add_message(user_msg, user_token_count)
            
            # プロンプト構築
            prompt = self.prompt_engineering.format_chat_prompt(
                self.conversation.get_context(),
                self.config
            )
            
            # 生成
            response_text = ""
            for token in self.generate_streaming(prompt):
                response_text += token
            
            # 最終的な安全性チェック
            is_safe, _ = self.safety_filter.is_safe(response_text)
            if not is_safe:
                response_text = "I apologize, but I cannot provide that response."
            
            # アシスタントメッセージ追加
            assistant_msg = Message(role='assistant', content=response_text)
            assistant_token_count = len(self.tokenizer.encode(response_text))
            self.conversation.add_message(assistant_msg, assistant_token_count)
            
            return response_text
    
    class ChatbotUI:
        """簡易的なチャットボットUI"""
        
        def __init__(self, chatbot: StreamingChatbot):
            self.chatbot = chatbot
            
        def run(self):
            """インタラクティブチャットループ"""
            print("=== Chatbot Started ===")
            print("Type 'quit' to exit, 'clear' to clear conversation history")
            print("-" * 50)
            
            while True:
                try:
                    user_input = input("\nYou: ").strip()
                    
                    if user_input.lower() == 'quit':
                        print("Goodbye!")
                        break
                    
                    if user_input.lower() == 'clear':
                        self.chatbot.conversation.clear()
                        print("Conversation history cleared.")
                        continue
                    
                    if not user_input:
                        continue
                    
                    print("\nAssistant: ", end='', flush=True)
                    
                    if self.chatbot.config.streaming:
                        # ストリーミング出力
                        for token in self.chatbot.generate_streaming(user_input):
                            print(token, end='', flush=True)
                        print()  # 改行
                    else:
                        # 通常の出力
                        response = self.chatbot.chat(user_input)
                        print(response)
                    
                except KeyboardInterrupt:
                    print("\n\nInterrupted. Type 'quit' to exit.")
                except Exception as e:
                    print(f"\nError: {e}")
    
    # 使用例
    def demo_chatbot():
        """チャットボットのデモ"""
        
        # モデルとトークナイザー（前の演習から）
        model = GPTWithPositionalEncoding(
            vocab_size=1000,
            d_model=256,
            n_heads=8,
            n_layers=4
        )
        
        tokenizer = SimpleTokenizer()
        
        # チャットボット設定
        config = ChatConfig(
            max_context_length=1024,
            temperature=0.8,
            top_p=0.95,
            system_prompt="You are a helpful AI assistant.",
            streaming=True
        )
        
        # チャットボット作成
        chatbot = StreamingChatbot(model, tokenizer, config)
        
        # デモ会話
        print("=== Chatbot Demo ===\n")
        
        test_conversations = [
            "Hello! How are you?",
            "Can you explain machine learning?",
            "What's the weather like?",
            "Tell me a joke.",
        ]
        
        for user_input in test_conversations:
            print(f"User: {user_input}")
            print("Assistant: ", end='')
            
            response = chatbot.chat(user_input)
            print(response)
            print("-" * 50)
        
        # 会話履歴の表示
        print("\n=== Conversation History ===")
        for msg in chatbot.conversation.get_context():
            print(f"{msg.role.upper()}: {msg.content[:50]}...")
        
        return chatbot
    
    # デモ実行
    chatbot_instance = demo_chatbot()
    
    # インタラクティブモードを開始する場合
    # ui = ChatbotUI(chatbot_instance)
    # ui.run()
    ```

## 次のステップ

これらの演習を完了したら、以下の発展的なトピックに挑戦してみてください：

1. **マルチモーダルTransformer**: 画像とテキストを同時に扱う
2. **効率的なアーキテクチャ**: Reformer、Linformer、Performerの実装
3. **強化学習との統合**: RLHF（Reinforcement Learning from Human Feedback）
4. **分散学習**: モデル並列、データ並列の実装

💡 **学習のアドバイス**:
- 各実装を小さな部分から始めて徐々に複雑にしていく
- 実際のデータセットで実験してみる
- 最新の研究論文を読んで新しい手法を試す
- コミュニティのコードを参考にしながら自分なりの改良を加える

頑張ってください！🚀