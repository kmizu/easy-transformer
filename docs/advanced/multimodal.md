# マルチモーダルTransformer

## はじめに：複数のモダリティを統合する

コンパイラが様々な入力形式（ソースコード、設定ファイル、リンカスクリプト）を処理して統一的な出力を生成するように、マルチモーダルTransformerは画像、テキスト、音声などの異なるモダリティを統合して処理します。

この章では、Vision Transformer (ViT)から始まり、CLIP、DALL-E、Flamingo などの最新のマルチモーダルモデルまでを実装します。

## 1. Vision Transformer (ViT)

### 1.1 画像のトークン化

画像を Transformer で処理するための最初のステップは、画像をトークンに変換することです。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class PatchEmbedding(nn.Module):
    """画像をパッチに分割してトークン化"""
    
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # パッチを線形変換で埋め込みに変換
        self.patch_embed = nn.Sequential(
            # (B, C, H, W) -> (B, num_patches, embed_dim)
            nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size),
            Rearrange('b c h w -> b (h w) c'),
        )
        
        # CLSトークン
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # 位置埋め込み
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # パッチ埋め込み
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        
        # CLSトークンを追加
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=batch_size)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, 1 + num_patches, embed_dim)
        
        # 位置埋め込みを追加
        x = x + self.pos_embed
        
        return x
    
    def visualize_patches(self, img):
        """パッチ分割の可視化"""
        import matplotlib.pyplot as plt
        
        # 画像をパッチに分割
        patches = rearrange(
            img, 
            'c (h p1) (w p2) -> (h w) p1 p2 c',
            p1=self.patch_size, 
            p2=self.patch_size
        )
        
        # グリッドで表示
        n = int(np.sqrt(patches.shape[0]))
        fig, axes = plt.subplots(n, n, figsize=(10, 10))
        
        for i, ax in enumerate(axes.flat):
            if i < patches.shape[0]:
                patch = patches[i].permute(1, 2, 0).numpy()
                ax.imshow(patch)
                ax.axis('off')
        
        plt.tight_layout()
        plt.show()

class VisionTransformer(nn.Module):
    """Vision Transformer (ViT) の実装"""
    
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        dropout=0.1,
        attention_dropout=0.1,
    ):
        super().__init__()
        
        # パッチ埋め込み
        self.patch_embed = PatchEmbedding(
            img_size, patch_size, in_channels, embed_dim
        )
        
        # Transformer エンコーダ
        self.transformer = nn.ModuleList([
            TransformerBlock(
                embed_dim, num_heads, 
                int(embed_dim * mlp_ratio),
                dropout, attention_dropout
            )
            for _ in range(depth)
        ])
        
        # 分類ヘッド
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x):
        # パッチ埋め込み
        x = self.patch_embed(x)
        
        # Transformer ブロック
        for block in self.transformer:
            x = block(x)
        
        # 分類用にCLSトークンを使用
        x = self.norm(x)
        cls_token = x[:, 0]
        
        # 分類
        return self.head(cls_token)
```

### 1.2 Vision Transformer の訓練

```python
class ViTTrainer:
    """Vision Transformer の訓練"""
    
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        
    def train(self, train_loader, val_loader, num_epochs=100, lr=1e-3):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.05)
        
        # Cosine annealing with warmup
        num_training_steps = num_epochs * len(train_loader)
        num_warmup_steps = int(0.1 * num_training_steps)
        
        scheduler = self.get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps
        )
        
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(num_epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Mixup augmentation
                if np.random.random() > 0.5:
                    images, labels_a, labels_b, lam = self.mixup_data(images, labels)
                    
                    outputs = self.model(images)
                    loss = lam * criterion(outputs, labels_a) + \
                           (1 - lam) * criterion(outputs, labels_b)
                else:
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
            
            # Validation
            val_acc = self.validate(val_loader)
            
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"Train Loss: {train_loss/len(train_loader):.4f}")
            print(f"Train Acc: {100.*train_correct/train_total:.2f}%")
            print(f"Val Acc: {val_acc:.2f}%")
    
    def validate(self, val_loader):
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        return 100. * correct / total
    
    @staticmethod
    def mixup_data(x, y, alpha=0.2):
        """Mixup augmentation"""
        lam = np.random.beta(alpha, alpha)
        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(x.device)
        
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam
    
    @staticmethod
    def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / \
                      float(max(1, num_training_steps - num_warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

## 2. CLIP: テキストと画像の統合

### 2.1 CLIP アーキテクチャ

```python
class CLIPModel(nn.Module):
    """CLIP (Contrastive Language-Image Pre-training) の実装"""
    
    def __init__(
        self,
        embed_dim=512,
        # Vision
        vision_width=768,
        vision_layers=12,
        vision_heads=12,
        vision_patch_size=16,
        image_size=224,
        # Text
        vocab_size=49408,
        text_width=512,
        text_layers=12,
        text_heads=8,
        text_max_length=77,
    ):
        super().__init__()
        
        # Vision encoder
        self.visual = VisionTransformer(
            img_size=image_size,
            patch_size=vision_patch_size,
            embed_dim=vision_width,
            depth=vision_layers,
            num_heads=vision_heads,
            num_classes=embed_dim,  # Project to shared embedding space
        )
        
        # Text encoder
        self.text = TextTransformer(
            vocab_size=vocab_size,
            embed_dim=text_width,
            num_layers=text_layers,
            num_heads=text_heads,
            max_length=text_max_length,
        )
        
        # Projection heads
        self.visual_projection = nn.Linear(vision_width, embed_dim, bias=False)
        self.text_projection = nn.Linear(text_width, embed_dim, bias=False)
        
        # Temperature parameter
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
    def encode_image(self, image):
        """画像をエンコード"""
        x = self.visual.patch_embed(image)
        
        for block in self.visual.transformer:
            x = block(x)
        
        x = self.visual.norm(x)
        x = x[:, 0]  # CLSトークン
        
        # 共有埋め込み空間に投影
        x = self.visual_projection(x)
        x = F.normalize(x, dim=-1)
        
        return x
    
    def encode_text(self, text):
        """テキストをエンコード"""
        x = self.text(text)
        
        # EOSトークンの位置を取得
        eos_indices = text.argmax(dim=-1)
        x = x[torch.arange(x.shape[0]), eos_indices]
        
        # 共有埋め込み空間に投影
        x = self.text_projection(x)
        x = F.normalize(x, dim=-1)
        
        return x
    
    def forward(self, image, text):
        """画像とテキストのペアに対する損失を計算"""
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)
        
        # コサイン類似度
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logits_per_image.T
        
        # 対照学習の損失
        batch_size = image.shape[0]
        labels = torch.arange(batch_size, device=image.device)
        
        loss_i = F.cross_entropy(logits_per_image, labels)
        loss_t = F.cross_entropy(logits_per_text, labels)
        loss = (loss_i + loss_t) / 2
        
        return loss, logits_per_image

class TextTransformer(nn.Module):
    """CLIP用のテキストエンコーダ"""
    
    def __init__(
        self,
        vocab_size,
        embed_dim=512,
        num_layers=12,
        num_heads=8,
        max_length=77,
    ):
        super().__init__()
        
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_embedding = nn.Parameter(
            torch.empty(max_length, embed_dim)
        )
        
        self.transformer = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, embed_dim * 4)
            for _ in range(num_layers)
        ])
        
        self.ln_final = nn.LayerNorm(embed_dim)
        
        # Initialize
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        
    def forward(self, text):
        x = self.token_embedding(text)
        x = x + self.positional_embedding[:text.shape[1]]
        
        # Causal mask for autoregressive
        mask = torch.triu(torch.ones(text.shape[1], text.shape[1]), diagonal=1)
        mask = mask.to(device=x.device, dtype=x.dtype)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        
        for block in self.transformer:
            x = block(x, mask)
        
        x = self.ln_final(x)
        
        return x
```

### 2.2 CLIP の訓練と応用

```python
class CLIPTrainer:
    """CLIP モデルの訓練"""
    
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        
    def train(self, dataloader, num_epochs=30, lr=5e-4):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.1)
        
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            
            for batch_idx, (images, texts) in enumerate(dataloader):
                images = images.to(self.device)
                texts = texts.to(self.device)
                
                loss, logits = self.model(images, texts)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                
                if batch_idx % 100 == 0:
                    # 精度を計算
                    accuracy = (logits.argmax(dim=1) == torch.arange(len(images), device=self.device)).float().mean()
                    print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}, Acc: {accuracy:.4f}")
            
            print(f"Epoch {epoch} completed. Average Loss: {total_loss/len(dataloader):.4f}")

class CLIPApplications:
    """CLIP の応用例"""
    
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.model.eval()
        
    def zero_shot_classification(self, images, class_names):
        """ゼロショット画像分類"""
        # クラス名をテンプレートに埋め込む
        text_prompts = [f"a photo of a {name}" for name in class_names]
        
        # テキストをトークン化（仮定）
        text_tokens = self.tokenize(text_prompts)
        
        with torch.no_grad():
            # 画像とテキストをエンコード
            image_features = self.model.encode_image(images)
            text_features = self.model.encode_text(text_tokens)
            
            # 類似度を計算
            similarity = image_features @ text_features.T
            
            # 予測
            predictions = similarity.argmax(dim=1)
            
        return predictions, similarity
    
    def image_retrieval(self, query_text, image_database):
        """テキストクエリによる画像検索"""
        # テキストをエンコード
        text_tokens = self.tokenize([query_text])
        
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            
            # すべての画像をエンコード
            all_image_features = []
            for images in image_database:
                image_features = self.model.encode_image(images)
                all_image_features.append(image_features)
            
            all_image_features = torch.cat(all_image_features, dim=0)
            
            # 類似度を計算
            similarities = text_features @ all_image_features.T
            
            # Top-k を取得
            top_k = 10
            top_indices = similarities.argsort(descending=True)[0, :top_k]
            
        return top_indices, similarities[0, top_indices]
    
    def tokenize(self, texts):
        """簡易的なトークン化（実際はCLIPトークナイザーを使用）"""
        # ダミー実装
        return torch.randint(0, 49408, (len(texts), 77), device=self.device)
```

## 3. 画像生成: DALL-E スタイル

### 3.1 VQ-VAE: 画像の離散表現

```python
class VectorQuantizer(nn.Module):
    """ベクトル量子化層"""
    
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
        
    def forward(self, inputs):
        # inputs: (B, C, H, W)
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self.embedding_dim)
        
        # 最近傍を見つける
        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(flat_input, self.embedding.weight.t())
        )
        
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # 量子化
        quantized = torch.matmul(encodings, self.embedding.weight).view(input_shape)
        
        # 勾配のコピー
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        
        return quantized, loss, encoding_indices.view(input_shape[0], -1)

class VQVAE(nn.Module):
    """Vector Quantized VAE"""
    
    def __init__(self, in_channels=3, hidden_dims=[128, 256], num_embeddings=512, embedding_dim=64):
        super().__init__()
        
        # Encoder
        modules = []
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, h_dim, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU()
                )
            )
            in_channels = h_dim
        
        modules.append(nn.Conv2d(hidden_dims[-1], embedding_dim, kernel_size=1))
        self.encoder = nn.Sequential(*modules)
        
        # Vector Quantizer
        self.vq = VectorQuantizer(num_embeddings, embedding_dim)
        
        # Decoder
        modules = []
        hidden_dims.reverse()
        
        modules.append(nn.Conv2d(embedding_dim, hidden_dims[0], kernel_size=1))
        
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i + 1],
                                     kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU()
                )
            )
        
        modules.append(
            nn.ConvTranspose2d(hidden_dims[-1], 3, kernel_size=4, stride=2, padding=1)
        )
        
        self.decoder = nn.Sequential(*modules)
        
    def forward(self, x):
        encoded = self.encoder(x)
        quantized, vq_loss, indices = self.vq(encoded)
        decoded = self.decoder(quantized)
        
        return decoded, vq_loss, indices
```

### 3.2 テキスト条件付き画像生成

```python
class TextToImageTransformer(nn.Module):
    """テキストから画像トークンを生成するTransformer"""
    
    def __init__(
        self,
        text_vocab_size,
        image_vocab_size,
        d_model=512,
        nhead=8,
        num_layers=12,
        max_text_len=256,
        max_image_len=1024,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.image_vocab_size = image_vocab_size
        
        # テキストと画像の埋め込み
        self.text_embedding = nn.Embedding(text_vocab_size, d_model)
        self.image_embedding = nn.Embedding(image_vocab_size, d_model)
        
        # 位置埋め込み
        self.text_pos_embedding = nn.Parameter(torch.randn(1, max_text_len, d_model))
        self.image_pos_embedding = nn.Parameter(torch.randn(1, max_image_len, d_model))
        
        # Transformer
        self.transformer = nn.ModuleList([
            TransformerBlock(d_model, nhead, d_model * 4)
            for _ in range(num_layers)
        ])
        
        # 出力ヘッド
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, image_vocab_size)
        
    def forward(self, text_tokens, image_tokens=None):
        """
        text_tokens: (B, text_len)
        image_tokens: (B, image_len) - 訓練時のみ
        """
        device = text_tokens.device
        batch_size = text_tokens.shape[0]
        
        # テキストの埋め込み
        text_emb = self.text_embedding(text_tokens)
        text_emb = text_emb + self.text_pos_embedding[:, :text_tokens.shape[1]]
        
        if image_tokens is not None:
            # 訓練時: Teacher forcing
            image_emb = self.image_embedding(image_tokens)
            image_emb = image_emb + self.image_pos_embedding[:, :image_tokens.shape[1]]
            
            # テキストと画像を結合
            x = torch.cat([text_emb, image_emb], dim=1)
            
            # Causal mask
            total_len = x.shape[1]
            mask = torch.ones(total_len, total_len, device=device)
            # テキスト部分は全て見える
            text_len = text_tokens.shape[1]
            mask[:text_len, :text_len] = 0
            # 画像部分は因果的
            mask[text_len:, text_len:] = torch.triu(
                torch.ones(total_len - text_len, total_len - text_len, device=device),
                diagonal=1
            )
            mask = mask.masked_fill(mask == 1, float('-inf'))
            
        else:
            # 推論時
            x = text_emb
            mask = None
        
        # Transformer layers
        for block in self.transformer:
            x = block(x, mask)
        
        # 出力
        x = self.ln_f(x)
        logits = self.head(x)
        
        if image_tokens is not None:
            # 画像部分のロジットのみ返す
            return logits[:, text_tokens.shape[1]:]
        else:
            return logits
    
    @torch.no_grad()
    def generate(self, text_tokens, max_length=256, temperature=1.0, top_k=100):
        """画像トークンを生成"""
        self.eval()
        device = text_tokens.device
        batch_size = text_tokens.shape[0]
        
        # 開始トークン
        generated = []
        
        # テキストエンコーディング
        text_emb = self.text_embedding(text_tokens)
        text_emb = text_emb + self.text_pos_embedding[:, :text_tokens.shape[1]]
        
        x = text_emb
        
        for i in range(max_length):
            # 位置埋め込みを追加
            if i > 0:
                image_tokens_so_far = torch.stack(generated, dim=1)
                image_emb = self.image_embedding(image_tokens_so_far)
                image_emb = image_emb + self.image_pos_embedding[:, :i]
                x = torch.cat([text_emb, image_emb], dim=1)
            
            # Forward pass
            for block in self.transformer:
                x_out = block(x)
            
            x_out = self.ln_f(x_out)
            logits = self.head(x_out[:, -1])  # 最後のトークンのみ
            
            # サンプリング
            if temperature > 0:
                logits = logits / temperature
                
                if top_k > 0:
                    # Top-k sampling
                    top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
                    logits = torch.full_like(logits, float('-inf'))
                    logits.scatter_(1, top_k_indices, top_k_logits)
                
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                # Greedy
                next_token = torch.argmax(logits, dim=-1)
            
            generated.append(next_token)
        
        return torch.stack(generated, dim=1)
```

## 4. Flamingo: 少数ショット学習

### 4.1 Perceiver Resampler

```python
class PerceiverResampler(nn.Module):
    """可変長の視覚特徴を固定長に変換"""
    
    def __init__(
        self,
        dim,
        depth=6,
        num_latents=64,
        dim_head=64,
        heads=8,
        ff_mult=4,
    ):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                # Cross attention (latents -> visual features)
                CrossAttention(dim, heads=heads, dim_head=dim_head),
                nn.LayerNorm(dim),
                # Self attention (latents)
                SelfAttention(dim, heads=heads, dim_head=dim_head),
                nn.LayerNorm(dim),
                # FFN
                FeedForward(dim, mult=ff_mult),
                nn.LayerNorm(dim)
            ]))
            
    def forward(self, x):
        """
        x: visual features (B, N, D)
        returns: resampled features (B, num_latents, D)
        """
        b = x.shape[0]
        
        # Repeat latents for batch
        latents = repeat(self.latents, 'n d -> b n d', b=b)
        
        for cross_attn, cross_norm, self_attn, self_norm, ff, ff_norm in self.layers:
            # Cross attention
            latents = latents + cross_attn(latents, context=x)
            latents = cross_norm(latents)
            
            # Self attention
            latents = latents + self_attn(latents)
            latents = self_norm(latents)
            
            # FFN
            latents = latents + ff(latents)
            latents = ff_norm(latents)
            
        return latents

class CrossAttention(nn.Module):
    """クロスアテンション層"""
    
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        
    def forward(self, x, context):
        h = self.heads
        
        q = self.to_q(x)
        k, v = self.to_kv(context).chunk(2, dim=-1)
        
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = dots.softmax(dim=-1)
        
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        return self.to_out(out)
```

### 4.2 Flamingo モデル

```python
class FlamingoModel(nn.Module):
    """Flamingo: 少数ショットビジョン言語モデル"""
    
    def __init__(
        self,
        vision_encoder,
        language_model,
        dim_visual=768,
        dim_text=768,
        resampler_depth=6,
        resampler_num_latents=64,
    ):
        super().__init__()
        
        self.vision_encoder = vision_encoder
        self.perceiver_resampler = PerceiverResampler(
            dim=dim_visual,
            depth=resampler_depth,
            num_latents=resampler_num_latents,
        )
        
        # Frozen language model with gated cross-attention
        self.language_model = language_model
        self._freeze_lm()
        self._add_gated_cross_attention()
        
        # Visual projection
        self.visual_projection = nn.Linear(dim_visual, dim_text)
        
    def _freeze_lm(self):
        """言語モデルを凍結"""
        for param in self.language_model.parameters():
            param.requires_grad = False
            
    def _add_gated_cross_attention(self):
        """ゲート付きクロスアテンションを追加"""
        for i, layer in enumerate(self.language_model.transformer.layers):
            layer.gated_cross_attention = GatedCrossAttention(
                layer.hidden_size,
                layer.num_attention_heads
            )
            
    def forward(self, images, text_tokens, image_positions):
        """
        images: List of images (different sizes ok)
        text_tokens: (B, L)
        image_positions: positions in text where images should be attended to
        """
        # Process all images
        visual_features = []
        for img in images:
            # Vision encoder
            vis_feat = self.vision_encoder(img.unsqueeze(0))
            # Perceiver resampler
            vis_feat = self.perceiver_resampler(vis_feat)
            # Project to text dimension
            vis_feat = self.visual_projection(vis_feat)
            visual_features.append(vis_feat)
        
        # Concatenate all visual features
        visual_features = torch.cat(visual_features, dim=0)
        
        # Run language model with cross-attention to visual features
        output = self.language_model(
            text_tokens,
            visual_features=visual_features,
            image_positions=image_positions
        )
        
        return output

class GatedCrossAttention(nn.Module):
    """ゲート付きクロスアテンション"""
    
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            batch_first=True
        )
        self.gate = nn.Parameter(torch.zeros(1))
        self.norm = nn.LayerNorm(hidden_size)
        
    def forward(self, x, visual_features, attention_mask=None):
        # Cross attention
        residual = x
        x = self.norm(x)
        x, _ = self.cross_attention(
            x, visual_features, visual_features,
            key_padding_mask=attention_mask
        )
        
        # Gated residual connection
        x = residual + self.gate.tanh() * x
        
        return x
```

## 5. 実践的な応用

### 5.1 マルチモーダル検索システム

```python
class MultimodalSearchEngine:
    """画像とテキストの統合検索"""
    
    def __init__(self, model, index_size=100000):
        self.model = model
        self.index_size = index_size
        
        # Feature database
        self.image_features = torch.zeros(index_size, 512)
        self.text_features = torch.zeros(index_size, 512)
        self.metadata = {}
        self.current_idx = 0
        
    def index_multimodal_data(self, images, texts, metadata):
        """マルチモーダルデータをインデックス"""
        with torch.no_grad():
            # Extract features
            img_feats = self.model.encode_image(images)
            txt_feats = self.model.encode_text(texts)
            
            # Store in database
            batch_size = len(images)
            self.image_features[self.current_idx:self.current_idx + batch_size] = img_feats
            self.text_features[self.current_idx:self.current_idx + batch_size] = txt_feats
            
            # Store metadata
            for i, meta in enumerate(metadata):
                self.metadata[self.current_idx + i] = meta
                
            self.current_idx += batch_size
            
    def search(self, query, modality='text', top_k=10):
        """マルチモーダル検索"""
        with torch.no_grad():
            if modality == 'text':
                query_feat = self.model.encode_text(query)
                # Search in both image and text features
                img_scores = query_feat @ self.image_features[:self.current_idx].T
                txt_scores = query_feat @ self.text_features[:self.current_idx].T
                scores = (img_scores + txt_scores) / 2
            else:  # image query
                query_feat = self.model.encode_image(query)
                img_scores = query_feat @ self.image_features[:self.current_idx].T
                txt_scores = query_feat @ self.text_features[:self.current_idx].T
                scores = (img_scores + txt_scores) / 2
                
            # Get top-k results
            top_scores, top_indices = scores.topk(top_k, dim=1)
            
            results = []
            for i in range(top_indices.shape[0]):
                batch_results = []
                for j in range(top_k):
                    idx = top_indices[i, j].item()
                    batch_results.append({
                        'score': top_scores[i, j].item(),
                        'metadata': self.metadata[idx]
                    })
                results.append(batch_results)
                
        return results
```

## まとめ

マルチモーダルTransformerは、異なるモダリティ間の関係を学習し、統一的に処理する強力なフレームワークです。主要な概念：

1. **モダリティ固有のエンコーダ**: 各モダリティに適したアーキテクチャ
2. **共有埋め込み空間**: 異なるモダリティを同じ空間で表現
3. **クロスモーダルアテンション**: モダリティ間の相互作用
4. **対照学習**: モダリティ間の対応関係を学習

これらの技術により、画像理解、画像生成、マルチモーダル検索など、様々な応用が可能になります。