# GPTアーキテクチャ

## はじめに：生成の革命

プログラミング言語の処理系で、REPLを実装したことを思い出してください。ユーザーが入力したコードを解析し、評価し、結果を返す。そして重要なのは、以前の文脈を記憶していることです。GPT（Generative Pre-trained Transformer）は、まさに言語のREPLのような存在です。

GPTは「次のトークンを予測する」というシンプルなタスクを通じて、驚くほど高度な言語理解と生成能力を獲得します。この章では、GPTアーキテクチャの詳細と、なぜそれが成功したのかを探ります。

## 17.1 GPTの基本構造

### Decoder-onlyアーキテクチャ

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, List, Dict, Any
import math
from dataclasses import dataclass
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle
import matplotlib.patches as mpatches

class GPTArchitectureOverview:
    """GPTアーキテクチャの概要"""
    
    def explain_gpt_philosophy(self):
        """GPTの設計哲学を説明"""
        print("=== GPTの設計哲学 ===\n")
        
        print("1. シンプルさの追求:")
        print("   - Decoder-onlyアーキテクチャ")
        print("   - 自己回帰的な生成")
        print("   - 統一されたタスク形式\n")
        
        print("2. スケーラビリティ:")
        print("   - パラメータ数の増加で性能向上")
        print("   - 計算効率の良い設計")
        print("   - 並列化可能な学習\n")
        
        print("3. 汎用性:")
        print("   - あらゆる言語タスクを「生成」として扱う")
        print("   - Few-shot学習能力")
        print("   - ゼロショット汎化\n")
        
        # アーキテクチャ図
        self._visualize_gpt_architecture()
    
    def _visualize_gpt_architecture(self):
        """GPTアーキテクチャを可視化"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # 左側：全体構造
        ax1.set_title('GPT Architecture Overview', fontsize=14, weight='bold')
        ax1.set_xlim(0, 10)
        ax1.set_ylim(0, 14)
        
        # コンポーネント
        components = [
            {"name": "Token Embeddings", "y": 1, "color": "lightgreen"},
            {"name": "Position Embeddings", "y": 2.5, "color": "lightyellow"},
            {"name": "Transformer Block 1", "y": 4.5, "color": "lightblue"},
            {"name": "Transformer Block 2", "y": 6, "color": "lightblue"},
            {"name": "...", "y": 7.5, "color": "white"},
            {"name": "Transformer Block N", "y": 9, "color": "lightblue"},
            {"name": "Layer Norm", "y": 10.5, "color": "lightgray"},
            {"name": "Output Projection", "y": 12, "color": "lightcoral"}
        ]
        
        for comp in components:
            if comp["name"] == "...":
                ax1.text(5, comp["y"], comp["name"], ha='center', 
                        va='center', fontsize=16)
            else:
                rect = FancyBboxPatch((2, comp["y"]-0.4), 6, 0.8,
                                     boxstyle="round,pad=0.1",
                                     facecolor=comp["color"],
                                     edgecolor='black', linewidth=2)
                ax1.add_patch(rect)
                ax1.text(5, comp["y"], comp["name"], ha='center', 
                        va='center', fontsize=10, weight='bold')
        
        # 矢印
        for i in range(len(components)-1):
            if components[i]["name"] != "...":
                ax1.arrow(5, components[i]["y"]+0.5, 0, 0.7,
                         head_width=0.3, head_length=0.2,
                         fc='black', ec='black')
        
        ax1.axis('off')
        
        # 右側：Transformer Block詳細
        ax2.set_title('Transformer Block Detail', fontsize=14, weight='bold')
        ax2.set_xlim(0, 10)
        ax2.set_ylim(0, 10)
        
        # Transformerブロックの詳細
        block_components = [
            {"name": "Input", "y": 1, "color": "white"},
            {"name": "Multi-Head Attention\n(Causal Mask)", "y": 3, "color": "lightcoral"},
            {"name": "Add & Norm", "y": 4.5, "color": "lightgray"},
            {"name": "Feed Forward", "y": 6, "color": "lightblue"},
            {"name": "Add & Norm", "y": 7.5, "color": "lightgray"},
            {"name": "Output", "y": 9, "color": "white"}
        ]
        
        for comp in block_components:
            if comp["color"] == "white":
                ax2.text(5, comp["y"], comp["name"], ha='center',
                        va='center', fontsize=10, style='italic')
            else:
                rect = FancyBboxPatch((2, comp["y"]-0.4), 6, 0.8,
                                     boxstyle="round,pad=0.1",
                                     facecolor=comp["color"],
                                     edgecolor='black', linewidth=1.5)
                ax2.add_patch(rect)
                ax2.text(5, comp["y"], comp["name"], ha='center',
                        va='center', fontsize=9)
        
        # 接続
        for i in range(len(block_components)-1):
            ax2.arrow(5, block_components[i]["y"]+0.3, 0, 
                     block_components[i+1]["y"]-block_components[i]["y"]-0.6,
                     head_width=0.2, head_length=0.15,
                     fc='black', ec='black', alpha=0.7)
        
        # 残差接続
        ax2.arrow(1.5, 2, 0, 2.2, head_width=0.15, head_length=0.1,
                 fc='blue', ec='blue', linestyle='--', linewidth=2)
        ax2.arrow(1.5, 5.5, 0, 1.7, head_width=0.15, head_length=0.1,
                 fc='blue', ec='blue', linestyle='--', linewidth=2)
        
        ax2.text(1, 3, 'Residual', rotation=90, va='center', color='blue')
        ax2.text(1, 6.5, 'Residual', rotation=90, va='center', color='blue')
        
        ax2.axis('off')
        
        plt.tight_layout()
        plt.show()

@dataclass
class GPTConfig:
    """GPTの設定"""
    vocab_size: int = 50257      # GPT-2のトークナイザーサイズ
    n_positions: int = 1024      # 最大シーケンス長
    n_embd: int = 768           # 埋め込み次元
    n_layer: int = 12           # Transformerブロック数
    n_head: int = 12            # 注意ヘッド数
    n_inner: int = None         # FFNの隠れ層サイズ（None = 4 * n_embd）
    activation: str = "gelu"     # 活性化関数
    dropout: float = 0.1        # ドロップアウト率
    layer_norm_epsilon: float = 1e-5
    initializer_range: float = 0.02
    
    def __post_init__(self):
        if self.n_inner is None:
            self.n_inner = 4 * self.n_embd

class GPTAttention(nn.Module):
    """GPTの注意機構（Causal Self-Attention）"""
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        
        # Q, K, Vを一度に計算（効率的）
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        
        # 出力投影
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        
        # ドロップアウト
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # Causalマスクを事前計算
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.n_positions, config.n_positions))
            .view(1, 1, config.n_positions, config.n_positions)
        )
        
    def forward(self, x: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.size()  # batch, sequence, embedding
        
        # Q, K, Vを計算
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        # ヘッドに分割
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        # Attention計算
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        
        if attention_mask is not None:
            att = att + attention_mask
            
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # 出力投影
        y = self.resid_dropout(self.c_proj(y))
        
        return y

class GPTMLP(nn.Module):
    """GPTのFeed Forward Network"""
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_inner)
        self.c_proj = nn.Linear(config.n_inner, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        
        # 活性化関数
        self.act = self._get_activation(config.activation)
    
    def _get_activation(self, activation: str):
        """活性化関数を取得"""
        if activation == "gelu":
            return nn.GELU()
        elif activation == "relu":
            return nn.ReLU()
        elif activation == "swish":
            return nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class GPTBlock(nn.Module):
    """GPTのTransformerブロック"""
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = GPTAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = GPTMLP(config)
        
    def forward(self, x: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Attention with residual
        x = x + self.attn(self.ln_1(x), attention_mask)
        # MLP with residual
        x = x + self.mlp(self.ln_2(x))
        return x

class GPTModel(nn.Module):
    """完全なGPTモデル"""
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        
        # 埋め込み層
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)  # token embeddings
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)  # position embeddings
        self.drop = nn.Dropout(config.dropout)
        
        # Transformerブロック
        self.h = nn.ModuleList([GPTBlock(config) for _ in range(config.n_layer)])
        
        # 最終層正規化
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        
        # 重みの初期化
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """重みの初期化"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            
    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        device = input_ids.device
        B, T = input_ids.size()
        
        # 位置IDの生成
        if position_ids is None:
            position_ids = torch.arange(0, T, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(B, T)
        
        # 埋め込み
        token_embeddings = self.wte(input_ids)
        position_embeddings = self.wpe(position_ids)
        hidden_states = self.drop(token_embeddings + position_embeddings)
        
        # Attention maskの準備
        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = (1.0 - attention_mask) * -10000.0
        
        # Transformerブロックを通過
        for block in self.h:
            hidden_states = block(hidden_states, attention_mask)
        
        # 最終層正規化
        hidden_states = self.ln_f(hidden_states)
        
        return hidden_states

class GPTLMHeadModel(nn.Module):
    """言語モデリング用のGPT"""
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.transformer = GPTModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # 重み共有（埋め込みと出力層）
        self.lm_head.weight = self.transformer.wte.weight
        
    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        # Transformer
        hidden_states = self.transformer(input_ids, attention_mask)
        
        # 言語モデルヘッド
        lm_logits = self.lm_head(hidden_states)
        
        outputs = {"logits": lm_logits}
        
        # 損失計算
        if labels is not None:
            # ラベルを左にシフト
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # 損失
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            outputs["loss"] = loss
        
        return outputs
    
    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor,
                max_new_tokens: int = 50,
                temperature: float = 1.0,
                top_k: Optional[int] = None,
                top_p: Optional[float] = None) -> torch.Tensor:
        """テキスト生成"""
        self.eval()
        
        for _ in range(max_new_tokens):
            # 最大長に制限
            idx_cond = input_ids if input_ids.size(1) <= self.transformer.config.n_positions else input_ids[:, -self.transformer.config.n_positions:]
            
            # 予測
            outputs = self.forward(idx_cond)
            logits = outputs["logits"][:, -1, :] / temperature
            
            # Top-kサンプリング
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # Top-pサンプリング（Nucleus sampling）
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # 累積確率がtop_pを超える位置を見つける
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits.scatter_(1, indices_to_remove, float('-inf'))
            
            # サンプリング
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # 追加
            input_ids = torch.cat((input_ids, idx_next), dim=1)
            
        return input_ids

## 17.2 GPTのスケーリング法則

class ScalingLawsDemo:
    """スケーリング法則のデモンストレーション"""
    
    def explain_scaling_laws(self):
        """スケーリング法則の説明"""
        print("=== GPTのスケーリング法則 ===\n")
        
        print("Kaplanらの発見（2020）:")
        print("  Loss ∝ N^(-α) × D^(-β) × C^(-γ)")
        print("  - N: モデルパラメータ数")
        print("  - D: データセットサイズ")
        print("  - C: 計算量\n")
        
        print("Chinchillaの法則（2022）:")
        print("  最適なモデルサイズとデータサイズの比率")
        print("  トークン数 ≈ 20 × パラメータ数\n")
        
        # スケーリング法則の可視化
        self._visualize_scaling_laws()
        
    def _visualize_scaling_laws(self):
        """スケーリング法則を可視化"""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # モデルサイズと性能
        model_sizes = np.logspace(6, 11, 50)  # 1M to 100B parameters
        alpha = 0.076  # 実験的に観測された値
        loss = 10 * model_sizes ** (-alpha)
        
        ax1.loglog(model_sizes, loss, 'b-', linewidth=2)
        ax1.set_xlabel('Model Parameters')
        ax1.set_ylabel('Loss')
        ax1.set_title('Model Size Scaling')
        ax1.grid(True, alpha=0.3)
        
        # 実際のGPTモデルをプロット
        gpt_models = {
            'GPT': 117e6,
            'GPT-2': 1.5e9,
            'GPT-3': 175e9,
            'GPT-4': 1e12  # 推定
        }
        
        for name, size in gpt_models.items():
            estimated_loss = 10 * size ** (-alpha)
            ax1.scatter(size, estimated_loss, s=100, zorder=5)
            ax1.annotate(name, (size, estimated_loss), 
                        xytext=(10, 10), textcoords='offset points')
        
        # データサイズと性能
        data_sizes = np.logspace(6, 12, 50)  # 1M to 1T tokens
        beta = 0.095
        loss_data = 8 * data_sizes ** (-beta)
        
        ax2.loglog(data_sizes, loss_data, 'g-', linewidth=2)
        ax2.set_xlabel('Dataset Size (tokens)')
        ax2.set_ylabel('Loss')
        ax2.set_title('Data Size Scaling')
        ax2.grid(True, alpha=0.3)
        
        # 計算量と性能
        compute = np.logspace(15, 25, 50)  # FLOPs
        gamma = 0.050
        loss_compute = 5 * compute ** (-gamma)
        
        ax3.loglog(compute, loss_compute, 'r-', linewidth=2)
        ax3.set_xlabel('Compute (FLOPs)')
        ax3.set_ylabel('Loss')
        ax3.set_title('Compute Scaling')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # 最適な配分の可視化
        self._visualize_optimal_allocation()
    
    def _visualize_optimal_allocation(self):
        """最適なリソース配分を可視化"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # パラメータ数の範囲
        params = np.logspace(7, 12, 100)  # 10M to 1T
        
        # Chinchillaの推奨
        optimal_tokens = 20 * params
        
        # 異なる配分戦略
        strategies = {
            'Chinchilla Optimal': optimal_tokens,
            'Compute Optimal': 10 * params,  # より少ないデータ
            'Over-trained': 100 * params,    # より多いデータ
        }
        
        for name, tokens in strategies.items():
            ax.loglog(params, tokens, linewidth=2, label=name)
        
        # 実際のモデル
        real_models = [
            ('GPT-3', 175e9, 300e9),
            ('Chinchilla', 70e9, 1.4e12),
            ('LLaMA', 7e9, 1e12),
            ('LLaMA-2', 70e9, 2e12)
        ]
        
        for name, param, token in real_models:
            ax.scatter(param, token, s=100, zorder=5)
            ax.annotate(name, (param, token), 
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=9)
        
        ax.set_xlabel('Model Parameters')
        ax.set_ylabel('Training Tokens')
        ax.set_title('Model Size vs Training Data')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 等計算量曲線
        compute_levels = [1e21, 1e22, 1e23, 1e24]
        for c in compute_levels:
            # 簡略化: compute ∝ params × tokens
            tokens_for_compute = c / params
            mask = tokens_for_compute < 1e15  # 現実的な範囲
            ax.plot(params[mask], tokens_for_compute[mask], 
                   'k--', alpha=0.3, linewidth=1)
            ax.text(params[mask][-1], tokens_for_compute[mask][-1],
                   f'{c:.0e} FLOPs', fontsize=8, alpha=0.7)
        
        plt.tight_layout()
        plt.show()

## 17.3 GPTの学習技術

class GPTTrainingTechniques:
    """GPTの学習技術"""
    
    def demonstrate_training_techniques(self):
        """主要な学習技術をデモ"""
        print("=== GPTの学習技術 ===\n")
        
        # 1. Learning Rate Schedule
        self._demonstrate_lr_schedule()
        
        # 2. Gradient Clipping
        print("\n2. Gradient Clipping:")
        print("   - 勾配爆発を防ぐ")
        print("   - 典型的な値: 1.0")
        print("   - 安定した学習を実現\n")
        
        # 3. Weight Decay
        print("3. Weight Decay:")
        print("   - 正則化効果")
        print("   - 埋め込み層とLayerNormには適用しない")
        print("   - 典型的な値: 0.1\n")
        
        # 4. Mixed Precision Training
        print("4. Mixed Precision Training:")
        print("   - FP16とFP32を併用")
        print("   - メモリ使用量を削減")
        print("   - 学習速度を向上")
    
    def _demonstrate_lr_schedule(self):
        """学習率スケジュールのデモ"""
        print("1. Learning Rate Schedule:")
        
        # Cosine schedule with warmup
        total_steps = 100000
        warmup_steps = 10000
        max_lr = 6e-4
        min_lr = 6e-5
        
        steps = np.arange(total_steps)
        lr = np.zeros_like(steps, dtype=float)
        
        # Warmup
        lr[:warmup_steps] = max_lr * steps[:warmup_steps] / warmup_steps
        
        # Cosine decay
        progress = (steps[warmup_steps:] - warmup_steps) / (total_steps - warmup_steps)
        lr[warmup_steps:] = min_lr + (max_lr - min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        
        # 可視化
        plt.figure(figsize=(10, 6))
        plt.plot(steps, lr, linewidth=2)
        plt.axvline(x=warmup_steps, color='red', linestyle='--', 
                   alpha=0.5, label='End of Warmup')
        plt.xlabel('Training Steps')
        plt.ylabel('Learning Rate')
        plt.title('GPT Learning Rate Schedule (Cosine with Warmup)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        print("   - Linear warmup: 学習初期の不安定性を回避")
        print("   - Cosine decay: スムーズな学習率の減衰")
        print("   - 最終的に小さな学習率で微調整")

class GPTOptimizationTricks:
    """GPTの最適化トリック"""
    
    def __init__(self):
        self.config = GPTConfig(n_layer=12, n_head=12, n_embd=768)
        
    def demonstrate_efficient_attention(self):
        """効率的な注意機構の実装"""
        print("=== 効率的な注意機構 ===\n")
        
        class FlashAttentionGPT(nn.Module):
            """Flash Attention風の最適化"""
            
            def __init__(self, config: GPTConfig):
                super().__init__()
                self.n_head = config.n_head
                self.n_embd = config.n_embd
                self.dropout = config.dropout
                
                # Fused QKV projection
                self.qkv_proj = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
                self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
                
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                B, T, C = x.shape
                
                # 効率的なQKV計算
                qkv = self.qkv_proj(x)
                qkv = qkv.reshape(B, T, 3, self.n_head, C // self.n_head)
                qkv = qkv.permute(2, 0, 3, 1, 4)
                q, k, v = qkv[0], qkv[1], qkv[2]
                
                # Flash Attentionのシミュレーション
                # 実際はカスタムCUDAカーネルを使用
                if T <= 128:  # 短いシーケンスは通常の計算
                    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
                    att = att.masked_fill(
                        torch.triu(torch.ones(T, T), diagonal=1).bool(), 
                        float('-inf')
                    )
                    att = F.softmax(att, dim=-1)
                    y = att @ v
                else:
                    # ブロック単位の計算（簡略版）
                    block_size = 64
                    y = torch.zeros_like(v)
                    
                    for i in range(0, T, block_size):
                        end_i = min(i + block_size, T)
                        for j in range(0, i + block_size, block_size):
                            end_j = min(j + block_size, T)
                            
                            if j <= i:  # Causal mask
                                q_block = q[:, :, i:end_i]
                                k_block = k[:, :, j:end_j]
                                v_block = v[:, :, j:end_j]
                                
                                att_block = (q_block @ k_block.transpose(-2, -1)) * \
                                          (1.0 / math.sqrt(k.size(-1)))
                                
                                # ブロック内のマスク
                                if i == j:
                                    block_mask = torch.triu(
                                        torch.ones(end_i-i, end_j-j), 
                                        diagonal=1
                                    ).bool()
                                    att_block = att_block.masked_fill(
                                        block_mask, float('-inf')
                                    )
                                
                                att_block = F.softmax(att_block, dim=-1)
                                y[:, :, i:end_i] += att_block @ v_block
                
                # 出力
                y = y.transpose(1, 2).contiguous().view(B, T, C)
                y = self.out_proj(y)
                
                return y
        
        print("Flash Attentionの利点:")
        print("✓ メモリ効率: O(n) instead of O(n²)")
        print("✓ より長いシーケンスの処理が可能")
        print("✓ メモリ帯域幅の有効活用")
        
        # パフォーマンス比較
        self._compare_attention_performance()
    
    def _compare_attention_performance(self):
        """注意機構のパフォーマンス比較"""
        seq_lengths = [128, 256, 512, 1024, 2048]
        standard_memory = []
        optimized_memory = []
        
        for seq_len in seq_lengths:
            # 標準的な注意のメモリ使用量（概算）
            # O(batch * heads * seq_len²)
            std_mem = 32 * 12 * seq_len * seq_len * 4 / 1024**2  # MB
            standard_memory.append(std_mem)
            
            # 最適化された注意のメモリ使用量
            # O(batch * heads * seq_len)
            opt_mem = 32 * 12 * seq_len * 64 * 4 / 1024**2  # MB (block_size=64)
            optimized_memory.append(opt_mem)
        
        plt.figure(figsize=(10, 6))
        plt.plot(seq_lengths, standard_memory, 'r-', marker='o', 
                label='Standard Attention', linewidth=2)
        plt.plot(seq_lengths, optimized_memory, 'g-', marker='s', 
                label='Optimized Attention', linewidth=2)
        
        plt.xlabel('Sequence Length')
        plt.ylabel('Memory Usage (MB)')
        plt.title('Attention Memory Usage Comparison')
        plt.legend()
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.show()

## 17.4 GPTの応用と発展

class GPTApplications:
    """GPTの応用例"""
    
    def demonstrate_applications(self):
        """様々な応用を実演"""
        print("=== GPTの応用 ===\n")
        
        # 1. Few-shot学習
        self._demonstrate_few_shot()
        
        # 2. Chain-of-Thought
        self._demonstrate_chain_of_thought()
        
        # 3. Instruction Tuning
        self._demonstrate_instruction_tuning()
    
    def _demonstrate_few_shot(self):
        """Few-shot学習のデモ"""
        print("1. Few-shot Learning:\n")
        
        few_shot_prompt = """
Task: Sentiment Analysis

Example 1:
Text: "This movie was fantastic! I loved every minute of it."
Sentiment: Positive

Example 2:
Text: "The service was terrible and the food was cold."
Sentiment: Negative

Example 3:
Text: "The weather is nice today."
Sentiment: """
        
        print("プロンプト:")
        print(few_shot_prompt)
        print("\nGPTは文脈から学習してタスクを実行")
        print("期待される出力: Neutral\n")
    
    def _demonstrate_chain_of_thought(self):
        """Chain-of-Thought推論のデモ"""
        print("2. Chain-of-Thought Reasoning:\n")
        
        cot_prompt = """
Q: Jack has 5 apples. He buys 3 more apples and then gives 2 apples to his friend. How many apples does Jack have now?

A: Let's think step by step:
1. Jack starts with 5 apples
2. He buys 3 more apples: 5 + 3 = 8 apples
3. He gives 2 apples to his friend: 8 - 2 = 6 apples
Therefore, Jack has 6 apples now.

Q: If a train travels at 60 mph for 2 hours, then at 40 mph for 1 hour, what is the total distance traveled?

A: Let's think step by step:"""
        
        print("プロンプト:")
        print(cot_prompt)
        print("\nGPTは段階的な推論プロセスを学習")
    
    def _demonstrate_instruction_tuning(self):
        """Instruction Tuningのデモ"""
        print("\n3. Instruction Tuning:\n")
        
        examples = [
            {
                "instruction": "Translate the following English text to French:",
                "input": "Hello, how are you?",
                "output": "Bonjour, comment allez-vous?"
            },
            {
                "instruction": "Summarize the following text in one sentence:",
                "input": "The quick brown fox jumps over the lazy dog. This pangram contains all letters of the alphabet.",
                "output": "This is a pangram that includes every letter of the alphabet."
            },
            {
                "instruction": "Convert the following number to binary:",
                "input": "42",
                "output": "101010"
            }
        ]
        
        print("Instruction-Response形式でのファインチューニング:")
        for ex in examples[:2]:
            print(f"\nInstruction: {ex['instruction']}")
            print(f"Input: {ex['input']}")
            print(f"Expected Output: {ex['output']}")

class GPTVariants:
    """GPTの派生モデル"""
    
    def explain_variants(self):
        """主要な派生モデルを説明"""
        print("=== GPTの派生モデル ===\n")
        
        variants = {
            "GPT-2": {
                "params": "1.5B",
                "context": "1024",
                "特徴": "Zero-shot性能の実証"
            },
            "GPT-3": {
                "params": "175B",
                "context": "2048",
                "特徴": "Few-shot学習の革命"
            },
            "GPT-4": {
                "params": "~1T (推定)",
                "context": "8K-32K",
                "特徴": "マルチモーダル、高度な推論"
            },
            "GPT-Neo/GPT-J": {
                "params": "2.7B/6B",
                "context": "2048",
                "特徴": "オープンソース実装"
            },
            "CodeGPT/Codex": {
                "params": "12B",
                "context": "4096", 
                "特徴": "コード生成に特化"
            }
        }
        
        # 比較表の作成
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('tight')
        ax.axis('off')
        
        # データの準備
        headers = ["Model", "Parameters", "Context Length", "Key Features"]
        cell_data = []
        
        for model, info in variants.items():
            cell_data.append([
                model,
                info["params"],
                info["context"],
                info["特徴"]
            ])
        
        # テーブルの作成
        table = ax.table(cellText=cell_data, colLabels=headers,
                        cellLoc='left', loc='center',
                        colWidths=[0.2, 0.2, 0.2, 0.4])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # スタイリング
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        plt.title('GPT Model Variants Comparison', fontsize=16, weight='bold', pad=20)
        plt.show()

# 実装例とデモ
def run_gpt_demo():
    """GPTのデモを実行"""
    print("=" * 70)
    print("GPTアーキテクチャのデモ")
    print("=" * 70 + "\n")
    
    # 1. アーキテクチャ概要
    overview = GPTArchitectureOverview()
    overview.explain_gpt_philosophy()
    
    # 2. 小さなGPTモデルの作成
    print("\n=== 小規模GPTモデルの作成 ===")
    config = GPTConfig(
        vocab_size=1000,
        n_positions=128,
        n_embd=128,
        n_layer=4,
        n_head=4
    )
    
    model = GPTLMHeadModel(config)
    
    # パラメータ数の計算
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nモデルパラメータ数: {total_params:,}")
    print(f"モデルサイズ: {total_params * 4 / 1024**2:.2f} MB (float32)")
    
    # 3. 推論デモ
    print("\n=== 推論デモ ===")
    
    # ダミー入力
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    # 順伝播
    with torch.no_grad():
        outputs = model(input_ids)
        print(f"入力形状: {input_ids.shape}")
        print(f"出力ロジット形状: {outputs['logits'].shape}")
    
    # 生成デモ
    print("\n生成例:")
    prompt = torch.tensor([[1, 2, 3]])  # ダミープロンプト
    generated = model.generate(prompt, max_new_tokens=10, temperature=0.8)
    print(f"生成されたトークンID: {generated.tolist()[0]}")
    
    # 4. スケーリング法則
    print("\n")
    scaling_demo = ScalingLawsDemo()
    scaling_demo.explain_scaling_laws()
    
    # 5. 学習技術
    print("\n")
    training_demo = GPTTrainingTechniques()
    training_demo.demonstrate_training_techniques()
    
    # 6. 最適化
    print("\n")
    optimization_demo = GPTOptimizationTricks()
    optimization_demo.demonstrate_efficient_attention()
    
    # 7. 応用
    print("\n")
    applications = GPTApplications()
    applications.demonstrate_applications()
    
    # 8. 派生モデル
    print("\n")
    variants = GPTVariants()
    variants.explain_variants()
    
    print("\n" + "=" * 70)
    print("まとめ")
    print("=" * 70)
    print("\nGPTの成功要因:")
    print("• シンプルで拡張可能なアーキテクチャ")
    print("• 大規模データでの事前学習")
    print("• 創発的な能力の獲得")
    print("• 汎用的なタスク形式（次トークン予測）")
    print("\nGPTは言語理解と生成の新時代を切り開きました。")

if __name__ == "__main__":
    run_gpt_demo()