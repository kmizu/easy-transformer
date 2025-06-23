# 各コンポーネントの実装

## はじめに：部品から全体へ

優れたコンパイラは、よく設計されたモジュールの組み合わせです。字句解析器、構文解析器、意味解析器、コード生成器—それぞれが独立して動作し、明確なインターフェースで接続されています。

Transformerも同じ設計哲学に従います。この章では、各コンポーネントを詳細に実装し、それらがどのように組み合わさって強力なモデルを形成するかを見ていきます。

## 14.1 Multi-Head Attentionの完全実装

### なぜMulti-Headが必要か

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, List, Dict, Any
import math
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle
import matplotlib.patches as mpatches

class MultiHeadAttentionImplementation:
    """Multi-Head Attentionの詳細実装"""
    
    def __init__(self):
        self.d_model = 512
        self.n_heads = 8
        self.d_k = self.d_model // self.n_heads  # 64
        
    def explain_multi_head_benefits(self):
        """Multi-Headの利点を説明"""
        print("=== Multi-Head Attentionの利点 ===\n")
        
        print("1. 並列的な表現学習:")
        print("   - 各ヘッドが異なる特徴に注目")
        print("   - 文法、意味、文脈など多様な関係を捉える\n")
        
        print("2. 計算効率:")
        print("   - ヘッドごとの次元削減で総計算量を抑制")
        print("   - 並列計算が可能\n")
        
        print("3. 表現力の向上:")
        print("   - 単一の注意機構より豊かな表現")
        print("   - アンサンブル効果による頑健性\n")
        
        # 視覚的説明
        self._visualize_multi_head_concept()
    
    def _visualize_multi_head_concept(self):
        """Multi-Headの概念を可視化"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Single Head
        ax1.set_title('Single-Head Attention', fontsize=14, weight='bold')
        ax1.set_xlim(0, 10)
        ax1.set_ylim(0, 10)
        
        # 入力
        input_rect = Rectangle((1, 2), 2, 6, facecolor='lightblue', 
                              edgecolor='darkblue', linewidth=2)
        ax1.add_patch(input_rect)
        ax1.text(2, 5, 'Input\n(d_model)', ha='center', va='center')
        
        # Attention
        attn_rect = Rectangle((4, 3), 3, 4, facecolor='lightcoral',
                             edgecolor='darkred', linewidth=2)
        ax1.add_patch(attn_rect)
        ax1.text(5.5, 5, 'Attention', ha='center', va='center')
        
        # 出力
        output_rect = Rectangle((8, 2), 2, 6, facecolor='lightgreen',
                               edgecolor='darkgreen', linewidth=2)
        ax1.add_patch(output_rect)
        ax1.text(9, 5, 'Output\n(d_model)', ha='center', va='center')
        
        # 矢印
        ax1.arrow(3, 5, 0.8, 0, head_width=0.3, head_length=0.2, 
                 fc='black', ec='black')
        ax1.arrow(7, 5, 0.8, 0, head_width=0.3, head_length=0.2,
                 fc='black', ec='black')
        
        ax1.axis('off')
        
        # Multi-Head
        ax2.set_title('Multi-Head Attention (8 heads)', fontsize=14, weight='bold')
        ax2.set_xlim(0, 12)
        ax2.set_ylim(0, 10)
        
        # 入力
        input_rect2 = Rectangle((1, 2), 2, 6, facecolor='lightblue',
                               edgecolor='darkblue', linewidth=2)
        ax2.add_patch(input_rect2)
        ax2.text(2, 5, 'Input\n(d_model)', ha='center', va='center')
        
        # 複数のヘッド
        colors = plt.cm.Set3(np.linspace(0, 1, 8))
        for i in range(8):
            y_pos = 1 + i * 0.9
            head_rect = Rectangle((4, y_pos), 2, 0.7, 
                                 facecolor=colors[i], alpha=0.7,
                                 edgecolor='black', linewidth=1)
            ax2.add_patch(head_rect)
            ax2.text(5, y_pos + 0.35, f'H{i+1}', ha='center', 
                    va='center', fontsize=8)
            
            # 矢印
            ax2.arrow(3, 5, 0.8, y_pos + 0.35 - 5, 
                     head_width=0.15, head_length=0.1,
                     fc='gray', ec='gray', alpha=0.5)
        
        # Concat
        concat_rect = Rectangle((7, 2), 2, 6, facecolor='lightyellow',
                               edgecolor='orange', linewidth=2)
        ax2.add_patch(concat_rect)
        ax2.text(8, 5, 'Concat', ha='center', va='center')
        
        # 出力
        output_rect2 = Rectangle((10, 2), 2, 6, facecolor='lightgreen',
                                edgecolor='darkgreen', linewidth=2)
        ax2.add_patch(output_rect2)
        ax2.text(11, 5, 'Output\n(d_model)', ha='center', va='center')
        
        # 矢印
        for i in range(8):
            y_pos = 1 + i * 0.9
            ax2.arrow(6, y_pos + 0.35, 0.8, 5 - (y_pos + 0.35),
                     head_width=0.15, head_length=0.1,
                     fc='gray', ec='gray', alpha=0.5)
        
        ax2.arrow(9, 5, 0.8, 0, head_width=0.3, head_length=0.2,
                 fc='black', ec='black')
        
        ax2.axis('off')
        
        plt.tight_layout()
        plt.show()

class OptimizedMultiHeadAttention(nn.Module):
    """最適化されたMulti-Head Attention実装"""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1,
                 use_bias: bool = True):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.d_k)
        
        # 効率的な実装：Q, K, Vを一つの行列で計算
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=use_bias)
        
        # 出力投影
        self.out_proj = nn.Linear(d_model, d_model, bias=use_bias)
        
        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        
        # 初期化
        self._init_weights()
        
    def _init_weights(self):
        """重みの初期化"""
        # Xavier初期化
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        
        if self.qkv_proj.bias is not None:
            nn.init.zeros_(self.qkv_proj.bias)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)
    
    def forward(self, query: torch.Tensor, 
                key: Optional[torch.Tensor] = None,
                value: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None,
                need_weights: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            query: [batch_size, seq_len, d_model]
            key: [batch_size, seq_len, d_model] (None for self-attention)
            value: [batch_size, seq_len, d_model] (None for self-attention)
            mask: [batch_size, seq_len, seq_len] or [seq_len, seq_len]
            need_weights: 注意重みを返すかどうか
        
        Returns:
            output: [batch_size, seq_len, d_model]
            attn_weights: [batch_size, n_heads, seq_len, seq_len] (if need_weights)
        """
        batch_size, seq_len, _ = query.shape
        
        # Self-attentionの場合
        if key is None:
            key = query
        if value is None:
            value = query
        
        # Q, K, Vを一度に計算（効率的）
        if key is query:  # self-attention
            qkv = self.qkv_proj(query)
            qkv = qkv.reshape(batch_size, seq_len, 3, self.n_heads, self.d_k)
            qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, L, D]
            q, k, v = qkv[0], qkv[1], qkv[2]
        else:  # cross-attention
            q = self.qkv_proj(query)[:, :, :self.d_model]
            k = self.qkv_proj(key)[:, :, self.d_model:2*self.d_model]
            v = self.qkv_proj(value)[:, :, 2*self.d_model:]
            
            q = q.reshape(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
            k = k.reshape(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
            v = v.reshape(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled Dot-Product Attention
        attn_output, attn_weights = self._scaled_dot_product_attention(
            q, k, v, mask, self.attn_dropout if self.training else None
        )
        
        # ヘッドを結合
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.d_model)
        
        # 出力投影
        output = self.out_proj(attn_output)
        output = self.proj_dropout(output)
        
        if need_weights:
            return output, attn_weights
        else:
            return output, None
    
    def _scaled_dot_product_attention(self, q: torch.Tensor, k: torch.Tensor,
                                     v: torch.Tensor, mask: Optional[torch.Tensor],
                                     dropout: Optional[nn.Dropout]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Scaled Dot-Product Attentionの計算"""
        # 注意スコアの計算
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # マスクの適用
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        
        # Dropout
        if dropout is not None:
            attn_weights = dropout(attn_weights)
        
        # 重み付き和
        attn_output = torch.matmul(attn_weights, v)
        
        return attn_output, attn_weights
    
    def get_attention_maps(self, query: torch.Tensor,
                          key: Optional[torch.Tensor] = None,
                          mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """注意マップの取得（可視化用）"""
        with torch.no_grad():
            _, attn_weights = self.forward(query, key, need_weights=True)
        return attn_weights

class FlashAttentionDemo:
    """Flash Attentionの概念説明"""
    
    def explain_flash_attention(self):
        """Flash Attentionの説明"""
        print("=== Flash Attention ===\n")
        
        print("問題: 標準的なAttentionのメモリボトルネック")
        print("- O(N²)のメモリ使用量")
        print("- GPUメモリ帯域幅の制約\n")
        
        print("Flash Attentionの解決策:")
        print("1. タイリング: 小さなブロックで計算")
        print("2. 再計算: 中間結果を保存せず再計算")
        print("3. カーネル融合: 複数の操作を1つのカーネルに\n")
        
        # 図解
        self._visualize_flash_attention()
        
    def _visualize_flash_attention(self):
        """Flash Attentionの動作を可視化"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # 標準的なAttention
        ax1.set_title('標準的なAttention', fontsize=12)
        ax1.set_xlim(0, 10)
        ax1.set_ylim(0, 10)
        
        # メモリ使用
        memory_blocks = [
            {"name": "Q", "pos": (1, 7), "size": (2, 2), "color": "lightblue"},
            {"name": "K", "pos": (1, 4), "size": (2, 2), "color": "lightgreen"},
            {"name": "QKᵀ", "pos": (4, 5.5), "size": (3, 3), "color": "yellow"},
            {"name": "Softmax", "pos": (8, 5.5), "size": (3, 3), "color": "orange"}
        ]
        
        for block in memory_blocks:
            rect = Rectangle(block["pos"], block["size"][0], block["size"][1],
                           facecolor=block["color"], edgecolor='black', linewidth=2)
            ax1.add_patch(rect)
            ax1.text(block["pos"][0] + block["size"][0]/2,
                    block["pos"][1] + block["size"][1]/2,
                    block["name"], ha='center', va='center')
        
        # メモリ使用量表示
        ax1.text(5, 1, 'メモリ: O(N²)', fontsize=14, weight='bold',
                ha='center', color='red')
        
        ax1.axis('off')
        
        # Flash Attention
        ax2.set_title('Flash Attention', fontsize=12)
        ax2.set_xlim(0, 10)
        ax2.set_ylim(0, 10)
        
        # タイリング
        tile_size = 1.5
        for i in range(2):
            for j in range(2):
                x = 2 + j * (tile_size + 0.5)
                y = 4 + i * (tile_size + 0.5)
                
                tile = Rectangle((x, y), tile_size, tile_size,
                               facecolor='lightcyan', edgecolor='darkblue',
                               linewidth=2, linestyle='--')
                ax2.add_patch(tile)
                ax2.text(x + tile_size/2, y + tile_size/2,
                        f'Tile\n{i},{j}', ha='center', va='center', fontsize=8)
        
        # On-chip SRAM
        sram = Rectangle((6, 4), 3, 3, facecolor='lightpink',
                        edgecolor='darkred', linewidth=2)
        ax2.add_patch(sram)
        ax2.text(7.5, 5.5, 'On-chip\nSRAM', ha='center', va='center')
        
        # メモリ使用量表示
        ax2.text(5, 1, 'メモリ: O(N)', fontsize=14, weight='bold',
                ha='center', color='green')
        
        ax2.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        print("利点:")
        print("✓ メモリ効率: O(N²) → O(N)")
        print("✓ 速度向上: メモリ帯域幅の有効活用")
        print("✓ 長いシーケンスの処理が可能")

## 14.2 位置エンコーディングの発展

class AdvancedPositionalEncoding:
    """高度な位置エンコーディング手法"""
    
    def __init__(self):
        self.d_model = 128
        self.max_len = 100
        
    def compare_encoding_methods(self):
        """異なる位置エンコーディング手法の比較"""
        print("=== 位置エンコーディング手法の比較 ===\n")
        
        methods = {
            "Sinusoidal": self._sinusoidal_encoding,
            "Learned": self._learned_encoding,
            "RoPE": self._rope_encoding,
            "ALiBi": self._alibi_encoding
        }
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for idx, (name, method) in enumerate(methods.items()):
            ax = axes[idx]
            encoding = method()
            
            # ヒートマップ
            im = ax.imshow(encoding[:20, :64], cmap='RdBu_r', 
                          aspect='auto', vmin=-1, vmax=1)
            
            ax.set_title(f'{name} Encoding', fontsize=12)
            ax.set_xlabel('Dimension')
            ax.set_ylabel('Position')
            
            # カラーバー
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.show()
        
        self._explain_each_method()
    
    def _sinusoidal_encoding(self):
        """正弦波位置エンコーディング"""
        pe = torch.zeros(self.max_len, self.d_model)
        position = torch.arange(0, self.max_len).unsqueeze(1).float()
        
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2).float() *
            -(math.log(10000.0) / self.d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.numpy()
    
    def _learned_encoding(self):
        """学習可能な位置エンコーディング"""
        # ランダム初期化（実際は学習される）
        pe = torch.randn(self.max_len, self.d_model) * 0.1
        return pe.numpy()
    
    def _rope_encoding(self):
        """Rotary Position Embedding (RoPE)"""
        # 簡略化した実装
        freqs = 1.0 / (10000 ** (torch.arange(0, self.d_model, 2).float() / self.d_model))
        positions = torch.arange(self.max_len).float()
        
        # 回転行列の要素
        angles = positions.unsqueeze(1) * freqs.unsqueeze(0)
        pe = torch.zeros(self.max_len, self.d_model)
        pe[:, 0::2] = torch.cos(angles)
        pe[:, 1::2] = torch.sin(angles)
        
        return pe.numpy()
    
    def _alibi_encoding(self):
        """Attention with Linear Biases (ALiBi)"""
        # 相対位置バイアス
        positions = torch.arange(self.max_len)
        relative_positions = positions.unsqueeze(1) - positions.unsqueeze(0)
        
        # 線形バイアス（ヘッドごとに異なるスロープ）
        slopes = torch.tensor([2**(-i/4) for i in range(8)])  # 8ヘッドの例
        biases = relative_positions.unsqueeze(0) * slopes.unsqueeze(1).unsqueeze(2)
        
        # 可視化用に最初のヘッドのバイアスを返す
        return biases[0].numpy()[:self.max_len, :self.d_model]
    
    def _explain_each_method(self):
        """各手法の説明"""
        print("\n手法の特徴:\n")
        
        print("1. Sinusoidal (正弦波):")
        print("   ✓ 学習不要")
        print("   ✓ 任意の長さに外挿可能")
        print("   ✓ 相対位置の計算が可能\n")
        
        print("2. Learned (学習型):")
        print("   ✓ タスクに最適化")
        print("   ✗ 固定長")
        print("   ✗ 外挿性能が低い\n")
        
        print("3. RoPE (回転位置埋め込み):")
        print("   ✓ 相対位置を自然に表現")
        print("   ✓ 長いシーケンスに強い")
        print("   ✓ 計算効率が良い\n")
        
        print("4. ALiBi (線形バイアス):")
        print("   ✓ 非常にシンプル")
        print("   ✓ 外挿性能が高い")
        print("   ✓ 埋め込みではなくバイアスとして作用")

class RoPEImplementation(nn.Module):
    """Rotary Position Embedding (RoPE) の実装"""
    
    def __init__(self, d_model: int, max_seq_len: int = 5000, base: float = 10000):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.base = base
        
        # 事前計算
        self._precompute_freqs()
        
    def _precompute_freqs(self):
        """周波数の事前計算"""
        # 周波数
        theta = torch.arange(0, self.d_model, 2).float()
        freqs = 1.0 / (self.base ** (theta / self.d_model))
        
        # 位置
        positions = torch.arange(self.max_seq_len).float()
        
        # 周波数と位置の積
        freqs = torch.outer(positions, freqs)
        
        # cosとsinを事前計算
        self.register_buffer('cos_cached', torch.cos(freqs))
        self.register_buffer('sin_cached', torch.sin(freqs))
        
    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> torch.Tensor:
        """
        RoPEを適用
        Args:
            x: [batch_size, seq_len, n_heads, d_head]
            seq_len: シーケンス長（Noneの場合はxから推定）
        """
        if seq_len is None:
            seq_len = x.shape[1]
            
        # 適切なサイズにスライス
        cos = self.cos_cached[:seq_len].unsqueeze(1)  # [seq_len, 1, d_model/2]
        sin = self.sin_cached[:seq_len].unsqueeze(1)
        
        # xを偶数・奇数インデックスに分割
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        
        # 回転を適用
        x_rotated = torch.stack([
            x_even * cos - x_odd * sin,
            x_even * sin + x_odd * cos
        ], dim=-1)
        
        # 元の形状に戻す
        x_rotated = x_rotated.flatten(-2)
        
        return x_rotated
    
    def rotate_queries_keys(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """クエリとキーにRoPEを適用"""
        q_rotated = self.forward(q)
        k_rotated = self.forward(k)
        return q_rotated, k_rotated

## 14.3 層正規化とその変種

class NormalizationTechniques:
    """正規化技術の実装と比較"""
    
    def compare_normalization_methods(self):
        """異なる正規化手法の比較"""
        print("=== 正規化手法の比較 ===\n")
        
        batch_size = 32
        seq_len = 100
        d_model = 512
        
        # ダミーデータ
        x = torch.randn(batch_size, seq_len, d_model) * 3 + 1
        
        # 各正規化手法
        methods = {
            "LayerNorm": nn.LayerNorm(d_model),
            "RMSNorm": self._create_rmsnorm(d_model),
            "GroupNorm": nn.GroupNorm(8, d_model)  # 8グループ
        }
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for idx, (name, norm) in enumerate(methods.items()):
            ax = axes[idx]
            
            # 正規化前後の分布
            with torch.no_grad():
                if name == "GroupNorm":
                    # GroupNormは異なる形状を期待
                    x_reshaped = x.transpose(1, 2)  # [B, D, L]
                    normalized = norm(x_reshaped)
                    normalized = normalized.transpose(1, 2)  # [B, L, D]に戻す
                else:
                    normalized = norm(x)
            
            # ヒストグラム
            ax.hist(x.flatten().numpy(), bins=50, alpha=0.5, 
                   label='Before', density=True, color='red')
            ax.hist(normalized.flatten().numpy(), bins=50, alpha=0.5,
                   label='After', density=True, color='blue')
            
            ax.set_title(f'{name}')
            ax.set_xlabel('Value')
            ax.set_ylabel('Density')
            ax.legend()
            
            # 統計情報
            mean_before = x.mean().item()
            std_before = x.std().item()
            mean_after = normalized.mean().item()
            std_after = normalized.std().item()
            
            ax.text(0.02, 0.98, 
                   f'Before: μ={mean_before:.2f}, σ={std_before:.2f}\n'
                   f'After: μ={mean_after:.2f}, σ={std_after:.2f}',
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.show()
        
        self._explain_normalization_differences()
    
    def _create_rmsnorm(self, d_model: int) -> nn.Module:
        """RMSNormの実装"""
        class RMSNorm(nn.Module):
            def __init__(self, d_model: int, eps: float = 1e-6):
                super().__init__()
                self.eps = eps
                self.weight = nn.Parameter(torch.ones(d_model))
                
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # RMS計算
                rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
                x = x / rms
                return x * self.weight
        
        return RMSNorm(d_model)
    
    def _explain_normalization_differences(self):
        """正規化手法の違いを説明"""
        print("\n各手法の特徴:\n")
        
        print("1. Layer Normalization:")
        print("   - 各サンプルの特徴次元で正規化")
        print("   - 平均と分散を使用")
        print("   - 学習可能なスケール・シフトパラメータ\n")
        
        print("2. RMS Normalization:")
        print("   - 平均を計算しない（計算効率が良い）")
        print("   - RMSのみで正規化")
        print("   - LLaMAなど最新モデルで採用\n")
        
        print("3. Group Normalization:")
        print("   - チャネルをグループに分けて正規化")
        print("   - バッチサイズに依存しない")
        print("   - Vision Transformerでよく使用")

class PrePostNormComparison:
    """Pre-Norm vs Post-Normの比較"""
    
    def __init__(self):
        self.d_model = 256
        self.n_heads = 8
        
    def create_pre_norm_block(self) -> nn.Module:
        """Pre-Normブロック"""
        class PreNormBlock(nn.Module):
            def __init__(self, d_model, n_heads):
                super().__init__()
                self.norm1 = nn.LayerNorm(d_model)
                self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
                self.norm2 = nn.LayerNorm(d_model)
                self.ffn = nn.Sequential(
                    nn.Linear(d_model, 4 * d_model),
                    nn.ReLU(),
                    nn.Linear(4 * d_model, d_model)
                )
                
            def forward(self, x):
                # Pre-Norm: 正規化してから処理
                attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
                x = x + attn_out
                
                ffn_out = self.ffn(self.norm2(x))
                x = x + ffn_out
                
                return x
        
        return PreNormBlock(self.d_model, self.n_heads)
    
    def create_post_norm_block(self) -> nn.Module:
        """Post-Normブロック"""
        class PostNormBlock(nn.Module):
            def __init__(self, d_model, n_heads):
                super().__init__()
                self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
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
        
        return PostNormBlock(self.d_model, self.n_heads)
    
    def compare_gradient_flow(self):
        """勾配フローの比較"""
        print("=== Pre-Norm vs Post-Norm 勾配フロー ===\n")
        
        # モデル作成
        pre_norm_model = nn.Sequential(*[
            self.create_pre_norm_block() for _ in range(12)
        ])
        post_norm_model = nn.Sequential(*[
            self.create_post_norm_block() for _ in range(12)
        ])
        
        # ダミーデータ
        batch_size = 8
        seq_len = 50
        x = torch.randn(batch_size, seq_len, self.d_model)
        
        # 勾配を計算
        models = {"Pre-Norm": pre_norm_model, "Post-Norm": post_norm_model}
        gradients = {}
        
        for name, model in models.items():
            model.zero_grad()
            output = model(x)
            loss = output.mean()
            loss.backward()
            
            # 各層の勾配ノルムを記録
            grad_norms = []
            for i, layer in enumerate(model):
                # 最初のLinear層の勾配を取得
                if hasattr(layer, 'attn'):
                    grad_norm = layer.attn.in_proj_weight.grad.norm().item()
                    grad_norms.append(grad_norm)
            
            gradients[name] = grad_norms
        
        # 可視化
        self._plot_gradient_comparison(gradients)
    
    def _plot_gradient_comparison(self, gradients: Dict[str, List[float]]):
        """勾配の比較をプロット"""
        plt.figure(figsize=(10, 6))
        
        for name, grad_norms in gradients.items():
            layers = range(1, len(grad_norms) + 1)
            plt.plot(layers, grad_norms, marker='o', label=name, linewidth=2)
        
        plt.xlabel('Layer')
        plt.ylabel('Gradient Norm')
        plt.title('勾配ノルムの層ごとの変化')
        plt.legend()
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print("観察:")
        print("• Pre-Norm: より安定した勾配フロー")
        print("• Post-Norm: 深い層で勾配が減衰しやすい")
        print("• 深いモデルではPre-Normが推奨される")

## 14.4 高度なFFN実装

class AdvancedFFNImplementations:
    """高度なFeed Forward Network実装"""
    
    def __init__(self):
        self.d_model = 512
        self.d_ff = 2048
        
    def implement_glu_variants(self):
        """GLU系の活性化関数の実装"""
        print("=== GLU系活性化関数 ===\n")
        
        class GLU(nn.Module):
            """Gated Linear Unit"""
            def __init__(self, d_model: int, d_ff: int):
                super().__init__()
                self.linear1 = nn.Linear(d_model, d_ff * 2)
                self.linear2 = nn.Linear(d_ff, d_model)
                
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = self.linear1(x)
                x, gate = x.chunk(2, dim=-1)
                x = x * torch.sigmoid(gate)
                x = self.linear2(x)
                return x
        
        class SwiGLU(nn.Module):
            """SwiGLU (Swish-Gated Linear Unit)"""
            def __init__(self, d_model: int, d_ff: int):
                super().__init__()
                self.linear1 = nn.Linear(d_model, d_ff * 2)
                self.linear2 = nn.Linear(d_ff, d_model)
                
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = self.linear1(x)
                x, gate = x.chunk(2, dim=-1)
                x = x * F.silu(gate)  # SiLU = x * sigmoid(x)
                x = self.linear2(x)
                return x
        
        class GeGLU(nn.Module):
            """GeGLU (GELU-Gated Linear Unit)"""
            def __init__(self, d_model: int, d_ff: int):
                super().__init__()
                self.linear1 = nn.Linear(d_model, d_ff * 2)
                self.linear2 = nn.Linear(d_ff, d_model)
                
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = self.linear1(x)
                x, gate = x.chunk(2, dim=-1)
                x = x * F.gelu(gate)
                x = self.linear2(x)
                return x
        
        # 比較
        self._compare_glu_variants(GLU, SwiGLU, GeGLU)
    
    def _compare_glu_variants(self, *glu_classes):
        """GLU変種の比較"""
        x = torch.randn(1, 100, self.d_model)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for idx, glu_class in enumerate(glu_classes):
            ax = axes[idx]
            
            # インスタンス化
            model = glu_class(self.d_model, self.d_ff)
            
            # 順伝播
            with torch.no_grad():
                output = model(x)
            
            # 活性化パターンを可視化
            # 中間層の活性化を取得するためのフック
            activations = []
            def hook(module, input, output):
                if hasattr(output, 'chunk'):
                    x, gate = output.chunk(2, dim=-1)
                    activations.append(gate)
            
            handle = model.linear1.register_forward_hook(hook)
            with torch.no_grad():
                _ = model(x)
            handle.remove()
            
            if activations:
                gate_values = activations[0][0, :, :100].numpy()
                im = ax.imshow(gate_values.T, cmap='RdBu_r', aspect='auto',
                              vmin=-2, vmax=2)
                ax.set_title(f'{glu_class.__name__}')
                ax.set_xlabel('Position')
                ax.set_ylabel('Hidden Dim (first 100)')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.show()
        
        print("GLU系の利点:")
        print("✓ より表現力の高い非線形変換")
        print("✓ 勾配フローの改善")
        print("✓ 学習の安定性向上")

class MixtureOfExperts(nn.Module):
    """Mixture of Experts (MoE) の実装"""
    
    def __init__(self, d_model: int, d_ff: int, n_experts: int = 8, 
                 top_k: int = 2):
        super().__init__()
        self.d_model = d_model
        self.n_experts = n_experts
        self.top_k = top_k
        
        # エキスパート（各エキスパートはFFN）
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.ReLU(),
                nn.Linear(d_ff, d_model)
            ) for _ in range(n_experts)
        ])
        
        # ゲーティングネットワーク
        self.gate = nn.Linear(d_model, n_experts)
        
        # ノイズ（学習時）
        self.noise_std = 0.1
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.shape
        
        # ゲート値の計算
        gate_logits = self.gate(x)  # [B, L, n_experts]
        
        # ノイズを追加（学習時）
        if self.training:
            noise = torch.randn_like(gate_logits) * self.noise_std
            gate_logits = gate_logits + noise
        
        # Top-kエキスパートを選択
        topk_gate_values, topk_indices = torch.topk(
            gate_logits, self.top_k, dim=-1
        )  # [B, L, top_k]
        
        # Softmaxで正規化
        topk_gate_values = F.softmax(topk_gate_values, dim=-1)
        
        # 出力の初期化
        output = torch.zeros_like(x)
        
        # 各トークンに対してTop-kエキスパートを適用
        for i in range(self.top_k):
            # 選択されたエキスパートのインデックス
            expert_idx = topk_indices[..., i]  # [B, L]
            gate_value = topk_gate_values[..., i:i+1]  # [B, L, 1]
            
            # バッチ処理のため、エキスパートごとにグループ化
            for e in range(self.n_experts):
                # このエキスパートが選択されたトークンのマスク
                mask = (expert_idx == e)
                if mask.any():
                    # マスクされたトークンを抽出
                    masked_x = x[mask]
                    # エキスパートを適用
                    expert_out = self.experts[e](masked_x)
                    # ゲート値で重み付けして出力に加算
                    output[mask] += expert_out * gate_value[mask]
        
        return output
    
    def analyze_expert_usage(self, x: torch.Tensor):
        """エキスパートの使用状況を分析"""
        with torch.no_grad():
            gate_logits = self.gate(x)
            _, topk_indices = torch.topk(gate_logits, self.top_k, dim=-1)
            
            # エキスパートごとの選択回数
            expert_counts = torch.zeros(self.n_experts)
            for i in range(self.n_experts):
                expert_counts[i] = (topk_indices == i).sum()
            
            # 可視化
            plt.figure(figsize=(10, 6))
            plt.bar(range(self.n_experts), expert_counts.numpy())
            plt.xlabel('Expert ID')
            plt.ylabel('Selection Count')
            plt.title('Expert Usage Distribution')
            plt.grid(True, alpha=0.3)
            plt.show()
            
            return expert_counts

# 実行例
def main():
    """メイン実行関数"""
    print("=" * 70)
    print("各コンポーネントの詳細実装")
    print("=" * 70 + "\n")
    
    # Multi-Head Attention
    mha_demo = MultiHeadAttentionImplementation()
    mha_demo.explain_multi_head_benefits()
    
    # 最適化されたMulti-Head Attention
    print("\n" + "=" * 70 + "\n")
    print("=== 最適化されたMulti-Head Attention ===\n")
    
    model = OptimizedMultiHeadAttention(d_model=512, n_heads=8)
    x = torch.randn(2, 100, 512)
    output, _ = model(x)
    print(f"入力形状: {x.shape}")
    print(f"出力形状: {output.shape}")
    
    # Flash Attention
    print("\n" + "=" * 70 + "\n")
    flash_demo = FlashAttentionDemo()
    flash_demo.explain_flash_attention()
    
    # 位置エンコーディング
    print("\n" + "=" * 70 + "\n")
    pos_demo = AdvancedPositionalEncoding()
    pos_demo.compare_encoding_methods()
    
    # RoPE実装
    print("\n" + "=" * 70 + "\n")
    print("=== RoPE実装例 ===\n")
    rope = RoPEImplementation(d_model=128)
    x = torch.randn(2, 50, 8, 16)  # [batch, seq_len, n_heads, d_head]
    x_rotated = rope(x)
    print(f"RoPE適用前: {x.shape}")
    print(f"RoPE適用後: {x_rotated.shape}")
    
    # 正規化手法
    print("\n" + "=" * 70 + "\n")
    norm_demo = NormalizationTechniques()
    norm_demo.compare_normalization_methods()
    
    # Pre-Norm vs Post-Norm
    print("\n" + "=" * 70 + "\n")
    norm_comparison = PrePostNormComparison()
    norm_comparison.compare_gradient_flow()
    
    # 高度なFFN
    print("\n" + "=" * 70 + "\n")
    ffn_demo = AdvancedFFNImplementations()
    ffn_demo.implement_glu_variants()
    
    # MoE
    print("\n" + "=" * 70 + "\n")
    print("=== Mixture of Experts ===\n")
    moe = MixtureOfExperts(d_model=256, d_ff=1024, n_experts=8, top_k=2)
    x = torch.randn(4, 50, 256)
    output = moe(x)
    print(f"MoE入力: {x.shape}")
    print(f"MoE出力: {output.shape}")
    moe.analyze_expert_usage(x)
    
    print("\n" + "=" * 70)
    print("まとめ:")
    print("• 各コンポーネントには多様な実装方法が存在")
    print("• タスクやリソースに応じて適切な手法を選択")
    print("• 最新の研究成果を取り入れることで性能向上")
    print("• 実装の詳細が性能に大きく影響")

if __name__ == "__main__":
    main()