# エンコーダー・デコーダー構造

## はじめに：変換の本質

コンパイラを思い出してください。ソースコード（入力）を機械語（出力）に変換する過程で、まず入力を完全に解析して中間表現（AST）を構築し、それから出力を生成します。この「理解してから生成する」という2段階のプロセスが、Transformerのエンコーダー・デコーダー構造の本質です。

エンコーダーは入力全体を理解し、豊かな内部表現を構築します。デコーダーはその表現を参照しながら、一つずつ出力を生成していきます。これは、プログラムの意味を完全に理解してから、ターゲット言語のコードを生成するコンパイラの動作とよく似ています。

この章では、エンコーダーとデコーダーがどのように協調して動作し、なぜこの構造が翻訳などのタスクに効果的なのかを詳しく見ていきます。

## 12.1 エンコーダーの役割

### 入力の完全な理解

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Optional, Any
import math
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle, FancyArrowPatch
from matplotlib.patches import ConnectionPatch
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

class TransformerEncoder:
    """Transformerエンコーダーの実装と解説"""
    
    def __init__(self, d_model: int = 512, n_heads: int = 8, 
                 d_ff: int = 2048, n_layers: int = 6):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.n_layers = n_layers
        
    def create_encoder_layer(self) -> nn.Module:
        """単一のエンコーダー層を作成"""
        class EncoderLayer(nn.Module):
            def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
                super().__init__()
                
                # Multi-Head Attention
                self.self_attn = nn.MultiheadAttention(
                    d_model, n_heads, dropout=dropout, batch_first=True
                )
                
                # Feed Forward Network
                self.feed_forward = nn.Sequential(
                    nn.Linear(d_model, d_ff),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_ff, d_model)
                )
                
                # Layer Normalization
                self.norm1 = nn.LayerNorm(d_model)
                self.norm2 = nn.LayerNorm(d_model)
                
                # Dropout
                self.dropout = nn.Dropout(dropout)
                
            def forward(self, x, mask=None):
                # Self-Attention with residual connection
                attn_output, attn_weights = self.self_attn(
                    x, x, x, attn_mask=mask
                )
                x = x + self.dropout(attn_output)
                x = self.norm1(x)
                
                # Feed Forward with residual connection
                ff_output = self.feed_forward(x)
                x = x + self.dropout(ff_output)
                x = self.norm2(x)
                
                return x, attn_weights
        
        return EncoderLayer(self.d_model, self.n_heads, self.d_ff)
    
    def create_full_encoder(self) -> nn.Module:
        """完全なエンコーダーを作成"""
        class Encoder(nn.Module):
            def __init__(self, n_layers, d_model, n_heads, d_ff, 
                        vocab_size, max_len=5000, dropout=0.1):
                super().__init__()
                
                # 埋め込み層
                self.embedding = nn.Embedding(vocab_size, d_model)
                self.pos_encoding = self._create_positional_encoding(
                    max_len, d_model
                )
                
                # エンコーダー層のスタック
                self.layers = nn.ModuleList([
                    EncoderLayer(d_model, n_heads, d_ff, dropout)
                    for _ in range(n_layers)
                ])
                
                self.dropout = nn.Dropout(dropout)
                self.scale = math.sqrt(d_model)
                
            def _create_positional_encoding(self, max_len, d_model):
                """位置エンコーディングの作成"""
                pe = torch.zeros(max_len, d_model)
                position = torch.arange(0, max_len).unsqueeze(1).float()
                
                div_term = torch.exp(
                    torch.arange(0, d_model, 2).float() * 
                    -(math.log(10000.0) / d_model)
                )
                
                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term)
                
                return nn.Parameter(pe.unsqueeze(0), requires_grad=False)
                
            def forward(self, src, mask=None):
                # 埋め込みと位置エンコーディング
                seq_len = src.size(1)
                x = self.embedding(src) * self.scale
                x = x + self.pos_encoding[:, :seq_len]
                x = self.dropout(x)
                
                # 各層の出力を保存（分析用）
                layer_outputs = []
                attention_weights = []
                
                # エンコーダー層を順次適用
                for layer in self.layers:
                    x, attn = layer(x, mask)
                    layer_outputs.append(x)
                    attention_weights.append(attn)
                
                return x, layer_outputs, attention_weights
        
        return Encoder(
            self.n_layers, self.d_model, self.n_heads, 
            self.d_ff, vocab_size=10000
        )
    
    def visualize_encoder_process(self):
        """エンコーダーの処理過程を可視化"""
        print("=== エンコーダーの処理過程 ===\n")
        
        # サンプル入力
        sample_text = "The cat sat on the mat"
        tokens = sample_text.split()
        seq_len = len(tokens)
        
        # ダミーのトークンID
        token_ids = torch.randint(0, 1000, (1, seq_len))
        
        # エンコーダーを作成して実行
        encoder = self.create_full_encoder()
        encoder.eval()
        
        with torch.no_grad():
            output, layer_outputs, attention_weights = encoder(token_ids)
        
        # 処理過程の可視化
        self._plot_encoding_progression(tokens, layer_outputs)
        
        print("\nエンコーダーの特徴:")
        print("✓ 双方向の文脈理解")
        print("✓ 並列処理可能")
        print("✓ 入力全体の俯瞰的理解")
        print("✓ 階層的な特徴抽出")
    
    def _plot_encoding_progression(self, tokens, layer_outputs):
        """エンコーディングの進行を可視化"""
        n_layers = len(layer_outputs)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for layer_idx in range(min(n_layers, 6)):
            ax = axes[layer_idx]
            
            # 各層の出力の活性化パターン
            activations = layer_outputs[layer_idx][0].mean(dim=-1).numpy()
            
            # ヒートマップとして表示
            im = ax.imshow(activations.reshape(-1, 1), cmap='RdBu_r', 
                          aspect='auto', vmin=-1, vmax=1)
            
            ax.set_yticks(range(len(tokens)))
            ax.set_yticklabels(tokens)
            ax.set_xticks([])
            ax.set_title(f'Layer {layer_idx + 1}')
            
            # カラーバー
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        plt.suptitle('エンコーダー各層での表現の変化', fontsize=16)
        plt.tight_layout()
        plt.show()

class TransformerDecoder:
    """Transformerデコーダーの実装と解説"""
    
    def __init__(self, d_model: int = 512, n_heads: int = 8, 
                 d_ff: int = 2048, n_layers: int = 6):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.n_layers = n_layers
    
    def create_decoder_layer(self) -> nn.Module:
        """単一のデコーダー層を作成"""
        class DecoderLayer(nn.Module):
            def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
                super().__init__()
                
                # Masked Self-Attention
                self.self_attn = nn.MultiheadAttention(
                    d_model, n_heads, dropout=dropout, batch_first=True
                )
                
                # Cross-Attention（エンコーダー出力への注意）
                self.cross_attn = nn.MultiheadAttention(
                    d_model, n_heads, dropout=dropout, batch_first=True
                )
                
                # Feed Forward Network
                self.feed_forward = nn.Sequential(
                    nn.Linear(d_model, d_ff),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_ff, d_model)
                )
                
                # Layer Normalization
                self.norm1 = nn.LayerNorm(d_model)
                self.norm2 = nn.LayerNorm(d_model)
                self.norm3 = nn.LayerNorm(d_model)
                
                # Dropout
                self.dropout = nn.Dropout(dropout)
                
            def forward(self, x, encoder_output, 
                       self_attn_mask=None, cross_attn_mask=None):
                # Masked Self-Attention
                self_attn_output, self_attn_weights = self.self_attn(
                    x, x, x, attn_mask=self_attn_mask
                )
                x = x + self.dropout(self_attn_output)
                x = self.norm1(x)
                
                # Cross-Attention
                cross_attn_output, cross_attn_weights = self.cross_attn(
                    x, encoder_output, encoder_output, 
                    attn_mask=cross_attn_mask
                )
                x = x + self.dropout(cross_attn_output)
                x = self.norm2(x)
                
                # Feed Forward
                ff_output = self.feed_forward(x)
                x = x + self.dropout(ff_output)
                x = self.norm3(x)
                
                return x, self_attn_weights, cross_attn_weights
        
        return DecoderLayer(self.d_model, self.n_heads, self.d_ff)
    
    def demonstrate_masked_attention(self):
        """Masked Attentionの動作を実証"""
        print("=== Masked Self-Attention ===\n")
        
        seq_len = 6
        d_model = 64
        
        # マスクの作成
        mask = self._create_causal_mask(seq_len)
        
        # 可視化
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # マスクパターン
        ax1.imshow(mask, cmap='RdBu_r', aspect='auto')
        ax1.set_title('Causal Mask (0: 可視, -∞: 不可視)')
        ax1.set_xlabel('Key位置')
        ax1.set_ylabel('Query位置')
        
        # 注意の流れ
        ax2.set_xlim(0, seq_len)
        ax2.set_ylim(0, seq_len)
        ax2.set_aspect('equal')
        
        # 各位置からの注意を矢印で表示
        for i in range(seq_len):
            for j in range(i + 1):  # i番目は0〜i番目までを見る
                ax2.arrow(j + 0.5, i + 0.5, 
                         (i - j) * 0.3, 0,
                         head_width=0.15, head_length=0.1,
                         fc='blue', ec='blue', alpha=0.5)
        
        ax2.set_title('注意の方向（過去のみ参照）')
        ax2.set_xlabel('位置')
        ax2.set_ylabel('現在の位置')
        ax2.invert_yaxis()
        
        plt.tight_layout()
        plt.show()
        
        print("特徴:")
        print("✓ 各位置は自分より前の位置のみ参照可能")
        print("✓ 自己回帰的な生成を可能にする")
        print("✓ 学習時と推論時の一貫性を保つ")
    
    def _create_causal_mask(self, size: int) -> torch.Tensor:
        """因果的マスクの作成"""
        mask = torch.triu(torch.ones(size, size) * float('-inf'), diagonal=1)
        return mask
    
    def visualize_cross_attention(self):
        """Cross-Attentionを可視化"""
        print("\n=== Cross-Attention ===\n")
        
        # ダミーデータ
        src_tokens = ["The", "cat", "is", "sleeping"]
        tgt_tokens = ["猫", "が", "寝て", "いる"]
        
        # ランダムな注意重み（実際は学習される）
        torch.manual_seed(42)
        attention_weights = torch.softmax(
            torch.randn(len(tgt_tokens), len(src_tokens)), dim=-1
        )
        
        # 可視化
        fig, ax = plt.subplots(figsize=(8, 6))
        
        im = ax.imshow(attention_weights, cmap='Blues', aspect='auto')
        
        # ラベル
        ax.set_xticks(range(len(src_tokens)))
        ax.set_xticklabels(src_tokens)
        ax.set_yticks(range(len(tgt_tokens)))
        ax.set_yticklabels(tgt_tokens)
        
        # 値を表示
        for i in range(len(tgt_tokens)):
            for j in range(len(src_tokens)):
                text = ax.text(j, i, f'{attention_weights[i, j]:.2f}',
                             ha="center", va="center", color="black")
        
        ax.set_xlabel('ソース（英語）')
        ax.set_ylabel('ターゲット（日本語）')
        ax.set_title('Cross-Attention: デコーダーがエンコーダー出力に注目')
        
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.show()
        
        print("Cross-Attentionの役割:")
        print("✓ エンコーダーの情報を選択的に利用")
        print("✓ ソースとターゲットの対応関係を学習")
        print("✓ 文脈に応じた動的なアライメント")

class EncoderDecoderArchitecture:
    """エンコーダー・デコーダー全体構造"""
    
    def __init__(self):
        self.encoder = TransformerEncoder()
        self.decoder = TransformerDecoder()
    
    def explain_information_flow(self):
        """情報の流れを説明"""
        print("=== エンコーダー・デコーダーの情報フロー ===\n")
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # エンコーダー部分
        encoder_x = 2
        encoder_y = 7
        encoder_width = 4
        encoder_height = 6
        
        # エンコーダーブロック
        encoder_rect = FancyBboxPatch(
            (encoder_x, encoder_y), encoder_width, encoder_height,
            boxstyle="round,pad=0.1", 
            facecolor='lightblue', edgecolor='darkblue', linewidth=2
        )
        ax.add_patch(encoder_rect)
        ax.text(encoder_x + encoder_width/2, encoder_y + encoder_height + 0.5,
                'エンコーダー', ha='center', fontsize=14, weight='bold')
        
        # エンコーダー層
        for i in range(3):
            layer_y = encoder_y + encoder_height - (i + 1) * 1.8
            layer_rect = Rectangle(
                (encoder_x + 0.5, layer_y), encoder_width - 1, 1.2,
                facecolor='white', edgecolor='darkblue'
            )
            ax.add_patch(layer_rect)
            ax.text(encoder_x + encoder_width/2, layer_y + 0.6,
                   f'エンコーダー層 {i+1}', ha='center', fontsize=10)
        
        # デコーダー部分
        decoder_x = 8
        decoder_y = 7
        decoder_width = 4
        decoder_height = 6
        
        # デコーダーブロック
        decoder_rect = FancyBboxPatch(
            (decoder_x, decoder_y), decoder_width, decoder_height,
            boxstyle="round,pad=0.1",
            facecolor='lightcoral', edgecolor='darkred', linewidth=2
        )
        ax.add_patch(decoder_rect)
        ax.text(decoder_x + decoder_width/2, decoder_y + decoder_height + 0.5,
                'デコーダー', ha='center', fontsize=14, weight='bold')
        
        # デコーダー層
        for i in range(3):
            layer_y = decoder_y + decoder_height - (i + 1) * 1.8
            layer_rect = Rectangle(
                (decoder_x + 0.5, layer_y), decoder_width - 1, 1.2,
                facecolor='white', edgecolor='darkred'
            )
            ax.add_patch(layer_rect)
            ax.text(decoder_x + decoder_width/2, layer_y + 0.6,
                   f'デコーダー層 {i+1}', ha='center', fontsize=10)
        
        # Cross-Attention接続
        for i in range(3):
            encoder_layer_y = encoder_y + encoder_height - (i + 1) * 1.8 + 0.6
            decoder_layer_y = decoder_y + decoder_height - (i + 1) * 1.8 + 0.6
            
            arrow = FancyArrowPatch(
                (encoder_x + encoder_width, encoder_layer_y),
                (decoder_x, decoder_layer_y),
                connectionstyle="arc3,rad=0.2",
                arrowstyle='->', mutation_scale=20,
                color='green', linewidth=2
            )
            ax.add_patch(arrow)
        
        # 入力と出力
        ax.text(encoder_x + encoder_width/2, encoder_y - 1,
                '入力シーケンス', ha='center', fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow'))
        
        ax.text(decoder_x + decoder_width/2, decoder_y - 1,
                '出力シーケンス', ha='center', fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow'))
        
        # エンコーダー出力
        ax.text(encoder_x + encoder_width/2, encoder_y + encoder_height + 1.5,
                'コンテキスト表現', ha='center', fontsize=11,
                style='italic', color='darkblue')
        
        # 凡例
        legend_elements = [
            Line2D([0], [0], color='green', linewidth=2, 
                   label='Cross-Attention'),
            mpatches.Patch(facecolor='lightblue', edgecolor='darkblue',
                          label='エンコーダー'),
            mpatches.Patch(facecolor='lightcoral', edgecolor='darkred',
                          label='デコーダー')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        ax.set_xlim(0, 14)
        ax.set_ylim(5, 15)
        ax.axis('off')
        ax.set_title('エンコーダー・デコーダー構造の情報フロー', 
                    fontsize=16, weight='bold', pad=20)
        
        plt.tight_layout()
        plt.show()
        
        print("情報の流れ:")
        print("1. エンコーダーが入力全体を処理")
        print("2. 各エンコーダー層が文脈情報を抽出・洗練")
        print("3. エンコーダーの最終出力が豊かな文脈表現に")
        print("4. デコーダーが出力を自己回帰的に生成")
        print("5. 各デコーダー層でCross-Attentionを通じてエンコーダー情報を参照")

class PracticalExample:
    """実践的な例：翻訳タスク"""
    
    def __init__(self):
        self.d_model = 256
        self.n_heads = 8
        self.n_layers = 3
        
    def demonstrate_translation_process(self):
        """翻訳プロセスの実演"""
        print("=== 翻訳タスクでの動作例 ===\n")
        
        # 簡単な語彙
        src_vocab = {
            '<pad>': 0, '<sos>': 1, '<eos>': 2,
            'the': 3, 'cat': 4, 'is': 5, 'sleeping': 6,
            'dog': 7, 'running': 8
        }
        
        tgt_vocab = {
            '<pad>': 0, '<sos>': 1, '<eos>': 2,
            '猫': 3, 'が': 4, '寝て': 5, 'いる': 6,
            '犬': 7, '走って': 8
        }
        
        # 例文
        src_sentence = "the cat is sleeping"
        src_tokens = ['<sos>'] + src_sentence.split() + ['<eos>']
        src_ids = torch.tensor([[src_vocab.get(t, 0) for t in src_tokens]])
        
        print(f"入力文: {src_sentence}")
        print(f"トークン: {src_tokens}")
        print(f"ID: {src_ids.tolist()[0]}\n")
        
        # モデルの作成（簡略版）
        class SimpleTranslator(nn.Module):
            def __init__(self, src_vocab_size, tgt_vocab_size, 
                        d_model, n_heads, n_layers):
                super().__init__()
                
                # エンコーダー
                self.src_embedding = nn.Embedding(src_vocab_size, d_model)
                self.encoder_layers = nn.ModuleList([
                    nn.TransformerEncoderLayer(
                        d_model, n_heads, d_model * 4, batch_first=True
                    ) for _ in range(n_layers)
                ])
                
                # デコーダー
                self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
                self.decoder_layers = nn.ModuleList([
                    nn.TransformerDecoderLayer(
                        d_model, n_heads, d_model * 4, batch_first=True
                    ) for _ in range(n_layers)
                ])
                
                # 出力層
                self.output_projection = nn.Linear(d_model, tgt_vocab_size)
                
            def encode(self, src):
                x = self.src_embedding(src)
                for layer in self.encoder_layers:
                    x = layer(x)
                return x
            
            def decode(self, tgt, memory):
                x = self.tgt_embedding(tgt)
                for layer in self.decoder_layers:
                    x = layer(x, memory)
                return self.output_projection(x)
        
        # モデルのインスタンス化
        model = SimpleTranslator(
            len(src_vocab), len(tgt_vocab),
            self.d_model, self.n_heads, self.n_layers
        )
        model.eval()
        
        # エンコーディング
        with torch.no_grad():
            encoder_output = model.encode(src_ids)
            print("エンコーダー出力の形状:", encoder_output.shape)
            print("(バッチサイズ, シーケンス長, 隠れ次元)\n")
        
        # デコーディング（貪欲法）
        self._demonstrate_greedy_decoding(model, encoder_output, tgt_vocab)
    
    def _demonstrate_greedy_decoding(self, model, encoder_output, tgt_vocab):
        """貪欲デコーディングの実演"""
        print("=== 貪欲デコーディング ===\n")
        
        # 逆引き辞書
        id_to_token = {v: k for k, v in tgt_vocab.items()}
        
        # 開始トークン
        decoder_input = torch.tensor([[tgt_vocab['<sos>']]])
        generated_tokens = ['<sos>']
        
        print("ステップごとの生成:")
        
        # 最大10ステップまで生成
        for step in range(10):
            with torch.no_grad():
                # デコーダーの予測
                output = model.decode(decoder_input, encoder_output)
                
                # 最後の位置の予測を取得
                next_token_logits = output[0, -1, :]
                next_token_probs = F.softmax(next_token_logits, dim=-1)
                next_token_id = torch.argmax(next_token_probs).item()
                
                # トークンに変換
                next_token = id_to_token.get(next_token_id, '<unk>')
                
                print(f"ステップ {step + 1}: {next_token} "
                      f"(確率: {next_token_probs[next_token_id]:.3f})")
                
                # 終了条件
                if next_token == '<eos>':
                    break
                
                # 次の入力に追加
                generated_tokens.append(next_token)
                decoder_input = torch.cat([
                    decoder_input,
                    torch.tensor([[next_token_id]])
                ], dim=1)
        
        print(f"\n生成された文: {' '.join(generated_tokens[1:])}")

class ComparisonWithOtherArchitectures:
    """他のアーキテクチャとの比較"""
    
    def compare_architectures(self):
        """異なるアーキテクチャの比較"""
        print("=== アーキテクチャの比較 ===\n")
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 6))
        
        # 1. Encoder-Decoder (Transformer)
        ax1 = axes[0]
        self._draw_encoder_decoder(ax1)
        ax1.set_title('Encoder-Decoder\n(翻訳・要約)', fontsize=12)
        
        # 2. Decoder-only (GPT)
        ax2 = axes[1]
        self._draw_decoder_only(ax2)
        ax2.set_title('Decoder-only\n(言語生成)', fontsize=12)
        
        # 3. Encoder-only (BERT)
        ax3 = axes[2]
        self._draw_encoder_only(ax3)
        ax3.set_title('Encoder-only\n(分類・理解)', fontsize=12)
        
        for ax in axes:
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 10)
            ax.axis('off')
        
        plt.suptitle('Transformerアーキテクチャの種類', fontsize=16, weight='bold')
        plt.tight_layout()
        plt.show()
        
        self._print_architecture_comparison()
    
    def _draw_encoder_decoder(self, ax):
        """エンコーダー・デコーダー構造を描画"""
        # エンコーダー
        encoder = FancyBboxPatch(
            (1, 4), 3, 4,
            boxstyle="round,pad=0.1",
            facecolor='lightblue', edgecolor='darkblue'
        )
        ax.add_patch(encoder)
        ax.text(2.5, 6, 'Encoder', ha='center', fontsize=10)
        
        # デコーダー
        decoder = FancyBboxPatch(
            (5, 4), 3, 4,
            boxstyle="round,pad=0.1",
            facecolor='lightcoral', edgecolor='darkred'
        )
        ax.add_patch(decoder)
        ax.text(6.5, 6, 'Decoder', ha='center', fontsize=10)
        
        # 接続
        arrow = FancyArrowPatch(
            (4, 6), (5, 6),
            arrowstyle='->', mutation_scale=20,
            color='green', linewidth=2
        )
        ax.add_patch(arrow)
        
        # 入出力
        ax.text(2.5, 2.5, '入力', ha='center', fontsize=9)
        ax.text(6.5, 2.5, '出力', ha='center', fontsize=9)
    
    def _draw_decoder_only(self, ax):
        """デコーダーのみ構造を描画"""
        # デコーダー
        decoder = FancyBboxPatch(
            (3, 4), 4, 4,
            boxstyle="round,pad=0.1",
            facecolor='lightcoral', edgecolor='darkred'
        )
        ax.add_patch(decoder)
        ax.text(5, 6, 'Decoder', ha='center', fontsize=10)
        
        # 自己回帰的な矢印
        arrow = FancyArrowPatch(
            (7, 5), (7.5, 5), 
            arrowstyle='->', mutation_scale=15,
            connectionstyle="arc3,rad=.5",
            color='darkred', linewidth=2
        )
        ax.add_patch(arrow)
        
        # 入出力
        ax.text(5, 2.5, '入力＋生成', ha='center', fontsize=9)
    
    def _draw_encoder_only(self, ax):
        """エンコーダーのみ構造を描画"""
        # エンコーダー
        encoder = FancyBboxPatch(
            (3, 4), 4, 4,
            boxstyle="round,pad=0.1",
            facecolor='lightblue', edgecolor='darkblue'
        )
        ax.add_patch(encoder)
        ax.text(5, 6, 'Encoder', ha='center', fontsize=10)
        
        # 双方向矢印
        arrow1 = FancyArrowPatch(
            (4, 5), (4.5, 5),
            arrowstyle='<->', mutation_scale=15,
            color='darkblue', linewidth=2
        )
        ax.add_patch(arrow1)
        
        arrow2 = FancyArrowPatch(
            (5.5, 5), (6, 5),
            arrowstyle='<->', mutation_scale=15,
            color='darkblue', linewidth=2
        )
        ax.add_patch(arrow2)
        
        # 入出力
        ax.text(5, 2.5, '入力→表現', ha='center', fontsize=9)
    
    def _print_architecture_comparison(self):
        """アーキテクチャの比較表を出力"""
        print("\n各アーキテクチャの特徴:\n")
        
        comparison = {
            "Encoder-Decoder": {
                "利点": "入出力が異なる形式、明示的な変換",
                "欠点": "計算コストが高い、複雑",
                "用途": "翻訳、要約、対話"
            },
            "Decoder-only": {
                "利点": "シンプル、強力な生成能力",
                "欠点": "双方向の文脈理解が困難",
                "用途": "テキスト生成、コード生成"
            },
            "Encoder-only": {
                "利点": "双方向の文脈理解、高速",
                "欠点": "生成タスクに不向き",
                "用途": "分類、固有表現認識、埋め込み"
            }
        }
        
        for arch, details in comparison.items():
            print(f"{arch}:")
            for key, value in details.items():
                print(f"  {key}: {value}")
            print()

# 実行例
def main():
    """メイン実行関数"""
    print("=" * 70)
    print("エンコーダー・デコーダー構造の詳細")
    print("=" * 70 + "\n")
    
    # エンコーダーの説明
    encoder_demo = TransformerEncoder()
    encoder_demo.visualize_encoder_process()
    
    print("\n" + "=" * 70 + "\n")
    
    # デコーダーの説明
    decoder_demo = TransformerDecoder()
    decoder_demo.demonstrate_masked_attention()
    decoder_demo.visualize_cross_attention()
    
    print("\n" + "=" * 70 + "\n")
    
    # 全体構造
    arch_demo = EncoderDecoderArchitecture()
    arch_demo.explain_information_flow()
    
    print("\n" + "=" * 70 + "\n")
    
    # 実践例
    practical = PracticalExample()
    practical.demonstrate_translation_process()
    
    print("\n" + "=" * 70 + "\n")
    
    # 比較
    comparison = ComparisonWithOtherArchitectures()
    comparison.compare_architectures()

if __name__ == "__main__":
    main()
```

## 12.2 デコーダーの特殊性

### Masked Self-Attentionの必要性

```python
class MaskedAttentionMechanics:
    """Masked Attentionの仕組みを詳細に解説"""
    
    def __init__(self):
        self.d_model = 128
        self.seq_len = 8
        
    def explain_why_masking_needed(self):
        """なぜマスキングが必要かを説明"""
        print("=== Masked Self-Attentionの必要性 ===\n")
        
        print("学習時の問題:")
        print("- Teacher Forcing: 正解の出力シーケンスを一度に入力")
        print("- しかし、未来の情報を見てはいけない")
        print("- 推論時と同じ条件で学習する必要がある\n")
        
        # 具体例で説明
        self._demonstrate_information_leakage()
        
    def _demonstrate_information_leakage(self):
        """情報漏洩の問題を実証"""
        print("例: 'I love cats' → '私は猫が好き'\n")
        
        # 正しいマスキング
        correct_example = [
            ["私", "？", "？", "？", "？"],
            ["私", "は", "？", "？", "？"],
            ["私", "は", "猫", "？", "？"],
            ["私", "は", "猫", "が", "？"],
            ["私", "は", "猫", "が", "好き"]
        ]
        
        # 間違い（マスキングなし）
        wrong_example = [
            ["私", "は", "猫", "が", "好き"],  # 全部見える！
            ["私", "は", "猫", "が", "好き"],
            ["私", "は", "猫", "が", "好き"],
            ["私", "は", "猫", "が", "好き"],
            ["私", "は", "猫", "が", "好き"]
        ]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # 正しいマスキング
        ax1.set_title('正しい: Masked Self-Attention', fontsize=12)
        for i, tokens in enumerate(correct_example):
            for j, token in enumerate(tokens):
                color = 'lightgreen' if token != "？" else 'lightgray'
                rect = Rectangle((j, 4-i), 1, 1, 
                               facecolor=color, edgecolor='black')
                ax1.add_patch(rect)
                ax1.text(j+0.5, 4-i+0.5, token, 
                        ha='center', va='center')
        
        ax1.set_xlim(0, 5)
        ax1.set_ylim(0, 5)
        ax1.set_xticks([])
        ax1.set_yticks(range(5))
        ax1.set_yticklabels([f'Step {i+1}' for i in range(5)])
        
        # 間違ったケース
        ax2.set_title('間違い: マスキングなし', fontsize=12)
        for i, tokens in enumerate(wrong_example):
            for j, token in enumerate(tokens):
                rect = Rectangle((j, 4-i), 1, 1,
                               facecolor='lightcoral', edgecolor='black')
                ax2.add_patch(rect)
                ax2.text(j+0.5, 4-i+0.5, token,
                        ha='center', va='center')
        
        ax2.set_xlim(0, 5)
        ax2.set_ylim(0, 5)
        ax2.set_xticks([])
        ax2.set_yticks(range(5))
        ax2.set_yticklabels([f'Step {i+1}' for i in range(5)])
        
        plt.tight_layout()
        plt.show()
        
        print("\n重要なポイント:")
        print("✓ 各ステップで見える情報を制限")
        print("✓ 学習時と推論時の一貫性を保つ")
        print("✓ 自己回帰的な生成を可能にする")

class CrossAttentionAnalysis:
    """Cross-Attentionの詳細分析"""
    
    def __init__(self):
        self.d_model = 256
        self.n_heads = 8
        
    def analyze_alignment_patterns(self):
        """アライメントパターンの分析"""
        print("=== Cross-Attentionのアライメント分析 ===\n")
        
        # 複数の翻訳例
        examples = [
            {
                "src": ["The", "quick", "brown", "fox"],
                "tgt": ["素早い", "茶色の", "狐"],
                "alignment": [
                    [0.1, 0.8, 0.05, 0.05],  # 素早い → quick
                    [0.05, 0.1, 0.8, 0.05],  # 茶色の → brown
                    [0.05, 0.05, 0.1, 0.8]   # 狐 → fox
                ]
            },
            {
                "src": ["I", "love", "you"],
                "tgt": ["私は", "あなたを", "愛して", "います"],
                "alignment": [
                    [0.9, 0.05, 0.05],      # 私は → I
                    [0.1, 0.1, 0.8],        # あなたを → you
                    [0.05, 0.9, 0.05],      # 愛して → love
                    [0.3, 0.4, 0.3]         # います → (auxiliary)
                ]
            }
        ]
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        for idx, example in enumerate(examples):
            ax = axes[idx]
            alignment = np.array(example["alignment"])
            
            im = ax.imshow(alignment, cmap='Blues', aspect='auto', vmin=0, vmax=1)
            
            # ラベル
            ax.set_xticks(range(len(example["src"])))
            ax.set_xticklabels(example["src"])
            ax.set_yticks(range(len(example["tgt"])))
            ax.set_yticklabels(example["tgt"])
            
            # アライメント強度を表示
            for i in range(len(example["tgt"])):
                for j in range(len(example["src"])):
                    ax.text(j, i, f'{alignment[i,j]:.2f}',
                           ha='center', va='center')
            
            ax.set_xlabel('ソース言語')
            ax.set_ylabel('ターゲット言語')
            ax.set_title(f'例 {idx+1}: Cross-Attention重み')
            
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.show()
        
        print("観察:")
        print("✓ 言語間の単語対応が学習される")
        print("✓ 語順の違いも適切に処理")
        print("✓ 文法的要素（助詞など）は分散的な注意")

class ImplementationDetails:
    """実装の詳細"""
    
    def create_complete_transformer(self):
        """完全なTransformerの実装"""
        print("=== 完全なTransformer実装 ===\n")
        
        class CompleteTransformer(nn.Module):
            def __init__(self, src_vocab_size, tgt_vocab_size,
                        d_model=512, n_heads=8, n_layers=6,
                        d_ff=2048, max_len=5000, dropout=0.1):
                super().__init__()
                
                self.d_model = d_model
                
                # ソース側埋め込み
                self.src_embedding = nn.Embedding(src_vocab_size, d_model)
                self.src_pos_encoding = self._create_positional_encoding(
                    max_len, d_model
                )
                
                # ターゲット側埋め込み
                self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
                self.tgt_pos_encoding = self._create_positional_encoding(
                    max_len, d_model
                )
                
                # Transformer本体
                self.transformer = nn.Transformer(
                    d_model=d_model,
                    nhead=n_heads,
                    num_encoder_layers=n_layers,
                    num_decoder_layers=n_layers,
                    dim_feedforward=d_ff,
                    dropout=dropout,
                    batch_first=True
                )
                
                # 出力層
                self.output_projection = nn.Linear(d_model, tgt_vocab_size)
                
                # その他
                self.dropout = nn.Dropout(dropout)
                self.scale = math.sqrt(d_model)
                
            def _create_positional_encoding(self, max_len, d_model):
                pe = torch.zeros(max_len, d_model)
                position = torch.arange(0, max_len).unsqueeze(1).float()
                
                div_term = torch.exp(
                    torch.arange(0, d_model, 2).float() *
                    -(math.log(10000.0) / d_model)
                )
                
                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term)
                
                return nn.Parameter(pe.unsqueeze(0), requires_grad=False)
            
            def create_masks(self, src, tgt):
                """マスクの作成"""
                # パディングマスク
                src_pad_mask = (src == 0)  # 0はパディングトークン
                tgt_pad_mask = (tgt == 0)
                
                # 因果的マスク（デコーダー用）
                tgt_len = tgt.size(1)
                tgt_mask = torch.triu(
                    torch.ones(tgt_len, tgt_len) * float('-inf'), 
                    diagonal=1
                ).to(tgt.device)
                
                return src_pad_mask, tgt_pad_mask, tgt_mask
            
            def forward(self, src, tgt):
                # マスクの作成
                src_pad_mask, tgt_pad_mask, tgt_mask = self.create_masks(src, tgt)
                
                # ソース側の埋め込み
                src_seq_len = src.size(1)
                src_emb = self.src_embedding(src) * self.scale
                src_emb = src_emb + self.src_pos_encoding[:, :src_seq_len]
                src_emb = self.dropout(src_emb)
                
                # ターゲット側の埋め込み
                tgt_seq_len = tgt.size(1)
                tgt_emb = self.tgt_embedding(tgt) * self.scale
                tgt_emb = tgt_emb + self.tgt_pos_encoding[:, :tgt_seq_len]
                tgt_emb = self.dropout(tgt_emb)
                
                # Transformerフォワードパス
                output = self.transformer(
                    src_emb, tgt_emb,
                    src_mask=None,
                    tgt_mask=tgt_mask,
                    src_key_padding_mask=src_pad_mask,
                    tgt_key_padding_mask=tgt_pad_mask
                )
                
                # 出力投影
                output = self.output_projection(output)
                
                return output
            
            def generate(self, src, max_len=50, temperature=1.0):
                """推論時の生成"""
                self.eval()
                device = src.device
                batch_size = src.size(0)
                
                # エンコーダーを一度だけ実行
                src_mask = (src == 0)
                src_seq_len = src.size(1)
                src_emb = self.src_embedding(src) * self.scale
                src_emb = src_emb + self.src_pos_encoding[:, :src_seq_len]
                
                memory = self.transformer.encoder(
                    src_emb,
                    src_key_padding_mask=src_mask
                )
                
                # 開始トークン
                tgt = torch.ones(batch_size, 1, dtype=torch.long).to(device)
                
                for _ in range(max_len):
                    # デコーダーの実行
                    tgt_seq_len = tgt.size(1)
                    tgt_mask = torch.triu(
                        torch.ones(tgt_seq_len, tgt_seq_len) * float('-inf'),
                        diagonal=1
                    ).to(device)
                    
                    tgt_emb = self.tgt_embedding(tgt) * self.scale
                    tgt_emb = tgt_emb + self.tgt_pos_encoding[:, :tgt_seq_len]
                    
                    output = self.transformer.decoder(
                        tgt_emb, memory,
                        tgt_mask=tgt_mask,
                        memory_key_padding_mask=src_mask
                    )
                    
                    # 最後の位置の予測
                    logits = self.output_projection(output[:, -1, :])
                    logits = logits / temperature
                    
                    # 次のトークンをサンプリング
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, 1)
                    
                    # 終了条件（EOSトークン = 2）
                    if (next_token == 2).all():
                        break
                    
                    tgt = torch.cat([tgt, next_token], dim=1)
                
                return tgt
        
        # モデルのインスタンス化と情報表示
        model = CompleteTransformer(
            src_vocab_size=10000,
            tgt_vocab_size=10000
        )
        
        # パラメータ数の計算
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() 
                             if p.requires_grad)
        
        print(f"モデルパラメータ:")
        print(f"- 総パラメータ数: {total_params:,}")
        print(f"- 学習可能パラメータ数: {trainable_params:,}")
        print(f"- モデルサイズ: {total_params * 4 / 1024**2:.1f} MB (float32)")
        
        return model

# 実用的なTipsとトリック
class PracticalTips:
    """実装時の実用的なヒント"""
    
    def share_best_practices(self):
        """ベストプラクティスの共有"""
        print("=== 実装時のベストプラクティス ===\n")
        
        tips = {
            "1. 初期化": [
                "Xavier/He初期化を使用",
                "埋め込み層は正規分布で初期化",
                "層正規化のパラメータは適切に初期化"
            ],
            
            "2. 学習の安定化": [
                "Learning rate warmupを使用",
                "Gradient clippingを適用",
                "Label smoothingで過学習を防ぐ"
            ],
            
            "3. 効率的な実装": [
                "Attention計算をバッチ化",
                "Key-Value cacheを使用（推論時）",
                "Mixed precision trainingを活用"
            ],
            
            "4. デバッグ": [
                "各層の出力分布を監視",
                "Attention重みを可視化",
                "勾配の流れを確認"
            ]
        }
        
        for category, items in tips.items():
            print(f"{category}:")
            for item in items:
                print(f"  • {item}")
            print()
        
        # コード例
        print("=== 実装例：Learning Rate Warmup ===\n")
        
        print("""
class WarmupScheduler:
    def __init__(self, optimizer, d_model, warmup_steps=4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0
        
    def step(self):
        self.step_num += 1
        lr = self._get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
    def _get_lr(self):
        # Attention is All You Need の式
        return self.d_model ** (-0.5) * min(
            self.step_num ** (-0.5),
            self.step_num * self.warmup_steps ** (-1.5)
        )
""")

def advanced_main():
    """メイン実行関数（詳細版）"""
    print("=" * 70)
    print("エンコーダー・デコーダー構造の深い理解")
    print("=" * 70 + "\n")
    
    # Masked Attentionの説明
    masked_demo = MaskedAttentionMechanics()
    masked_demo.explain_why_masking_needed()
    
    print("\n" + "=" * 70 + "\n")
    
    # Cross-Attentionの分析
    cross_demo = CrossAttentionAnalysis()
    cross_demo.analyze_alignment_patterns()
    
    print("\n" + "=" * 70 + "\n")
    
    # 完全な実装
    impl_demo = ImplementationDetails()
    model = impl_demo.create_complete_transformer()
    
    print("\n" + "=" * 70 + "\n")
    
    # 実用的なヒント
    tips = PracticalTips()
    tips.share_best_practices()
    
    print("\n" + "=" * 70)
    print("まとめ")
    print("=" * 70 + "\n")
    
    print("エンコーダー・デコーダー構造の要点:")
    print("• エンコーダー: 入力全体を理解し、豊かな表現を作成")
    print("• デコーダー: エンコーダーの情報を参照しながら出力を生成")
    print("• Cross-Attention: 両者を繋ぐ重要な機構")
    print("• Masked Attention: 自己回帰的生成を可能にする")
    print("\nこの構造により、高品質な系列変換タスクが実現されます。")

if __name__ == "__main__":
    # 基本的な説明
    main()
    
    print("\n" + "=" * 70 + "\n")
    
    # 詳細な説明
    advanced_main()