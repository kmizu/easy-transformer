# 最小限のTransformer実装

## はじめに：本質を掴む

コンパイラを初めて実装するとき、いきなりGCCやLLVMのような巨大なシステムを作ろうとはしません。まず、字句解析器と簡単な構文解析器を作り、四則演算ができる電卓を実装します。動くものを作ってから、徐々に機能を追加していくのです。

Transformerも同じアプローチで学びましょう。この章では、最小限の機能に絞った「動くTransformer」を実装します。余計な最適化や複雑な機能は後回しにして、コアとなる仕組みを理解することに集中します。

## 13.1 設計方針

### 何を作るか

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import math
from typing import Optional, Tuple, List, Dict
import time

class MinimalTransformerDesign:
    """最小限のTransformer設計"""
    
    def explain_design_principles(self):
        """設計方針の説明"""
        print("=== 最小限のTransformerの設計方針 ===\n")
        
        print("目標:")
        print("• シンプルで理解しやすい実装")
        print("• 動作の検証が容易")
        print("• コア概念の明確な表現")
        print("• 拡張可能な構造\n")
        
        print("含めるもの:")
        print("✓ Self-Attention機構")
        print("✓ 位置エンコーディング") 
        print("✓ Feed Forward Network")
        print("✓ 残差接続と層正規化")
        print("✓ 基本的なトークン化\n")
        
        print("省略するもの:")
        print("✗ Multi-Head（最初は Single-Head）")
        print("✗ エンコーダー・デコーダー構造（デコーダーのみ）")
        print("✗ 複雑な最適化")
        print("✗ 高度なトークン化（BPEなど）\n")
        
        # アーキテクチャ図
        self._visualize_minimal_architecture()
    
    def _visualize_minimal_architecture(self):
        """最小限のアーキテクチャを可視化"""
        fig, ax = plt.subplots(figsize=(10, 12))
        
        # コンポーネントの位置
        components = [
            {"name": "入力トークン", "y": 1, "color": "lightgreen"},
            {"name": "埋め込み層", "y": 2.5, "color": "lightblue"},
            {"name": "位置エンコーディング", "y": 4, "color": "lightyellow"},
            {"name": "Self-Attention", "y": 6, "color": "lightcoral"},
            {"name": "残差接続 + 層正規化", "y": 7.5, "color": "lightgray"},
            {"name": "Feed Forward", "y": 9, "color": "lightcoral"},
            {"name": "残差接続 + 層正規化", "y": 10.5, "color": "lightgray"},
            {"name": "出力層", "y": 12, "color": "lightgreen"}
        ]
        
        # コンポーネントを描画
        for comp in components:
            rect = plt.Rectangle((2, comp["y"]-0.4), 6, 0.8,
                               facecolor=comp["color"], 
                               edgecolor='black', linewidth=2)
            ax.add_patch(rect)
            ax.text(5, comp["y"], comp["name"], 
                   ha='center', va='center', fontsize=12, weight='bold')
        
        # 矢印で接続
        for i in range(len(components)-1):
            ax.arrow(5, components[i]["y"]+0.5, 0, 
                    components[i+1]["y"]-components[i]["y"]-1,
                    head_width=0.3, head_length=0.2, 
                    fc='black', ec='black')
        
        # 残差接続の矢印
        # Attention周りの残差
        ax.arrow(1.5, 5.5, 0, 2.5, head_width=0.2, head_length=0.1,
                fc='blue', ec='blue', linestyle='--', linewidth=2)
        # FFN周りの残差
        ax.arrow(1.5, 8.5, 0, 2.5, head_width=0.2, head_length=0.1,
                fc='blue', ec='blue', linestyle='--', linewidth=2)
        
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 13)
        ax.axis('off')
        ax.set_title('最小限のTransformerアーキテクチャ', 
                    fontsize=16, weight='bold', pad=20)
        
        plt.tight_layout()
        plt.show()

# 実装開始
class MinimalSelfAttention(nn.Module):
    """最小限のSelf-Attention実装"""
    
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        
        # Q, K, V の線形変換
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        
        # 出力投影
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        
        # スケーリング係数
        self.scale = 1.0 / math.sqrt(d_model)
        
    def forward(self, x: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
            mask: [batch_size, seq_len, seq_len] or None
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.shape
        
        # Q, K, V を計算
        Q = self.w_q(x)  # [batch_size, seq_len, d_model]
        K = self.w_k(x)  # [batch_size, seq_len, d_model]
        V = self.w_v(x)  # [batch_size, seq_len, d_model]
        
        # 注意スコアを計算
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        # scores: [batch_size, seq_len, seq_len]
        
        # マスクを適用（オプション）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmaxで注意重みを計算
        attn_weights = F.softmax(scores, dim=-1)
        
        # 重み付き平均を計算
        attn_output = torch.matmul(attn_weights, V)
        # attn_output: [batch_size, seq_len, d_model]
        
        # 出力投影
        output = self.w_o(attn_output)
        
        return output

class MinimalFeedForward(nn.Module):
    """最小限のFeed Forward Network"""
    
    def __init__(self, d_model: int, d_ff: Optional[int] = None):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model  # 一般的な設定
            
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.activation = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        # 第一層：拡張
        x = self.linear1(x)
        x = self.activation(x)
        
        # 第二層：圧縮
        x = self.linear2(x)
        
        return x

class MinimalTransformerBlock(nn.Module):
    """最小限のTransformerブロック"""
    
    def __init__(self, d_model: int, d_ff: Optional[int] = None):
        super().__init__()
        
        # Self-Attention
        self.attention = MinimalSelfAttention(d_model)
        
        # Feed Forward
        self.feed_forward = MinimalFeedForward(d_model, d_ff)
        
        # 層正規化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
            mask: [batch_size, seq_len, seq_len] or None
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        # Self-Attention with residual connection
        attn_output = self.attention(x, mask)
        x = self.norm1(x + attn_output)
        
        # Feed Forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        return x

class PositionalEncoding(nn.Module):
    """位置エンコーディング"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        # 位置エンコーディングを事前計算
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        # 周波数項を計算
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            -(math.log(10000.0) / d_model)
        )
        
        # sin/cos を適用
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # バッファとして登録（学習されない）
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]

class MinimalTransformer(nn.Module):
    """最小限の完全なTransformer"""
    
    def __init__(self, vocab_size: int, d_model: int = 256, 
                 n_layers: int = 4, d_ff: Optional[int] = None,
                 max_len: int = 1024):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # 埋め込み層
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # 位置エンコーディング
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # Transformerブロックのスタック
        self.layers = nn.ModuleList([
            MinimalTransformerBlock(d_model, d_ff)
            for _ in range(n_layers)
        ])
        
        # 出力層
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # 重みの初期化
        self._init_weights()
        
    def _init_weights(self):
        """重みの初期化"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """因果的マスクの作成"""
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device), 
            diagonal=1
        )
        return mask == 0  # 0がマスクされる位置
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len] - トークンID
        Returns:
            output: [batch_size, seq_len, vocab_size] - ロジット
        """
        batch_size, seq_len = x.shape
        device = x.device
        
        # 埋め込み
        x = self.embedding(x) * math.sqrt(self.d_model)  # スケーリング
        
        # 位置エンコーディング
        x = self.pos_encoding(x)
        
        # 因果的マスクを作成
        mask = self.create_causal_mask(seq_len, device)
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
        
        # 各層を通過
        for layer in self.layers:
            x = layer(x, mask)
        
        # 出力投影
        output = self.output_projection(x)
        
        return output
    
    @torch.no_grad()
    def generate(self, prompt: torch.Tensor, max_new_tokens: int = 50,
                temperature: float = 1.0, top_k: Optional[int] = None) -> torch.Tensor:
        """
        テキスト生成
        Args:
            prompt: [batch_size, seq_len] - 開始トークン
            max_new_tokens: 生成する最大トークン数
            temperature: サンプリング温度
            top_k: Top-kサンプリング（Noneの場合は使用しない）
        Returns:
            generated: [batch_size, seq_len + max_new_tokens]
        """
        self.eval()
        generated = prompt
        
        for _ in range(max_new_tokens):
            # 現在のシーケンスで予測
            logits = self(generated)
            
            # 最後のトークンの予測を取得
            next_token_logits = logits[:, -1, :] / temperature
            
            # Top-kサンプリング（オプション）
            if top_k is not None:
                values, indices = next_token_logits.topk(top_k)
                next_token_logits = torch.full_like(
                    next_token_logits, float('-inf')
                )
                next_token_logits.scatter_(1, indices, values)
            
            # 確率分布に変換してサンプリング
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # 生成されたトークンを追加
            generated = torch.cat([generated, next_token], dim=1)
            
        return generated

# 動作確認
class MinimalTransformerDemo:
    """最小限のTransformerのデモ"""
    
    def __init__(self):
        # 簡単な語彙
        self.vocab = ['<pad>', '<unk>', 'the', 'cat', 'sat', 'on', 
                     'mat', 'dog', 'runs', 'fast', '.']
        self.token_to_id = {token: i for i, token in enumerate(self.vocab)}
        self.id_to_token = {i: token for i, token in enumerate(self.vocab)}
        
        # モデルの作成
        self.model = MinimalTransformer(
            vocab_size=len(self.vocab),
            d_model=64,
            n_layers=2,
            d_ff=256,
            max_len=32
        )
        
    def tokenize(self, text: str) -> List[int]:
        """簡単なトークン化"""
        tokens = text.lower().split()
        return [self.token_to_id.get(token, 1) for token in tokens]  # 1 = <unk>
    
    def decode(self, token_ids: List[int]) -> str:
        """トークンIDを文字列に変換"""
        tokens = [self.id_to_token.get(id, '<unk>') for id in token_ids]
        return ' '.join(tokens)
    
    def demonstrate_forward_pass(self):
        """順伝播のデモ"""
        print("=== 順伝播のデモ ===\n")
        
        # サンプル入力
        text = "the cat sat on the mat"
        token_ids = self.tokenize(text)
        print(f"入力テキスト: {text}")
        print(f"トークンID: {token_ids}\n")
        
        # バッチ化してテンソルに変換
        input_tensor = torch.tensor([token_ids])
        
        # 順伝播
        with torch.no_grad():
            output = self.model(input_tensor)
        
        print(f"入力形状: {input_tensor.shape}")
        print(f"出力形状: {output.shape}")
        print(f"  (batch_size={output.shape[0]}, "
              f"seq_len={output.shape[1]}, "
              f"vocab_size={output.shape[2]})\n")
        
        # 各位置での予測を表示
        print("各位置での上位予測:")
        for pos in range(len(token_ids)):
            probs = F.softmax(output[0, pos], dim=0)
            top_probs, top_indices = probs.topk(3)
            
            print(f"\n位置 {pos} ('{self.vocab[token_ids[pos]]}' の後):")
            for prob, idx in zip(top_probs, top_indices):
                token = self.vocab[idx]
                print(f"  {token}: {prob:.3f}")
    
    def visualize_attention_patterns(self):
        """注意パターンの可視化"""
        print("\n=== 注意パターンの可視化 ===\n")
        
        # 入力を準備
        text = "the cat sat on the mat"
        token_ids = self.tokenize(text)
        input_tensor = torch.tensor([token_ids])
        
        # 最初のブロックの注意重みを取得するためのフック
        attention_weights = None
        
        def hook_fn(module, input, output):
            nonlocal attention_weights
            # MinimalSelfAttentionの出力を取得
            Q = module.w_q(input[0])
            K = module.w_k(input[0])
            scores = torch.matmul(Q, K.transpose(-2, -1)) * module.scale
            
            # マスクがある場合は適用
            if input[1] is not None:
                scores = scores.masked_fill(input[1] == 0, -1e9)
                
            attention_weights = F.softmax(scores, dim=-1)
        
        # フックを登録
        handle = self.model.layers[0].attention.register_forward_hook(hook_fn)
        
        # 順伝播
        with torch.no_grad():
            _ = self.model(input_tensor)
        
        # フックを削除
        handle.remove()
        
        # 可視化
        if attention_weights is not None:
            plt.figure(figsize=(8, 6))
            
            # 注意重みをプロット
            attn = attention_weights[0].cpu().numpy()
            im = plt.imshow(attn, cmap='Blues', aspect='auto')
            
            # ラベル
            tokens = [self.vocab[id] for id in token_ids]
            plt.xticks(range(len(tokens)), tokens, rotation=45)
            plt.yticks(range(len(tokens)), tokens)
            
            plt.xlabel('Key (参照される位置)')
            plt.ylabel('Query (現在の位置)')
            plt.title('Self-Attention重み（最初の層）')
            
            # カラーバー
            plt.colorbar(im)
            
            # 因果的マスクの境界を表示
            for i in range(len(tokens)):
                for j in range(i+1, len(tokens)):
                    plt.gca().add_patch(plt.Rectangle(
                        (j-0.5, i-0.5), 1, 1, 
                        fill=True, color='gray', alpha=0.3
                    ))
            
            plt.tight_layout()
            plt.show()
            
            print("グレーの領域：因果的マスクによりアクセス不可")
            print("青の濃さ：注意の強さ")

class DebugTools:
    """デバッグ用ツール"""
    
    def __init__(self, model: MinimalTransformer):
        self.model = model
        
    def check_gradient_flow(self):
        """勾配の流れをチェック"""
        print("=== 勾配フローのチェック ===\n")
        
        # ダミーデータ
        batch_size = 2
        seq_len = 10
        input_data = torch.randint(0, self.model.vocab_size, 
                                  (batch_size, seq_len))
        target = torch.randint(0, self.model.vocab_size, 
                              (batch_size, seq_len))
        
        # 順伝播
        output = self.model(input_data)
        loss = F.cross_entropy(
            output.reshape(-1, self.model.vocab_size),
            target.reshape(-1)
        )
        
        # 逆伝播
        loss.backward()
        
        # 各層の勾配をチェック
        print("層ごとの勾配ノルム:")
        for i, layer in enumerate(self.model.layers):
            grad_norms = {}
            
            # Attention層の勾配
            for name, param in layer.attention.named_parameters():
                if param.grad is not None:
                    grad_norms[f"attention.{name}"] = param.grad.norm().item()
            
            # FFN層の勾配
            for name, param in layer.feed_forward.named_parameters():
                if param.grad is not None:
                    grad_norms[f"ffn.{name}"] = param.grad.norm().item()
            
            print(f"\nLayer {i}:")
            for name, norm in grad_norms.items():
                print(f"  {name}: {norm:.4f}")
        
        # 埋め込み層の勾配
        if self.model.embedding.weight.grad is not None:
            print(f"\nEmbedding: {self.model.embedding.weight.grad.norm().item():.4f}")
        
        # 出力層の勾配
        if self.model.output_projection.weight.grad is not None:
            print(f"Output: {self.model.output_projection.weight.grad.norm().item():.4f}")
    
    def profile_inference(self):
        """推論時間のプロファイリング"""
        print("\n=== 推論パフォーマンス ===\n")
        
        # ウォームアップ
        dummy_input = torch.randint(0, self.model.vocab_size, (1, 50))
        for _ in range(10):
            with torch.no_grad():
                _ = self.model(dummy_input)
        
        # 異なるシーケンス長でテスト
        seq_lengths = [10, 50, 100, 200]
        times = []
        
        for seq_len in seq_lengths:
            input_data = torch.randint(0, self.model.vocab_size, (1, seq_len))
            
            # 時間計測
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()
            
            with torch.no_grad():
                for _ in range(100):
                    _ = self.model(input_data)
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.time()
            
            avg_time = (end_time - start_time) / 100 * 1000  # ミリ秒
            times.append(avg_time)
            
            print(f"シーケンス長 {seq_len}: {avg_time:.2f} ms")
        
        # 複雑度の確認
        print(f"\n時間複雑度: O(n²) - シーケンス長の2乗に比例")
        print(f"比率確認: {times[1]/times[0]:.1f}x @ 5x長, "
              f"{times[2]/times[0]:.1f}x @ 10x長")

# 実行例
def main():
    """メイン実行関数"""
    print("=" * 70)
    print("最小限のTransformer実装")
    print("=" * 70 + "\n")
    
    # 設計説明
    design = MinimalTransformerDesign()
    design.explain_design_principles()
    
    print("\n" + "=" * 70 + "\n")
    
    # デモの実行
    demo = MinimalTransformerDemo()
    demo.demonstrate_forward_pass()
    demo.visualize_attention_patterns()
    
    print("\n" + "=" * 70 + "\n")
    
    # デバッグツール
    debug = DebugTools(demo.model)
    debug.check_gradient_flow()
    debug.profile_inference()
    
    print("\n" + "=" * 70 + "\n")
    print("まとめ:")
    print("• 最小限の実装でTransformerの動作を確認")
    print("• 各コンポーネントの役割が明確")
    print("• デバッグとプロファイリングが容易")
    print("• この基盤の上に、より高度な機能を追加可能")

if __name__ == "__main__":
    main()
```

## 13.2 ステップバイステップ実装

### 1. データの準備

```python
import torch
from torch.utils.data import Dataset, DataLoader
import random
from typing import List, Tuple, Dict

class SimpleTextDataset(Dataset):
    """シンプルなテキストデータセット"""
    
    def __init__(self, texts: List[str], vocab: Dict[str, int], 
                 seq_len: int = 32):
        self.vocab = vocab
        self.seq_len = seq_len
        self.data = []
        
        # テキストをトークン化
        for text in texts:
            tokens = self._tokenize(text)
            # 固定長のシーケンスに分割
            for i in range(0, len(tokens) - seq_len, seq_len // 2):
                self.data.append(tokens[i:i + seq_len + 1])
    
    def _tokenize(self, text: str) -> List[int]:
        """簡単なトークン化"""
        words = text.lower().split()
        return [self.vocab.get(word, self.vocab['<unk>']) for word in words]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        tokens = self.data[idx]
        # 入力と目標を作成（目標は1つずつずれている）
        input_tokens = tokens[:-1]
        target_tokens = tokens[1:]
        
        # パディング
        if len(input_tokens) < self.seq_len:
            pad_len = self.seq_len - len(input_tokens)
            input_tokens += [self.vocab['<pad>']] * pad_len
            target_tokens += [self.vocab['<pad>']] * pad_len
        
        return (torch.tensor(input_tokens), torch.tensor(target_tokens))

class TrainingPipeline:
    """学習パイプライン"""
    
    def __init__(self, model: MinimalTransformer, vocab: Dict[str, int]):
        self.model = model
        self.vocab = vocab
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def train_step(self, batch: Tuple[torch.Tensor, torch.Tensor], 
                   optimizer: torch.optim.Optimizer) -> float:
        """1ステップの学習"""
        inputs, targets = batch
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        
        # 順伝播
        outputs = self.model(inputs)
        
        # 損失計算（パディングトークンは無視）
        loss = F.cross_entropy(
            outputs.reshape(-1, len(self.vocab)),
            targets.reshape(-1),
            ignore_index=self.vocab['<pad>']
        )
        
        # 逆伝播
        optimizer.zero_grad()
        loss.backward()
        
        # 勾配クリッピング
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        # パラメータ更新
        optimizer.step()
        
        return loss.item()
    
    def evaluate(self, dataloader: DataLoader) -> float:
        """評価"""
        self.model.eval()
        total_loss = 0
        n_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = F.cross_entropy(
                    outputs.reshape(-1, len(self.vocab)),
                    targets.reshape(-1),
                    ignore_index=self.vocab['<pad>']
                )
                
                total_loss += loss.item()
                n_batches += 1
        
        self.model.train()
        return total_loss / n_batches
    
    def train(self, train_dataloader: DataLoader, 
              val_dataloader: DataLoader,
              n_epochs: int = 10,
              learning_rate: float = 1e-3):
        """学習ループ"""
        print("=== 学習開始 ===\n")
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        train_losses = []
        val_losses = []
        
        for epoch in range(n_epochs):
            # 学習
            epoch_loss = 0
            n_batches = 0
            
            for batch in train_dataloader:
                loss = self.train_step(batch, optimizer)
                epoch_loss += loss
                n_batches += 1
            
            avg_train_loss = epoch_loss / n_batches
            train_losses.append(avg_train_loss)
            
            # 評価
            val_loss = self.evaluate(val_dataloader)
            val_losses.append(val_loss)
            
            print(f"Epoch {epoch+1}/{n_epochs}")
            print(f"  Train Loss: {avg_train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            
            # 生成サンプル
            if (epoch + 1) % 5 == 0:
                self._generate_sample()
                print()
        
        # 学習曲線をプロット
        self._plot_learning_curve(train_losses, val_losses)
        
    def _generate_sample(self):
        """サンプル生成"""
        print("\n  生成サンプル:")
        
        # プロンプト
        prompt = "the cat"
        prompt_tokens = [self.vocab.get(w, self.vocab['<unk>']) 
                        for w in prompt.split()]
        prompt_tensor = torch.tensor([prompt_tokens]).to(self.device)
        
        # 生成
        generated = self.model.generate(
            prompt_tensor, 
            max_new_tokens=10,
            temperature=0.8,
            top_k=5
        )
        
        # デコード
        generated_tokens = generated[0].cpu().tolist()
        id_to_token = {v: k for k, v in self.vocab.items()}
        generated_text = ' '.join([id_to_token.get(id, '<unk>') 
                                  for id in generated_tokens])
        
        print(f"  '{generated_text}'")
    
    def _plot_learning_curve(self, train_losses: List[float], 
                            val_losses: List[float]):
        """学習曲線のプロット"""
        plt.figure(figsize=(10, 6))
        
        epochs = range(1, len(train_losses) + 1)
        plt.plot(epochs, train_losses, 'b-', label='Train Loss')
        plt.plot(epochs, val_losses, 'r-', label='Val Loss')
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('学習曲線')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# 実際の学習例
def training_example():
    """学習の実例"""
    print("=== 最小限のTransformerの学習例 ===\n")
    
    # 簡単なデータセット
    texts = [
        "the cat sat on the mat",
        "the dog runs fast",
        "the cat runs on the mat",
        "the dog sat on the floor",
        "cats and dogs are animals",
        "the mat is on the floor",
        "animals run and play",
        "the fast cat runs",
        "dogs play on the mat",
        "the floor is clean"
    ] * 10  # データを増やす
    
    # 語彙の構築
    vocab = {'<pad>': 0, '<unk>': 1}
    for text in texts:
        for word in text.lower().split():
            if word not in vocab:
                vocab[word] = len(vocab)
    
    print(f"語彙サイズ: {len(vocab)}")
    print(f"語彙: {list(vocab.keys())[:10]}...\n")
    
    # データセットの作成
    dataset = SimpleTextDataset(texts, vocab, seq_len=8)
    
    # 訓練・検証に分割
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # データローダー
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # モデルの作成
    model = MinimalTransformer(
        vocab_size=len(vocab),
        d_model=64,
        n_layers=2,
        d_ff=256,
        max_len=32
    )
    
    print(f"モデルパラメータ数: {sum(p.numel() for p in model.parameters()):,}\n")
    
    # 学習
    pipeline = TrainingPipeline(model, vocab)
    pipeline.train(train_dataloader, val_dataloader, n_epochs=20)

# 詳細な分析
class DetailedAnalysis:
    """詳細な分析ツール"""
    
    def __init__(self, model: MinimalTransformer):
        self.model = model
        
    def analyze_attention_heads(self):
        """注意パターンの詳細分析"""
        print("=== 注意パターンの詳細分析 ===\n")
        
        # テスト入力
        test_sentences = [
            "the cat sat",
            "cat sat on", 
            "on the mat"
        ]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for idx, sentence in enumerate(test_sentences):
            tokens = sentence.split()
            token_ids = [2 + i for i in range(len(tokens))]  # 仮のID
            input_tensor = torch.tensor([token_ids])
            
            # 注意重みを取得
            with torch.no_grad():
                # 埋め込み
                x = self.model.embedding(input_tensor) * math.sqrt(self.model.d_model)
                x = self.model.pos_encoding(x)
                
                # 最初の層のAttentionを計算
                layer = self.model.layers[0]
                Q = layer.attention.w_q(x)
                K = layer.attention.w_k(x)
                
                scores = torch.matmul(Q, K.transpose(-2, -1)) * layer.attention.scale
                seq_len = len(tokens)
                mask = self.model.create_causal_mask(seq_len, x.device)
                scores = scores.masked_fill(~mask.unsqueeze(0), -1e9)
                
                attn_weights = F.softmax(scores, dim=-1)[0]
            
            # 可視化
            ax = axes[idx]
            im = ax.imshow(attn_weights.cpu().numpy(), cmap='Blues', 
                          vmin=0, vmax=1)
            
            ax.set_xticks(range(len(tokens)))
            ax.set_xticklabels(tokens)
            ax.set_yticks(range(len(tokens)))
            ax.set_yticklabels(tokens)
            ax.set_title(f'"{sentence}"')
            
            # 値を表示
            for i in range(len(tokens)):
                for j in range(len(tokens)):
                    if j <= i:  # 因果的マスク
                        value = attn_weights[i, j].item()
                        ax.text(j, i, f'{value:.2f}', 
                               ha='center', va='center',
                               color='white' if value > 0.5 else 'black')
        
        plt.suptitle('異なる入力での注意パターン', fontsize=14)
        plt.tight_layout()
        plt.show()
        
        print("観察:")
        print("• 直前のトークンへの強い注意")
        print("• 位置が離れるほど注意が弱まる傾向")
        print("• 文脈によって注意パターンが変化")
    
    def parameter_statistics(self):
        """パラメータの統計情報"""
        print("\n=== パラメータ統計 ===\n")
        
        stats = {}
        
        for name, param in self.model.named_parameters():
            stats[name] = {
                'shape': list(param.shape),
                'mean': param.mean().item(),
                'std': param.std().item(),
                'min': param.min().item(),
                'max': param.max().item()
            }
        
        # 表形式で出力
        print(f"{'Layer':<40} {'Shape':<20} {'Mean':<10} {'Std':<10}")
        print("-" * 80)
        
        for name, stat in stats.items():
            shape_str = str(stat['shape'])
            print(f"{name:<40} {shape_str:<20} "
                  f"{stat['mean']:<10.4f} {stat['std']:<10.4f}")

# 最適化のヒント
def optimization_tips():
    """最適化のヒント"""
    print("\n=== 最適化のヒント ===\n")
    
    tips = {
        "メモリ効率": [
            "勾配累積で実効バッチサイズを増やす",
            "Mixed Precision Training (AMP) の使用",
            "勾配チェックポイントで中間活性化を削減"
        ],
        
        "計算効率": [
            "Flash Attentionなどの最適化されたAttention実装",
            "Key-Value キャッシュの活用（推論時）",
            "量子化による高速化"
        ],
        
        "学習の安定性": [
            "Learning rate schedulerの使用",
            "Gradient clippingの適切な設定", 
            "Weight decayの調整"
        ]
    }
    
    for category, items in tips.items():
        print(f"{category}:")
        for item in items:
            print(f"  • {item}")
        print()
    
    # コード例
    print("=== コード例：勾配累積 ===")
    print("""
    accumulation_steps = 4
    optimizer.zero_grad()
    
    for i, batch in enumerate(dataloader):
        loss = compute_loss(batch)
        loss = loss / accumulation_steps  # 正規化
        loss.backward()
        
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
    """)

# メイン実行
if __name__ == "__main__":
    # 基本的なデモ
    main()
    
    print("\n" + "=" * 70 + "\n")
    
    # 学習例
    training_example()
    
    print("\n" + "=" * 70 + "\n")
    
    # 詳細分析
    model = MinimalTransformer(vocab_size=100, d_model=64, n_layers=2)
    analysis = DetailedAnalysis(model)
    analysis.analyze_attention_heads()
    analysis.parameter_statistics()
    
    # 最適化のヒント
    optimization_tips()
    
    print("\n" + "=" * 70)
    print("次のステップ:")
    print("• Multi-Head Attentionの追加")
    print("• より高度な位置エンコーディング")
    print("• エンコーダー・デコーダー構造への拡張")
    print("• 実用的なトークナイザーの統合")