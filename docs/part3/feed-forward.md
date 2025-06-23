# Feed Forward Network

## はじめに：位置ごとの深い思考

コンパイラの最適化において、各命令や式に対して個別に最適化を適用することがあります。例えば、定数畳み込みでは各式を独立に評価し、最適化可能かを判断します。この「位置ごとの独立した処理」という考え方が、TransformerのFeed Forward Network（FFN）の本質です。

Multi-Head Attentionが「単語間の関係」を学習するのに対し、FFNは「各位置での深い特徴変換」を担当します。これは、文脈情報を統合した後、その情報をより豊かな表現に変換する役割を果たします。

この章では、一見シンプルに見えるFFNが、なぜTransformerに不可欠なのか、そしてどのような計算を行っているのかを詳しく見ていきます。

## 10.1 FFNの役割と必要性

### なぜAttentionだけでは不十分なのか

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Optional, Callable
import math
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle, FancyArrowPatch
from matplotlib.patches import ConnectionPatch
import matplotlib.patches as mpatches

class FFNMotivation:
    """Feed Forward Networkの動機を説明"""
    
    def __init__(self):
        self.d_model = 512
        self.d_ff = 2048
    
    def explain_limitations_of_attention_only(self):
        """Attentionのみの限界を説明"""
        print("=== Attention のみの限界 ===\n")
        
        print("Multi-Head Attentionの特徴:")
        print("✓ 単語間の関係を学習")
        print("✓ 文脈情報の統合")
        print("✓ 並列処理可能\n")
        
        print("しかし、以下の限界がある:")
        print("✗ 線形変換のみ（非線形性がない）")
        print("✗ 位置ごとの深い特徴変換ができない")
        print("✗ 表現力が制限される\n")
        
        # 実験で実証
        self._demonstrate_linearity_limitation()
    
    def _demonstrate_linearity_limitation(self):
        """線形性の限界を実証"""
        # 簡単な分類タスク
        print("実験：XOR問題（線形分離不可能）")
        
        # データ
        X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
        y = torch.tensor([0, 1, 1, 0], dtype=torch.float32)
        
        # 線形モデル（Attentionのみに相当）
        linear_model = nn.Linear(2, 1)
        
        # 非線形モデル（Attention + FFN）
        nonlinear_model = nn.Sequential(
            nn.Linear(2, 4),
            nn.ReLU(),
            nn.Linear(4, 1)
        )
        
        # 簡易的な学習
        for model, name in [(linear_model, "線形"), (nonlinear_model, "非線形")]:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
            
            for _ in range(1000):
                optimizer.zero_grad()
                pred = model(X).squeeze()
                loss = F.binary_cross_entropy_with_logits(pred, y)
                loss.backward()
                optimizer.step()
            
            # 結果
            with torch.no_grad():
                pred = torch.sigmoid(model(X).squeeze())
                accuracy = ((pred > 0.5).float() == y).float().mean()
                print(f"\n{name}モデルの精度: {accuracy:.1%}")
        
        # 決定境界の可視化
        self._visualize_decision_boundaries(linear_model, nonlinear_model)
    
    def _visualize_decision_boundaries(self, linear_model, nonlinear_model):
        """決定境界の可視化"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # メッシュグリッド
        x_min, x_max = -0.5, 1.5
        y_min, y_max = -0.5, 1.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                            np.linspace(y_min, y_max, 100))
        
        # 各モデルの予測
        for ax, model, title in [(ax1, linear_model, "線形モデル（Attentionのみ）"),
                                 (ax2, nonlinear_model, "非線形モデル（Attention + FFN）")]:
            
            # 予測
            with torch.no_grad():
                Z = model(torch.tensor(np.c_[xx.ravel(), yy.ravel()], 
                                      dtype=torch.float32))
                Z = torch.sigmoid(Z).numpy().reshape(xx.shape)
            
            # コンター
            contour = ax.contourf(xx, yy, Z, levels=20, cmap='RdBu', alpha=0.8)
            ax.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)
            
            # データ点
            colors = ['red', 'blue']
            markers = ['o', 'x']
            for i in range(2):
                mask = (torch.tensor([0, 1, 1, 0]) == i).numpy()
                ax.scatter(torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])[mask, 0],
                          torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])[mask, 1],
                          c=colors[i], marker=markers[i], s=200, edgecolors='black')
            
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('XOR問題：非線形性の必要性', fontsize=16)
        plt.tight_layout()
        plt.show()
```

### FFNの構造と特徴

```python
class FFNStructure:
    """Feed Forward Networkの構造"""
    
    def __init__(self, d_model: int = 512, d_ff: int = 2048):
        self.d_model = d_model
        self.d_ff = d_ff
    
    def explain_structure(self):
        """FFNの構造を説明"""
        print("=== Feed Forward Network の構造 ===\n")
        
        print("基本構造:")
        print(f"1. 線形層: {self.d_model} → {self.d_ff} (拡張)")
        print(f"2. 活性化関数: ReLU または GELU")
        print(f"3. 線形層: {self.d_ff} → {self.d_model} (圧縮)")
        print(f"4. Dropout (オプション)\n")
        
        print("特徴:")
        print(f"- 拡張率: {self.d_ff / self.d_model:.0f}x (通常4x)")
        print("- 位置ごとに独立（position-wise）")
        print("- パラメータ共有（全位置で同じ重み）")
        print("- 非線形変換による表現力")
        
        # 構造の可視化
        self._visualize_ffn_structure()
    
    def _visualize_ffn_structure(self):
        """FFN構造の可視化"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 各層のボックス
        layers = [
            {"name": f"Input\n({self.d_model})", "x": 0.1, "width": 0.15, "color": "lightblue"},
            {"name": f"Linear 1\n({self.d_model}→{self.d_ff})", "x": 0.3, "width": 0.15, "color": "lightgreen"},
            {"name": "ReLU/GELU", "x": 0.5, "width": 0.1, "color": "yellow"},
            {"name": f"Linear 2\n({self.d_ff}→{self.d_model})", "x": 0.65, "width": 0.15, "color": "lightcoral"},
            {"name": f"Output\n({self.d_model})", "x": 0.85, "width": 0.15, "color": "lightblue"}
        ]
        
        for layer in layers:
            rect = FancyBboxPatch(
                (layer["x"], 0.3), layer["width"], 0.4,
                boxstyle="round,pad=0.02",
                facecolor=layer["color"],
                edgecolor='black',
                linewidth=2
            )
            ax.add_patch(rect)
            ax.text(layer["x"] + layer["width"]/2, 0.5, layer["name"],
                   ha='center', va='center', fontsize=10, fontweight='bold')
        
        # 矢印
        arrow_pairs = [(0, 1), (1, 2), (2, 3), (3, 4)]
        for i, j in arrow_pairs:
            start_x = layers[i]["x"] + layers[i]["width"]
            end_x = layers[j]["x"]
            ax.arrow(start_x, 0.5, end_x - start_x - 0.01, 0,
                    head_width=0.05, head_length=0.01,
                    fc='black', ec='black')
        
        # 次元の変化を表示
        ax.text(0.375, 0.8, f"拡張\n{self.d_ff/self.d_model:.0f}x",
               ha='center', fontsize=10, color='green', fontweight='bold')
        ax.text(0.725, 0.8, f"圧縮\n1/{self.d_ff/self.d_model:.0f}x",
               ha='center', fontsize=10, color='red', fontweight='bold')
        
        ax.set_xlim(0, 1.1)
        ax.set_ylim(0, 1)
        ax.set_title('Feed Forward Network の構造', fontsize=16)
        ax.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def compare_parameter_counts(self):
        """パラメータ数の比較"""
        print("\n=== パラメータ数の分析 ===")
        
        # FFNのパラメータ数
        ffn_params = 2 * self.d_model * self.d_ff
        
        # Multi-Head Attentionのパラメータ数（8ヘッドと仮定）
        n_heads = 8
        mha_params = 4 * self.d_model * self.d_model  # Q, K, V, O
        
        print(f"\nFFN:")
        print(f"  Linear1: {self.d_model} × {self.d_ff} = {self.d_model * self.d_ff:,}")
        print(f"  Linear2: {self.d_ff} × {self.d_model} = {self.d_ff * self.d_model:,}")
        print(f"  合計: {ffn_params:,}")
        
        print(f"\nMulti-Head Attention (8 heads):")
        print(f"  Q, K, V, O: 4 × {self.d_model} × {self.d_model} = {mha_params:,}")
        
        print(f"\n比率: FFN / MHA = {ffn_params / mha_params:.1f}")
        
        # 可視化
        self._visualize_parameter_distribution()
    
    def _visualize_parameter_distribution(self):
        """パラメータ分布の可視化"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # パラメータ数の比較
        ffn_params = 2 * self.d_model * self.d_ff
        mha_params = 4 * self.d_model * self.d_model
        
        # 円グラフ
        sizes = [mha_params, ffn_params]
        labels = ['Multi-Head\nAttention', 'Feed Forward\nNetwork']
        colors = ['lightblue', 'lightcoral']
        
        ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
               startangle=90, textprops={'fontsize': 12})
        ax1.set_title('Transformerブロック内のパラメータ分布')
        
        # 棒グラフ（詳細）
        components = ['Q', 'K', 'V', 'O', 'FFN Linear1', 'FFN Linear2']
        params = [
            self.d_model * self.d_model,
            self.d_model * self.d_model,
            self.d_model * self.d_model,
            self.d_model * self.d_model,
            self.d_model * self.d_ff,
            self.d_ff * self.d_model
        ]
        
        bars = ax2.bar(components, params, color=['lightblue']*4 + ['lightcoral']*2)
        ax2.set_ylabel('パラメータ数')
        ax2.set_title('各コンポーネントのパラメータ数')
        ax2.tick_params(axis='x', rotation=45)
        
        # 値を表示
        for bar, param in zip(bars, params):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{param/1000:.0f}K',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
```

## 10.2 位置ごとの計算（Position-wise）

### Position-wise の意味

```python
class PositionWiseComputation:
    """位置ごとの計算の理解"""
    
    def explain_position_wise(self):
        """Position-wiseの概念を説明"""
        print("=== Position-wise Feed Forward ===\n")
        
        print("「位置ごと」の意味:")
        print("- 各位置（トークン）に対して独立に適用")
        print("- 位置間の相互作用はない")
        print("- すべての位置で同じ重みを共有")
        print("- 1×1 Convolutionと等価\n")
        
        # 比較を可視化
        self._visualize_position_wise_vs_fully_connected()
    
    def _visualize_position_wise_vs_fully_connected(self):
        """Position-wise vs Fully Connectedの比較"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        seq_len = 4
        d_model = 3
        
        # Position-wise（実際のFFN）
        ax1.set_title('Position-wise FFN\n（各位置で独立）', fontsize=14)
        
        # 各位置を独立に処理
        for pos in range(seq_len):
            # 入力
            for dim in range(d_model):
                circle = Circle((0, pos * 1.5 + dim * 0.4), 0.15,
                              color='lightblue', ec='black')
                ax1.add_patch(circle)
                ax1.text(0, pos * 1.5 + dim * 0.4, f'x{pos},{dim}',
                        ha='center', va='center', fontsize=8)
            
            # FFN（各位置で同じ）
            rect = Rectangle((1, pos * 1.5 - 0.2), 1, 1,
                           facecolor='lightgreen', edgecolor='black')
            ax1.add_patch(rect)
            ax1.text(1.5, pos * 1.5 + 0.3, 'FFN',
                    ha='center', va='center', fontsize=10)
            
            # 出力
            for dim in range(d_model):
                circle = Circle((3, pos * 1.5 + dim * 0.4), 0.15,
                              color='lightcoral', ec='black')
                ax1.add_patch(circle)
                ax1.text(3, pos * 1.5 + dim * 0.4, f'y{pos},{dim}',
                        ha='center', va='center', fontsize=8)
            
            # 矢印
            ax1.arrow(0.2, pos * 1.5 + 0.3, 0.7, 0,
                     head_width=0.1, head_length=0.05,
                     fc='gray', ec='gray')
            ax1.arrow(2.1, pos * 1.5 + 0.3, 0.7, 0,
                     head_width=0.1, head_length=0.05,
                     fc='gray', ec='gray')
        
        ax1.set_xlim(-0.5, 3.5)
        ax1.set_ylim(-0.5, seq_len * 1.5)
        ax1.axis('off')
        
        # Fully Connected（仮想的な代替案）
        ax2.set_title('Fully Connected\n（全位置が相互作用）', fontsize=14)
        
        # すべての入力
        all_inputs = []
        for pos in range(seq_len):
            for dim in range(d_model):
                y = pos * 0.8 + dim * 0.25
                circle = Circle((0, y), 0.1,
                              color='lightblue', ec='black')
                ax2.add_patch(circle)
                all_inputs.append((0, y))
        
        # 中央の処理
        rect = Rectangle((1.5, 0), 1, seq_len * 0.8 + 0.2,
                       facecolor='yellow', edgecolor='black')
        ax2.add_patch(rect)
        ax2.text(2, seq_len * 0.4, 'FC',
                ha='center', va='center', fontsize=12)
        
        # すべての出力
        all_outputs = []
        for pos in range(seq_len):
            for dim in range(d_model):
                y = pos * 0.8 + dim * 0.25
                circle = Circle((3.5, y), 0.1,
                              color='lightcoral', ec='black')
                ax2.add_patch(circle)
                all_outputs.append((3.5, y))
        
        # 全結合を表す線（一部のみ）
        for i in range(0, len(all_inputs), 3):
            for j in range(0, len(all_outputs), 3):
                ax2.plot([all_inputs[i][0] + 0.1, 1.5],
                        [all_inputs[i][1], seq_len * 0.4],
                        'gray', alpha=0.3, linewidth=0.5)
                ax2.plot([2.5, all_outputs[j][0] - 0.1],
                        [seq_len * 0.4, all_outputs[j][1]],
                        'gray', alpha=0.3, linewidth=0.5)
        
        ax2.set_xlim(-0.5, 4)
        ax2.set_ylim(-0.5, seq_len * 0.8 + 0.5)
        ax2.axis('off')
        
        plt.suptitle('FFNの「位置ごと」処理 vs 全結合', fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def demonstrate_weight_sharing(self):
        """重み共有の実演"""
        print("\n=== 重み共有の実演 ===")
        
        # サンプルデータ
        batch_size = 2
        seq_len = 3
        d_model = 4
        d_ff = 8
        
        # FFN層
        ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        
        # 入力
        x = torch.randn(batch_size, seq_len, d_model)
        print(f"入力形状: {x.shape}")
        
        # Position-wise適用（PyTorchは自動的に処理）
        output = ffn(x)
        print(f"出力形状: {output.shape}")
        
        # 手動で各位置に適用して同じ結果を確認
        manual_output = torch.zeros_like(x)
        for pos in range(seq_len):
            manual_output[:, pos, :] = ffn(x[:, pos, :])
        
        # 結果が同じことを確認
        difference = (output - manual_output).abs().max().item()
        print(f"\n自動処理と手動処理の差: {difference:.6f}")
        print("→ 同じ重みが全位置で共有されている")
```

### 1×1 Convolution との等価性

```python
class ConvolutionEquivalence:
    """1×1 Convolutionとの等価性"""
    
    def demonstrate_equivalence(self):
        """FFNと1×1 Convの等価性を実証"""
        print("=== FFN と 1×1 Convolution の等価性 ===\n")
        
        # パラメータ
        batch_size = 2
        seq_len = 10
        d_model = 256
        d_ff = 1024
        
        # データ（同じ初期値）
        torch.manual_seed(42)
        x = torch.randn(batch_size, seq_len, d_model)
        
        # 方法1: FFN（Linear層）
        ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        
        # 方法2: 1×1 Convolution
        # Conv1dは(batch, channels, length)の順序を期待
        conv = nn.Sequential(
            nn.Conv1d(d_model, d_ff, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(d_ff, d_model, kernel_size=1)
        )
        
        # 重みをコピーして同じにする
        with torch.no_grad():
            # Linear: (out_features, in_features)
            # Conv1d: (out_channels, in_channels, kernel_size)
            conv[0].weight.data = ffn[0].weight.data.unsqueeze(-1)
            conv[0].bias.data = ffn[0].bias.data
            conv[2].weight.data = ffn[2].weight.data.unsqueeze(-1)
            conv[2].bias.data = ffn[2].bias.data
        
        # 計算
        ffn_output = ffn(x)
        
        # Conv1dのための転置
        x_conv = x.transpose(1, 2)  # (batch, d_model, seq_len)
        conv_output = conv(x_conv)
        conv_output = conv_output.transpose(1, 2)  # (batch, seq_len, d_model)
        
        # 結果の比較
        difference = (ffn_output - conv_output).abs().max().item()
        print(f"FFN出力形状: {ffn_output.shape}")
        print(f"Conv出力形状: {conv_output.shape}")
        print(f"最大差: {difference:.6f}")
        print("\n→ FFNと1×1 Convolutionは数学的に等価！")
        
        # 計算効率の比較
        self._compare_computation_efficiency()
    
    def _compare_computation_efficiency(self):
        """計算効率の比較"""
        import time
        
        print("\n=== 計算効率の比較 ===")
        
        # 大きめのデータ
        batch_size = 32
        seq_len = 512
        d_model = 768
        d_ff = 3072
        
        x = torch.randn(batch_size, seq_len, d_model)
        
        # FFN
        ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        
        # 1×1 Conv
        conv = nn.Sequential(
            nn.Conv1d(d_model, d_ff, 1),
            nn.ReLU(),
            nn.Conv1d(d_ff, d_model, 1)
        )
        
        # FFNの時間測定
        start = time.time()
        for _ in range(10):
            _ = ffn(x)
        ffn_time = time.time() - start
        
        # Convの時間測定
        x_conv = x.transpose(1, 2)
        start = time.time()
        for _ in range(10):
            _ = conv(x_conv).transpose(1, 2)
        conv_time = time.time() - start
        
        print(f"FFN時間: {ffn_time:.3f}秒")
        print(f"Conv時間: {conv_time:.3f}秒")
        print(f"比率: {conv_time/ffn_time:.2f}x")
```

## 10.3 活性化関数の選択

### ReLU vs GELU

```python
class ActivationFunctions:
    """活性化関数の比較と分析"""
    
    def compare_activation_functions(self):
        """主要な活性化関数の比較"""
        print("=== 活性化関数の比較 ===\n")
        
        # 活性化関数の定義
        x = torch.linspace(-3, 3, 1000)
        
        activations = {
            'ReLU': F.relu(x),
            'GELU': F.gelu(x),
            'SiLU/Swish': F.silu(x),
            'Mish': x * torch.tanh(F.softplus(x))
        }
        
        # 導関数（近似）
        x.requires_grad_(True)
        derivatives = {}
        
        for name, act_func in [
            ('ReLU', lambda x: F.relu(x)),
            ('GELU', lambda x: F.gelu(x)),
            ('SiLU/Swish', lambda x: F.silu(x)),
            ('Mish', lambda x: x * torch.tanh(F.softplus(x)))
        ]:
            y = act_func(x.clone())
            y.sum().backward()
            derivatives[name] = x.grad.clone()
            x.grad.zero_()
        
        # 可視化
        self._visualize_activations(x.detach(), activations, derivatives)
    
    def _visualize_activations(self, x, activations, derivatives):
        """活性化関数の可視化"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # 活性化関数
        for name, y in activations.items():
            ax1.plot(x, y, label=name, linewidth=2)
        
        ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax1.axvline(x=0, color='k', linestyle='--', alpha=0.3)
        ax1.set_xlabel('x')
        ax1.set_ylabel('f(x)')
        ax1.set_title('活性化関数')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(-3, 3)
        
        # 導関数
        for name, dy in derivatives.items():
            ax2.plot(x, dy, label=name, linewidth=2)
        
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax2.axvline(x=0, color='k', linestyle='--', alpha=0.3)
        ax2.set_xlabel('x')
        ax2.set_ylabel("f'(x)")
        ax2.set_title('導関数')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(-3, 3)
        
        plt.suptitle('Transformerで使用される活性化関数', fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def explain_gelu_advantage(self):
        """GELUの利点を説明"""
        print("\n=== GELU (Gaussian Error Linear Unit) の利点 ===\n")
        
        print("ReLUの特徴:")
        print("✓ シンプルで高速")
        print("✓ 勾配消失を防ぐ")
        print("✗ x<0で勾配が0（Dead ReLU問題）")
        print("✗ 原点で微分不可能\n")
        
        print("GELUの特徴:")
        print("✓ 滑らかで微分可能")
        print("✓ 確率的な解釈が可能")
        print("✓ 負の値も一部通す")
        print("✓ 実験的に優れた性能")
        print("✗ 計算コストがやや高い\n")
        
        # 実験：勾配の流れ
        self._compare_gradient_flow()
    
    def _compare_gradient_flow(self):
        """勾配の流れを比較"""
        print("実験：勾配の流れの比較")
        
        # 深いネットワークを構築
        depth = 20
        d_model = 128
        
        # ReLUネットワーク
        relu_net = nn.Sequential(*[
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU()
            ) for _ in range(depth)
        ])
        
        # GELUネットワーク
        gelu_net = nn.Sequential(*[
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU()
            ) for _ in range(depth)
        ])
        
        # 同じ初期化
        for relu_layer, gelu_layer in zip(relu_net, gelu_net):
            gelu_layer[0].weight.data = relu_layer[0].weight.data.clone()
            gelu_layer[0].bias.data = relu_layer[0].bias.data.clone()
        
        # 入力
        x = torch.randn(32, d_model, requires_grad=True)
        
        # 順伝播と逆伝播
        for net, name in [(relu_net, "ReLU"), (gelu_net, "GELU")]:
            x_copy = x.clone()
            output = net(x_copy)
            loss = output.mean()
            loss.backward()
            
            # 勾配の統計
            grad_norms = []
            for i, layer in enumerate(net):
                if isinstance(layer, nn.Sequential):
                    grad_norm = layer[0].weight.grad.norm().item()
                    grad_norms.append(grad_norm)
            
            print(f"\n{name}ネットワーク:")
            print(f"  最初の層の勾配ノルム: {grad_norms[0]:.4f}")
            print(f"  最後の層の勾配ノルム: {grad_norms[-1]:.4f}")
            print(f"  勾配の減衰率: {grad_norms[-1]/grad_norms[0]:.6f}")
```

### 活性化関数の実装

```python
class CustomActivations:
    """カスタム活性化関数の実装"""
    
    def implement_activations(self):
        """各種活性化関数の実装"""
        print("=== 活性化関数の実装 ===\n")
        
        class CustomGELU(nn.Module):
            """GELUのカスタム実装"""
            def forward(self, x):
                # GELU(x) = x * Φ(x)
                # Φ(x) は標準正規分布の累積分布関数
                return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
        
        class CustomSwish(nn.Module):
            """Swish/SiLUのカスタム実装"""
            def __init__(self, beta=1.0):
                super().__init__()
                self.beta = beta
            
            def forward(self, x):
                return x * torch.sigmoid(self.beta * x)
        
        class CustomMish(nn.Module):
            """Mishのカスタム実装"""
            def forward(self, x):
                return x * torch.tanh(F.softplus(x))
        
        # テスト
        x = torch.randn(10, 20)
        
        # カスタム実装
        custom_gelu = CustomGELU()
        custom_swish = CustomSwish()
        custom_mish = CustomMish()
        
        # PyTorch実装との比較
        print("カスタム実装とPyTorch実装の差:")
        
        # GELU
        diff_gelu = (custom_gelu(x) - F.gelu(x)).abs().max().item()
        print(f"GELU: {diff_gelu:.6f}")
        
        # Swish/SiLU
        diff_swish = (custom_swish(x) - F.silu(x)).abs().max().item()
        print(f"Swish: {diff_swish:.6f}")
        
        # 近似版GELU（高速化）
        self._implement_approximate_gelu()
    
    def _implement_approximate_gelu(self):
        """近似版GELUの実装"""
        print("\n=== 近似版GELU ===")
        
        class ApproximateGELU(nn.Module):
            """高速な近似GELU"""
            def forward(self, x):
                # tanh近似
                # GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
                return 0.5 * x * (1 + torch.tanh(
                    math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))
                ))
        
        # 比較
        x = torch.linspace(-3, 3, 1000)
        
        exact_gelu = F.gelu(x)
        approx_gelu = ApproximateGELU()(x)
        
        # 誤差の可視化
        plt.figure(figsize=(10, 6))
        
        plt.subplot(2, 1, 1)
        plt.plot(x, exact_gelu, label='Exact GELU', linewidth=2)
        plt.plot(x, approx_gelu, label='Approximate GELU', linestyle='--', linewidth=2)
        plt.xlabel('x')
        plt.ylabel('GELU(x)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 1, 2)
        plt.plot(x, (approx_gelu - exact_gelu).abs(), 'r-', linewidth=2)
        plt.xlabel('x')
        plt.ylabel('|Approximate - Exact|')
        plt.title('近似誤差')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        max_error = (approx_gelu - exact_gelu).abs().max().item()
        print(f"\n最大近似誤差: {max_error:.6f}")
```

## 10.4 FFNの最適化手法

### GLU変種の実装

```python
class GLUVariants:
    """Gated Linear Unit (GLU) の変種"""
    
    def explain_glu_family(self):
        """GLUファミリーの説明"""
        print("=== GLU (Gated Linear Unit) ファミリー ===\n")
        
        print("基本的なGLU:")
        print("GLU(x) = (xW + b) ⊗ σ(xV + c)")
        print("  ⊗: 要素ごとの積")
        print("  σ: 活性化関数（sigmoid）\n")
        
        print("変種:")
        variants = {
            "GLU": "sigmoid gate",
            "ReGLU": "ReLU gate",
            "GEGLU": "GELU gate",
            "SwiGLU": "Swish gate",
            "Linear": "no gate (standard FFN)"
        }
        
        for name, description in variants.items():
            print(f"  {name}: {description}")
    
    def implement_glu_variants(self):
        """GLU変種の実装"""
        
        class GLU(nn.Module):
            """基本的なGLU"""
            def __init__(self, d_model, d_ff):
                super().__init__()
                self.linear = nn.Linear(d_model, d_ff * 2)
            
            def forward(self, x):
                x = self.linear(x)
                x, gate = x.chunk(2, dim=-1)
                return x * torch.sigmoid(gate)
        
        class SwiGLU(nn.Module):
            """SwiGLU（LLaMAで使用）"""
            def __init__(self, d_model, d_ff):
                super().__init__()
                self.w1 = nn.Linear(d_model, d_ff)
                self.w2 = nn.Linear(d_model, d_ff)
                self.w3 = nn.Linear(d_ff, d_model)
            
            def forward(self, x):
                return self.w3(F.silu(self.w1(x)) * self.w2(x))
        
        class GEGLU(nn.Module):
            """GEGLU"""
            def __init__(self, d_model, d_ff):
                super().__init__()
                self.linear = nn.Linear(d_model, d_ff * 2)
                self.output = nn.Linear(d_ff, d_model)
            
            def forward(self, x):
                x = self.linear(x)
                x, gate = x.chunk(2, dim=-1)
                x = x * F.gelu(gate)
                return self.output(x)
        
        # 性能比較
        self._compare_glu_performance()
    
    def _compare_glu_performance(self):
        """GLU変種の性能比較"""
        print("\n=== GLU変種の比較 ===")
        
        d_model = 512
        d_ff = 2048
        batch_size = 32
        seq_len = 128
        
        # 標準FFN
        standard_ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        
        # GLU変種
        class SwiGLU(nn.Module):
            def __init__(self, d_model, d_ff):
                super().__init__()
                # d_ffを調整してパラメータ数を合わせる
                self.d_ff_adjusted = int(d_ff * 2 / 3)
                self.w1 = nn.Linear(d_model, self.d_ff_adjusted)
                self.w2 = nn.Linear(d_model, self.d_ff_adjusted)
                self.w3 = nn.Linear(self.d_ff_adjusted, d_model)
            
            def forward(self, x):
                return self.w3(F.silu(self.w1(x)) * self.w2(x))
        
        swiglu = SwiGLU(d_model, d_ff)
        
        # パラメータ数の比較
        standard_params = sum(p.numel() for p in standard_ffn.parameters())
        swiglu_params = sum(p.numel() for p in swiglu.parameters())
        
        print(f"標準FFN パラメータ数: {standard_params:,}")
        print(f"SwiGLU パラメータ数: {swiglu_params:,}")
        print(f"比率: {swiglu_params / standard_params:.2f}")
        
        # 実際の計算で表現力を比較（簡易テスト）
        x = torch.randn(batch_size, seq_len, d_model)
        
        with torch.no_grad():
            standard_out = standard_ffn(x)
            swiglu_out = swiglu(x)
            
            # 出力の統計
            print(f"\n出力の統計:")
            print(f"標準FFN - 平均: {standard_out.mean():.4f}, 標準偏差: {standard_out.std():.4f}")
            print(f"SwiGLU - 平均: {swiglu_out.mean():.4f}, 標準偏差: {swiglu_out.std():.4f}")
```

### MoE（Mixture of Experts）FFN

```python
class MixtureOfExpertsFFN:
    """Mixture of Experts FFN"""
    
    def explain_moe_concept(self):
        """MoEの概念を説明"""
        print("=== Mixture of Experts (MoE) FFN ===\n")
        
        print("基本アイデア:")
        print("- 複数の「専門家」FFNを用意")
        print("- 各トークンに対して適切な専門家を選択")
        print("- スパース性により計算効率を維持\n")
        
        print("利点:")
        print("✓ パラメータ数を増やしても計算量は一定")
        print("✓ 各専門家が特定のパターンに特化")
        print("✓ モデル容量の効率的な拡張\n")
        
        print("課題:")
        print("✗ 負荷分散の問題")
        print("✗ 学習の不安定性")
        print("✗ 実装の複雑さ")
        
        self._visualize_moe_concept()
    
    def _visualize_moe_concept(self):
        """MoEの概念を可視化"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 入力トークン
        tokens = ["The", "cat", "sat", "on", "mat"]
        for i, token in enumerate(tokens):
            circle = Circle((0.1, 0.8 - i * 0.15), 0.05,
                          color='lightblue', ec='black')
            ax.add_patch(circle)
            ax.text(0.1, 0.8 - i * 0.15, token,
                   ha='center', va='center', fontsize=10)
        
        # ゲーティングネットワーク
        gate_rect = FancyBboxPatch((0.25, 0.3), 0.15, 0.4,
                                  boxstyle="round,pad=0.02",
                                  facecolor='yellow',
                                  edgecolor='black')
        ax.add_patch(gate_rect)
        ax.text(0.325, 0.5, 'Gating\nNetwork',
               ha='center', va='center', fontsize=10)
        
        # 専門家FFN
        experts = ["Expert 1\n(名詞)", "Expert 2\n(動詞)", 
                  "Expert 3\n(前置詞)", "Expert 4\n(一般)"]
        colors = ['lightgreen', 'lightcoral', 'lightyellow', 'lightgray']
        
        for i, (expert, color) in enumerate(zip(experts, colors)):
            expert_rect = FancyBboxPatch((0.5, 0.7 - i * 0.15), 0.2, 0.1,
                                       boxstyle="round,pad=0.02",
                                       facecolor=color,
                                       edgecolor='black')
            ax.add_patch(expert_rect)
            ax.text(0.6, 0.75 - i * 0.15, expert,
                   ha='center', va='center', fontsize=9)
        
        # 選択された経路を表示
        token_to_expert = {
            "The": 3, "cat": 0, "sat": 1, "on": 2, "mat": 0
        }
        
        for i, (token, expert_idx) in enumerate(token_to_expert.items()):
            # ゲートへの矢印
            ax.arrow(0.15, 0.8 - i * 0.15, 0.08, 0,
                    head_width=0.02, head_length=0.01,
                    fc='gray', ec='gray', alpha=0.5)
            
            # ゲートから専門家への矢印
            start_y = 0.5
            end_y = 0.75 - expert_idx * 0.15
            ax.annotate('', xy=(0.5, end_y), xytext=(0.4, start_y),
                       arrowprops=dict(arrowstyle='->',
                                     color='red' if i == 1 else 'blue',
                                     linewidth=2,
                                     alpha=0.7))
            
            # ゲート値を表示
            ax.text(0.45, (start_y + end_y) / 2,
                   f'{0.8:.1f}' if expert_idx == token_to_expert[tokens[i]] else '',
                   fontsize=8, color='red' if i == 1 else 'blue')
        
        # 出力
        output_rect = FancyBboxPatch((0.8, 0.4), 0.1, 0.2,
                                   boxstyle="round,pad=0.02",
                                   facecolor='lightsteelblue',
                                   edgecolor='black')
        ax.add_patch(output_rect)
        ax.text(0.85, 0.5, 'Output',
               ha='center', va='center', fontsize=10)
        
        # 専門家から出力への矢印
        for i in range(4):
            ax.arrow(0.7, 0.75 - i * 0.15, 0.08, 0,
                    head_width=0.02, head_length=0.01,
                    fc='gray', ec='gray', alpha=0.3)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0.2, 0.9)
        ax.set_title('Mixture of Experts FFN', fontsize=16)
        ax.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def implement_simple_moe(self):
        """シンプルなMoEの実装"""
        print("\n=== シンプルなMoE実装 ===")
        
        class SimpleMoE(nn.Module):
            def __init__(self, d_model, d_ff, num_experts=4, top_k=2):
                super().__init__()
                self.num_experts = num_experts
                self.top_k = top_k
                
                # 専門家FFN
                self.experts = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(d_model, d_ff),
                        nn.GELU(),
                        nn.Linear(d_ff, d_model)
                    ) for _ in range(num_experts)
                ])
                
                # ゲーティングネットワーク
                self.gate = nn.Linear(d_model, num_experts)
            
            def forward(self, x):
                batch_size, seq_len, d_model = x.shape
                
                # ゲート値の計算
                gate_scores = self.gate(x)  # [batch, seq, num_experts]
                
                # Top-k専門家の選択
                topk_scores, topk_indices = torch.topk(
                    gate_scores, self.top_k, dim=-1
                )
                
                # Softmaxで正規化
                topk_scores = F.softmax(topk_scores, dim=-1)
                
                # 出力の初期化
                output = torch.zeros_like(x)
                
                # 各専門家の処理
                for i in range(self.top_k):
                    # 選択された専門家のインデックス
                    expert_idx = topk_indices[:, :, i]  # [batch, seq]
                    
                    # 各専門家に対して処理
                    for e in range(self.num_experts):
                        # この専門家が選ばれたトークン
                        mask = (expert_idx == e)
                        
                        if mask.any():
                            # 専門家の出力
                            expert_out = self.experts[e](x)
                            
                            # ゲート値で重み付け
                            scores = topk_scores[:, :, i].unsqueeze(-1)
                            output += torch.where(
                                mask.unsqueeze(-1),
                                expert_out * scores,
                                torch.zeros_like(expert_out)
                            )
                
                return output, gate_scores
        
        # テスト
        moe = SimpleMoE(d_model=256, d_ff=1024, num_experts=4, top_k=2)
        x = torch.randn(2, 10, 256)
        output, gate_scores = moe(x)
        
        print(f"入力形状: {x.shape}")
        print(f"出力形状: {output.shape}")
        print(f"ゲートスコア形状: {gate_scores.shape}")
        
        # 専門家の選択パターンを分析
        topk_scores, topk_indices = torch.topk(gate_scores, 2, dim=-1)
        print(f"\n最初のバッチ、最初の5トークンの専門家選択:")
        print(topk_indices[0, :5])
```

## 10.5 FFNの実装と統合

### 完全なFFN実装

```python
class CompleteFeedForward:
    """完全なFeed Forward実装"""
    
    def create_feed_forward(self, 
                           d_model: int = 512,
                           d_ff: int = 2048,
                           activation: str = 'gelu',
                           dropout: float = 0.1,
                           use_glu: bool = False) -> nn.Module:
        """柔軟なFFNモジュールを作成"""
        
        class FeedForward(nn.Module):
            def __init__(self):
                super().__init__()
                
                if use_glu:
                    # GLU変種（SwiGLU）
                    self.w1 = nn.Linear(d_model, d_ff)
                    self.w2 = nn.Linear(d_model, d_ff)
                    self.w3 = nn.Linear(d_ff, d_model)
                    self.activation = self._get_activation(activation)
                else:
                    # 標準FFN
                    self.linear1 = nn.Linear(d_model, d_ff)
                    self.activation = self._get_activation(activation)
                    self.dropout1 = nn.Dropout(dropout)
                    self.linear2 = nn.Linear(d_ff, d_model)
                    self.dropout2 = nn.Dropout(dropout)
                
                self.use_glu = use_glu
                self._init_weights()
            
            def _get_activation(self, name):
                activations = {
                    'relu': nn.ReLU(),
                    'gelu': nn.GELU(),
                    'silu': nn.SiLU(),
                    'mish': nn.Mish()
                }
                return activations.get(name, nn.GELU())
            
            def _init_weights(self):
                # Heの初期化（ReLU系）またはXavierの初期化
                for m in self.modules():
                    if isinstance(m, nn.Linear):
                        if isinstance(self.activation, (nn.ReLU, nn.GELU)):
                            nn.init.kaiming_normal_(m.weight, mode='fan_in')
                        else:
                            nn.init.xavier_uniform_(m.weight)
                        
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
            
            def forward(self, x):
                if self.use_glu:
                    # SwiGLU: x = W3(SiLU(W1(x)) * W2(x))
                    return self.w3(self.activation(self.w1(x)) * self.w2(x))
                else:
                    # 標準FFN
                    x = self.linear1(x)
                    x = self.activation(x)
                    x = self.dropout1(x)
                    x = self.linear2(x)
                    x = self.dropout2(x)
                    return x
        
        return FeedForward()
    
    def test_implementations(self):
        """実装のテスト"""
        print("=== FFN実装のテスト ===\n")
        
        # 設定
        batch_size = 2
        seq_len = 10
        d_model = 512
        d_ff = 2048
        
        # 入力
        x = torch.randn(batch_size, seq_len, d_model)
        
        # 各種FFN
        implementations = [
            ("Standard ReLU", self.create_feed_forward(activation='relu')),
            ("Standard GELU", self.create_feed_forward(activation='gelu')),
            ("SwiGLU", self.create_feed_forward(use_glu=True, activation='silu'))
        ]
        
        for name, ffn in implementations:
            output = ffn(x)
            
            # パラメータ数
            params = sum(p.numel() for p in ffn.parameters())
            
            print(f"{name}:")
            print(f"  出力形状: {output.shape}")
            print(f"  パラメータ数: {params:,}")
            print(f"  出力統計 - 平均: {output.mean():.4f}, 標準偏差: {output.std():.4f}\n")
```

## まとめ：FFNの重要性

この章で学んだFeed Forward Networkの重要なポイント：

1. **必要性**：
   - Attentionの線形性を補完する非線形変換
   - 位置ごとの深い特徴抽出
   - モデルの表現力を大幅に向上

2. **構造**：
   - シンプルな2層MLP
   - 拡張→活性化→圧縮のパターン
   - Position-wise（位置ごとに独立）

3. **活性化関数**：
   - ReLU：シンプルで高速
   - GELU：滑らかで高性能
   - GLU変種：さらなる表現力

4. **最新の手法**：
   - SwiGLU：LLaMAで採用
   - MoE：スパースな専門家モデル
   - 効率的な実装

FFNは、Transformerブロックの約2/3のパラメータを占める重要なコンポーネントです。次章では、これらの要素を安定して深く積み重ねるための技術、残差接続と層正規化について詳しく見ていきます。

## 演習問題

1. **実装課題**：GeGLU（GELU Gate）を実装し、標準FFNと性能を比較してください。

2. **分析課題**：異なる拡張率（2x, 4x, 8x）でFFNを作成し、性能とパラメータ数のトレードオフを分析してください。

3. **最適化課題**：Sparse FFNを実装し、計算効率を測定してください。

4. **理論課題**：なぜFFNの拡張率は通常4xなのか、理論的・実験的に考察してください。

---

次章「残差接続と層正規化」へ続く。