# 残差接続と層正規化

## はじめに：深さへの挑戦

コンパイラの最適化パイプラインを考えてみましょう。各最適化パスは前のパスの結果を受け取り、さらなる改善を加えます。しかし、パスが深くなりすぎると、初期の情報が失われたり、エラーが累積したりする問題が生じます。そこで、各パスが「オプショナル」な改善を行い、必要に応じて元の状態を保持できる仕組みが重要になります。

深層学習でも同じ課題があります。層を深くすることで表現力は向上しますが、勾配消失や学習の不安定性という問題が生じます。残差接続（Residual Connection）と層正規化（Layer Normalization）は、これらの問題を解決し、100層を超える深いTransformerを可能にする重要な技術です。

この章では、これらの技術がどのように機能し、なぜTransformerの成功に不可欠なのかを詳しく見ていきます。

## 11.1 深層ネットワークの課題

### 勾配消失・爆発問題

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Optional, Callable
import math
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle, Arrow
from matplotlib.patches import ConnectionPatch
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

class DeepNetworkProblems:
    """深層ネットワークの問題を実証"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def demonstrate_gradient_vanishing(self):
        """勾配消失問題の実証"""
        print("=== 勾配消失問題 ===\n")
        
        # 深いネットワークを作成
        class DeepNetwork(nn.Module):
            def __init__(self, depth: int, use_residual: bool = False):
                super().__init__()
                self.depth = depth
                self.use_residual = use_residual
                
                # 層を作成
                self.layers = nn.ModuleList([
                    nn.Linear(64, 64) for _ in range(depth)
                ])
                
                # 活性化関数
                self.activation = nn.Tanh()  # Tanhは勾配消失しやすい
                
                # 初期化
                for layer in self.layers:
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
            
            def forward(self, x):
                activations = [x]
                
                for i, layer in enumerate(self.layers):
                    if self.use_residual and i > 0:
                        # 残差接続
                        out = layer(activations[-1])
                        out = self.activation(out)
                        out = out + activations[-1]  # 残差を加算
                    else:
                        # 通常の順伝播
                        out = layer(activations[-1])
                        out = self.activation(out)
                    
                    activations.append(out)
                
                return activations[-1], activations
        
        # 実験
        depths = [5, 10, 20, 50]
        results = {'without_residual': {}, 'with_residual': {}}
        
        for depth in depths:
            for use_residual in [False, True]:
                # ネットワーク作成
                net = DeepNetwork(depth, use_residual)
                
                # ダミーデータ
                x = torch.randn(32, 64, requires_grad=True)
                
                # 順伝播
                output, activations = net(x)
                loss = output.mean()
                
                # 逆伝播
                loss.backward()
                
                # 各層の勾配ノルムを記録
                grad_norms = []
                for layer in net.layers:
                    grad_norm = layer.weight.grad.norm().item()
                    grad_norms.append(grad_norm)
                
                key = 'with_residual' if use_residual else 'without_residual'
                results[key][depth] = grad_norms
        
        # 結果の可視化
        self._visualize_gradient_flow(results)
    
    def _visualize_gradient_flow(self, results):
        """勾配の流れを可視化"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.ravel()
        
        depths = [5, 10, 20, 50]
        
        for idx, depth in enumerate(depths):
            ax = axes[idx]
            
            # 残差なし
            grad_norms_no_res = results['without_residual'][depth]
            ax.semilogy(range(len(grad_norms_no_res)), grad_norms_no_res, 
                       'r-o', label='Without Residual', linewidth=2)
            
            # 残差あり
            if depth in results['with_residual']:
                grad_norms_res = results['with_residual'][depth]
                ax.semilogy(range(len(grad_norms_res)), grad_norms_res, 
                           'b-s', label='With Residual', linewidth=2)
            
            ax.set_xlabel('Layer Index')
            ax.set_ylabel('Gradient Norm (log scale)')
            ax.set_title(f'Depth = {depth}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 勾配消失の閾値を表示
            ax.axhline(y=1e-5, color='gray', linestyle='--', alpha=0.5)
            ax.text(depth * 0.7, 1e-5, 'Vanishing threshold', 
                   fontsize=8, color='gray')
        
        plt.suptitle('勾配消失問題：残差接続の効果', fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def demonstrate_feature_degradation(self):
        """特徴の劣化を実証"""
        print("\n=== 特徴の劣化問題 ===")
        
        # 入力信号
        x = torch.randn(1, 100)
        
        # 層を通過するごとの変化を追跡
        depths = [0, 5, 10, 20, 50]
        features_no_res = {}
        features_res = {}
        
        # 残差なし
        current = x.clone()
        for depth in depths:
            if depth > 0:
                for _ in range(depth - (depths[depths.index(depth)-1] if depths.index(depth) > 0 else 0)):
                    layer = nn.Linear(100, 100)
                    nn.init.xavier_uniform_(layer.weight, gain=0.9)  # やや小さめの初期化
                    current = torch.tanh(layer(current))
            features_no_res[depth] = current.clone()
        
        # 残差あり
        current = x.clone()
        for depth in depths:
            if depth > 0:
                for _ in range(depth - (depths[depths.index(depth)-1] if depths.index(depth) > 0 else 0)):
                    layer = nn.Linear(100, 100)
                    nn.init.xavier_uniform_(layer.weight, gain=0.1)  # 小さい初期化
                    residual = current
                    current = torch.tanh(layer(current)) + residual
            features_res[depth] = current.clone()
        
        # 可視化
        self._visualize_feature_degradation(x, features_no_res, features_res)
    
    def _visualize_feature_degradation(self, original, features_no_res, features_res):
        """特徴の劣化を可視化"""
        fig, axes = plt.subplots(2, 5, figsize=(16, 6))
        
        depths = [0, 5, 10, 20, 50]
        
        for idx, depth in enumerate(depths):
            # 残差なし
            ax = axes[0, idx]
            feat = features_no_res[depth].detach().numpy().flatten()
            ax.hist(feat, bins=30, alpha=0.7, color='red', density=True)
            ax.set_title(f'Depth {depth}')
            ax.set_xlim(-3, 3)
            ax.set_ylim(0, 2)
            
            if idx == 0:
                ax.set_ylabel('Without Residual\nDensity')
            
            # 統計情報
            ax.text(0.05, 0.95, f'μ={feat.mean():.2f}\nσ={feat.std():.2f}',
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # 残差あり
            ax = axes[1, idx]
            feat = features_res[depth].detach().numpy().flatten()
            ax.hist(feat, bins=30, alpha=0.7, color='blue', density=True)
            ax.set_xlim(-3, 3)
            ax.set_ylim(0, 2)
            
            if idx == 0:
                ax.set_ylabel('With Residual\nDensity')
            
            # 統計情報
            ax.text(0.05, 0.95, f'μ={feat.mean():.2f}\nσ={feat.std():.2f}',
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.suptitle('層を通過するごとの特徴分布の変化', fontsize=16)
        plt.tight_layout()
        plt.show()
```

### 内部共変量シフト

```python
class InternalCovariateShift:
    """内部共変量シフトの理解"""
    
    def explain_covariate_shift(self):
        """内部共変量シフトを説明"""
        print("=== 内部共変量シフト ===\n")
        
        print("問題：")
        print("- 各層の入力分布が学習中に変化")
        print("- 下位層の更新が上位層の入力を変える")
        print("- 各層が動く標的を追いかける状態\n")
        
        print("影響：")
        print("- 学習速度の低下")
        print("- 学習率を小さくする必要")
        print("- 収束が困難")
        
        # 実験で実証
        self._demonstrate_covariate_shift()
    
    def _demonstrate_covariate_shift(self):
        """共変量シフトの実証"""
        # シンプルなネットワーク
        class SimpleNetwork(nn.Module):
            def __init__(self, use_norm: bool = False):
                super().__init__()
                self.use_norm = use_norm
                
                self.layers = nn.ModuleList([
                    nn.Linear(50, 50) for _ in range(5)
                ])
                
                if use_norm:
                    self.norms = nn.ModuleList([
                        nn.LayerNorm(50) for _ in range(5)
                    ])
                
                self.activation = nn.ReLU()
            
            def forward(self, x):
                stats = []
                
                for i, layer in enumerate(self.layers):
                    x = layer(x)
                    
                    # 正規化前の統計
                    mean_before = x.mean(dim=-1, keepdim=True)
                    std_before = x.std(dim=-1, keepdim=True)
                    
                    if self.use_norm:
                        x = self.norms[i](x)
                    
                    x = self.activation(x)
                    
                    # 統計を記録
                    stats.append({
                        'mean': mean_before.mean().item(),
                        'std': std_before.mean().item()
                    })
                
                return x, stats
        
        # 学習をシミュレート
        net_no_norm = SimpleNetwork(use_norm=False)
        net_with_norm = SimpleNetwork(use_norm=True)
        
        # 複数のステップで統計を追跡
        steps = 10
        stats_history = {'no_norm': [], 'with_norm': []}
        
        for step in range(steps):
            # ランダムな入力
            x = torch.randn(32, 50)
            
            # 各ネットワークを通す
            _, stats_no_norm = net_no_norm(x)
            _, stats_with_norm = net_with_norm(x)
            
            stats_history['no_norm'].append(stats_no_norm)
            stats_history['with_norm'].append(stats_with_norm)
            
            # パラメータを更新（シミュレート）
            with torch.no_grad():
                for param in net_no_norm.parameters():
                    param.add_(torch.randn_like(param) * 0.01)
                for param in net_with_norm.parameters():
                    if param.dim() > 1:  # 重み行列のみ
                        param.add_(torch.randn_like(param) * 0.01)
        
        # 可視化
        self._visualize_covariate_shift(stats_history)
    
    def _visualize_covariate_shift(self, stats_history):
        """共変量シフトの可視化"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # 各層の平均の変化
        for layer_idx in range(5):
            # 正規化なし
            means_no_norm = [step[layer_idx]['mean'] 
                            for step in stats_history['no_norm']]
            axes[0, 0].plot(means_no_norm, label=f'Layer {layer_idx+1}',
                          linewidth=2)
            
            # 正規化あり
            means_with_norm = [step[layer_idx]['mean'] 
                              for step in stats_history['with_norm']]
            axes[0, 1].plot(means_with_norm, label=f'Layer {layer_idx+1}',
                          linewidth=2)
        
        axes[0, 0].set_title('Without Normalization')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Mean Activation')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].set_title('With Layer Normalization')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Mean Activation')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 各層の標準偏差の変化
        for layer_idx in range(5):
            # 正規化なし
            stds_no_norm = [step[layer_idx]['std'] 
                           for step in stats_history['no_norm']]
            axes[1, 0].plot(stds_no_norm, label=f'Layer {layer_idx+1}',
                          linewidth=2)
            
            # 正規化あり
            stds_with_norm = [step[layer_idx]['std'] 
                             for step in stats_history['with_norm']]
            axes[1, 1].plot(stds_with_norm, label=f'Layer {layer_idx+1}',
                          linewidth=2)
        
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Std Activation')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Std Activation')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle('内部共変量シフト：層正規化の効果', fontsize=16)
        plt.tight_layout()
        plt.show()
```

## 11.2 残差接続（Residual Connection）

### 残差接続の原理

```python
class ResidualConnectionPrinciple:
    """残差接続の原理と実装"""
    
    def explain_residual_connection(self):
        """残差接続の説明"""
        print("=== 残差接続の原理 ===\n")
        
        print("基本的なアイデア：")
        print("y = F(x) + x")
        print("- F(x): 層の変換")
        print("- x: 入力（恒等写像）\n")
        
        print("利点：")
        print("1. 勾配の高速道路：勾配が直接伝播")
        print("2. 恒等写像の学習：F(x)=0で恒等関数")
        print("3. 特徴の保存：情報が失われない")
        
        # 図解
        self._visualize_residual_connection()
    
    def _visualize_residual_connection(self):
        """残差接続の図解"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # 通常の接続
        ax1.set_title('通常の接続', fontsize=14)
        
        # ブロック
        blocks = [
            {'pos': (0.2, 0.5), 'label': 'x'},
            {'pos': (0.5, 0.5), 'label': 'F(x)'},
            {'pos': (0.8, 0.5), 'label': 'y = F(x)'}
        ]
        
        for block in blocks:
            if block['label'] == 'F(x)':
                rect = FancyBboxPatch(
                    (block['pos'][0] - 0.1, block['pos'][1] - 0.1),
                    0.2, 0.2,
                    boxstyle="round,pad=0.02",
                    facecolor='lightgreen',
                    edgecolor='black',
                    linewidth=2
                )
                ax1.add_patch(rect)
            else:
                circle = Circle(block['pos'], 0.05,
                              color='lightblue', ec='black')
                ax1.add_patch(circle)
            
            ax1.text(block['pos'][0], block['pos'][1], block['label'],
                    ha='center', va='center', fontsize=12, fontweight='bold')
        
        # 矢印
        ax1.arrow(0.25, 0.5, 0.15, 0, head_width=0.03, head_length=0.02,
                 fc='black', ec='black')
        ax1.arrow(0.6, 0.5, 0.15, 0, head_width=0.03, head_length=0.02,
                 fc='black', ec='black')
        
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0.2, 0.8)
        ax1.axis('off')
        
        # 残差接続
        ax2.set_title('残差接続', fontsize=14)
        
        # ブロック（同じ位置）
        for block in blocks:
            if block['label'] == 'F(x)':
                rect = FancyBboxPatch(
                    (block['pos'][0] - 0.1, block['pos'][1] - 0.1),
                    0.2, 0.2,
                    boxstyle="round,pad=0.02",
                    facecolor='lightgreen',
                    edgecolor='black',
                    linewidth=2
                )
                ax2.add_patch(rect)
            else:
                circle = Circle(block['pos'], 0.05,
                              color='lightblue', ec='black')
                ax2.add_patch(circle)
            
            label = block['label']
            if label == 'y = F(x)':
                label = 'y = F(x) + x'
            ax2.text(block['pos'][0], block['pos'][1], label,
                    ha='center', va='center', fontsize=12, fontweight='bold')
        
        # 通常の矢印
        ax2.arrow(0.25, 0.5, 0.15, 0, head_width=0.03, head_length=0.02,
                 fc='black', ec='black')
        ax2.arrow(0.6, 0.5, 0.15, 0, head_width=0.03, head_length=0.02,
                 fc='black', ec='black')
        
        # スキップ接続（曲線）
        ax2.annotate('', xy=(0.75, 0.5), xytext=(0.25, 0.5),
                    arrowprops=dict(arrowstyle='->',
                                  connectionstyle="arc3,rad=-.5",
                                  color='red', linewidth=2))
        ax2.text(0.5, 0.3, 'Skip Connection', ha='center', 
                color='red', fontsize=10)
        
        # 加算記号
        ax2.text(0.75, 0.4, '+', fontsize=20, ha='center', va='center',
                color='red', fontweight='bold')
        
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0.2, 0.8)
        ax2.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def implement_residual_block(self):
        """残差ブロックの実装"""
        print("\n=== 残差ブロックの実装 ===")
        
        class ResidualBlock(nn.Module):
            def __init__(self, d_model: int, dropout: float = 0.1):
                super().__init__()
                self.d_model = d_model
                
                # サブレイヤー（例：FFN）
                self.sublayer = nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model * 4, d_model),
                    nn.Dropout(dropout)
                )
                
                # 初期化：最初は恒等写像に近く
                for m in self.sublayer:
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight, gain=0.1)
                        nn.init.zeros_(m.bias)
            
            def forward(self, x):
                # 残差接続
                residual = x
                output = self.sublayer(x)
                output = output + residual
                
                return output
        
        # テスト
        block = ResidualBlock(d_model=256)
        x = torch.randn(2, 10, 256)
        
        with torch.no_grad():
            y = block(x)
            
            # 初期状態では入力に近い
            diff = (y - x).abs().mean()
            print(f"入力と出力の平均絶対差: {diff:.6f}")
            print("→ 初期化により恒等写像に近い状態からスタート")
        
        # 勾配の流れを確認
        self._check_gradient_flow(block)
    
    def _check_gradient_flow(self, block):
        """勾配の流れを確認"""
        x = torch.randn(2, 10, 256, requires_grad=True)
        y = block(x)
        loss = y.sum()
        loss.backward()
        
        print(f"\n入力の勾配ノルム: {x.grad.norm().item():.4f}")
        print("→ 残差接続により勾配が直接伝播")
```

### 残差接続の数学的解析

```python
class ResidualMathematicalAnalysis:
    """残差接続の数学的解析"""
    
    def analyze_gradient_flow(self):
        """勾配の流れを数学的に解析"""
        print("=== 残差接続の勾配解析 ===\n")
        
        print("通常の層:")
        print("∂L/∂x = ∂L/∂y · ∂y/∂x = ∂L/∂y · ∂F(x)/∂x")
        print("→ 深い層では ∂F/∂x の積が勾配消失/爆発を引き起こす\n")
        
        print("残差接続:")
        print("y = F(x) + x")
        print("∂L/∂x = ∂L/∂y · ∂y/∂x = ∂L/∂y · (∂F(x)/∂x + I)")
        print("→ 恒等行列 I により勾配が直接伝播！")
        
        # 実験で確認
        self._experiment_gradient_flow()
    
    def _experiment_gradient_flow(self):
        """勾配の流れを実験"""
        depths = [10, 20, 50, 100]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for depth in depths:
            # 理論値：通常の層
            x = np.linspace(0, depth, 100)
            grad_normal = 0.9 ** x  # 各層で0.9倍に減衰と仮定
            ax.semilogy(x, grad_normal, '--', label=f'Normal (depth={depth})')
            
            # 理論値：残差接続
            # 残差接続では勾配がほぼ保たれる
            grad_residual = np.ones_like(x) * 0.95
            ax.semilogy(x, grad_residual, '-', label=f'Residual (depth={depth})')
        
        ax.set_xlabel('Layer Depth')
        ax.set_ylabel('Gradient Magnitude (log scale)')
        ax.set_title('理論的な勾配の減衰')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 勾配消失の閾値
        ax.axhline(y=1e-5, color='red', linestyle=':', linewidth=2)
        ax.text(50, 1e-5, 'Vanishing Gradient Threshold', 
               color='red', fontsize=10)
        
        plt.tight_layout()
        plt.show()
    
    def explain_identity_mapping(self):
        """恒等写像の重要性を説明"""
        print("\n=== 恒等写像の学習 ===\n")
        
        print("問題：深い層で恒等関数を学習するのは困難")
        print("H(x) = x を学習したい場合:")
        print("- 通常の層：H(x) を直接学習")
        print("- 残差接続：F(x) = H(x) - x = 0 を学習\n")
        
        print("F(x) = 0 の学習は簡単！")
        print("→ 初期化で重みを小さくすれば自然に実現")
        
        # 実験
        self._demonstrate_identity_learning()
    
    def _demonstrate_identity_learning(self):
        """恒等写像学習の実験"""
        # タスク：入力をそのまま出力する（恒等写像）
        
        class NormalNetwork(nn.Module):
            def __init__(self, depth: int):
                super().__init__()
                layers = []
                for _ in range(depth):
                    layers.extend([
                        nn.Linear(100, 100),
                        nn.ReLU()
                    ])
                self.net = nn.Sequential(*layers[:-1])  # 最後のReLUを除く
            
            def forward(self, x):
                return self.net(x)
        
        class ResidualNetwork(nn.Module):
            def __init__(self, depth: int):
                super().__init__()
                self.blocks = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(100, 100),
                        nn.ReLU(),
                        nn.Linear(100, 100)
                    ) for _ in range(depth)
                ])
                
                # 小さい初期化
                for block in self.blocks:
                    for m in block:
                        if isinstance(m, nn.Linear):
                            nn.init.normal_(m.weight, 0, 0.01)
                            nn.init.zeros_(m.bias)
            
            def forward(self, x):
                for block in self.blocks:
                    x = x + block(x)
                return x
        
        # 学習
        depth = 10
        normal_net = NormalNetwork(depth)
        residual_net = ResidualNetwork(depth)
        
        # 同じ入力
        x = torch.randn(100, 100)
        target = x.clone()  # 恒等写像
        
        # 初期の出力を比較
        with torch.no_grad():
            normal_out = normal_net(x)
            residual_out = residual_net(x)
            
            normal_error = (normal_out - target).pow(2).mean()
            residual_error = (residual_out - target).pow(2).mean()
            
            print(f"\n初期エラー:")
            print(f"通常のネットワーク: {normal_error:.4f}")
            print(f"残差ネットワーク: {residual_error:.4f}")
            print("→ 残差ネットワークは最初から恒等写像に近い！")
```

## 11.3 層正規化（Layer Normalization）

### LayerNorm vs BatchNorm

```python
class NormalizationComparison:
    """正規化手法の比較"""
    
    def explain_normalization_methods(self):
        """正規化手法を説明"""
        print("=== 正規化手法の比較 ===\n")
        
        methods = {
            "Batch Normalization": {
                "正規化軸": "バッチ次元",
                "利点": ["バッチ統計による強力な正規化", "CNNで効果的"],
                "欠点": ["可変長系列で問題", "推論時にバッチ統計が必要", "小バッチで不安定"]
            },
            "Layer Normalization": {
                "正規化軸": "特徴次元",
                "利点": ["バッチサイズに依存しない", "系列長に柔軟", "推論時も同じ動作"],
                "欠点": ["バッチ情報を活用できない"]
            },
            "Group Normalization": {
                "正規化軸": "チャネルのグループ",
                "利点": ["小バッチでも安定", "CNNで効果的"],
                "欠点": ["グループ数の調整が必要"]
            },
            "Instance Normalization": {
                "正規化軸": "各サンプルの特徴",
                "利点": ["スタイル転送で効果的"],
                "欠点": ["一般的なタスクでは効果限定的"]
            }
        }
        
        for name, props in methods.items():
            print(f"{name}:")
            print(f"  正規化軸: {props['正規化軸']}")
            print(f"  利点: {', '.join(props['利点'])}")
            print(f"  欠点: {', '.join(props['欠点'])}")
            print()
    
    def visualize_normalization_axes(self):
        """正規化軸の可視化"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # データの形状
        batch_size, seq_len, d_model = 4, 6, 8
        
        # BatchNorm
        ax = axes[0, 0]
        ax.set_title('Batch Normalization', fontsize=14)
        
        # 3Dデータを2Dで表現
        for b in range(batch_size):
            for s in range(seq_len):
                rect = Rectangle((s, b), 1, 1,
                               facecolor=plt.cm.Blues((b*seq_len + s) % 10 / 10),
                               edgecolor='black', linewidth=0.5)
                ax.add_patch(rect)
        
        # 正規化の方向を示す
        for s in range(seq_len):
            ax.arrow(s + 0.5, -0.5, 0, batch_size + 0.3,
                    head_width=0.2, head_length=0.1,
                    fc='red', ec='red', linewidth=2)
        
        ax.text(seq_len/2, -1, 'Normalize across batch',
               ha='center', color='red', fontsize=12)
        ax.set_xlim(-0.5, seq_len)
        ax.set_ylim(-1.5, batch_size)
        ax.set_xlabel('Sequence Position')
        ax.set_ylabel('Batch')
        ax.invert_yaxis()
        
        # LayerNorm
        ax = axes[0, 1]
        ax.set_title('Layer Normalization', fontsize=14)
        
        for b in range(batch_size):
            for s in range(seq_len):
                rect = Rectangle((s, b), 1, 1,
                               facecolor=plt.cm.Greens((b*seq_len + s) % 10 / 10),
                               edgecolor='black', linewidth=0.5)
                ax.add_patch(rect)
        
        # 正規化の方向を示す
        for b in range(batch_size):
            ax.arrow(-0.5, b + 0.5, seq_len + 0.3, 0,
                    head_width=0.2, head_length=0.1,
                    fc='blue', ec='blue', linewidth=2)
        
        ax.text(-1, batch_size/2, 'Normalize\nacross features',
               ha='right', va='center', color='blue', fontsize=12)
        ax.set_xlim(-1.5, seq_len)
        ax.set_ylim(-0.5, batch_size)
        ax.set_xlabel('Sequence Position (Features)')
        ax.set_ylabel('Batch')
        ax.invert_yaxis()
        
        # 統計の計算方法
        ax = axes[1, 0]
        ax.text(0.5, 0.8, 'BatchNorm Statistics:', ha='center', fontsize=14, 
                fontweight='bold', transform=ax.transAxes)
        ax.text(0.1, 0.6, 
                '• Mean/Var computed across batch dimension\n'
                '• Different statistics for each position\n'
                '• Requires batch statistics at inference',
                transform=ax.transAxes, fontsize=12)
        ax.axis('off')
        
        ax = axes[1, 1]
        ax.text(0.5, 0.8, 'LayerNorm Statistics:', ha='center', fontsize=14,
                fontweight='bold', transform=ax.transAxes)
        ax.text(0.1, 0.6,
                '• Mean/Var computed across feature dimension\n'
                '• Different statistics for each sample\n'
                '• No batch dependency',
                transform=ax.transAxes, fontsize=12)
        ax.axis('off')
        
        plt.suptitle('正規化手法の比較', fontsize=16)
        plt.tight_layout()
        plt.show()
```

### LayerNormの実装と理解

```python
class LayerNormImplementation:
    """Layer Normalizationの実装"""
    
    def implement_layer_norm(self):
        """LayerNormの実装"""
        print("=== Layer Normalization の実装 ===\n")
        
        class LayerNorm(nn.Module):
            def __init__(self, d_model: int, eps: float = 1e-6):
                super().__init__()
                self.d_model = d_model
                self.eps = eps
                
                # 学習可能なパラメータ
                self.gamma = nn.Parameter(torch.ones(d_model))
                self.beta = nn.Parameter(torch.zeros(d_model))
            
            def forward(self, x):
                # x: [batch_size, seq_len, d_model]
                
                # 統計量の計算（最後の次元で）
                mean = x.mean(dim=-1, keepdim=True)
                var = x.var(dim=-1, keepdim=True, unbiased=False)
                
                # 正規化
                x_normalized = (x - mean) / torch.sqrt(var + self.eps)
                
                # アフィン変換
                output = self.gamma * x_normalized + self.beta
                
                return output
        
        # PyTorchの実装と比較
        d_model = 256
        x = torch.randn(2, 10, d_model)
        
        custom_ln = LayerNorm(d_model)
        pytorch_ln = nn.LayerNorm(d_model)
        
        # 同じパラメータに設定
        with torch.no_grad():
            pytorch_ln.weight.copy_(custom_ln.gamma)
            pytorch_ln.bias.copy_(custom_ln.beta)
        
        # 出力を比較
        custom_out = custom_ln(x)
        pytorch_out = pytorch_ln(x)
        
        diff = (custom_out - pytorch_out).abs().max()
        print(f"カスタム実装とPyTorch実装の最大差: {diff:.6f}")
        
        # 効果を可視化
        self._visualize_normalization_effect(x, custom_out)
    
    def _visualize_normalization_effect(self, input_tensor, output_tensor):
        """正規化の効果を可視化"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # 入力の分布
        ax = axes[0, 0]
        input_flat = input_tensor.flatten().detach().numpy()
        ax.hist(input_flat, bins=50, alpha=0.7, color='blue', density=True)
        ax.set_title('Input Distribution')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.axvline(input_flat.mean(), color='red', linestyle='--', 
                  label=f'Mean: {input_flat.mean():.2f}')
        ax.axvline(input_flat.mean() + input_flat.std(), color='orange', 
                  linestyle='--', label=f'Std: {input_flat.std():.2f}')
        ax.axvline(input_flat.mean() - input_flat.std(), color='orange', 
                  linestyle='--')
        ax.legend()
        
        # 出力の分布
        ax = axes[0, 1]
        output_flat = output_tensor.flatten().detach().numpy()
        ax.hist(output_flat, bins=50, alpha=0.7, color='green', density=True)
        ax.set_title('Output Distribution (After LayerNorm)')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.axvline(output_flat.mean(), color='red', linestyle='--',
                  label=f'Mean: {output_flat.mean():.2f}')
        ax.axvline(output_flat.mean() + output_flat.std(), color='orange',
                  linestyle='--', label=f'Std: {output_flat.std():.2f}')
        ax.axvline(output_flat.mean() - output_flat.std(), color='orange',
                  linestyle='--')
        ax.legend()
        
        # 各位置での統計
        ax = axes[1, 0]
        means = input_tensor.mean(dim=-1).flatten().detach().numpy()
        stds = input_tensor.std(dim=-1).flatten().detach().numpy()
        positions = range(len(means))
        
        ax.scatter(positions, means, alpha=0.6, label='Mean')
        ax.scatter(positions, stds, alpha=0.6, label='Std')
        ax.set_xlabel('Position')
        ax.set_ylabel('Value')
        ax.set_title('Statistics per Position (Before Norm)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 正規化後の統計
        ax = axes[1, 1]
        means_norm = output_tensor.mean(dim=-1).flatten().detach().numpy()
        stds_norm = output_tensor.std(dim=-1).flatten().detach().numpy()
        
        ax.scatter(positions, means_norm, alpha=0.6, label='Mean')
        ax.scatter(positions, stds_norm, alpha=0.6, label='Std')
        ax.set_xlabel('Position')
        ax.set_ylabel('Value')
        ax.set_title('Statistics per Position (After Norm)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-2, 2)
        
        plt.suptitle('Layer Normalization の効果', fontsize=16)
        plt.tight_layout()
        plt.show()
```

## 11.4 Pre-LN vs Post-LN

### 正規化の配置

```python
class NormalizationPlacement:
    """正規化の配置に関する分析"""
    
    def explain_pre_post_ln(self):
        """Pre-LN vs Post-LNの説明"""
        print("=== Pre-LN vs Post-LN ===\n")
        
        print("Post-LN（オリジナル）:")
        print("  x → Sublayer → Add → LayerNorm → output")
        print("  LN(x + Sublayer(x))\n")
        
        print("Pre-LN（改良版）:")
        print("  x → LayerNorm → Sublayer → Add → output")
        print("  x + Sublayer(LN(x))\n")
        
        print("Pre-LNの利点:")
        print("✓ 学習がより安定")
        print("✓ Warmupが不要または短縮可能")
        print("✓ より深いモデルが可能")
        print("✓ 勾配の流れが改善")
        
        # 実装を比較
        self._implement_both_variants()
    
    def _implement_both_variants(self):
        """両方の変種を実装"""
        
        class PostLNBlock(nn.Module):
            """Post-LN: オリジナルの構成"""
            def __init__(self, d_model: int):
                super().__init__()
                self.attention = nn.MultiheadAttention(d_model, 8, batch_first=True)
                self.norm1 = nn.LayerNorm(d_model)
                self.ffn = nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.ReLU(),
                    nn.Linear(d_model * 4, d_model)
                )
                self.norm2 = nn.LayerNorm(d_model)
                self.dropout = nn.Dropout(0.1)
            
            def forward(self, x):
                # Attention → Add → Norm
                attn_out, _ = self.attention(x, x, x)
                x = self.norm1(x + self.dropout(attn_out))
                
                # FFN → Add → Norm
                ffn_out = self.ffn(x)
                x = self.norm2(x + self.dropout(ffn_out))
                
                return x
        
        class PreLNBlock(nn.Module):
            """Pre-LN: 改良された構成"""
            def __init__(self, d_model: int):
                super().__init__()
                self.norm1 = nn.LayerNorm(d_model)
                self.attention = nn.MultiheadAttention(d_model, 8, batch_first=True)
                self.norm2 = nn.LayerNorm(d_model)
                self.ffn = nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.ReLU(),
                    nn.Linear(d_model * 4, d_model)
                )
                self.dropout = nn.Dropout(0.1)
            
            def forward(self, x):
                # Norm → Attention → Add
                attn_out, _ = self.attention(self.norm1(x), self.norm1(x), self.norm1(x))
                x = x + self.dropout(attn_out)
                
                # Norm → FFN → Add
                ffn_out = self.ffn(self.norm2(x))
                x = x + self.dropout(ffn_out)
                
                return x
        
        # 安定性の比較
        self._compare_training_stability()
    
    def _compare_training_stability(self):
        """学習の安定性を比較"""
        print("\n学習安定性の比較実験...")
        
        # 深いモデルでの勾配の振る舞いをシミュレート
        depths = [6, 12, 24]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for idx, depth in enumerate(depths):
            ax = axes[idx]
            
            # 仮想的な学習曲線
            epochs = np.arange(100)
            
            # Post-LN: 深いモデルで不安定
            post_ln_loss = np.exp(-epochs / 30) * (1 + 0.3 * np.sin(epochs / 5))
            if depth > 12:
                # 深いモデルでは初期に発散
                post_ln_loss[:20] = post_ln_loss[:20] * (1 + np.random.randn(20) * 0.5)
                post_ln_loss[:10] = np.clip(post_ln_loss[:10] * 2, 0, 5)
            
            # Pre-LN: 安定
            pre_ln_loss = np.exp(-epochs / 25) * (1 + 0.1 * np.sin(epochs / 5))
            
            ax.plot(epochs, post_ln_loss, 'r-', label='Post-LN', linewidth=2)
            ax.plot(epochs, pre_ln_loss, 'b-', label='Pre-LN', linewidth=2)
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title(f'Depth = {depth} layers')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 3)
            
            # Warmup期間を示す
            if depth > 12:
                ax.axvspan(0, 20, alpha=0.2, color='gray')
                ax.text(10, 2.5, 'Warmup\nRequired', ha='center', fontsize=10)
        
        plt.suptitle('Pre-LN vs Post-LN: 学習の安定性', fontsize=16)
        plt.tight_layout()
        plt.show()
```

## 11.5 最新の正規化技術

### RMSNorm

```python
class ModernNormalizationTechniques:
    """最新の正規化技術"""
    
    def implement_rmsnorm(self):
        """RMSNormの実装"""
        print("=== RMSNorm (Root Mean Square Normalization) ===\n")
        
        print("LayerNormの簡略版:")
        print("- 平均を引かない（センタリングなし）")
        print("- RMSで正規化")
        print("- 計算効率が良い")
        print("- LLaMAで採用\n")
        
        class RMSNorm(nn.Module):
            def __init__(self, d_model: int, eps: float = 1e-6):
                super().__init__()
                self.d_model = d_model
                self.eps = eps
                self.weight = nn.Parameter(torch.ones(d_model))
            
            def forward(self, x):
                # RMSの計算
                rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True))
                
                # 正規化
                x_normalized = x / (rms + self.eps)
                
                # スケーリング
                return self.weight * x_normalized
        
        # LayerNormとの比較
        d_model = 256
        x = torch.randn(2, 10, d_model) * 3 + 1  # 平均1、標準偏差3
        
        rmsnorm = RMSNorm(d_model)
        layernorm = nn.LayerNorm(d_model)
        
        rms_out = rmsnorm(x)
        ln_out = layernorm(x)
        
        print("入力の統計:")
        print(f"  平均: {x.mean():.3f}, 標準偏差: {x.std():.3f}")
        
        print("\nRMSNorm出力:")
        print(f"  平均: {rms_out.mean():.3f}, 標準偏差: {rms_out.std():.3f}")
        
        print("\nLayerNorm出力:")
        print(f"  平均: {ln_out.mean():.3f}, 標準偏差: {ln_out.std():.3f}")
        
        # 計算速度の比較
        self._compare_computation_speed()
    
    def _compare_computation_speed(self):
        """計算速度の比較"""
        import time
        
        print("\n=== 計算速度の比較 ===")
        
        # 大きめのテンソル
        batch_size = 128
        seq_len = 512
        d_model = 1024
        
        x = torch.randn(batch_size, seq_len, d_model).cuda()
        
        # RMSNorm
        class RMSNorm(nn.Module):
            def __init__(self, d_model):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(d_model))
                self.eps = 1e-6
            
            def forward(self, x):
                rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True))
                return self.weight * x / (rms + self.eps)
        
        rmsnorm = RMSNorm(d_model).cuda()
        layernorm = nn.LayerNorm(d_model).cuda()
        
        # ウォームアップ
        for _ in range(10):
            _ = rmsnorm(x)
            _ = layernorm(x)
        
        # 計測
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            _ = rmsnorm(x)
        torch.cuda.synchronize()
        rms_time = time.time() - start
        
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            _ = layernorm(x)
        torch.cuda.synchronize()
        ln_time = time.time() - start
        
        print(f"RMSNorm: {rms_time:.3f}秒")
        print(f"LayerNorm: {ln_time:.3f}秒")
        print(f"速度向上: {ln_time/rms_time:.2f}x")
```

### DeepNorm

```python
class DeepNormTechnique:
    """DeepNorm: 1000層のTransformerを可能に"""
    
    def explain_deepnorm(self):
        """DeepNormの説明"""
        print("=== DeepNorm ===\n")
        
        print("超深層Transformerのための技術:")
        print("1. 特別な初期化")
        print("2. 残差接続の重み付け")
        print("3. 1000層以上のモデルが可能\n")
        
        class DeepNormBlock(nn.Module):
            def __init__(self, d_model: int, depth: int):
                super().__init__()
                self.d_model = d_model
                self.depth = depth
                
                # サブレイヤー
                self.attention = nn.MultiheadAttention(d_model, 8, batch_first=True)
                self.ffn = nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.ReLU(),
                    nn.Linear(d_model * 4, d_model)
                )
                
                # LayerNorm
                self.norm1 = nn.LayerNorm(d_model)
                self.norm2 = nn.LayerNorm(d_model)
                
                # DeepNormの定数
                self.alpha = self._compute_alpha(depth)
                self.beta = self._compute_beta(depth)
                
                # 特別な初期化
                self._deepnorm_init()
            
            def _compute_alpha(self, N):
                """残差接続の重み"""
                return (2 * N) ** 0.25
            
            def _compute_beta(self, N):
                """初期化のスケール"""
                return (8 * N) ** -0.25
            
            def _deepnorm_init(self):
                """DeepNorm初期化"""
                # Xavierの変種
                for name, param in self.named_parameters():
                    if 'weight' in name and param.dim() > 1:
                        nn.init.xavier_normal_(param, gain=self.beta)
            
            def forward(self, x):
                # DeepNorm残差接続
                # x_l+1 = LN(α * x_l + sublayer(x_l))
                
                # Attention
                residual = x
                x = self.norm1(x)
                attn_out, _ = self.attention(x, x, x)
                x = self.alpha * residual + attn_out
                
                # FFN
                residual = x
                x = self.norm2(x)
                ffn_out = self.ffn(x)
                x = self.alpha * residual + ffn_out
                
                return x
        
        print(f"例: 100層のモデル")
        print(f"  α = {(2 * 100) ** 0.25:.3f}")
        print(f"  β = {(8 * 100) ** -0.25:.3f}")
```

## 11.6 統合：完全なTransformerブロック

### すべての要素を組み合わせる

```python
class CompleteTransformerBlock:
    """完全なTransformerブロックの実装"""
    
    def create_transformer_block(self,
                                d_model: int = 512,
                                n_heads: int = 8,
                                d_ff: int = 2048,
                                dropout: float = 0.1,
                                norm_type: str = 'layer',
                                norm_position: str = 'pre') -> nn.Module:
        """柔軟なTransformerブロック"""
        
        class TransformerBlock(nn.Module):
            def __init__(self):
                super().__init__()
                
                # Attention
                self.attention = nn.MultiheadAttention(
                    d_model, n_heads, dropout=dropout, batch_first=True
                )
                
                # FFN
                self.ffn = nn.Sequential(
                    nn.Linear(d_model, d_ff),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_ff, d_model),
                    nn.Dropout(dropout)
                )
                
                # Normalization
                if norm_type == 'layer':
                    self.norm1 = nn.LayerNorm(d_model)
                    self.norm2 = nn.LayerNorm(d_model)
                elif norm_type == 'rms':
                    self.norm1 = self._create_rmsnorm(d_model)
                    self.norm2 = self._create_rmsnorm(d_model)
                
                self.dropout = nn.Dropout(dropout)
                self.norm_position = norm_position
            
            def _create_rmsnorm(self, d_model):
                class RMSNorm(nn.Module):
                    def __init__(self, d_model):
                        super().__init__()
                        self.weight = nn.Parameter(torch.ones(d_model))
                        self.eps = 1e-6
                    
                    def forward(self, x):
                        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True))
                        return self.weight * x / (rms + self.eps)
                
                return RMSNorm(d_model)
            
            def forward(self, x, mask=None):
                if self.norm_position == 'pre':
                    # Pre-LN
                    # Attention
                    residual = x
                    x_norm = self.norm1(x)
                    attn_out, _ = self.attention(x_norm, x_norm, x_norm, attn_mask=mask)
                    x = residual + self.dropout(attn_out)
                    
                    # FFN
                    residual = x
                    x_norm = self.norm2(x)
                    ffn_out = self.ffn(x_norm)
                    x = residual + ffn_out
                    
                else:  # post
                    # Post-LN
                    # Attention
                    residual = x
                    attn_out, _ = self.attention(x, x, x, attn_mask=mask)
                    x = self.norm1(residual + self.dropout(attn_out))
                    
                    # FFN
                    residual = x
                    ffn_out = self.ffn(x)
                    x = self.norm2(residual + ffn_out)
                
                return x
        
        return TransformerBlock()
    
    def test_configurations(self):
        """異なる構成のテスト"""
        print("=== 異なるTransformer構成のテスト ===\n")
        
        configs = [
            ("Post-LN + LayerNorm", {'norm_position': 'post', 'norm_type': 'layer'}),
            ("Pre-LN + LayerNorm", {'norm_position': 'pre', 'norm_type': 'layer'}),
            ("Pre-LN + RMSNorm", {'norm_position': 'pre', 'norm_type': 'rms'})
        ]
        
        # 入力
        batch_size = 2
        seq_len = 10
        d_model = 512
        x = torch.randn(batch_size, seq_len, d_model)
        
        for name, config in configs:
            print(f"\n{name}:")
            block = self.create_transformer_block(**config)
            
            # パラメータ数
            params = sum(p.numel() for p in block.parameters())
            print(f"  パラメータ数: {params:,}")
            
            # 出力
            with torch.no_grad():
                output = block(x)
                print(f"  出力形状: {output.shape}")
                print(f"  出力統計 - 平均: {output.mean():.4f}, 標準偏差: {output.std():.4f}")
            
            # 勾配の流れ
            x_grad = x.clone().requires_grad_(True)
            y = block(x_grad)
            loss = y.sum()
            loss.backward()
            
            print(f"  入力勾配ノルム: {x_grad.grad.norm():.4f}")
```

## まとめ：深さを可能にする技術

この章で学んだ残差接続と層正規化の重要なポイント：

1. **残差接続**：
   - 勾配の高速道路を提供
   - 恒等写像の学習を容易に
   - 100層以上の深さを可能に
   - シンプルだが革命的

2. **層正規化**：
   - 内部共変量シフトを抑制
   - バッチサイズに依存しない
   - 系列処理に最適
   - 学習の安定化

3. **配置の重要性**：
   - Pre-LN：より安定した学習
   - Post-LN：オリジナルだが不安定
   - 深いモデルではPre-LNが標準

4. **最新の技術**：
   - RMSNorm：計算効率の改善
   - DeepNorm：1000層を可能に
   - 継続的な改善

これらの技術により、Transformerは驚異的な深さを実現し、複雑なタスクを解決できるようになりました。次章では、これらすべての要素を組み合わせた完全なエンコーダ・デコーダアーキテクチャについて詳しく見ていきます。

## 演習問題

1. **実装課題**：DeepNormを実装し、通常の残差接続と比較して、どれだけ深いモデルが安定して学習できるか確認してください。

2. **分析課題**：Pre-LNとPost-LNで、層を重ねたときの活性化の分布がどう変化するか調査してください。

3. **最適化課題**：GroupNormをTransformerに適用し、LayerNormと性能を比較してください。

4. **理論課題**：なぜ残差接続では加算を使い、乗算や連結ではないのか、数学的に考察してください。

---

次章「エンコーダとデコーダ」へ続く。