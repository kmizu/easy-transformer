# デバッグと可視化

## はじめに：見えないものを見る

コンパイラのデバッグを思い出してください。構文木を可視化し、中間表現をダンプし、最適化の各段階を追跡することで、複雑な変換過程を理解できます。print文デバッグから始めて、最終的には洗練されたデバッガーやプロファイラーを使うようになります。

深層学習モデル、特にTransformerのような複雑なアーキテクチャでは、内部で何が起きているかを理解することが成功の鍵です。この章では、Transformerをデバッグし、その動作を可視化するための実践的な技術を学びます。

## 15.1 注意機構の可視化

### Attention Weightの詳細分析

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch
from matplotlib.collections import LineCollection
import matplotlib.cm as cm
from IPython.display import HTML, display
import ipywidgets as widgets
import warnings
warnings.filterwarnings('ignore')

class AttentionVisualizer:
    """注意機構の包括的な可視化ツール"""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.attention_weights = {}
        self.hooks = []
        
    def register_hooks(self):
        """注意重みを記録するフックを登録"""
        def create_hook(name):
            def hook_fn(module, input, output):
                # outputは(output, attention_weights)のタプル
                if isinstance(output, tuple) and len(output) == 2:
                    _, attn_weights = output
                    if attn_weights is not None:
                        self.attention_weights[name] = attn_weights.detach()
                return output
            return hook_fn
        
        # すべてのMultiHeadAttentionモジュールにフックを登録
        for name, module in self.model.named_modules():
            if isinstance(module, nn.MultiheadAttention):
                hook = module.register_forward_hook(create_hook(name))
                self.hooks.append(hook)
    
    def remove_hooks(self):
        """フックを削除"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def visualize_attention_pattern(self, tokens: List[str], 
                                  layer_name: str = None):
        """注意パターンを可視化"""
        if layer_name is None:
            # 最初の層を使用
            layer_name = list(self.attention_weights.keys())[0]
        
        attn_weights = self.attention_weights[layer_name]
        
        # バッチの最初のサンプル、すべてのヘッドの平均
        if attn_weights.dim() == 4:  # [batch, heads, seq, seq]
            attn_weights = attn_weights[0].mean(dim=0)
        elif attn_weights.dim() == 3:  # [batch, seq, seq]
            attn_weights = attn_weights[0]
        
        attn_weights = attn_weights.cpu().numpy()
        
        # ヒートマップ
        plt.figure(figsize=(10, 8))
        sns.heatmap(attn_weights, xticklabels=tokens, yticklabels=tokens,
                   cmap='Blues', cbar_kws={'label': 'Attention Weight'})
        plt.title(f'Attention Pattern - {layer_name}')
        plt.xlabel('Keys (Attended to)')
        plt.ylabel('Queries (Attending from)')
        plt.tight_layout()
        plt.show()
    
    def visualize_head_diversity(self, tokens: List[str], 
                               layer_name: str = None):
        """各ヘッドの多様性を可視化"""
        if layer_name is None:
            layer_name = list(self.attention_weights.keys())[0]
        
        attn_weights = self.attention_weights[layer_name]
        
        if attn_weights.dim() == 4:
            attn_weights = attn_weights[0]  # 最初のバッチ
        else:
            print("Multi-head情報がありません")
            return
        
        n_heads = attn_weights.shape[0]
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        for head_idx in range(min(n_heads, 8)):
            ax = axes[head_idx]
            head_weights = attn_weights[head_idx].cpu().numpy()
            
            im = ax.imshow(head_weights, cmap='Blues', aspect='auto')
            ax.set_title(f'Head {head_idx + 1}')
            
            # 簡略化のため、ラベルは最初と最後のヘッドのみ
            if head_idx == 0:
                ax.set_yticks(range(len(tokens)))
                ax.set_yticklabels(tokens, fontsize=8)
            else:
                ax.set_yticks([])
            
            if head_idx >= 4:
                ax.set_xticks(range(len(tokens)))
                ax.set_xticklabels(tokens, rotation=45, fontsize=8)
            else:
                ax.set_xticks([])
        
        plt.suptitle(f'Head Diversity - {layer_name}', fontsize=14)
        plt.tight_layout()
        plt.show()
        
        # ヘッド間の類似性を計算
        self._compute_head_similarity(attn_weights)
    
    def _compute_head_similarity(self, attn_weights: torch.Tensor):
        """ヘッド間の類似性を計算"""
        n_heads = attn_weights.shape[0]
        similarity_matrix = torch.zeros(n_heads, n_heads)
        
        for i in range(n_heads):
            for j in range(n_heads):
                # コサイン類似度
                similarity = F.cosine_similarity(
                    attn_weights[i].flatten(),
                    attn_weights[j].flatten(),
                    dim=0
                )
                similarity_matrix[i, j] = similarity
        
        # 類似性マトリックスを表示
        plt.figure(figsize=(8, 6))
        sns.heatmap(similarity_matrix.numpy(), annot=True, fmt='.2f',
                   cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                   xticklabels=[f'H{i+1}' for i in range(n_heads)],
                   yticklabels=[f'H{i+1}' for i in range(n_heads)])
        plt.title('Head Similarity Matrix')
        plt.tight_layout()
        plt.show()

class AttentionFlowVisualizer:
    """注意の流れを可視化"""
    
    def create_attention_flow_diagram(self, tokens: List[str], 
                                    attention_weights: np.ndarray):
        """注意の流れ図を作成"""
        seq_len = len(tokens)
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # トークンの配置
        y_positions = np.linspace(0.1, 0.9, seq_len)
        x_left = 0.2
        x_right = 0.8
        
        # 左側（Query）と右側（Key）にトークンを配置
        for i, (token, y) in enumerate(zip(tokens, y_positions)):
            # Query側
            ax.text(x_left, y, token, ha='right', va='center',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue'),
                   fontsize=12)
            # Key側
            ax.text(x_right, y, token, ha='left', va='center',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen'),
                   fontsize=12)
        
        # 注意の矢印を描画
        for i in range(seq_len):
            for j in range(seq_len):
                weight = attention_weights[i, j]
                if weight > 0.1:  # 閾値
                    # 矢印の太さと透明度を重みに応じて調整
                    arrow = mpatches.FancyArrowPatch(
                        (x_left + 0.05, y_positions[i]),
                        (x_right - 0.05, y_positions[j]),
                        connectionstyle="arc3,rad=0.2",
                        arrowstyle='->', 
                        mutation_scale=20,
                        linewidth=weight * 5,
                        alpha=weight,
                        color='purple'
                    )
                    ax.add_patch(arrow)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('Attention Flow Visualization', fontsize=16, pad=20)
        
        # 凡例
        ax.text(0.5, 0.02, 'Query → Key (arrow thickness = attention weight)',
               ha='center', fontsize=10, style='italic')
        
        plt.tight_layout()
        plt.show()

class InteractiveAttentionExplorer:
    """インタラクティブな注意探索ツール"""
    
    def __init__(self, model: nn.Module, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.visualizer = AttentionVisualizer(model)
        
    def create_interactive_widget(self):
        """対話型ウィジェットを作成"""
        # テキスト入力
        text_input = widgets.Textarea(
            value='The cat sat on the mat.',
            description='Text:',
            layout=widgets.Layout(width='500px', height='80px')
        )
        
        # 層選択
        layer_dropdown = widgets.Dropdown(
            options=['Layer 1', 'Layer 2', 'Layer 3', 'Layer 4'],
            value='Layer 1',
            description='Layer:'
        )
        
        # ヘッド選択
        head_slider = widgets.IntSlider(
            value=1,
            min=1,
            max=8,
            step=1,
            description='Head:',
            continuous_update=False
        )
        
        # 可視化タイプ
        viz_type = widgets.RadioButtons(
            options=['Heatmap', 'Flow Diagram', 'Head Comparison'],
            value='Heatmap',
            description='Viz Type:'
        )
        
        # 出力エリア
        output = widgets.Output()
        
        def update_visualization(change):
            with output:
                output.clear_output(wait=True)
                
                # トークン化
                tokens = text_input.value.split()  # 簡易版
                
                # ダミーの注意重み（実際はモデルから取得）
                seq_len = len(tokens)
                attention_weights = np.random.rand(seq_len, seq_len)
                attention_weights = attention_weights / attention_weights.sum(axis=1, keepdims=True)
                
                if viz_type.value == 'Heatmap':
                    self._plot_heatmap(tokens, attention_weights)
                elif viz_type.value == 'Flow Diagram':
                    flow_viz = AttentionFlowVisualizer()
                    flow_viz.create_attention_flow_diagram(tokens, attention_weights)
                else:
                    self._plot_head_comparison(tokens)
        
        # イベントハンドラを登録
        text_input.observe(update_visualization, names='value')
        layer_dropdown.observe(update_visualization, names='value')
        head_slider.observe(update_visualization, names='value')
        viz_type.observe(update_visualization, names='value')
        
        # 初期表示
        update_visualization(None)
        
        # レイアウト
        controls = widgets.VBox([text_input, layer_dropdown, head_slider, viz_type])
        return widgets.HBox([controls, output])
    
    def _plot_heatmap(self, tokens: List[str], weights: np.ndarray):
        """ヒートマップをプロット"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(weights, xticklabels=tokens, yticklabels=tokens,
                   cmap='Blues', cbar=True)
        plt.title('Attention Weights Heatmap')
        plt.tight_layout()
        plt.show()
    
    def _plot_head_comparison(self, tokens: List[str]):
        """ヘッド比較をプロット"""
        # ダミーデータで8つのヘッドを表示
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        for i in range(8):
            ax = axes[i]
            # 各ヘッドで異なるパターンを生成
            if i < 2:  # 局所的な注意
                weights = np.eye(len(tokens))
                for j in range(1, 2):
                    weights += np.eye(len(tokens), k=j) * 0.5
                    weights += np.eye(len(tokens), k=-j) * 0.5
            elif i < 4:  # 長距離の注意
                weights = np.random.rand(len(tokens), len(tokens))
                weights = weights / weights.sum(axis=1, keepdims=True)
            else:  # 特定パターン
                weights = np.zeros((len(tokens), len(tokens)))
                weights[:, 0] = 0.5  # 最初のトークンに注目
                weights[:, -1] = 0.5  # 最後のトークンに注目
            
            im = ax.imshow(weights, cmap='Blues')
            ax.set_title(f'Head {i+1}')
            ax.set_xticks([])
            ax.set_yticks([])
        
        plt.suptitle('Attention Head Patterns')
        plt.tight_layout()
        plt.show()

## 15.2 勾配フローの追跡

class GradientFlowAnalyzer:
    """勾配フローの分析ツール"""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.gradient_data = {}
        self.activation_data = {}
        
    def register_gradient_hooks(self):
        """勾配を記録するフックを登録"""
        def create_grad_hook(name):
            def hook_fn(grad):
                self.gradient_data[name] = {
                    'mean': grad.mean().item(),
                    'std': grad.std().item(),
                    'max': grad.max().item(),
                    'min': grad.min().item(),
                    'norm': grad.norm().item()
                }
            return hook_fn
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.register_hook(create_grad_hook(name))
    
    def analyze_gradient_flow(self, loss: torch.Tensor):
        """勾配フローを分析"""
        # 逆伝播
        loss.backward()
        
        # 勾配統計を可視化
        self._plot_gradient_statistics()
        
        # 勾配消失・爆発の検出
        self._detect_gradient_issues()
        
    def _plot_gradient_statistics(self):
        """勾配統計をプロット"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # レイヤー名を整理
        layer_names = list(self.gradient_data.keys())
        layer_indices = range(len(layer_names))
        
        # 平均勾配
        ax = axes[0, 0]
        means = [self.gradient_data[name]['mean'] for name in layer_names]
        ax.bar(layer_indices, means)
        ax.set_title('Mean Gradient per Layer')
        ax.set_xlabel('Layer Index')
        ax.set_ylabel('Mean Gradient')
        ax.set_yscale('symlog')
        
        # 勾配ノルム
        ax = axes[0, 1]
        norms = [self.gradient_data[name]['norm'] for name in layer_names]
        ax.plot(layer_indices, norms, 'o-')
        ax.set_title('Gradient Norm per Layer')
        ax.set_xlabel('Layer Index')
        ax.set_ylabel('Gradient Norm')
        ax.set_yscale('log')
        
        # 勾配の分散
        ax = axes[1, 0]
        stds = [self.gradient_data[name]['std'] for name in layer_names]
        ax.bar(layer_indices, stds, color='orange')
        ax.set_title('Gradient Std per Layer')
        ax.set_xlabel('Layer Index')
        ax.set_ylabel('Gradient Std')
        
        # 勾配の最大値・最小値
        ax = axes[1, 1]
        maxs = [self.gradient_data[name]['max'] for name in layer_names]
        mins = [self.gradient_data[name]['min'] for name in layer_names]
        ax.plot(layer_indices, maxs, 'g-', label='Max')
        ax.plot(layer_indices, mins, 'r-', label='Min')
        ax.set_title('Gradient Range per Layer')
        ax.set_xlabel('Layer Index')
        ax.set_ylabel('Gradient Value')
        ax.legend()
        ax.set_yscale('symlog')
        
        plt.tight_layout()
        plt.show()
        
        # 詳細なレポート
        self._print_gradient_report()
    
    def _detect_gradient_issues(self):
        """勾配の問題を検出"""
        issues = []
        
        for name, stats in self.gradient_data.items():
            # 勾配消失
            if stats['norm'] < 1e-6:
                issues.append(f"勾配消失の可能性: {name} (norm={stats['norm']:.2e})")
            
            # 勾配爆発
            if stats['norm'] > 1e3:
                issues.append(f"勾配爆発の可能性: {name} (norm={stats['norm']:.2e})")
            
            # 不安定な勾配
            if stats['std'] / (abs(stats['mean']) + 1e-8) > 10:
                issues.append(f"不安定な勾配: {name} (変動係数が大きい)")
        
        if issues:
            print("=== 検出された問題 ===")
            for issue in issues:
                print(f"⚠️  {issue}")
        else:
            print("✅ 勾配フローは正常です")
    
    def _print_gradient_report(self):
        """勾配レポートを出力"""
        print("\n=== 勾配フロー詳細レポート ===\n")
        
        # 最も大きい/小さい勾配を持つ層
        sorted_by_norm = sorted(self.gradient_data.items(), 
                               key=lambda x: x[1]['norm'])
        
        print("勾配ノルムが最も小さい層 (Top 5):")
        for name, stats in sorted_by_norm[:5]:
            print(f"  {name}: {stats['norm']:.2e}")
        
        print("\n勾配ノルムが最も大きい層 (Top 5):")
        for name, stats in sorted_by_norm[-5:]:
            print(f"  {name}: {stats['norm']:.2e}")

class ActivationAnalyzer:
    """活性化の分析ツール"""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.activation_data = {}
        self.hooks = []
        
    def register_activation_hooks(self):
        """活性化を記録するフックを登録"""
        def create_hook(name):
            def hook_fn(module, input, output):
                if isinstance(output, torch.Tensor):
                    self.activation_data[name] = {
                        'mean': output.mean().item(),
                        'std': output.std().item(),
                        'zeros': (output == 0).float().mean().item(),
                        'shape': list(output.shape),
                        'histogram': output.detach().cpu().numpy().flatten()
                    }
            return hook_fn
        
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # リーフモジュール
                hook = module.register_forward_hook(create_hook(name))
                self.hooks.append(hook)
    
    def analyze_activations(self, input_data: torch.Tensor):
        """活性化を分析"""
        # 順伝播
        with torch.no_grad():
            _ = self.model(input_data)
        
        # 活性化の分布を可視化
        self._plot_activation_distributions()
        
        # デッドニューロンの検出
        self._detect_dead_neurons()
        
    def _plot_activation_distributions(self):
        """活性化分布をプロット"""
        n_layers = len(self.activation_data)
        n_cols = 4
        n_rows = (n_layers + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes]
        
        for idx, (name, data) in enumerate(self.activation_data.items()):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            
            # ヒストグラム
            hist_data = data['histogram']
            if len(hist_data) > 10000:
                # サンプリング
                indices = np.random.choice(len(hist_data), 10000, replace=False)
                hist_data = hist_data[indices]
            
            ax.hist(hist_data, bins=50, alpha=0.7, density=True)
            ax.axvline(data['mean'], color='red', linestyle='--', 
                      label=f'Mean: {data["mean"]:.3f}')
            ax.set_title(f'{name}\n(zeros: {data["zeros"]*100:.1f}%)')
            ax.set_xlabel('Activation Value')
            ax.set_ylabel('Density')
            ax.legend()
        
        # 余ったaxesを非表示
        for idx in range(len(self.activation_data), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def _detect_dead_neurons(self):
        """デッドニューロンを検出"""
        print("\n=== デッドニューロン検出 ===\n")
        
        dead_threshold = 0.9  # 90%以上がゼロ
        
        for name, data in self.activation_data.items():
            if data['zeros'] > dead_threshold:
                print(f"⚠️  {name}: {data['zeros']*100:.1f}% がゼロ (デッドニューロンの可能性)")

## 15.3 学習過程の監視

class TrainingMonitor:
    """学習過程の包括的な監視ツール"""
    
    def __init__(self):
        self.metrics = {
            'loss': [],
            'learning_rate': [],
            'gradient_norm': [],
            'weight_update_ratio': [],
            'val_loss': [],
            'val_accuracy': []
        }
        self.batch_metrics = {
            'loss': [],
            'gradient_norm': []
        }
        
    def log_batch(self, loss: float, model: nn.Module, 
                  optimizer: torch.optim.Optimizer):
        """バッチごとのメトリクスを記録"""
        # 損失
        self.batch_metrics['loss'].append(loss)
        
        # 勾配ノルム
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.norm().item() ** 2
        total_norm = total_norm ** 0.5
        self.batch_metrics['gradient_norm'].append(total_norm)
        
        # 重み更新比率
        if len(self.batch_metrics['loss']) % 100 == 0:
            self._compute_weight_update_ratio(model, optimizer)
    
    def log_epoch(self, epoch: int, train_loss: float, val_loss: float,
                  val_accuracy: float, learning_rate: float):
        """エポックごとのメトリクスを記録"""
        self.metrics['loss'].append(train_loss)
        self.metrics['val_loss'].append(val_loss)
        self.metrics['val_accuracy'].append(val_accuracy)
        self.metrics['learning_rate'].append(learning_rate)
        
        # バッチメトリクスの平均
        if self.batch_metrics['gradient_norm']:
            avg_grad_norm = np.mean(self.batch_metrics['gradient_norm'])
            self.metrics['gradient_norm'].append(avg_grad_norm)
    
    def _compute_weight_update_ratio(self, model: nn.Module,
                                   optimizer: torch.optim.Optimizer):
        """重み更新比率を計算"""
        ratios = []
        
        for group in optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    # 更新量 / パラメータのノルム
                    update = group['lr'] * p.grad
                    ratio = update.norm().item() / (p.norm().item() + 1e-8)
                    ratios.append(ratio)
        
        if ratios:
            self.metrics['weight_update_ratio'].append(np.mean(ratios))
    
    def plot_training_curves(self):
        """学習曲線をプロット"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 損失
        ax = axes[0, 0]
        epochs = range(1, len(self.metrics['loss']) + 1)
        ax.plot(epochs, self.metrics['loss'], 'b-', label='Train Loss')
        ax.plot(epochs, self.metrics['val_loss'], 'r-', label='Val Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training and Validation Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 精度
        ax = axes[0, 1]
        ax.plot(epochs, self.metrics['val_accuracy'], 'g-')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Validation Accuracy')
        ax.grid(True, alpha=0.3)
        
        # 学習率
        ax = axes[0, 2]
        ax.plot(epochs, self.metrics['learning_rate'], 'orange')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        # 勾配ノルム
        ax = axes[1, 0]
        ax.plot(epochs, self.metrics['gradient_norm'], 'purple')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Gradient Norm')
        ax.set_title('Average Gradient Norm')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        # 重み更新比率
        ax = axes[1, 1]
        if self.metrics['weight_update_ratio']:
            ax.plot(self.metrics['weight_update_ratio'], 'brown')
            ax.set_xlabel('Update Step')
            ax.set_ylabel('Update/Weight Ratio')
            ax.set_title('Weight Update Ratio')
            ax.axhline(y=1e-3, color='r', linestyle='--', 
                      label='Typical Good Range')
            ax.legend()
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
        
        # バッチごとの損失
        ax = axes[1, 2]
        ax.plot(self.batch_metrics['loss'][:1000], alpha=0.5)  # 最初の1000バッチ
        ax.set_xlabel('Batch')
        ax.set_ylabel('Loss')
        ax.set_title('Batch Loss (First 1000 batches)')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def generate_training_report(self):
        """学習レポートを生成"""
        print("=== 学習サマリーレポート ===\n")
        
        # 最終的なメトリクス
        print(f"最終エポック:")
        print(f"  訓練損失: {self.metrics['loss'][-1]:.4f}")
        print(f"  検証損失: {self.metrics['val_loss'][-1]:.4f}")
        print(f"  検証精度: {self.metrics['val_accuracy'][-1]:.4f}")
        
        # 最良のエポック
        best_val_loss_epoch = np.argmin(self.metrics['val_loss']) + 1
        best_val_acc_epoch = np.argmax(self.metrics['val_accuracy']) + 1
        
        print(f"\n最良のパフォーマンス:")
        print(f"  最小検証損失: {min(self.metrics['val_loss']):.4f} (Epoch {best_val_loss_epoch})")
        print(f"  最高検証精度: {max(self.metrics['val_accuracy']):.4f} (Epoch {best_val_acc_epoch})")
        
        # 過学習の検出
        if len(self.metrics['loss']) > 5:
            recent_train = np.mean(self.metrics['loss'][-5:])
            recent_val = np.mean(self.metrics['val_loss'][-5:])
            
            if recent_val > recent_train * 1.5:
                print("\n⚠️  過学習の兆候が見られます")
        
        # 学習の安定性
        if self.batch_metrics['gradient_norm']:
            grad_std = np.std(self.batch_metrics['gradient_norm'])
            grad_mean = np.mean(self.batch_metrics['gradient_norm'])
            
            if grad_std / grad_mean > 2:
                print("\n⚠️  勾配が不安定です")

## 15.4 モデル診断ツール

class ModelDiagnostics:
    """モデルの包括的な診断ツール"""
    
    def __init__(self, model: nn.Module):
        self.model = model
        
    def run_diagnostics(self, sample_input: torch.Tensor):
        """完全な診断を実行"""
        print("=== モデル診断開始 ===\n")
        
        # 1. モデル構造の分析
        self._analyze_model_structure()
        
        # 2. パラメータ分析
        self._analyze_parameters()
        
        # 3. 計算量分析
        self._analyze_computation(sample_input)
        
        # 4. メモリ使用量分析
        self._analyze_memory(sample_input)
        
        # 5. 推論速度測定
        self._measure_inference_speed(sample_input)
        
        print("\n=== 診断完了 ===")
    
    def _analyze_model_structure(self):
        """モデル構造を分析"""
        print("1. モデル構造分析")
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() 
                             if p.requires_grad)
        
        print(f"  総パラメータ数: {total_params:,}")
        print(f"  学習可能パラメータ数: {trainable_params:,}")
        print(f"  固定パラメータ数: {total_params - trainable_params:,}")
        
        # 層ごとのパラメータ数
        print("\n  層ごとのパラメータ数:")
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # リーフモジュール
                params = sum(p.numel() for p in module.parameters())
                if params > 0:
                    print(f"    {name}: {params:,}")
    
    def _analyze_parameters(self):
        """パラメータを分析"""
        print("\n2. パラメータ分析")
        
        # パラメータの統計
        all_params = []
        for p in self.model.parameters():
            all_params.extend(p.detach().cpu().numpy().flatten())
        
        all_params = np.array(all_params)
        
        print(f"  平均: {np.mean(all_params):.4f}")
        print(f"  標準偏差: {np.std(all_params):.4f}")
        print(f"  最小値: {np.min(all_params):.4f}")
        print(f"  最大値: {np.max(all_params):.4f}")
        
        # スパース性
        sparsity = (np.abs(all_params) < 1e-6).mean()
        print(f"  スパース性: {sparsity*100:.2f}%")
        
        # パラメータ分布のプロット
        plt.figure(figsize=(10, 4))
        plt.hist(all_params, bins=100, alpha=0.7, density=True)
        plt.xlabel('Parameter Value')
        plt.ylabel('Density')
        plt.title('Parameter Distribution')
        plt.axvline(x=0, color='red', linestyle='--', alpha=0.5)
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def _analyze_computation(self, sample_input: torch.Tensor):
        """計算量を分析"""
        print("\n3. 計算量分析")
        
        # FLOPsを概算（簡易版）
        total_mult_adds = 0
        
        def count_operations(module, input, output):
            nonlocal total_mult_adds
            
            if isinstance(module, nn.Linear):
                # Linear層: input_features * output_features
                total_mult_adds += input[0].numel() * module.out_features
            elif isinstance(module, nn.MultiheadAttention):
                # Attention: O(n^2 * d)の計算量
                seq_len = input[0].shape[1]
                d_model = input[0].shape[2]
                total_mult_adds += seq_len * seq_len * d_model
        
        # フックを登録
        hooks = []
        for module in self.model.modules():
            if isinstance(module, (nn.Linear, nn.MultiheadAttention)):
                hook = module.register_forward_hook(count_operations)
                hooks.append(hook)
        
        # 順伝播
        with torch.no_grad():
            _ = self.model(sample_input)
        
        # フックを削除
        for hook in hooks:
            hook.remove()
        
        print(f"  推定FLOP数: {total_mult_adds:,}")
        print(f"  GFLOP: {total_mult_adds / 1e9:.2f}")
    
    def _analyze_memory(self, sample_input: torch.Tensor):
        """メモリ使用量を分析"""
        print("\n4. メモリ使用量分析")
        
        # パラメータのメモリ
        param_memory = sum(p.numel() * p.element_size() 
                          for p in self.model.parameters())
        print(f"  パラメータメモリ: {param_memory / 1024**2:.2f} MB")
        
        # 勾配のメモリ（学習時）
        grad_memory = sum(p.numel() * p.element_size() 
                         for p in self.model.parameters() 
                         if p.requires_grad)
        print(f"  勾配メモリ: {grad_memory / 1024**2:.2f} MB")
        
        # 活性化のメモリ（概算）
        # 実際の測定にはメモリプロファイラーが必要
        print(f"  活性化メモリ: 入力サイズに依存")
    
    def _measure_inference_speed(self, sample_input: torch.Tensor):
        """推論速度を測定"""
        print("\n5. 推論速度測定")
        
        # ウォームアップ
        for _ in range(10):
            with torch.no_grad():
                _ = self.model(sample_input)
        
        # 測定
        import time
        n_runs = 100
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        for _ in range(n_runs):
            with torch.no_grad():
                _ = self.model(sample_input)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        
        avg_time = (end_time - start_time) / n_runs * 1000  # ms
        
        print(f"  平均推論時間: {avg_time:.2f} ms")
        print(f"  スループット: {1000/avg_time:.2f} samples/sec")

# 実行例とデモ
def run_comprehensive_demo():
    """包括的なデモを実行"""
    print("=" * 70)
    print("Transformerデバッグ・可視化ツールのデモ")
    print("=" * 70 + "\n")
    
    # ダミーモデルの作成
    class DummyTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(1000, 256)
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=256, nhead=8, dim_feedforward=1024,
                    batch_first=True
                ),
                num_layers=4
            )
            self.output = nn.Linear(256, 1000)
            
        def forward(self, x):
            x = self.embedding(x)
            x = self.transformer(x)
            x = self.output(x)
            return x
    
    model = DummyTransformer()
    
    # 1. 注意の可視化
    print("=== 1. 注意機構の可視化 ===\n")
    visualizer = AttentionVisualizer(model)
    
    # ダミーデータで注意パターンを生成
    tokens = ["The", "cat", "sat", "on", "the", "mat"]
    dummy_attention = torch.rand(1, 8, len(tokens), len(tokens))
    dummy_attention = F.softmax(dummy_attention, dim=-1)
    
    # 注意フロー図
    flow_viz = AttentionFlowVisualizer()
    flow_viz.create_attention_flow_diagram(tokens, dummy_attention[0, 0].numpy())
    
    # 2. 勾配フロー分析
    print("\n=== 2. 勾配フロー分析 ===\n")
    grad_analyzer = GradientFlowAnalyzer(model)
    grad_analyzer.register_gradient_hooks()
    
    # ダミーの順伝播と逆伝播
    input_ids = torch.randint(0, 1000, (2, 10))
    output = model(input_ids)
    loss = output.mean()
    grad_analyzer.analyze_gradient_flow(loss)
    
    # 3. 活性化分析
    print("\n=== 3. 活性化分析 ===\n")
    act_analyzer = ActivationAnalyzer(model)
    act_analyzer.register_activation_hooks()
    act_analyzer.analyze_activations(input_ids)
    
    # 4. 学習監視
    print("\n=== 4. 学習過程の監視 ===\n")
    monitor = TrainingMonitor()
    
    # ダミーの学習データ
    for epoch in range(10):
        # エポックごとのダミーメトリクス
        train_loss = 2.5 * np.exp(-0.3 * epoch) + np.random.normal(0, 0.1)
        val_loss = 2.5 * np.exp(-0.25 * epoch) + np.random.normal(0, 0.15)
        val_acc = 1 - np.exp(-0.5 * epoch) + np.random.normal(0, 0.05)
        lr = 0.001 * (0.1 ** (epoch // 3))
        
        monitor.log_epoch(epoch, train_loss, val_loss, val_acc, lr)
        
        # バッチごとのダミーメトリクス
        for batch in range(50):
            batch_loss = train_loss + np.random.normal(0, 0.2)
            monitor.log_batch(batch_loss, model, 
                            torch.optim.Adam(model.parameters()))
    
    monitor.plot_training_curves()
    monitor.generate_training_report()
    
    # 5. モデル診断
    print("\n=== 5. モデル診断 ===\n")
    diagnostics = ModelDiagnostics(model)
    diagnostics.run_diagnostics(input_ids)
    
    print("\n" + "=" * 70)
    print("デモ完了")
    print("=" * 70)

if __name__ == "__main__":
    run_comprehensive_demo()