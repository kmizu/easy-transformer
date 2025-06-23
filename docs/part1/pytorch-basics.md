# PyTorchの最小限の使い方

## はじめに：なぜPyTorchなのか

プログラミング言語を実装する際、あなたはCやRustのような低レベル言語を選ぶでしょう。速度と制御性を重視するからです。しかし、ディープラーニングの世界では、異なる要求があります：

1. **自動微分**: 手動で微分を計算するのは非現実的
2. **GPU対応**: 行列演算の並列化が不可欠
3. **豊富なエコシステム**: 既存のモデルやツールとの連携

PyTorchは、これらすべてを提供しながら、Pythonの直感的な文法を維持しています。コンパイラ開発者にとって、PyTorchは「ディープラーニングのためのLLVM」のような存在です。

## 4.1 テンソルの完全理解

### テンソル = 多次元配列 + 自動微分 + GPU対応

まず、なぜNumPyではダメなのかを理解しましょう：

```python
import numpy as np
import torch
import time
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional

class TensorVsNumPy:
    """NumPyとPyTorchテンソルの違いを実証"""
    
    def performance_comparison(self, size: int = 10000) -> None:
        """大規模行列演算での性能比較"""
        # NumPy
        np_a = np.random.randn(size, size).astype(np.float32)
        np_b = np.random.randn(size, size).astype(np.float32)
        
        start = time.time()
        np_c = np.matmul(np_a, np_b)
        numpy_time = time.time() - start
        
        # PyTorch CPU
        torch_a = torch.randn(size, size)
        torch_b = torch.randn(size, size)
        
        start = time.time()
        torch_c = torch.matmul(torch_a, torch_b)
        torch_cpu_time = time.time() - start
        
        # PyTorch GPU（利用可能な場合）
        if torch.cuda.is_available():
            torch_a_gpu = torch_a.cuda()
            torch_b_gpu = torch_b.cuda()
            
            # ウォームアップ（GPUの初期化）
            _ = torch.matmul(torch_a_gpu, torch_b_gpu)
            torch.cuda.synchronize()
            
            start = time.time()
            torch_c_gpu = torch.matmul(torch_a_gpu, torch_b_gpu)
            torch.cuda.synchronize()  # GPU演算の完了を待つ
            torch_gpu_time = time.time() - start
        else:
            torch_gpu_time = float('inf')
        
        print(f"=== 行列積 ({size}x{size}) の性能比較 ===")
        print(f"NumPy: {numpy_time:.3f}秒")
        print(f"PyTorch (CPU): {torch_cpu_time:.3f}秒")
        if torch.cuda.is_available():
            print(f"PyTorch (GPU): {torch_gpu_time:.3f}秒")
            print(f"GPU高速化: {torch_cpu_time/torch_gpu_time:.1f}倍")
    
    def gradient_capability(self) -> None:
        """自動微分の能力"""
        # NumPyでは手動計算が必要
        def manual_gradient():
            # f(x, y) = x²y + xy²
            x, y = 3.0, 4.0
            
            # 手動で偏微分を計算
            df_dx = 2*x*y + y**2  # ∂f/∂x = 2xy + y²
            df_dy = x**2 + 2*x*y  # ∂f/∂y = x² + 2xy
            
            return df_dx, df_dy
        
        # PyTorchでは自動計算
        def auto_gradient():
            x = torch.tensor(3.0, requires_grad=True)
            y = torch.tensor(4.0, requires_grad=True)
            
            # 計算グラフの構築
            f = x**2 * y + x * y**2
            
            # 自動微分
            f.backward()
            
            return x.grad.item(), y.grad.item()
        
        manual = manual_gradient()
        auto = auto_gradient()
        
        print("\n=== 自動微分の比較 ===")
        print(f"手動計算: ∂f/∂x = {manual[0]}, ∂f/∂y = {manual[1]}")
        print(f"自動微分: ∂f/∂x = {auto[0]}, ∂f/∂y = {auto[1]}")
```

### テンソルの作成と基本操作

```python
class TensorBasics:
    """テンソルの基本操作を体系的に学習"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用デバイス: {self.device}")
    
    def tensor_creation_methods(self) -> dict:
        """様々なテンソル作成方法"""
        tensors = {}
        
        # 1. Pythonリストから
        tensors['from_list'] = torch.tensor([[1, 2, 3], [4, 5, 6]])
        
        # 2. NumPy配列から（メモリ共有に注意）
        np_array = np.array([[1, 2], [3, 4]], dtype=np.float32)
        tensors['from_numpy_shared'] = torch.from_numpy(np_array)  # メモリ共有
        tensors['from_numpy_copy'] = torch.tensor(np_array)  # コピー
        
        # 3. 特殊なテンソル
        shape = (3, 4)
        tensors['zeros'] = torch.zeros(shape)
        tensors['ones'] = torch.ones(shape)
        tensors['eye'] = torch.eye(4)  # 単位行列
        tensors['full'] = torch.full(shape, 3.14)  # 定数で埋める
        
        # 4. ランダムテンソル
        tensors['uniform'] = torch.rand(shape)  # [0, 1)の一様分布
        tensors['normal'] = torch.randn(shape)  # 標準正規分布
        tensors['int_random'] = torch.randint(0, 10, shape)  # 整数乱数
        
        # 5. 既存テンソルと同じ属性
        reference = torch.randn(2, 3, dtype=torch.float64, device=self.device)
        tensors['zeros_like'] = torch.zeros_like(reference)
        tensors['ones_like'] = torch.ones_like(reference)
        tensors['randn_like'] = torch.randn_like(reference)
        
        # 6. 範囲テンソル
        tensors['arange'] = torch.arange(0, 10, 2)  # [0, 2, 4, 6, 8]
        tensors['linspace'] = torch.linspace(0, 1, 5)  # 等間隔の5点
        tensors['logspace'] = torch.logspace(0, 2, 5)  # 対数スケール
        
        return tensors
    
    def tensor_properties_demo(self) -> None:
        """テンソルの属性を詳しく調査"""
        t = torch.randn(2, 3, 4, dtype=torch.float32, requires_grad=True)
        
        print("\n=== テンソルの属性 ===")
        print(f"形状 (shape): {t.shape}")
        print(f"サイズ (size): {t.size()}")  # shapeと同じ
        print(f"次元数 (ndim): {t.ndim}")
        print(f"要素数 (numel): {t.numel()}")
        print(f"データ型 (dtype): {t.dtype}")
        print(f"デバイス (device): {t.device}")
        print(f"勾配追跡 (requires_grad): {t.requires_grad}")
        print(f"勾配関数 (grad_fn): {t.grad_fn}")
        
        # メモリレイアウト
        print(f"\n--- メモリレイアウト ---")
        print(f"連続性 (is_contiguous): {t.is_contiguous()}")
        print(f"ストライド (stride): {t.stride()}")
        print(f"メモリ上のバイト数: {t.element_size() * t.numel()}")
        
        # 形状変更後の連続性
        t_reshaped = t.transpose(0, 1)
        print(f"\n転置後の連続性: {t_reshaped.is_contiguous()}")
        t_contiguous = t_reshaped.contiguous()
        print(f"contiguous()後: {t_contiguous.is_contiguous()}")
    
    def broadcasting_deep_dive(self) -> None:
        """ブロードキャスティングの詳細理解"""
        print("\n=== ブロードキャスティングルール ===")
        
        # ルール1: 次元数を揃える（右側から）
        a = torch.ones(3, 4)      # shape: (3, 4)
        b = torch.ones(4)         # shape: (4,) → (1, 4) → (3, 4)
        c = a + b
        print(f"行列 + ベクトル: {a.shape} + {b.shape} → {c.shape}")
        
        # ルール2: サイズ1の次元は拡張可能
        a = torch.ones(3, 1, 5)   # shape: (3, 1, 5)
        b = torch.ones(1, 4, 5)   # shape: (1, 4, 5)
        c = a + b                 # shape: (3, 4, 5)
        print(f"次元拡張: {a.shape} + {b.shape} → {c.shape}")
        
        # 可視化
        self._visualize_broadcasting()
        
        # 注意：メモリ効率
        print("\n--- メモリ効率の観点 ---")
        large_tensor = torch.randn(1000, 1000)
        small_vector = torch.randn(1000)
        
        # 効率的：ブロードキャスティング（メモリコピーなし）
        result_efficient = large_tensor + small_vector.unsqueeze(0)
        
        # 非効率：明示的な拡張（メモリコピーあり）
        expanded_vector = small_vector.unsqueeze(0).expand(1000, 1000)
        result_inefficient = large_tensor + expanded_vector
        
        print(f"small_vector のメモリ: {small_vector.numel() * 4} bytes")
        print(f"expanded_vector のメモリ: {expanded_vector.numel() * 4} bytes")
    
    def _visualize_broadcasting(self) -> None:
        """ブロードキャスティングの可視化"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        
        # ケース1: スカラー × 行列
        ax = axes[0, 0]
        matrix = torch.ones(3, 4)
        scalar = torch.tensor(2.0)
        self._draw_broadcast(ax, matrix.shape, scalar.shape, "スカラー × 行列")
        
        # ケース2: ベクトル × 行列（行方向）
        ax = axes[0, 1]
        matrix = torch.ones(3, 4)
        row_vector = torch.ones(1, 4)
        self._draw_broadcast(ax, matrix.shape, row_vector.shape, "行ベクトル × 行列")
        
        # ケース3: ベクトル × 行列（列方向）
        ax = axes[0, 2]
        matrix = torch.ones(3, 4)
        col_vector = torch.ones(3, 1)
        self._draw_broadcast(ax, matrix.shape, col_vector.shape, "列ベクトル × 行列")
        
        # ケース4: 3次元の例
        ax = axes[1, 0]
        tensor3d = torch.ones(2, 3, 4)
        matrix = torch.ones(3, 4)
        self._draw_broadcast(ax, tensor3d.shape, matrix.shape, "3次元 × 2次元")
        
        # ケース5: エラーケース
        ax = axes[1, 1]
        ax.text(0.5, 0.5, "ブロードキャスト不可能な例:\n(3, 4) × (3, 3)\n→ エラー", 
                ha='center', va='center', fontsize=12)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        # ケース6: 複雑な例
        ax = axes[1, 2]
        a = torch.ones(5, 1, 4, 1)
        b = torch.ones(3, 1, 6)
        # 結果: (5, 3, 4, 6)
        ax.text(0.5, 0.5, f"複雑な例:\n{a.shape} × {b.shape}\n→ (5, 3, 4, 6)", 
                ha='center', va='center', fontsize=12)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def _draw_broadcast(self, ax, shape1, shape2, title):
        """ブロードキャスティングの図示補助関数"""
        ax.set_title(title)
        ax.text(0.5, 0.5, f"{shape1} × {shape2}", ha='center', va='center')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
```

### 高度なインデックスとスライシング

```python
class AdvancedIndexing:
    """Transformerで必要となる高度なインデックス操作"""
    
    def basic_indexing(self) -> None:
        """基本的なインデックス操作"""
        print("=== 基本的なインデックス ===")
        
        # 3次元テンソル: (batch, sequence, features)
        tensor = torch.randn(4, 5, 6)  # 4バッチ、5トークン、6次元特徴
        
        # スライシング
        print(f"元の形状: {tensor.shape}")
        print(f"最初のバッチ: {tensor[0].shape}")
        print(f"全バッチの最初のトークン: {tensor[:, 0].shape}")
        print(f"最初の2バッチの3-5トークン: {tensor[:2, 2:5].shape}")
        
        # 省略記号（...）の使用
        print(f"全次元の最後の特徴: {tensor[..., -1].shape}")
        print(f"最初と最後のトークン: {tensor[:, [0, -1]].shape}")
    
    def advanced_indexing(self) -> None:
        """高度なインデックス操作"""
        print("\n=== 高度なインデックス ===")
        
        # 整数配列インデックス
        batch_size, seq_len, hidden_dim = 8, 10, 16
        hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
        
        # 特定の位置のトークンを選択
        positions = torch.tensor([1, 3, 5, 7, 2, 4, 6, 8])  # 各バッチで選択する位置
        batch_indices = torch.arange(batch_size)
        
        selected = hidden_states[batch_indices, positions]
        print(f"選択されたトークン: {selected.shape}")  # (8, 16)
        
        # ブールマスク
        mask = hidden_states > 0.5
        positive_values = hidden_states[mask]
        print(f"0.5より大きい値の数: {positive_values.numel()}")
        
        # マスクを使った値の置換
        hidden_states_copy = hidden_states.clone()
        hidden_states_copy[mask] = 1.0
        print(f"マスク適用後の平均値: {hidden_states_copy.mean():.3f}")
    
    def gather_scatter_operations(self) -> None:
        """gather/scatter操作（Transformerで重要）"""
        print("\n=== Gather/Scatter操作 ===")
        
        # Gather: 特定のインデックスの値を収集
        # 例：各バッチで最も注目すべきトークンを選択
        batch_size, seq_len, hidden_dim = 4, 6, 8
        hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
        
        # 各バッチで注目すべきトークンのインデックス
        attention_indices = torch.tensor([[2], [0], [4], [1]])  # shape: (4, 1)
        
        # gatherで選択
        attention_indices_expanded = attention_indices.unsqueeze(-1).expand(-1, -1, hidden_dim)
        selected_tokens = torch.gather(hidden_states, 1, attention_indices_expanded)
        print(f"Gather結果: {selected_tokens.shape}")  # (4, 1, 8)
        
        # Scatter: 値を特定の位置に配置
        # 例：one-hotエンコーディング
        num_classes = 10
        batch_size = 5
        labels = torch.tensor([3, 7, 1, 9, 0])
        
        one_hot = torch.zeros(batch_size, num_classes)
        one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
        print(f"\nOne-hotエンコーディング:")
        print(one_hot)
        
        # Scatter_add: 値を加算
        # 例：トークンの埋め込みを位置ごとに集計
        embeddings = torch.randn(10, 8)  # 10個の8次元埋め込み
        positions = torch.tensor([0, 1, 0, 2, 1, 2, 0, 1, 2, 0])  # 各埋め込みの位置
        
        aggregated = torch.zeros(3, 8)  # 3つの位置
        aggregated.scatter_add_(0, positions.unsqueeze(1).expand(-1, 8), embeddings)
        print(f"\nScatter_add結果: {aggregated.shape}")
    
    def index_operations_visualization(self) -> None:
        """インデックス操作の可視化"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. 基本的なスライシング
        ax = axes[0, 0]
        tensor = torch.arange(20).reshape(4, 5)
        ax.imshow(tensor, cmap='viridis')
        ax.set_title("元のテンソル")
        for i in range(4):
            for j in range(5):
                ax.text(j, i, str(tensor[i, j].item()), ha='center', va='center')
        
        # 2. 高度なインデックス
        ax = axes[0, 1]
        indices = torch.tensor([0, 2, 3])
        selected = tensor[indices]
        ax.imshow(selected, cmap='plasma')
        ax.set_title("行インデックス [0, 2, 3]")
        for i in range(3):
            for j in range(5):
                ax.text(j, i, str(selected[i, j].item()), ha='center', va='center')
        
        # 3. ブールマスク
        ax = axes[1, 0]
        mask = tensor > 10
        masked = torch.where(mask, tensor, torch.tensor(-1))
        ax.imshow(masked, cmap='RdBu')
        ax.set_title("ブールマスク (>10)")
        for i in range(4):
            for j in range(5):
                ax.text(j, i, str(masked[i, j].item()), ha='center', va='center')
        
        # 4. Gather操作
        ax = axes[1, 1]
        indices = torch.tensor([[0, 2, 4], [1, 3, 0], [2, 4, 1], [3, 0, 2]])
        gathered = torch.gather(tensor, 1, indices)
        ax.imshow(gathered, cmap='coolwarm')
        ax.set_title("Gather操作")
        for i in range(4):
            for j in range(3):
                ax.text(j, i, str(gathered[i, j].item()), ha='center', va='center')
        
        plt.tight_layout()
        plt.show()
```

## 4.2 自動微分の仕組みと実践

### 計算グラフの理解

```python
class AutogradDeepDive:
    """自動微分の仕組みを深く理解"""
    
    def computational_graph_basics(self) -> None:
        """計算グラフの基本"""
        print("=== 計算グラフの構築 ===")
        
        # 入力
        x = torch.tensor(2.0, requires_grad=True)
        y = torch.tensor(3.0, requires_grad=True)
        
        # 計算グラフの構築
        z = x * y
        w = z + x
        loss = w ** 2
        
        # グラフ情報の表示
        print(f"x.requires_grad: {x.requires_grad}")
        print(f"z.grad_fn: {z.grad_fn}")  # MulBackward
        print(f"w.grad_fn: {w.grad_fn}")  # AddBackward
        print(f"loss.grad_fn: {loss.grad_fn}")  # PowBackward
        
        # 逆伝播
        loss.backward()
        
        # 勾配の確認
        print(f"\n勾配:")
        print(f"∂loss/∂x = {x.grad}")
        print(f"∂loss/∂y = {y.grad}")
        
        # 手動計算との比較
        # loss = (xy + x)² = (2*3 + 2)² = 64
        # ∂loss/∂x = 2(xy + x)(y + 1) = 2*8*4 = 64
        # ∂loss/∂y = 2(xy + x)x = 2*8*2 = 32
        print(f"\n手動計算:")
        print(f"∂loss/∂x = 2*(x*y + x)*(y + 1) = {2*(2*3 + 2)*(3 + 1)}")
        print(f"∂loss/∂y = 2*(x*y + x)*x = {2*(2*3 + 2)*2}")
    
    def gradient_accumulation_detailed(self) -> None:
        """勾配累積の詳細"""
        print("\n=== 勾配累積 ===")
        
        x = torch.tensor(1.0, requires_grad=True)
        
        # 最初の計算
        y1 = x ** 2
        y1.backward()
        print(f"最初のbackward後: x.grad = {x.grad}")
        
        # 二回目の計算（勾配が累積される）
        y2 = x ** 3
        y2.backward()
        print(f"二回目のbackward後: x.grad = {x.grad}")
        # 期待値: 2*1 + 3*1² = 5
        
        # 勾配のリセット
        x.grad.zero_()
        y3 = x ** 4
        y3.backward()
        print(f"リセット後のbackward: x.grad = {x.grad}")
    
    def gradient_flow_control(self) -> None:
        """勾配の流れを制御"""
        print("\n=== 勾配フローの制御 ===")
        
        # 1. torch.no_grad()コンテキスト
        x = torch.tensor(1.0, requires_grad=True)
        
        with torch.no_grad():
            y = x * 2
            z = y * 3
        
        print(f"no_grad内での計算:")
        print(f"y.requires_grad: {y.requires_grad}")
        print(f"z.requires_grad: {z.requires_grad}")
        
        # 2. detach()メソッド
        x = torch.tensor(1.0, requires_grad=True)
        y = x * 2
        z = y.detach() * 3  # ここで勾配の流れを切断
        w = z + x  # xからの勾配は流れるが、yからは流れない
        
        w.backward()
        print(f"\ndetach使用時:")
        print(f"x.grad: {x.grad}")  # 1.0（w = z + xのxの項から）
        
        # 3. requires_grad_()メソッド
        a = torch.tensor(1.0)
        print(f"\n初期状態: a.requires_grad = {a.requires_grad}")
        a.requires_grad_(True)
        print(f"requires_grad_()後: a.requires_grad = {a.requires_grad}")
    
    def higher_order_derivatives(self) -> None:
        """高階微分の計算"""
        print("\n=== 高階微分 ===")
        
        x = torch.tensor(2.0, requires_grad=True)
        
        # f(x) = x⁴
        y = x ** 4
        
        # 一階微分: f'(x) = 4x³
        first_grad = torch.autograd.grad(y, x, create_graph=True)[0]
        print(f"f'(2) = {first_grad}")  # 32
        
        # 二階微分: f''(x) = 12x²
        second_grad = torch.autograd.grad(first_grad, x, create_graph=True)[0]
        print(f"f''(2) = {second_grad}")  # 48
        
        # 三階微分: f'''(x) = 24x
        third_grad = torch.autograd.grad(second_grad, x, create_graph=True)[0]
        print(f"f'''(2) = {third_grad}")  # 48
        
        # 四階微分: f''''(x) = 24
        fourth_grad = torch.autograd.grad(third_grad, x)[0]
        print(f"f''''(2) = {fourth_grad}")  # 24
    
    def custom_autograd_function(self) -> None:
        """カスタム自動微分関数の実装"""
        print("\n=== カスタム自動微分関数 ===")
        
        class ReLUCustom(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input):
                # 逆伝播で必要な情報を保存
                ctx.save_for_backward(input)
                return input.clamp(min=0)
            
            @staticmethod
            def backward(ctx, grad_output):
                input, = ctx.saved_tensors
                grad_input = grad_output.clone()
                grad_input[input < 0] = 0
                return grad_input
        
        # 使用例
        relu_custom = ReLUCustom.apply
        
        x = torch.randn(5, requires_grad=True)
        y = relu_custom(x)
        loss = y.sum()
        loss.backward()
        
        print(f"入力: {x.data}")
        print(f"出力: {y.data}")
        print(f"勾配: {x.grad}")
```

### 実践的な勾配処理

```python
class PracticalGradientHandling:
    """実践的な勾配処理テクニック"""
    
    def gradient_clipping_strategies(self) -> None:
        """様々な勾配クリッピング戦略"""
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 10)
        )
        
        # ダミーデータで勾配を生成
        x = torch.randn(32, 10)
        y = model(x).sum()
        y.backward()
        
        print("=== 勾配クリッピング戦略 ===")
        
        # 1. ノルムによるクリッピング
        print("\n1. ノルムクリッピング:")
        total_norm_before = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        print(f"クリッピング前のノルム: {total_norm_before:.3f}")
        
        # 2. 値によるクリッピング
        print("\n2. 値クリッピング:")
        torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.1)
        
        # 3. 適応的クリッピング
        print("\n3. 適応的クリッピング:")
        self._adaptive_clipping(model)
    
    def _adaptive_clipping(self, model):
        """適応的な勾配クリッピング"""
        # 各層ごとに異なるクリッピング値を設定
        for name, param in model.named_parameters():
            if param.grad is not None:
                if 'weight' in name:
                    max_norm = 1.0
                else:  # bias
                    max_norm = 0.5
                
                grad_norm = param.grad.norm()
                if grad_norm > max_norm:
                    param.grad.mul_(max_norm / grad_norm)
                
                print(f"{name}: norm = {grad_norm:.3f}")
    
    def gradient_checkpointing_example(self) -> None:
        """勾配チェックポイント（メモリ節約）"""
        import torch.utils.checkpoint as checkpoint
        
        class DeepModel(torch.nn.Module):
            def __init__(self, use_checkpoint=False):
                super().__init__()
                self.use_checkpoint = use_checkpoint
                self.layers = torch.nn.ModuleList([
                    torch.nn.Sequential(
                        torch.nn.Linear(100, 100),
                        torch.nn.ReLU(),
                        torch.nn.Dropout(0.1)
                    ) for _ in range(10)
                ])
            
            def forward(self, x):
                for layer in self.layers:
                    if self.use_checkpoint:
                        x = checkpoint.checkpoint(layer, x)
                    else:
                        x = layer(x)
                return x
        
        # メモリ使用量の比較
        print("\n=== 勾配チェックポイント ===")
        
        # 通常のモデル
        model_normal = DeepModel(use_checkpoint=False)
        x = torch.randn(100, 100, requires_grad=True)
        y = model_normal(x).sum()
        
        # メモリ使用量を推定
        memory_normal = sum(p.numel() * p.element_size() for p in model_normal.parameters())
        
        # チェックポイント使用
        model_checkpoint = DeepModel(use_checkpoint=True)
        y_checkpoint = model_checkpoint(x).sum()
        
        print(f"通常モデルのパラメータメモリ: {memory_normal / 1024:.1f} KB")
        print("チェックポイント使用時: 中間活性化のメモリを節約")
```

## 4.3 実践的なニューラルネットワーク構築

### モジュールの設計哲学

```python
class ModuleDesignPhilosophy:
    """PyTorchモジュール設計の哲学と実践"""
    
    def module_as_computation_unit(self) -> None:
        """モジュール = 計算の単位"""
        print("=== モジュールの設計哲学 ===")
        
        # 1. 単一責任の原則
        class AttentionHead(torch.nn.Module):
            """単一のアテンションヘッド（単一責任）"""
            def __init__(self, d_model: int, d_k: int):
                super().__init__()
                self.d_k = d_k
                self.W_q = torch.nn.Linear(d_model, d_k, bias=False)
                self.W_k = torch.nn.Linear(d_model, d_k, bias=False)
                self.W_v = torch.nn.Linear(d_model, d_k, bias=False)
            
            def forward(self, query, key, value, mask=None):
                Q = self.W_q(query)
                K = self.W_k(key)
                V = self.W_v(value)
                
                scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
                
                if mask is not None:
                    scores = scores.masked_fill(mask == 0, -1e9)
                
                attention_weights = torch.softmax(scores, dim=-1)
                context = torch.matmul(attention_weights, V)
                
                return context, attention_weights
        
        # 2. 組み合わせ可能性
        class MultiHeadAttention(torch.nn.Module):
            """複数のAttentionHeadを組み合わせ"""
            def __init__(self, d_model: int, num_heads: int):
                super().__init__()
                assert d_model % num_heads == 0
                self.d_k = d_model // num_heads
                self.num_heads = num_heads
                
                # 複数のヘッドを作成
                self.heads = torch.nn.ModuleList([
                    AttentionHead(d_model, self.d_k) 
                    for _ in range(num_heads)
                ])
                self.W_o = torch.nn.Linear(d_model, d_model)
            
            def forward(self, query, key, value, mask=None):
                # 各ヘッドで計算
                head_outputs = []
                for head in self.heads:
                    context, _ = head(query, key, value, mask)
                    head_outputs.append(context)
                
                # 結合と線形変換
                concatenated = torch.cat(head_outputs, dim=-1)
                output = self.W_o(concatenated)
                
                return output
        
        # 3. 再利用性
        class TransformerBlock(torch.nn.Module):
            """再利用可能なTransformerブロック"""
            def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
                super().__init__()
                self.attention = MultiHeadAttention(d_model, num_heads)
                self.norm1 = torch.nn.LayerNorm(d_model)
                self.norm2 = torch.nn.LayerNorm(d_model)
                
                self.feed_forward = torch.nn.Sequential(
                    torch.nn.Linear(d_model, d_ff),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(dropout),
                    torch.nn.Linear(d_ff, d_model)
                )
                self.dropout = torch.nn.Dropout(dropout)
            
            def forward(self, x, mask=None):
                # Self-attention with residual
                attn_output = self.attention(x, x, x, mask)
                x = self.norm1(x + self.dropout(attn_output))
                
                # Feed-forward with residual
                ff_output = self.feed_forward(x)
                x = self.norm2(x + self.dropout(ff_output))
                
                return x
    
    def initialization_strategies(self) -> None:
        """初期化戦略の重要性"""
        print("\n=== パラメータ初期化戦略 ===")
        
        class ProperlyInitializedModel(torch.nn.Module):
            def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
                super().__init__()
                self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
                self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
                self.fc3 = torch.nn.Linear(hidden_dim, output_dim)
                self.relu = torch.nn.ReLU()
                
                # 適切な初期化
                self._initialize_weights()
            
            def _initialize_weights(self):
                for module in self.modules():
                    if isinstance(module, torch.nn.Linear):
                        # Xavier/Glorot初期化（活性化関数に応じて選択）
                        if self._next_is_relu(module):
                            # ReLU用のHe初期化
                            torch.nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                        else:
                            # その他の活性化関数用
                            torch.nn.init.xavier_normal_(module.weight)
                        
                        if module.bias is not None:
                            torch.nn.init.constant_(module.bias, 0)
            
            def _next_is_relu(self, module):
                """次の層がReLUかどうかを判定（簡易版）"""
                return True  # この例では全てReLU
            
            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.relu(self.fc2(x))
                x = self.fc3(x)
                return x
        
        # 初期化の影響を可視化
        self._visualize_initialization_effects()
    
    def _visualize_initialization_effects(self):
        """初期化の効果を可視化"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 異なる初期化方法
        init_methods = {
            'Zero': lambda w: torch.nn.init.constant_(w, 0),
            'Normal(0.01)': lambda w: torch.nn.init.normal_(w, std=0.01),
            'Xavier': lambda w: torch.nn.init.xavier_normal_(w),
            'He': lambda w: torch.nn.init.kaiming_normal_(w, nonlinearity='relu')
        }
        
        for idx, (name, init_fn) in enumerate(init_methods.items()):
            ax = axes[idx // 2, idx % 2]
            
            # シンプルなネットワーク
            model = torch.nn.Sequential(
                torch.nn.Linear(100, 50),
                torch.nn.ReLU(),
                torch.nn.Linear(50, 10)
            )
            
            # 初期化
            for layer in model:
                if isinstance(layer, torch.nn.Linear):
                    init_fn(layer.weight)
            
            # 活性化の分布を調査
            x = torch.randn(1000, 100)
            activations = []
            
            def hook_fn(module, input, output):
                activations.append(output.detach())
            
            for layer in model:
                if isinstance(layer, torch.nn.ReLU):
                    layer.register_forward_hook(hook_fn)
            
            with torch.no_grad():
                _ = model(x)
            
            # 可視化
            if activations:
                act = activations[0].flatten().numpy()
                ax.hist(act, bins=50, alpha=0.7, density=True)
                ax.set_title(f'{name} 初期化')
                ax.set_xlabel('活性化値')
                ax.set_ylabel('密度')
                ax.axvline(0, color='red', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.show()
```

### 学習ループの完全実装

```python
class ComprehensiveTrainingLoop:
    """包括的な学習ループの実装"""
    
    def __init__(self, model: torch.nn.Module, device: str = 'cuda'):
        self.model = model.to(device)
        self.device = torch.device(device)
        self.history = defaultdict(list)
        self.best_model_state = None
    
    def train_with_all_bells_and_whistles(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        gradient_clip: float = 1.0,
        patience: int = 10,
        warmup_steps: int = 1000
    ):
        """完全な機能を持つ学習ループ"""
        
        # オプティマイザの設定
        optimizer = self._setup_optimizer(learning_rate, weight_decay)
        
        # スケジューラの設定
        scheduler = self._setup_scheduler(optimizer, num_epochs, warmup_steps)
        
        # 損失関数
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # 早期終了の設定
        early_stopping = EarlyStopping(patience=patience)
        
        # 学習ループ
        for epoch in range(num_epochs):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"{'='*50}")
            
            # 学習
            train_metrics = self._train_epoch(
                train_loader, optimizer, criterion, scheduler, gradient_clip
            )
            
            # 検証
            val_metrics = self._validate_epoch(val_loader, criterion)
            
            # 履歴の記録
            self._update_history(train_metrics, val_metrics, optimizer)
            
            # 最良モデルの保存
            if val_metrics['accuracy'] > self.best_val_accuracy:
                self.best_val_accuracy = val_metrics['accuracy']
                self.best_model_state = self.model.state_dict().copy()
                print(f"✓ 最良モデル更新! Val Acc: {val_metrics['accuracy']:.4f}")
            
            # 早期終了のチェック
            if early_stopping(val_metrics['loss']):
                print(f"\n早期終了: {epoch+1}エポックで停止")
                break
            
            # 定期的な可視化
            if (epoch + 1) % 5 == 0:
                self._visualize_training_progress()
    
    def _setup_optimizer(self, learning_rate: float, weight_decay: float):
        """オプティマイザの設定（層ごとに異なる学習率）"""
        # 層ごとにパラメータをグループ化
        param_groups = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            # 層の種類に応じて学習率を調整
            if 'embedding' in name:
                lr_scale = 0.1  # 埋め込み層は小さい学習率
            elif 'classifier' in name or 'output' in name:
                lr_scale = 10.0  # 出力層は大きい学習率
            else:
                lr_scale = 1.0
            
            # 重み減衰の調整（バイアスとLayerNormには適用しない）
            wd = 0 if 'bias' in name or 'norm' in name else weight_decay
            
            param_groups.append({
                'params': param,
                'lr': learning_rate * lr_scale,
                'weight_decay': wd,
                'name': name
            })
        
        return torch.optim.AdamW(param_groups)
    
    def _setup_scheduler(self, optimizer, num_epochs: int, warmup_steps: int):
        """学習率スケジューラの設定"""
        # カスタムスケジューラ：ウォームアップ + コサイン減衰
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                progress = (step - warmup_steps) / (num_epochs * 1000 - warmup_steps)
                return 0.5 * (1 + math.cos(math.pi * progress))
        
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    def _train_epoch(self, loader, optimizer, criterion, scheduler, gradient_clip):
        """1エポックの学習"""
        self.model.train()
        
        total_loss = 0
        correct = 0
        total = 0
        
        # プログレスバー
        pbar = tqdm(loader, desc='Training', leave=False)
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            # Mixed Precision Training
            with torch.cuda.amp.autocast():
                output = self.model(data)
                loss = criterion(output, target)
            
            # 逆伝播
            optimizer.zero_grad()
            loss.backward()
            
            # 勾配クリッピング
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), gradient_clip
            )
            
            # 更新
            optimizer.step()
            scheduler.step()
            
            # 統計
            total_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            # プログレスバー更新
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100. * correct / total:.2f}%',
                'grad_norm': f'{grad_norm:.3f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        return {
            'loss': total_loss / total,
            'accuracy': correct / total
        }
    
    def _visualize_training_progress(self):
        """学習進捗の可視化"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Loss
        ax = axes[0, 0]
        ax.plot(epochs, self.history['train_loss'], 'b-', label='Train')
        ax.plot(epochs, self.history['val_loss'], 'r-', label='Val')
        ax.set_title('Loss')
        ax.set_xlabel('Epoch')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Accuracy
        ax = axes[0, 1]
        ax.plot(epochs, self.history['train_acc'], 'b-', label='Train')
        ax.plot(epochs, self.history['val_acc'], 'r-', label='Val')
        ax.set_title('Accuracy')
        ax.set_xlabel('Epoch')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Learning Rate
        ax = axes[0, 2]
        ax.plot(epochs, self.history['lr'])
        ax.set_title('Learning Rate')
        ax.set_xlabel('Epoch')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        # Gradient Norm
        ax = axes[1, 0]
        ax.plot(epochs, self.history['grad_norm'])
        ax.set_title('Gradient Norm')
        ax.set_xlabel('Epoch')
        ax.grid(True, alpha=0.3)
        
        # 過学習度
        ax = axes[1, 1]
        overfit = np.array(self.history['train_acc']) - np.array(self.history['val_acc'])
        ax.plot(epochs, overfit)
        ax.axhline(y=0, color='k', linestyle='--')
        ax.set_title('過学習度 (Train - Val Acc)')
        ax.set_xlabel('Epoch')
        ax.grid(True, alpha=0.3)
        
        # 最後のエポックの詳細
        ax = axes[1, 2]
        ax.text(0.1, 0.9, f"最終エポック統計:", transform=ax.transAxes, fontsize=12, weight='bold')
        stats_text = f"""
Train Loss: {self.history['train_loss'][-1]:.4f}
Val Loss: {self.history['val_loss'][-1]:.4f}
Train Acc: {self.history['train_acc'][-1]:.4f}
Val Acc: {self.history['val_acc'][-1]:.4f}
Best Val Acc: {max(self.history['val_acc']):.4f}
Learning Rate: {self.history['lr'][-1]:.6f}
        """
        ax.text(0.1, 0.1, stats_text, transform=ax.transAxes, fontsize=10)
        ax.axis('off')
        
        plt.tight_layout()
        plt.show()

class EarlyStopping:
    """早期終了の実装"""
    def __init__(self, patience: int = 10, min_delta: float = 0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        
        return self.early_stop
```

## まとめ：PyTorchマスターへの道

この章で学んだPyTorchの基礎は、Transformerを実装する上での土台となります：

1. **テンソル操作**: NumPy感覚で使えるが、自動微分とGPU対応という強力な機能を持つ
2. **自動微分**: 計算グラフの概念を理解し、勾配の流れを制御できる
3. **モジュール設計**: 再利用可能で組み合わせ可能な部品として実装
4. **学習ループ**: 実践的な機能（早期終了、勾配クリッピング、スケジューリング等）を含む

次章では、これらの知識を活用して、いよいよTransformerの中核である「単語の数値表現」に取り組みます。

## 演習問題

1. **テンソル操作**: バッチ処理されたシーケンスデータ（shape: [batch, seq_len, features]）から、各バッチの最長シーケンスの最後のトークンを効率的に抽出する関数を実装してください。

2. **自動微分**: カスタム活性化関数「Swish (x * sigmoid(x))」を、自動微分対応で実装してください。

3. **モジュール設計**: 残差接続とドロップアウトを含む、深さ可変のMLPを実装してください。

4. **学習最適化**: 学習率のウォームアップと再スタート（SGDR）を組み合わせたスケジューラを実装してください。

---

次章「第2部：Transformerへの道のり - 単語の数値表現」へ続く。