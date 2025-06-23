# Transformer解説プロジェクト - 超詳細計画

## 第1部：導入と基礎概念（30-35ページ）の詳細

### 1.4 PyTorchの最小限の使い方（8-10ページ）

#### 1.4.1 テンソルの完全理解（3ページ）

##### なぜテンソルが必要か
```python
# 従来のNumPyアプローチ
import numpy as np

# CPUでの計算
data = np.array([[1, 2, 3], [4, 5, 6]])
result = np.dot(data, data.T)

# 問題点：
# 1. GPU計算ができない
# 2. 自動微分がない
# 3. 並列処理の最適化が限定的
```

```python
# PyTorchのテンソル
import torch

# GPU対応、自動微分可能
data = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
data = data.cuda()  # GPUへ転送（利用可能な場合）
result = torch.mm(data, data.T)
```

##### テンソルの詳細な操作
```python
class TensorOperations:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def tensor_basics(self):
        # テンソルの作成方法いろいろ
        # 1. リストから
        t1 = torch.tensor([1, 2, 3])
        
        # 2. NumPy配列から
        np_array = np.array([1, 2, 3])
        t2 = torch.from_numpy(np_array)
        
        # 3. 特殊なテンソル
        zeros = torch.zeros(3, 4)
        ones = torch.ones(3, 4)
        rand = torch.rand(3, 4)  # 一様分布 [0, 1)
        randn = torch.randn(3, 4)  # 標準正規分布
        
        # 4. 既存テンソルと同じ形状
        t3 = torch.zeros_like(t1)
        t4 = torch.randn_like(t1, dtype=torch.float32)
        
        return t1, t2, t3, t4
    
    def tensor_properties(self):
        t = torch.randn(3, 4, 5)
        
        print(f"形状: {t.shape}")  # torch.Size([3, 4, 5])
        print(f"次元数: {t.ndim}")  # 3
        print(f"要素数: {t.numel()}")  # 60
        print(f"データ型: {t.dtype}")  # torch.float32
        print(f"デバイス: {t.device}")  # cpu or cuda:0
        print(f"勾配が必要か: {t.requires_grad}")  # False
        
        # メモリレイアウト
        print(f"連続か: {t.is_contiguous()}")
        print(f"ストライド: {t.stride()}")
    
    def broadcasting_explained(self):
        """
        NumPyと同じブロードキャスティングルール
        プログラミング言語の暗黙の型変換に似ている
        """
        # スカラーとベクトル
        scalar = torch.tensor(2.0)
        vector = torch.tensor([1.0, 2.0, 3.0])
        result = scalar * vector  # [2.0, 4.0, 6.0]
        
        # 行列とベクトル
        matrix = torch.randn(3, 4)
        row_vector = torch.randn(1, 4)
        col_vector = torch.randn(3, 1)
        
        # ブロードキャスティングが適用される
        result1 = matrix + row_vector  # (3, 4) + (1, 4) -> (3, 4)
        result2 = matrix + col_vector  # (3, 4) + (3, 1) -> (3, 4)
        
        # 可視化
        self.visualize_broadcasting()
    
    def visualize_broadcasting(self):
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 元の行列
        ax = axes[0]
        ax.set_title("元の行列 (3×4)")
        for i in range(3):
            for j in range(4):
                rect = patches.Rectangle((j, 2-i), 1, 1, 
                                       linewidth=1, edgecolor='black', 
                                       facecolor='lightblue')
                ax.add_patch(rect)
                ax.text(j+0.5, 2-i+0.5, f'{i},{j}', ha='center', va='center')
        
        # 行ベクトル
        ax = axes[1]
        ax.set_title("行ベクトル (1×4) のブロードキャスト")
        for j in range(4):
            for i in range(3):
                if i == 0:
                    color = 'lightgreen'
                else:
                    color = 'lightgray'
                rect = patches.Rectangle((j, 2-i), 1, 1,
                                       linewidth=1, edgecolor='black',
                                       facecolor=color)
                ax.add_patch(rect)
        
        # 列ベクトル
        ax = axes[2]
        ax.set_title("列ベクトル (3×1) のブロードキャスト")
        for i in range(3):
            for j in range(4):
                if j == 0:
                    color = 'lightcoral'
                else:
                    color = 'lightgray'
                rect = patches.Rectangle((j, 2-i), 1, 1,
                                       linewidth=1, edgecolor='black',
                                       facecolor=color)
                ax.add_patch(rect)
        
        for ax in axes:
            ax.set_xlim(0, 4)
            ax.set_ylim(0, 3)
            ax.set_aspect('equal')
            ax.axis('off')
```

##### インデックスとスライシングの詳細
```python
def advanced_indexing():
    # 基本的なインデックス
    t = torch.randn(5, 4, 3)
    
    # NumPyと同じ記法
    print(t[0])        # 最初の行列
    print(t[:, 0])     # すべての行列の最初の行
    print(t[..., -1])  # すべての要素の最後の列
    
    # 高度なインデックス
    # 1. 整数配列によるインデックス
    indices = torch.tensor([0, 2, 4])
    selected = t[indices]  # 0, 2, 4番目の行列を選択
    
    # 2. ブールマスク
    mask = t > 0
    positive_values = t[mask]  # 正の値のみ抽出
    
    # 3. gather操作（Transformerで重要）
    # 特定のインデックスの値を収集
    batch_size, seq_len, hidden_dim = 2, 5, 4
    hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
    
    # 各バッチで特定の位置の隠れ状態を取得
    positions = torch.tensor([[1], [3]])  # shape: (batch_size, 1)
    positions = positions.unsqueeze(-1).expand(-1, -1, hidden_dim)
    selected_hidden = torch.gather(hidden_states, 1, positions)
    
    # 4. scatter操作（値の散布）
    # one-hotエンコーディングの実装
    num_classes = 10
    labels = torch.tensor([3, 7, 1, 9])
    one_hot = torch.zeros(len(labels), num_classes)
    one_hot.scatter_(1, labels.unsqueeze(1), 1)
    
    return selected, positive_values, selected_hidden, one_hot
```

#### 1.4.2 自動微分の完全理解（3ページ）

##### 計算グラフの構築と可視化
```python
class ComputationalGraph:
    def __init__(self):
        self.nodes = []
        self.edges = []
    
    def simple_example(self):
        # 手動での微分計算との比較
        # f(x, y) = x²y + y³
        
        # 手動計算
        def f(x, y):
            return x**2 * y + y**3
        
        def df_dx(x, y):
            return 2 * x * y
        
        def df_dy(x, y):
            return x**2 + 3 * y**2
        
        # PyTorchでの自動微分
        x = torch.tensor(2.0, requires_grad=True)
        y = torch.tensor(3.0, requires_grad=True)
        
        # 順伝播（計算グラフの構築）
        z1 = x ** 2      # z1 = x²
        z2 = z1 * y      # z2 = x²y
        z3 = y ** 3      # z3 = y³
        f = z2 + z3      # f = x²y + y³
        
        # 逆伝播
        f.backward()
        
        print(f"手動計算: df/dx = {df_dx(2, 3)}, df/dy = {df_dy(2, 3)}")
        print(f"自動微分: df/dx = {x.grad}, df/dy = {y.grad}")
        
        # 計算グラフの可視化
        self.visualize_graph(f)
    
    def gradient_accumulation(self):
        """
        勾配の累積：大きなバッチを小さく分割して処理
        メモリ制約下でのトレーニングで重要
        """
        # モデルとデータの準備
        model = torch.nn.Linear(10, 1)
        total_batch_size = 100
        micro_batch_size = 10
        
        # 勾配をゼロ化
        model.zero_grad()
        
        # 大きなバッチを小さく分割
        for i in range(0, total_batch_size, micro_batch_size):
            # ミニバッチのデータ
            x = torch.randn(micro_batch_size, 10)
            y = torch.randn(micro_batch_size, 1)
            
            # 順伝播
            output = model(x)
            loss = ((output - y) ** 2).mean()
            
            # 勾配を累積（スケーリングが必要）
            loss = loss / (total_batch_size / micro_batch_size)
            loss.backward()
        
        # パラメータ更新
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        optimizer.step()
    
    def gradient_checkpointing(self):
        """
        メモリと計算のトレードオフ
        大規模モデルのトレーニングで使用
        """
        import torch.utils.checkpoint as checkpoint
        
        class CheckpointedLayer(torch.nn.Module):
            def __init__(self, hidden_dim):
                super().__init__()
                self.linear1 = torch.nn.Linear(hidden_dim, hidden_dim * 4)
                self.linear2 = torch.nn.Linear(hidden_dim * 4, hidden_dim)
                self.relu = torch.nn.ReLU()
            
            def forward(self, x):
                # 通常の順伝播
                # x = self.relu(self.linear1(x))
                # x = self.linear2(x)
                
                # チェックポイント版（メモリ節約）
                def forward_fn(x):
                    x = self.relu(self.linear1(x))
                    x = self.linear2(x)
                    return x
                
                return checkpoint.checkpoint(forward_fn, x)
    
    def higher_order_gradients(self):
        """
        高階微分の計算
        最適化アルゴリズムの研究で使用
        """
        x = torch.tensor(2.0, requires_grad=True)
        
        # f(x) = x³
        y = x ** 3
        
        # 一階微分: dy/dx = 3x²
        first_grad = torch.autograd.grad(y, x, create_graph=True)[0]
        print(f"一階微分: {first_grad}")  # 12.0
        
        # 二階微分: d²y/dx² = 6x
        second_grad = torch.autograd.grad(first_grad, x, create_graph=True)[0]
        print(f"二階微分: {second_grad}")  # 12.0
        
        # 三階微分: d³y/dx³ = 6
        third_grad = torch.autograd.grad(second_grad, x)[0]
        print(f"三階微分: {third_grad}")  # 6.0
```

##### 勾配の問題と対策
```python
class GradientProblems:
    def __init__(self):
        self.history = {"gradients": [], "losses": []}
    
    def vanishing_gradient(self):
        """
        勾配消失問題の実演
        深いネットワークでシグモイド活性化を使った場合
        """
        # 深いネットワーク（シグモイド活性化）
        class DeepSigmoidNet(torch.nn.Module):
            def __init__(self, num_layers=10):
                super().__init__()
                self.layers = torch.nn.ModuleList([
                    torch.nn.Linear(10, 10) for _ in range(num_layers)
                ])
                self.sigmoid = torch.nn.Sigmoid()
            
            def forward(self, x):
                activations = []
                for layer in self.layers:
                    x = self.sigmoid(layer(x))
                    activations.append(x)
                return x, activations
        
        model = DeepSigmoidNet(num_layers=20)
        x = torch.randn(32, 10)
        output, activations = model(x)
        loss = output.sum()
        loss.backward()
        
        # 各層の勾配の大きさを可視化
        gradient_norms = []
        for i, layer in enumerate(model.layers):
            grad_norm = layer.weight.grad.norm().item()
            gradient_norms.append(grad_norm)
        
        plt.figure(figsize=(10, 6))
        plt.semilogy(gradient_norms)
        plt.xlabel('層の深さ')
        plt.ylabel('勾配のノルム（対数スケール）')
        plt.title('勾配消失問題：深い層ほど勾配が小さくなる')
        plt.grid(True)
    
    def gradient_clipping(self):
        """
        勾配クリッピング：勾配爆発を防ぐ
        """
        model = torch.nn.LSTM(10, 20, 2)
        optimizer = torch.optim.Adam(model.parameters())
        
        # 極端な入力で勾配爆発を誘発
        x = torch.randn(100, 32, 10) * 10
        h0 = torch.randn(2, 32, 20) * 10
        c0 = torch.randn(2, 32, 20) * 10
        
        output, _ = model(x, (h0, c0))
        loss = output.sum()
        loss.backward()
        
        # クリッピング前の勾配
        total_norm_before = 0
        for p in model.parameters():
            if p.grad is not None:
                total_norm_before += p.grad.norm().item() ** 2
        total_norm_before = total_norm_before ** 0.5
        
        # 勾配クリッピング
        max_norm = 1.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        
        # クリッピング後の勾配
        total_norm_after = 0
        for p in model.parameters():
            if p.grad is not None:
                total_norm_after += p.grad.norm().item() ** 2
        total_norm_after = total_norm_after ** 0.5
        
        print(f"クリッピング前の勾配ノルム: {total_norm_before:.2f}")
        print(f"クリッピング後の勾配ノルム: {total_norm_after:.2f}")
```

#### 1.4.3 実践的なニューラルネットワーク構築（4ページ）

##### モジュールの設計パターン
```python
class ModuleDesignPatterns:
    """
    PyTorchのnn.Moduleはコンパイラの抽象構文木（AST）のようなもの
    各モジュールは計算の単位を表す
    """
    
    def basic_module(self):
        """基本的なモジュールの作り方"""
        class SimpleLinear(torch.nn.Module):
            def __init__(self, in_features, out_features, bias=True):
                super().__init__()
                # パラメータの初期化
                self.weight = torch.nn.Parameter(
                    torch.randn(out_features, in_features) * 0.01
                )
                if bias:
                    self.bias = torch.nn.Parameter(torch.zeros(out_features))
                else:
                    self.register_parameter('bias', None)
                
                # バッファ（学習しないが保存される値）
                self.register_buffer('running_mean', torch.zeros(out_features))
                self.register_buffer('num_calls', torch.tensor(0))
            
            def forward(self, x):
                # 線形変換
                output = torch.matmul(x, self.weight.t())
                if self.bias is not None:
                    output += self.bias
                
                # 統計情報の更新（学習時のみ）
                if self.training:
                    self.running_mean = 0.9 * self.running_mean + 0.1 * output.mean(0)
                    self.num_calls += 1
                
                return output
            
            def extra_repr(self):
                """print時の表示をカスタマイズ"""
                return f'in_features={self.weight.shape[1]}, out_features={self.weight.shape[0]}'
    
    def composite_module(self):
        """複合モジュールの設計"""
        class ResidualBlock(torch.nn.Module):
            def __init__(self, channels, kernel_size=3):
                super().__init__()
                # サブモジュールの定義
                self.conv1 = torch.nn.Conv2d(channels, channels, kernel_size, padding=kernel_size//2)
                self.bn1 = torch.nn.BatchNorm2d(channels)
                self.conv2 = torch.nn.Conv2d(channels, channels, kernel_size, padding=kernel_size//2)
                self.bn2 = torch.nn.BatchNorm2d(channels)
                self.relu = torch.nn.ReLU(inplace=True)
                
                # 初期化
                self._init_weights()
            
            def _init_weights(self):
                for m in self.modules():
                    if isinstance(m, torch.nn.Conv2d):
                        torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    elif isinstance(m, torch.nn.BatchNorm2d):
                        torch.nn.init.constant_(m.weight, 1)
                        torch.nn.init.constant_(m.bias, 0)
            
            def forward(self, x):
                identity = x
                
                out = self.conv1(x)
                out = self.bn1(out)
                out = self.relu(out)
                
                out = self.conv2(out)
                out = self.bn2(out)
                
                # 残差接続
                out += identity
                out = self.relu(out)
                
                return out
    
    def dynamic_module(self):
        """動的なモジュール構築"""
        class DynamicMLP(torch.nn.Module):
            def __init__(self, layer_sizes, activation='relu', dropout=0.0):
                super().__init__()
                
                # 活性化関数の辞書
                activation_fns = {
                    'relu': torch.nn.ReLU(),
                    'tanh': torch.nn.Tanh(),
                    'sigmoid': torch.nn.Sigmoid(),
                    'gelu': torch.nn.GELU()
                }
                
                layers = []
                for i in range(len(layer_sizes) - 1):
                    layers.append(torch.nn.Linear(layer_sizes[i], layer_sizes[i+1]))
                    
                    # 最後の層以外に活性化関数を追加
                    if i < len(layer_sizes) - 2:
                        layers.append(activation_fns[activation])
                        if dropout > 0:
                            layers.append(torch.nn.Dropout(dropout))
                
                self.model = torch.nn.Sequential(*layers)
            
            def forward(self, x):
                return self.model(x)
        
        # 使用例
        model = DynamicMLP([784, 256, 128, 10], activation='gelu', dropout=0.2)
        return model
```

##### 学習ループの詳細実装
```python
class TrainingLoop:
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'learning_rates': []
        }
    
    def train_epoch(self, dataloader, optimizer, criterion, scheduler=None):
        """1エポックの学習"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        # プログレスバー
        from tqdm import tqdm
        pbar = tqdm(dataloader, desc='Training')
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            # 勾配をゼロ化
            optimizer.zero_grad()
            
            # 順伝播
            output = self.model(data)
            loss = criterion(output, target)
            
            # 逆伝播
            loss.backward()
            
            # 勾配クリッピング（オプション）
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # パラメータ更新
            optimizer.step()
            
            # 統計情報の更新
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # プログレスバーの更新
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })
            
            # 学習率スケジューラ（バッチごと）
            if scheduler is not None and hasattr(scheduler, 'step_batch'):
                scheduler.step_batch(batch_idx)
        
        return total_loss / len(dataloader), correct / total
    
    def validate(self, dataloader, criterion):
        """検証"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
                total_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        return total_loss / len(dataloader), correct / total
    
    def fit(self, train_loader, val_loader, epochs, optimizer, criterion, scheduler=None):
        """完全な学習ループ"""
        best_val_acc = 0
        patience = 10
        patience_counter = 0
        
        for epoch in range(epochs):
            print(f'\nEpoch {epoch+1}/{epochs}')
            
            # 学習
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion, scheduler)
            
            # 検証
            val_loss, val_acc = self.validate(val_loader, criterion)
            
            # 学習率スケジューラ（エポックごと）
            if scheduler is not None and hasattr(scheduler, 'step'):
                scheduler.step(val_loss)
            
            # 履歴の保存
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(optimizer.param_groups[0]['lr'])
            
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
            print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
            
            # 早期終了
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # モデルの保存
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                }, 'best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'Early stopping at epoch {epoch+1}')
                    break
        
        self.plot_history()
    
    def plot_history(self):
        """学習履歴の可視化"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Loss
        axes[0, 0].plot(self.history['train_loss'], label='Train')
        axes[0, 0].plot(self.history['val_loss'], label='Validation')
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].legend()
        
        # Accuracy
        axes[0, 1].plot(self.history['train_acc'], label='Train')
        axes[0, 1].plot(self.history['val_acc'], label='Validation')
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].legend()
        
        # Learning Rate
        axes[1, 0].plot(self.history['learning_rates'])
        axes[1, 0].set_title('Learning Rate')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_yscale('log')
        
        # 過学習の検出
        train_acc = np.array(self.history['train_acc'])
        val_acc = np.array(self.history['val_acc'])
        overfit = train_acc - val_acc
        axes[1, 1].plot(overfit)
        axes[1, 1].set_title('過学習度 (Train Acc - Val Acc)')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].axhline(y=0, color='k', linestyle='--')
        
        plt.tight_layout()
        plt.show()
```

##### カスタム最適化とスケジューリング
```python
class OptimizationTechniques:
    def custom_optimizer(self):
        """カスタム最適化アルゴリズムの実装"""
        class SignSGD(torch.optim.Optimizer):
            """符号付きSGD：勾配の符号のみを使用"""
            def __init__(self, params, lr=0.01):
                defaults = dict(lr=lr)
                super().__init__(params, defaults)
            
            def step(self, closure=None):
                loss = None
                if closure is not None:
                    loss = closure()
                
                for group in self.param_groups:
                    for p in group['params']:
                        if p.grad is None:
                            continue
                        
                        # 勾配の符号のみを使用
                        p.data.add_(torch.sign(p.grad.data), alpha=-group['lr'])
                
                return loss
    
    def learning_rate_schedules(self):
        """様々な学習率スケジュール"""
        model = torch.nn.Linear(10, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # 1. ステップ減衰
        step_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=30, gamma=0.1
        )
        
        # 2. 指数減衰
        exp_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=0.95
        )
        
        # 3. コサインアニーリング
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=100, eta_min=0.00001
        )
        
        # 4. ウォームアップ付きコサインスケジュール
        class WarmupCosineSchedule(torch.optim.lr_scheduler._LRScheduler):
            def __init__(self, optimizer, warmup_steps, total_steps, min_lr=0):
                self.warmup_steps = warmup_steps
                self.total_steps = total_steps
                self.min_lr = min_lr
                super().__init__(optimizer)
            
            def get_lr(self):
                if self.last_epoch < self.warmup_steps:
                    # ウォームアップ期間
                    return [base_lr * self.last_epoch / self.warmup_steps 
                           for base_lr in self.base_lrs]
                else:
                    # コサイン減衰
                    progress = (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps)
                    return [self.min_lr + (base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
                           for base_lr in self.base_lrs]
        
        # スケジュールの可視化
        self.visualize_schedules([step_scheduler, exp_scheduler, cosine_scheduler])
    
    def gradient_accumulation_example(self):
        """
        勾配累積：大きなバッチサイズをシミュレート
        メモリ制約下での学習
        """
        model = torch.nn.Linear(100, 10)
        optimizer = torch.optim.Adam(model.parameters())
        
        # 設定
        true_batch_size = 128
        accumulation_steps = 4
        micro_batch_size = true_batch_size // accumulation_steps
        
        for epoch in range(10):
            for i, (data, target) in enumerate(dataloader):
                # マイクロバッチの処理
                data = data[:micro_batch_size]
                target = target[:micro_batch_size]
                
                output = model(data)
                loss = criterion(output, target)
                
                # 勾配の正規化
                loss = loss / accumulation_steps
                loss.backward()
                
                # accumulation_steps回ごとに更新
                if (i + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
```

## 第2部以降の詳細計画の続き...

（第2部〜第5部も同様に、各セクションを2倍以上に詳細化して記述）