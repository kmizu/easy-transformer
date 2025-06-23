# 必要な数学的基礎

## はじめに：プログラマーのための数学

「数学は苦手...」そんな声が聞こえてきそうですが、心配はいりません。プログラミング言語を実装できるあなたは、すでに多くの数学的概念を使いこなしています。再帰、グラフ理論、計算量解析...これらはすべて数学です。

この章では、Transformerを理解するために必要な数学を、プログラマーの視点から解説します。抽象的な定理ではなく、実装可能なコードとして数学を理解していきましょう。

## 3.1 線形代数の本質的理解

### ベクトル：データの基本単位

プログラミングでは配列やリストを日常的に扱います。ベクトルは、これらに「幾何学的な意味」を与えたものです：

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import torch.nn as nn
from typing import List, Tuple, Optional

class VectorBasics:
    """ベクトルの基本概念を実装で理解"""
    
    def __init__(self):
        self.examples = {
            "word_embedding": "単語の意味をベクトルで表現",
            "program_state": "プログラムの状態をベクトルで表現",
            "feature_vector": "特徴量をベクトルで表現"
        }
    
    def vector_as_array(self):
        """ベクトルは単なる数値の配列"""
        # プログラマーにとって馴染みのある表現
        array = [1.0, 2.0, 3.0]
        
        # NumPyベクトル
        np_vector = np.array(array)
        
        # PyTorchテンソル
        torch_vector = torch.tensor(array)
        
        print("=== ベクトルの表現 ===")
        print(f"Python list: {array}")
        print(f"NumPy array: {np_vector}")
        print(f"PyTorch tensor: {torch_vector}")
        
        return array, np_vector, torch_vector
    
    def vector_as_point(self):
        """ベクトルは空間上の点"""
        fig = plt.figure(figsize=(15, 5))
        
        # 2次元ベクトル
        ax1 = fig.add_subplot(131)
        vectors_2d = [
            ([0, 0], [3, 4], 'v1=(3,4)'),
            ([0, 0], [5, 2], 'v2=(5,2)'),
            ([0, 0], [-2, 3], 'v3=(-2,3)')
        ]
        
        for start, end, label in vectors_2d:
            ax1.arrow(start[0], start[1], 
                     end[0]-start[0], end[1]-start[1],
                     head_width=0.3, head_length=0.2, 
                     fc='blue', ec='blue')
            ax1.text(end[0]+0.2, end[1]+0.2, label)
        
        ax1.set_xlim(-3, 6)
        ax1.set_ylim(-1, 5)
        ax1.grid(True)
        ax1.set_title('2次元ベクトル')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        
        # 3次元ベクトル
        ax2 = fig.add_subplot(132, projection='3d')
        vectors_3d = [
            ([0, 0, 0], [3, 4, 2], 'v1'),
            ([0, 0, 0], [5, 2, 4], 'v2'),
            ([0, 0, 0], [-2, 3, 1], 'v3')
        ]
        
        for start, end, label in vectors_3d:
            ax2.quiver(start[0], start[1], start[2],
                      end[0], end[1], end[2],
                      arrow_length_ratio=0.1)
            ax2.text(end[0], end[1], end[2], label)
        
        ax2.set_xlim(-3, 6)
        ax2.set_ylim(-1, 5)
        ax2.set_zlim(0, 5)
        ax2.set_title('3次元ベクトル')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_zlabel('z')
        
        # 高次元ベクトルの可視化（次元削減）
        ax3 = fig.add_subplot(133)
        
        # 仮想的な100次元ベクトルを2次元に投影
        np.random.seed(42)
        high_dim_vectors = np.random.randn(50, 100)
        
        # PCA風の次元削減（簡易版）
        projection_matrix = np.random.randn(100, 2)
        projected = high_dim_vectors @ projection_matrix
        
        ax3.scatter(projected[:, 0], projected[:, 1], alpha=0.6)
        ax3.set_title('100次元ベクトルの2次元投影')
        ax3.set_xlabel('第1主成分')
        ax3.set_ylabel('第2主成分')
        ax3.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def vector_operations(self):
        """ベクトル演算の意味"""
        
        # 単語ベクトルの例
        word_vectors = {
            "king": np.array([1.0, 0.5, 0.2, 0.8]),
            "queen": np.array([0.9, 0.6, 0.8, 0.7]),
            "man": np.array([0.8, 0.3, 0.1, 0.9]),
            "woman": np.array([0.7, 0.4, 0.7, 0.8]),
            "prince": np.array([0.9, 0.4, 0.15, 0.75]),
            "princess": np.array([0.85, 0.45, 0.65, 0.65])
        }
        
        print("=== ベクトル演算の意味 ===")
        
        # 加法：概念の組み合わせ
        print("\n1. ベクトルの加法（概念の組み合わせ）")
        result = word_vectors["king"] - word_vectors["man"] + word_vectors["woman"]
        print(f"'king' - 'man' + 'woman' = {result}")
        print(f"'queen' = {word_vectors['queen']}")
        similarity = np.dot(result, word_vectors["queen"]) / (np.linalg.norm(result) * np.linalg.norm(word_vectors["queen"]))
        print(f"類似度: {similarity:.3f}")
        
        # スカラー倍：強度の調整
        print("\n2. スカラー倍（強度の調整）")
        strong_king = 2.0 * word_vectors["king"]
        weak_king = 0.5 * word_vectors["king"]
        print(f"通常の'king': {word_vectors['king']}")
        print(f"強い'king' (2x): {strong_king}")
        print(f"弱い'king' (0.5x): {weak_king}")
        
        # 内積：類似度の計算
        print("\n3. 内積（類似度の計算）")
        for word1 in ["king", "queen", "man"]:
            for word2 in ["queen", "woman", "prince"]:
                if word1 != word2:
                    dot_product = np.dot(word_vectors[word1], word_vectors[word2])
                    print(f"'{word1}' · '{word2}' = {dot_product:.3f}")
        
        # 可視化
        self.visualize_vector_operations(word_vectors)
    
    def visualize_vector_operations(self, word_vectors):
        """ベクトル演算の可視化"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 2次元に投影（最初の2成分）
        words = list(word_vectors.keys())
        vectors = np.array([word_vectors[w][:2] for w in words])
        
        # 1. ベクトル空間での単語の配置
        ax = axes[0]
        for i, (word, vec) in enumerate(zip(words, vectors)):
            ax.arrow(0, 0, vec[0], vec[1], 
                    head_width=0.05, head_length=0.05,
                    fc=f'C{i}', ec=f'C{i}')
            ax.text(vec[0]+0.05, vec[1]+0.05, word, fontsize=10)
        
        ax.set_xlim(-0.2, 1.2)
        ax.set_ylim(-0.1, 0.8)
        ax.grid(True)
        ax.set_title('単語ベクトル')
        ax.set_xlabel('次元1')
        ax.set_ylabel('次元2')
        
        # 2. ベクトル演算
        ax = axes[1]
        
        # king - man + woman ≈ queen
        king = word_vectors["king"][:2]
        man = word_vectors["man"][:2]
        woman = word_vectors["woman"][:2]
        queen = word_vectors["queen"][:2]
        
        # 演算の可視化
        ax.arrow(0, 0, king[0], king[1], 
                head_width=0.05, head_length=0.05,
                fc='red', ec='red', label='king')
        
        # king - man
        diff = king - man
        ax.arrow(king[0], king[1], -man[0], -man[1],
                head_width=0.05, head_length=0.05,
                fc='blue', ec='blue', linestyle='--', alpha=0.5)
        
        # + woman
        result = diff + woman
        ax.arrow(diff[0], diff[1], woman[0], woman[1],
                head_width=0.05, head_length=0.05,
                fc='green', ec='green', alpha=0.5)
        
        # 結果とqueenの比較
        ax.arrow(0, 0, result[0], result[1],
                head_width=0.05, head_length=0.05,
                fc='purple', ec='purple', linewidth=2, label='result')
        ax.arrow(0, 0, queen[0], queen[1],
                head_width=0.05, head_length=0.05,
                fc='orange', ec='orange', linestyle=':', linewidth=2, label='queen')
        
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(-0.5, 1.0)
        ax.grid(True)
        ax.set_title('king - man + woman ≈ queen')
        ax.legend()
        
        # 3. 内積と角度
        ax = axes[2]
        
        # いくつかのベクトルペアの角度を可視化
        pairs = [("king", "queen"), ("king", "woman"), ("man", "woman")]
        
        for i, (w1, w2) in enumerate(pairs):
            v1 = word_vectors[w1][:2]
            v2 = word_vectors[w2][:2]
            
            # ベクトルを描画
            ax.arrow(0, 0, v1[0], v1[1],
                    head_width=0.05, head_length=0.05,
                    fc=f'C{i*2}', ec=f'C{i*2}', alpha=0.7)
            ax.arrow(0, 0, v2[0], v2[1],
                    head_width=0.05, head_length=0.05,
                    fc=f'C{i*2+1}', ec=f'C{i*2+1}', alpha=0.7)
            
            # 角度を計算
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            angle = np.arccos(np.clip(cos_angle, -1, 1))
            
            # 角度を表示
            ax.text(0.1, 0.7 - i*0.15, 
                   f'{w1}-{w2}: {np.degrees(angle):.1f}°',
                   fontsize=10)
        
        ax.set_xlim(-0.2, 1.2)
        ax.set_ylim(-0.1, 0.8)
        ax.grid(True)
        ax.set_title('ベクトル間の角度（類似度）')
        
        plt.tight_layout()
        plt.show()

class MatrixOperations:
    """行列演算の直感的理解"""
    
    def __init__(self):
        self.examples = {
            "linear_transform": "線形変換",
            "weight_matrix": "重み行列",
            "attention_matrix": "注意行列"
        }
    
    def matrix_as_transformation(self):
        """行列を変換として理解"""
        
        print("=== 行列は変換 ===")
        
        # 基本的な変換行列
        transformations = {
            "恒等変換": np.array([[1, 0], [0, 1]]),
            "拡大": np.array([[2, 0], [0, 2]]),
            "回転(45°)": np.array([[np.cos(np.pi/4), -np.sin(np.pi/4)],
                                   [np.sin(np.pi/4), np.cos(np.pi/4)]]),
            "せん断": np.array([[1, 0.5], [0, 1]]),
            "反射": np.array([[1, 0], [0, -1]])
        }
        
        # 元のベクトル集合（正方形）
        square = np.array([
            [0, 0], [1, 0], [1, 1], [0, 1], [0, 0]
        ]).T
        
        # 可視化
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, (name, matrix) in enumerate(transformations.items()):
            ax = axes[i]
            
            # 元の形
            ax.plot(square[0], square[1], 'b-', linewidth=2, 
                   label='元の形', alpha=0.5)
            
            # 変換後
            transformed = matrix @ square
            ax.plot(transformed[0], transformed[1], 'r-', linewidth=2,
                   label='変換後')
            
            # 行列を表示
            ax.text(0.5, 1.5, f'{name}\n{matrix}', 
                   ha='center', va='bottom', fontsize=10,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
            
            ax.set_xlim(-2, 2)
            ax.set_ylim(-2, 2)
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='k', linewidth=0.5)
            ax.axvline(x=0, color='k', linewidth=0.5)
            ax.legend()
            ax.set_aspect('equal')
        
        # 合成変換
        ax = axes[5]
        
        # 回転してから拡大
        rotate = transformations["回転(45°)"]
        scale = transformations["拡大"]
        composed = scale @ rotate  # 行列の積は変換の合成
        
        transformed = composed @ square
        ax.plot(square[0], square[1], 'b-', linewidth=2, 
               label='元の形', alpha=0.5)
        ax.plot(transformed[0], transformed[1], 'g-', linewidth=2,
               label='回転→拡大')
        
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.axvline(x=0, color='k', linewidth=0.5)
        ax.legend()
        ax.set_aspect('equal')
        ax.set_title('合成変換')
        
        plt.tight_layout()
        plt.show()
    
    def matrix_multiplication_intuition(self):
        """行列積の直感的理解"""
        
        print("\n=== 行列積の意味 ===")
        
        # ニューラルネットワークの文脈で
        class SimpleLayer:
            def __init__(self, input_dim, output_dim):
                # 重み行列
                self.W = np.random.randn(output_dim, input_dim) * 0.1
                self.b = np.zeros(output_dim)
            
            def forward(self, x):
                """
                x: [batch_size, input_dim]
                W: [output_dim, input_dim]
                output: [batch_size, output_dim]
                """
                # 行列積の各要素は内積
                output = x @ self.W.T + self.b
                return output
            
            def visualize_computation(self, x):
                """計算過程を可視化"""
                batch_size = x.shape[0]
                
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                # 1. 入力
                ax = axes[0]
                im1 = ax.imshow(x, cmap='Blues', aspect='auto')
                ax.set_title('入力 x')
                ax.set_ylabel('バッチ')
                ax.set_xlabel('入力次元')
                plt.colorbar(im1, ax=ax)
                
                # 2. 重み行列
                ax = axes[1]
                im2 = ax.imshow(self.W, cmap='RdBu', aspect='auto')
                ax.set_title('重み行列 W')
                ax.set_ylabel('出力次元')
                ax.set_xlabel('入力次元')
                plt.colorbar(im2, ax=ax)
                
                # 3. 出力
                ax = axes[2]
                output = self.forward(x)
                im3 = ax.imshow(output, cmap='Greens', aspect='auto')
                ax.set_title('出力 y = xW^T + b')
                ax.set_ylabel('バッチ')
                ax.set_xlabel('出力次元')
                plt.colorbar(im3, ax=ax)
                
                plt.tight_layout()
                plt.show()
                
                # 1つの出力要素の計算を詳細に表示
                print("\n出力の1要素の計算詳細:")
                print(f"output[0,0] = Σ(x[0,i] * W[0,i]) + b[0]")
                dot_product = sum(x[0, i] * self.W[0, i] for i in range(x.shape[1]))
                print(f"= {dot_product:.3f} + {self.b[0]:.3f}")
                print(f"= {dot_product + self.b[0]:.3f}")
                print(f"実際の値: {output[0, 0]:.3f}")
        
        # デモ
        layer = SimpleLayer(input_dim=5, output_dim=3)
        x = np.random.randn(4, 5)  # バッチサイズ4、入力次元5
        layer.visualize_computation(x)
    
    def eigenvalues_and_eigenvectors(self):
        """固有値と固有ベクトル：行列の「本質」"""
        
        print("\n=== 固有値と固有ベクトル ===")
        print("固有ベクトル v に対して: Av = λv")
        print("つまり、行列Aは固有ベクトルの方向を変えない（大きさだけ変える）")
        
        # 例：共分散行列（データの主要な方向を表す）
        # データ生成
        np.random.seed(42)
        mean = [2, 3]
        cov = [[2, 1.5], [1.5, 1]]
        data = np.random.multivariate_normal(mean, cov, 1000)
        
        # 共分散行列を計算
        data_centered = data - np.mean(data, axis=0)
        cov_matrix = np.cov(data_centered.T)
        
        # 固有値と固有ベクトルを計算
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        
        # 可視化
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 1. データと主成分
        ax = axes[0]
        ax.scatter(data[:, 0], data[:, 1], alpha=0.5, s=10)
        
        # 固有ベクトルを表示（主成分の方向）
        center = np.mean(data, axis=0)
        for i in range(2):
            # 固有値の大きさに比例した長さで表示
            scale = np.sqrt(eigenvalues[i]) * 2
            eigvec = eigenvectors[:, i]
            ax.arrow(center[0], center[1],
                    eigvec[0] * scale, eigvec[1] * scale,
                    head_width=0.2, head_length=0.1,
                    fc=f'C{i}', ec=f'C{i}', linewidth=2,
                    label=f'固有値 {eigenvalues[i]:.2f}')
        
        ax.set_title('データの主成分（固有ベクトル）')
        ax.legend()
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # 2. 固有値の解釈
        ax = axes[1]
        ax.bar([0, 1], eigenvalues)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['第1主成分', '第2主成分'])
        ax.set_ylabel('固有値（分散）')
        ax.set_title('各主成分の重要度')
        
        # 寄与率を表示
        total_var = sum(eigenvalues)
        for i, (x, y) in enumerate(zip([0, 1], eigenvalues)):
            ratio = y / total_var * 100
            ax.text(x, y + 0.05, f'{ratio:.1f}%', ha='center')
        
        plt.tight_layout()
        plt.show()
        
        print(f"\n共分散行列:\n{cov_matrix}")
        print(f"\n固有値: {eigenvalues}")
        print(f"固有ベクトル:\n{eigenvectors}")
        print(f"\n第1主成分の寄与率: {eigenvalues[0]/sum(eigenvalues)*100:.1f}%")

class AttentionMathematics:
    """Attention機構の数学的基礎"""
    
    def __init__(self, d_model=64):
        self.d_model = d_model
        self.scale = np.sqrt(d_model)
    
    def dot_product_attention(self):
        """内積注意の数学"""
        
        print("=== 内積注意（Dot Product Attention）===")
        
        # 簡単な例で説明
        seq_len = 5
        d_k = 4
        
        # Query, Key, Value
        Q = np.random.randn(seq_len, d_k)
        K = np.random.randn(seq_len, d_k)
        V = np.random.randn(seq_len, d_k)
        
        # ステップ1: QとKの内積でスコアを計算
        scores = Q @ K.T  # [seq_len, seq_len]
        print(f"1. スコア行列の形状: {scores.shape}")
        print(f"   scores[i,j] = Q[i] · K[j] （クエリiとキーjの類似度）")
        
        # ステップ2: スケーリング
        scaled_scores = scores / self.scale
        print(f"\n2. スケーリング: 除算 by sqrt({d_k}) = {self.scale:.2f}")
        print(f"   理由: 内積の値が大きくなりすぎるのを防ぐ")
        
        # ステップ3: Softmax
        attention_weights = self.softmax(scaled_scores)
        print(f"\n3. Softmax: 各行の和が1になる確率分布に変換")
        print(f"   attention_weights[i] = softmax(scaled_scores[i])")
        
        # ステップ4: 重み付き和
        output = attention_weights @ V
        print(f"\n4. 出力: 重み付き和")
        print(f"   output[i] = Σ_j (attention_weights[i,j] * V[j])")
        
        # 可視化
        self.visualize_attention_computation(Q, K, V, scores, attention_weights, output)
        
        return output, attention_weights
    
    def softmax(self, x):
        """Softmax関数の実装"""
        # 数値的安定性のため最大値を引く
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def visualize_attention_computation(self, Q, K, V, scores, weights, output):
        """Attention計算の可視化"""
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Query
        ax = axes[0, 0]
        im = ax.imshow(Q, cmap='Blues', aspect='auto')
        ax.set_title('Query (Q)')
        ax.set_ylabel('位置')
        ax.set_xlabel('次元')
        plt.colorbar(im, ax=ax)
        
        # Key
        ax = axes[0, 1]
        im = ax.imshow(K, cmap='Oranges', aspect='auto')
        ax.set_title('Key (K)')
        ax.set_ylabel('位置')
        ax.set_xlabel('次元')
        plt.colorbar(im, ax=ax)
        
        # Value
        ax = axes[0, 2]
        im = ax.imshow(V, cmap='Greens', aspect='auto')
        ax.set_title('Value (V)')
        ax.set_ylabel('位置')
        ax.set_xlabel('次元')
        plt.colorbar(im, ax=ax)
        
        # スコア行列
        ax = axes[1, 0]
        im = ax.imshow(scores, cmap='RdBu', aspect='auto')
        ax.set_title('スコア (Q·K^T)')
        ax.set_ylabel('Query位置')
        ax.set_xlabel('Key位置')
        plt.colorbar(im, ax=ax)
        
        # Attention重み
        ax = axes[1, 1]
        im = ax.imshow(weights, cmap='hot', aspect='auto')
        ax.set_title('Attention重み (Softmax後)')
        ax.set_ylabel('Query位置')
        ax.set_xlabel('Key位置')
        plt.colorbar(im, ax=ax)
        
        # 出力
        ax = axes[1, 2]
        im = ax.imshow(output, cmap='Purples', aspect='auto')
        ax.set_title('出力 (重み付きValue)')
        ax.set_ylabel('位置')
        ax.set_xlabel('次元')
        plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        plt.show()
    
    def scaled_attention_importance(self):
        """スケーリングの重要性を実証"""
        
        print("\n=== スケーリングの重要性 ===")
        
        d_k_values = [4, 64, 512]
        fig, axes = plt.subplots(1, len(d_k_values), figsize=(15, 4))
        
        for idx, d_k in enumerate(d_k_values):
            ax = axes[idx]
            
            # ランダムなベクトル
            q = np.random.randn(d_k)
            k = np.random.randn(d_k)
            
            # 内積
            dot_product = np.dot(q, k)
            
            # Softmaxの入力値の範囲
            x_range = np.linspace(-20, 20, 1000)
            
            # スケーリングなし
            scores_no_scale = x_range
            probs_no_scale = self.softmax(np.array([dot_product, 0]))
            
            # スケーリングあり
            scale = np.sqrt(d_k)
            scores_scaled = x_range / scale
            probs_scaled = self.softmax(np.array([dot_product / scale, 0]))
            
            # Softmax関数をプロット
            y_no_scale = np.exp(x_range) / (np.exp(x_range) + 1)
            y_scaled = np.exp(x_range / scale) / (np.exp(x_range / scale) + 1)
            
            ax.plot(x_range, y_no_scale, 'r-', label='スケーリングなし', alpha=0.7)
            ax.plot(x_range, y_scaled, 'b-', label=f'スケーリングあり (÷√{d_k})', alpha=0.7)
            
            # 実際の内積値での確率を表示
            ax.axvline(x=dot_product, color='red', linestyle='--', alpha=0.5)
            ax.axvline(x=dot_product/scale, color='blue', linestyle='--', alpha=0.5)
            
            ax.set_title(f'd_k = {d_k}')
            ax.set_xlabel('スコア')
            ax.set_ylabel('Softmax出力')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 確率値を表示
            ax.text(0.5, 0.9, f'内積: {dot_product:.2f}', 
                   transform=ax.transAxes, ha='center')
            ax.text(0.5, 0.85, f'P(スケールなし): {probs_no_scale[0]:.3f}', 
                   transform=ax.transAxes, ha='center', color='red')
            ax.text(0.5, 0.8, f'P(スケールあり): {probs_scaled[0]:.3f}', 
                   transform=ax.transAxes, ha='center', color='blue')
        
        plt.tight_layout()
        plt.show()
        
        print("\n観察:")
        print("- d_kが大きいほど、内積の値が大きくなる傾向")
        print("- スケーリングなしでは、Softmaxが飽和して勾配消失")
        print("- スケーリングにより、適切な勾配が保たれる")

## 3.2 確率・統計の実践的理解

### 確率分布：不確実性の表現

```python
class ProbabilityDistributions:
    """確率分布の直感的理解"""
    
    def __init__(self):
        self.examples = {
            "discrete": "離散分布（単語の出現確率など）",
            "continuous": "連続分布（パラメータの分布など）",
            "multivariate": "多変量分布（埋め込みベクトルの分布など）"
        }
    
    def discrete_distributions(self):
        """離散確率分布"""
        
        print("=== 離散確率分布 ===")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. カテゴリカル分布（単語の出現確率）
        ax = axes[0, 0]
        words = ['the', 'is', 'a', 'of', 'and', 'to', 'in', 'that']
        probs = [0.25, 0.15, 0.12, 0.10, 0.08, 0.08, 0.07, 0.05]
        remaining = 1 - sum(probs)
        words.append('others')
        probs.append(remaining)
        
        ax.bar(words, probs, color='skyblue')
        ax.set_title('カテゴリカル分布（単語の出現確率）')
        ax.set_ylabel('確率')
        ax.set_xticklabels(words, rotation=45)
        
        # エントロピーを計算
        entropy = -sum(p * np.log(p) for p in probs if p > 0)
        ax.text(0.7, 0.9, f'エントロピー: {entropy:.2f}', 
               transform=ax.transAxes,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow"))
        
        # 2. 二項分布（成功/失敗の回数）
        ax = axes[0, 1]
        n = 20  # 試行回数
        p = 0.3  # 成功確率
        x = np.arange(0, n+1)
        
        from scipy import stats
        pmf = stats.binom.pmf(x, n, p)
        
        ax.bar(x, pmf, color='lightgreen')
        ax.set_title(f'二項分布 (n={n}, p={p})')
        ax.set_xlabel('成功回数')
        ax.set_ylabel('確率')
        
        # 期待値と分散
        mean = n * p
        var = n * p * (1 - p)
        ax.axvline(x=mean, color='red', linestyle='--', 
                  label=f'期待値: {mean:.1f}')
        ax.legend()
        
        # 3. ポアソン分布（稀な事象の発生回数）
        ax = axes[1, 0]
        lambda_ = 3  # 平均発生率
        x = np.arange(0, 15)
        pmf = stats.poisson.pmf(x, lambda_)
        
        ax.bar(x, pmf, color='lightcoral')
        ax.set_title(f'ポアソン分布 (λ={lambda_})')
        ax.set_xlabel('発生回数')
        ax.set_ylabel('確率')
        
        # 4. 実際のテキストデータでの単語長分布
        ax = axes[1, 1]
        text = """Transformers have revolutionized natural language processing 
                  by introducing self-attention mechanisms that capture long-range 
                  dependencies in text without recurrence or convolution."""
        
        words = text.split()
        lengths = [len(word) for word in words]
        
        unique_lengths, counts = np.unique(lengths, return_counts=True)
        probs = counts / len(lengths)
        
        ax.bar(unique_lengths, probs, color='mediumpurple')
        ax.set_title('実際のテキストの単語長分布')
        ax.set_xlabel('単語長')
        ax.set_ylabel('相対頻度')
        
        # 統計量
        mean_length = np.mean(lengths)
        ax.axvline(x=mean_length, color='red', linestyle='--',
                  label=f'平均: {mean_length:.1f}')
        ax.legend()
        
        plt.tight_layout()
        plt.show()
    
    def continuous_distributions(self):
        """連続確率分布"""
        
        print("\n=== 連続確率分布 ===")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. 正規分布（ガウス分布）
        ax = axes[0, 0]
        x = np.linspace(-4, 4, 100)
        
        # 異なるパラメータでの正規分布
        params = [(0, 1, 'μ=0, σ=1'), (0, 2, 'μ=0, σ=2'), (2, 1, 'μ=2, σ=1')]
        
        for mu, sigma, label in params:
            y = stats.norm.pdf(x, mu, sigma)
            ax.plot(x, y, label=label, linewidth=2)
        
        ax.set_title('正規分布')
        ax.set_xlabel('x')
        ax.set_ylabel('確率密度')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. パラメータ初期化での利用
        ax = axes[0, 1]
        
        # Xavier/He初期化の比較
        fan_in = 100
        fan_out = 50
        
        # Xavier初期化（活性化関数: tanh, sigmoid）
        xavier_std = np.sqrt(2 / (fan_in + fan_out))
        
        # He初期化（活性化関数: ReLU）
        he_std = np.sqrt(2 / fan_in)
        
        x = np.linspace(-0.5, 0.5, 1000)
        
        xavier_dist = stats.norm.pdf(x, 0, xavier_std)
        he_dist = stats.norm.pdf(x, 0, he_std)
        uniform_dist = stats.uniform.pdf(x, -0.5, 1)
        
        ax.plot(x, xavier_dist, label=f'Xavier (σ={xavier_std:.3f})')
        ax.plot(x, he_dist, label=f'He (σ={he_std:.3f})')
        ax.plot(x, uniform_dist * 2, label='一様分布', linestyle='--')
        
        ax.set_title('パラメータ初期化の分布')
        ax.set_xlabel('重みの値')
        ax.set_ylabel('確率密度')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. t分布（ロバストな推定）
        ax = axes[1, 0]
        x = np.linspace(-4, 4, 100)
        
        # 自由度による形状の変化
        dfs = [1, 3, 10, 30]
        
        for df in dfs:
            y = stats.t.pdf(x, df)
            ax.plot(x, y, label=f'df={df}', linewidth=2)
        
        # 正規分布と比較
        y_norm = stats.norm.pdf(x, 0, 1)
        ax.plot(x, y_norm, 'k--', label='正規分布', linewidth=2)
        
        ax.set_title('t分布（裾が厚い分布）')
        ax.set_xlabel('x')
        ax.set_ylabel('確率密度')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. 実際のモデルパラメータの分布
        ax = axes[1, 1]
        
        # 仮想的な学習済みモデルのパラメータ
        np.random.seed(42)
        
        # 初期化時
        initial_params = np.random.normal(0, 0.1, 10000)
        
        # 学習後（仮想的）
        trained_params = np.concatenate([
            np.random.normal(-0.5, 0.05, 3000),  # 負の重み
            np.random.normal(0, 0.02, 4000),     # ゼロ付近
            np.random.normal(0.5, 0.05, 3000)    # 正の重み
        ])
        
        ax.hist(initial_params, bins=50, alpha=0.5, density=True, 
               label='初期化時', color='blue')
        ax.hist(trained_params, bins=50, alpha=0.5, density=True,
               label='学習後', color='red')
        
        ax.set_title('モデルパラメータの分布変化')
        ax.set_xlabel('パラメータ値')
        ax.set_ylabel('密度')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def softmax_as_probability(self):
        """Softmax関数と確率分布"""
        
        print("\n=== Softmax関数：スコアから確率へ ===")
        
        def softmax(x, temperature=1.0):
            """温度付きSoftmax"""
            x = x / temperature
            exp_x = np.exp(x - np.max(x))
            return exp_x / np.sum(exp_x)
        
        # ロジット（スコア）
        logits = np.array([2.0, 1.0, 0.1, -1.0, -2.0])
        labels = ['very_pos', 'pos', 'neutral', 'neg', 'very_neg']
        
        # 異なる温度でのSoftmax
        temperatures = [0.1, 0.5, 1.0, 2.0, 5.0]
        
        fig, axes = plt.subplots(1, len(temperatures), figsize=(20, 4))
        
        for ax, temp in zip(axes, temperatures):
            probs = softmax(logits, temperature=temp)
            
            bars = ax.bar(labels, probs, color='lightblue')
            ax.set_title(f'Temperature = {temp}')
            ax.set_ylabel('確率')
            ax.set_ylim(0, 1)
            
            # 値を表示
            for bar, prob in zip(bars, probs):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{prob:.3f}', ha='center', va='bottom')
            
            # エントロピーを計算
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            ax.text(0.5, 0.9, f'H = {entropy:.2f}',
                   transform=ax.transAxes, ha='center',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow"))
        
        plt.tight_layout()
        plt.show()
        
        print("\n温度パラメータの効果:")
        print("- 低温（T < 1）: より確信的な分布（エントロピー低）")
        print("- 高温（T > 1）: より一様な分布（エントロピー高）")
        print("- T = 1: 標準的なSoftmax")
    
    def cross_entropy_loss(self):
        """交差エントロピー損失の理解"""
        
        print("\n=== 交差エントロピー損失 ===")
        
        # 真の分布（one-hot）
        true_dist = np.array([0, 0, 1, 0, 0])  # クラス2が正解
        
        # 予測分布の例
        predictions = [
            np.array([0.1, 0.1, 0.6, 0.1, 0.1]),  # 良い予測
            np.array([0.2, 0.2, 0.2, 0.2, 0.2]),  # 不確実な予測
            np.array([0.6, 0.1, 0.1, 0.1, 0.1]),  # 悪い予測
        ]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for ax, pred in zip(axes, predictions):
            # 交差エントロピーを計算
            ce_loss = -np.sum(true_dist * np.log(pred + 1e-10))
            
            # 可視化
            x = np.arange(len(true_dist))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, true_dist, width, 
                           label='真の分布', alpha=0.7)
            bars2 = ax.bar(x + width/2, pred, width,
                           label='予測分布', alpha=0.7)
            
            ax.set_title(f'交差エントロピー: {ce_loss:.3f}')
            ax.set_xlabel('クラス')
            ax.set_ylabel('確率')
            ax.set_xticks(x)
            ax.legend()
            
            # 正解クラスの予測確率を強調
            correct_class = np.argmax(true_dist)
            ax.text(correct_class, pred[correct_class] + 0.05,
                   f'{pred[correct_class]:.2f}',
                   ha='center', fontweight='bold', color='red')
        
        plt.tight_layout()
        plt.show()
        
        print("\n交差エントロピーの式:")
        print("H(p, q) = -Σ p(x) log q(x)")
        print("- p: 真の分布（通常one-hot）")
        print("- q: 予測分布")
        print("- 正解クラスの予測確率が高いほど損失が小さい")

class StatisticalConcepts:
    """統計的概念の実装"""
    
    def __init__(self):
        pass
    
    def expectation_and_variance(self):
        """期待値と分散"""
        
        print("=== 期待値と分散 ===")
        
        # Layer Normalizationでの使用例
        class LayerNorm:
            def __init__(self, eps=1e-6):
                self.eps = eps
            
            def forward(self, x):
                """
                x: [batch_size, seq_len, d_model]
                """
                # 最後の次元で統計量を計算
                mean = np.mean(x, axis=-1, keepdims=True)
                var = np.var(x, axis=-1, keepdims=True)
                
                # 正規化
                x_normalized = (x - mean) / np.sqrt(var + self.eps)
                
                return x_normalized, mean, var
            
            def visualize_normalization(self, x):
                """正規化の効果を可視化"""
                x_norm, mean, var = self.forward(x)
                
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                
                # 元の分布
                ax = axes[0, 0]
                ax.hist(x.flatten(), bins=50, alpha=0.7, density=True)
                ax.set_title('元の分布')
                ax.set_xlabel('値')
                ax.set_ylabel('密度')
                
                orig_mean = np.mean(x)
                orig_std = np.std(x)
                ax.axvline(x=orig_mean, color='red', linestyle='--',
                          label=f'μ={orig_mean:.2f}')
                ax.axvline(x=orig_mean + orig_std, color='green', linestyle='--',
                          label=f'σ={orig_std:.2f}')
                ax.axvline(x=orig_mean - orig_std, color='green', linestyle='--')
                ax.legend()
                
                # 正規化後の分布
                ax = axes[0, 1]
                ax.hist(x_norm.flatten(), bins=50, alpha=0.7, density=True)
                ax.set_title('正規化後の分布')
                ax.set_xlabel('値')
                ax.set_ylabel('密度')
                
                norm_mean = np.mean(x_norm)
                norm_std = np.std(x_norm)
                ax.axvline(x=norm_mean, color='red', linestyle='--',
                          label=f'μ={norm_mean:.2f}')
                ax.axvline(x=norm_mean + norm_std, color='green', linestyle='--',
                          label=f'σ={norm_std:.2f}')
                ax.axvline(x=norm_mean - norm_std, color='green', linestyle='--')
                ax.legend()
                
                # 各サンプルの統計量
                ax = axes[1, 0]
                sample_means = mean.flatten()
                sample_vars = var.flatten()
                
                scatter = ax.scatter(sample_means, sample_vars, alpha=0.6)
                ax.set_xlabel('平均')
                ax.set_ylabel('分散')
                ax.set_title('各サンプルの統計量')
                ax.grid(True, alpha=0.3)
                
                # 正規化の安定性
                ax = axes[1, 1]
                
                # 異なるスケールのデータで比較
                scales = [0.1, 1.0, 10.0, 100.0]
                colors = plt.cm.viridis(np.linspace(0, 1, len(scales)))
                
                for scale, color in zip(scales, colors):
                    x_scaled = x * scale
                    x_scaled_norm, _, _ = self.forward(x_scaled)
                    
                    ax.hist(x_scaled_norm.flatten(), bins=30, alpha=0.5,
                           density=True, color=color,
                           label=f'scale={scale}')
                
                ax.set_title('異なるスケールでも同じ分布に正規化')
                ax.set_xlabel('正規化後の値')
                ax.set_ylabel('密度')
                ax.legend()
                
                plt.tight_layout()
                plt.show()
        
        # デモ
        ln = LayerNorm()
        
        # バッチデータ（異なる統計量を持つ）
        batch_size, seq_len, d_model = 32, 10, 64
        x = np.random.randn(batch_size, seq_len, d_model)
        
        # 一部のサンプルに異なるスケールを適用
        x[10:15] *= 5.0  # 大きな値
        x[20:25] *= 0.1  # 小さな値
        
        ln.visualize_normalization(x)
    
    def correlation_and_covariance(self):
        """相関と共分散"""
        
        print("\n=== 相関と共分散 ===")
        
        # Attention での相関の重要性
        def attention_as_correlation():
            """Attentionを相関として理解"""
            
            # 例：文中の単語間の関係
            sentence = "The cat sat on the mat"
            words = sentence.split()
            n_words = len(words)
            
            # 仮想的な単語埋め込み
            np.random.seed(42)
            d_model = 64
            embeddings = np.random.randn(n_words, d_model)
            
            # 特定の単語を類似させる
            embeddings[1] *= 0.8  # cat
            embeddings[5] *= 0.8  # mat（catと韻を踏む）
            
            # 相関行列を計算
            correlation_matrix = np.corrcoef(embeddings)
            
            # Attention スコア（正規化前）
            attention_scores = embeddings @ embeddings.T
            
            # 可視化
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # 相関行列
            ax = axes[0]
            im = ax.imshow(correlation_matrix, cmap='RdBu', vmin=-1, vmax=1)
            ax.set_xticks(range(n_words))
            ax.set_yticks(range(n_words))
            ax.set_xticklabels(words, rotation=45)
            ax.set_yticklabels(words)
            ax.set_title('単語埋め込みの相関行列')
            plt.colorbar(im, ax=ax)
            
            # Attentionスコア
            ax = axes[1]
            im = ax.imshow(attention_scores, cmap='hot')
            ax.set_xticks(range(n_words))
            ax.set_yticks(range(n_words))
            ax.set_xticklabels(words, rotation=45)
            ax.set_yticklabels(words)
            ax.set_title('Attentionスコア（内積）')
            plt.colorbar(im, ax=ax)
            
            plt.tight_layout()
            plt.show()
            
            print("観察:")
            print("- 相関が高い単語ペアは、Attentionスコアも高い")
            print("- 'cat'と'mat'のように韻を踏む単語は相関が高い")
            print("- Attentionは単語間の意味的類似性を捉える")
        
        attention_as_correlation()

## 3.3 微分とバックプロパゲーション

### 勾配：関数の変化の方向

```python
class GradientConcepts:
    """勾配の概念を実装で理解"""
    
    def __init__(self):
        self.history = []
    
    def gradient_visualization(self):
        """勾配の可視化"""
        
        print("=== 勾配：最適化の方向 ===")
        
        # 1次元の例
        def f1d(x):
            return x**2 - 4*x + 3
        
        def df1d_dx(x):
            return 2*x - 4
        
        # 2次元の例（損失関数の景観）
        def f2d(x, y):
            return (x - 2)**2 + (y - 3)**2 + 0.5*x*y
        
        def grad_f2d(x, y):
            df_dx = 2*(x - 2) + 0.5*y
            df_dy = 2*(y - 3) + 0.5*x
            return df_dx, df_dy
        
        fig = plt.figure(figsize=(15, 5))
        
        # 1次元関数と勾配
        ax1 = fig.add_subplot(131)
        x = np.linspace(-2, 6, 100)
        y = f1d(x)
        ax1.plot(x, y, 'b-', linewidth=2, label='f(x) = x² - 4x + 3')
        
        # いくつかの点での勾配
        sample_points = [-1, 0, 2, 4, 5]
        for x_point in sample_points:
            y_point = f1d(x_point)
            grad = df1d_dx(x_point)
            
            # 接線を描画
            tangent_x = np.linspace(x_point - 0.5, x_point + 0.5, 10)
            tangent_y = y_point + grad * (tangent_x - x_point)
            ax1.plot(tangent_x, tangent_y, 'r-', linewidth=1, alpha=0.7)
            
            # 勾配の方向を矢印で表示
            arrow_scale = 0.3
            ax1.arrow(x_point, y_point, 
                     arrow_scale, arrow_scale * grad,
                     head_width=0.1, head_length=0.05,
                     fc='green', ec='green')
            
            ax1.plot(x_point, y_point, 'ro', markersize=8)
            ax1.text(x_point, y_point - 0.5, f'∇={grad:.1f}',
                    ha='center', fontsize=8)
        
        ax1.set_xlabel('x')
        ax1.set_ylabel('f(x)')
        ax1.set_title('1次元関数の勾配')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 2次元関数の等高線と勾配
        ax2 = fig.add_subplot(132)
        x = np.linspace(-1, 5, 100)
        y = np.linspace(-1, 7, 100)
        X, Y = np.meshgrid(x, y)
        Z = f2d(X, Y)
        
        # 等高線
        contour = ax2.contour(X, Y, Z, levels=20, alpha=0.6)
        ax2.clabel(contour, inline=True, fontsize=8)
        
        # 勾配ベクトル場
        x_sparse = np.linspace(-1, 5, 10)
        y_sparse = np.linspace(-1, 7, 10)
        X_sparse, Y_sparse = np.meshgrid(x_sparse, y_sparse)
        
        # 各点での勾配
        U = np.zeros_like(X_sparse)
        V = np.zeros_like(Y_sparse)
        
        for i in range(len(x_sparse)):
            for j in range(len(y_sparse)):
                grad_x, grad_y = grad_f2d(X_sparse[j, i], Y_sparse[j, i])
                U[j, i] = -grad_x  # 負の勾配（降下方向）
                V[j, i] = -grad_y
        
        # 勾配ベクトルを描画
        ax2.quiver(X_sparse, Y_sparse, U, V, color='red', alpha=0.7)
        
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_title('2次元関数の勾配ベクトル場')
        ax2.set_aspect('equal')
        
        # 最適化の軌跡
        ax3 = fig.add_subplot(133)
        
        # 勾配降下法の実装
        def gradient_descent(start_point, learning_rate=0.1, n_steps=50):
            trajectory = [start_point]
            point = np.array(start_point)
            
            for _ in range(n_steps):
                grad = np.array(grad_f2d(point[0], point[1]))
                point = point - learning_rate * grad
                trajectory.append(point.copy())
            
            return np.array(trajectory)
        
        # 異なる開始点からの軌跡
        start_points = [(-0.5, 6), (4.5, 0), (4, 6)]
        colors = ['blue', 'green', 'purple']
        
        # 等高線を再描画
        contour = ax3.contour(X, Y, Z, levels=20, alpha=0.3)
        
        for start, color in zip(start_points, colors):
            trajectory = gradient_descent(start, learning_rate=0.1)
            ax3.plot(trajectory[:, 0], trajectory[:, 1], 
                    f'{color[0]}-', linewidth=2, label=f'開始: {start}')
            ax3.plot(trajectory[0, 0], trajectory[0, 1], 
                    f'{color[0]}o', markersize=10)
            ax3.plot(trajectory[-1, 0], trajectory[-1, 1], 
                    f'{color[0]}*', markersize=15)
        
        # 最小値の位置
        ax3.plot(2, 3, 'r*', markersize=20, label='最小値')
        
        ax3.set_xlabel('x')
        ax3.set_ylabel('y')
        ax3.set_title('勾配降下法の軌跡')
        ax3.legend()
        ax3.set_aspect('equal')
        
        plt.tight_layout()
        plt.show()
    
    def chain_rule_demonstration(self):
        """連鎖律の実演"""
        
        print("\n=== 連鎖律：複合関数の微分 ===")
        
        # 簡単なニューラルネットワークで説明
        class SimpleNetwork:
            def __init__(self):
                # パラメータ
                self.W1 = np.array([[0.5, -0.3], [0.2, 0.8]])
                self.b1 = np.array([0.1, -0.1])
                self.W2 = np.array([[0.7], [-0.4]])
                self.b2 = np.array([0.2])
                
                # 中間値を保存
                self.cache = {}
            
            def relu(self, x):
                return np.maximum(0, x)
            
            def relu_derivative(self, x):
                return (x > 0).astype(float)
            
            def forward(self, x):
                """順伝播"""
                # 層1: z1 = W1 @ x + b1
                self.cache['x'] = x
                self.cache['z1'] = self.W1 @ x + self.b1
                
                # 活性化: a1 = ReLU(z1)
                self.cache['a1'] = self.relu(self.cache['z1'])
                
                # 層2: z2 = W2 @ a1 + b2
                self.cache['z2'] = self.W2 @ self.cache['a1'] + self.b2
                
                # 出力（線形）
                y = self.cache['z2']
                
                return y
            
            def backward(self, y, y_true):
                """逆伝播（連鎖律を使用）"""
                # 損失: L = 0.5 * (y - y_true)^2
                
                # ∂L/∂y
                dL_dy = y - y_true
                
                # ∂L/∂z2 = ∂L/∂y * ∂y/∂z2 = dL_dy * 1
                dL_dz2 = dL_dy
                
                # ∂L/∂W2 = ∂L/∂z2 * ∂z2/∂W2 = dL_dz2 * a1^T
                dL_dW2 = dL_dz2 @ self.cache['a1'].reshape(1, -1)
                
                # ∂L/∂b2 = ∂L/∂z2 * ∂z2/∂b2 = dL_dz2 * 1
                dL_db2 = dL_dz2
                
                # ∂L/∂a1 = ∂L/∂z2 * ∂z2/∂a1 = W2^T @ dL_dz2
                dL_da1 = self.W2.T @ dL_dz2
                
                # ∂L/∂z1 = ∂L/∂a1 * ∂a1/∂z1 = dL_da1 * ReLU'(z1)
                dL_dz1 = dL_da1.flatten() * self.relu_derivative(self.cache['z1'])
                
                # ∂L/∂W1 = ∂L/∂z1 * ∂z1/∂W1 = dL_dz1 * x^T
                dL_dW1 = np.outer(dL_dz1, self.cache['x'])
                
                # ∂L/∂b1 = ∂L/∂z1 * ∂z1/∂b1 = dL_dz1 * 1
                dL_db1 = dL_dz1
                
                gradients = {
                    'W1': dL_dW1,
                    'b1': dL_db1,
                    'W2': dL_dW2,
                    'b2': dL_db2
                }
                
                return gradients
            
            def visualize_computation_graph(self):
                """計算グラフの可視化"""
                import networkx as nx
                
                G = nx.DiGraph()
                
                # ノードを追加
                nodes = ['x', 'W1', 'b1', 'z1', 'a1', 'W2', 'b2', 'z2', 'y', 'L']
                node_colors = {
                    'x': 'lightblue',
                    'W1': 'lightgreen', 'b1': 'lightgreen',
                    'W2': 'lightgreen', 'b2': 'lightgreen',
                    'z1': 'lightyellow', 'a1': 'lightyellow',
                    'z2': 'lightyellow', 'y': 'lightyellow',
                    'L': 'lightcoral'
                }
                
                for node in nodes:
                    G.add_node(node)
                
                # エッジを追加（計算の依存関係）
                edges = [
                    ('x', 'z1'), ('W1', 'z1'), ('b1', 'z1'),
                    ('z1', 'a1'),
                    ('a1', 'z2'), ('W2', 'z2'), ('b2', 'z2'),
                    ('z2', 'y'),
                    ('y', 'L')
                ]
                
                G.add_edges_from(edges)
                
                # レイアウト
                pos = {
                    'x': (0, 2),
                    'W1': (-1, 1), 'b1': (1, 1),
                    'z1': (0, 1),
                    'a1': (0, 0),
                    'W2': (-1, -1), 'b2': (1, -1),
                    'z2': (0, -1),
                    'y': (0, -2),
                    'L': (0, -3)
                }
                
                plt.figure(figsize=(10, 8))
                
                # ノードを描画
                for node in G.nodes():
                    nx.draw_networkx_nodes(G, pos, nodelist=[node],
                                         node_color=node_colors[node],
                                         node_size=1000)
                
                # エッジを描画
                nx.draw_networkx_edges(G, pos, edge_color='gray',
                                     arrows=True, arrowsize=20)
                
                # ラベルを追加
                nx.draw_networkx_labels(G, pos, font_size=12)
                
                # 勾配の流れを表示
                gradient_edges = [
                    ('L', 'y', '∂L/∂y'),
                    ('y', 'z2', '∂L/∂z2'),
                    ('z2', 'W2', '∂L/∂W2'),
                    ('z2', 'a1', '∂L/∂a1'),
                    ('a1', 'z1', '∂L/∂z1'),
                    ('z1', 'W1', '∂L/∂W1')
                ]
                
                for src, dst, label in gradient_edges:
                    # 逆方向の矢印で勾配を表示
                    if src in pos and dst in pos:
                        x1, y1 = pos[dst]
                        x2, y2 = pos[src]
                        plt.annotate('', xy=(x1, y1), xytext=(x2, y2),
                                   arrowprops=dict(arrowstyle='<-',
                                                 color='red',
                                                 lw=2,
                                                 alpha=0.7))
                
                plt.title('計算グラフと勾配の逆伝播')
                plt.axis('off')
                plt.tight_layout()
                plt.show()
        
        # 実演
        net = SimpleNetwork()
        
        # 入力
        x = np.array([1.0, 0.5])
        y_true = np.array([0.8])
        
        # 順伝播
        y = net.forward(x)
        print(f"入力: {x}")
        print(f"出力: {y}")
        print(f"目標: {y_true}")
        
        # 逆伝播
        gradients = net.backward(y, y_true)
        
        print("\n勾配:")
        for param, grad in gradients.items():
            print(f"∂L/∂{param} = \n{grad}")
        
        # 計算グラフを可視化
        net.visualize_computation_graph()
    
    def automatic_differentiation(self):
        """自動微分の仕組み"""
        
        print("\n=== 自動微分 ===")
        
        # PyTorchでの自動微分
        import torch
        
        # 計算グラフの構築と自動微分
        x = torch.tensor(2.0, requires_grad=True)
        y = torch.tensor(3.0, requires_grad=True)
        
        # 複雑な関数
        z = x ** 2 + 2 * x * y + y ** 2  # (x + y)^2
        w = torch.sin(z) + torch.cos(x * y)
        loss = w ** 2
        
        # 逆伝播
        loss.backward()
        
        print("自動微分の結果:")
        print(f"x = {x.item():.3f}, ∂loss/∂x = {x.grad.item():.3f}")
        print(f"y = {y.item():.3f}, ∂loss/∂y = {y.grad.item():.3f}")
        
        # 数値微分との比較
        def numerical_gradient(f, x, h=1e-5):
            """数値微分（有限差分法）"""
            return (f(x + h) - f(x - h)) / (2 * h)
        
        # 同じ関数を通常のPythonで定義
        def f(x_val, y_val):
            z = x_val ** 2 + 2 * x_val * y_val + y_val ** 2
            w = np.sin(z) + np.cos(x_val * y_val)
            return w ** 2
        
        # 数値微分
        dx_numerical = numerical_gradient(lambda x: f(x, 3.0), 2.0)
        dy_numerical = numerical_gradient(lambda y: f(2.0, y), 3.0)
        
        print(f"\n数値微分:")
        print(f"∂loss/∂x ≈ {dx_numerical:.3f}")
        print(f"∂loss/∂y ≈ {dy_numerical:.3f}")
        
        print(f"\n誤差:")
        print(f"|自動微分 - 数値微分|_x = {abs(x.grad.item() - dx_numerical):.6f}")
        print(f"|自動微分 - 数値微分|_y = {abs(y.grad.item() - dy_numerical):.6f}")

## 3.4 実践：数学をコードで確認

### 完全なTransformerブロックの数学

```python
class TransformerMathematics:
    """Transformerの数学的構成要素"""
    
    def __init__(self, d_model=512, n_heads=8, d_ff=2048):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.d_k = d_model // n_heads
    
    def complete_transformer_block(self):
        """完全なTransformerブロックの数学"""
        
        print("=== Transformerブロックの数学的構成 ===")
        
        # 各コンポーネントの数式
        equations = {
            "Multi-Head Attention": [
                "Q = XW_Q, K = XW_K, V = XW_V",
                "head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)",
                "MultiHead(Q,K,V) = Concat(head_1,...,head_h)W_O"
            ],
            "Scaled Dot-Product Attention": [
                "Attention(Q,K,V) = softmax(QK^T / √d_k)V"
            ],
            "Position-wise Feed Forward": [
                "FFN(x) = max(0, xW_1 + b_1)W_2 + b_2"
            ],
            "Residual Connection": [
                "output = LayerNorm(x + Sublayer(x))"
            ]
        }
        
        # 数式を表示
        for component, eqs in equations.items():
            print(f"\n{component}:")
            for eq in eqs:
                print(f"  {eq}")
        
        # 実際の計算をステップバイステップで
        self.step_by_step_computation()
    
    def step_by_step_computation(self):
        """ステップバイステップの計算"""
        
        # 小さな例で計算過程を追跡
        batch_size = 2
        seq_len = 4
        d_model = 8  # 小さくして見やすく
        n_heads = 2
        d_k = d_model // n_heads
        
        print("\n=== 具体例での計算 ===")
        print(f"バッチサイズ: {batch_size}")
        print(f"シーケンス長: {seq_len}")
        print(f"モデル次元: {d_model}")
        print(f"ヘッド数: {n_heads}")
        
        # 入力
        X = torch.randn(batch_size, seq_len, d_model)
        
        # 重み行列
        W_Q = torch.randn(d_model, d_model)
        W_K = torch.randn(d_model, d_model)
        W_V = torch.randn(d_model, d_model)
        W_O = torch.randn(d_model, d_model)
        
        # Step 1: Linear projections
        Q = X @ W_Q
        K = X @ W_K
        V = X @ W_V
        
        print(f"\nStep 1: 線形投影")
        print(f"Q shape: {Q.shape}")
        print(f"K shape: {K.shape}")
        print(f"V shape: {V.shape}")
        
        # Step 2: Reshape for multi-head
        Q = Q.view(batch_size, seq_len, n_heads, d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, n_heads, d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, n_heads, d_k).transpose(1, 2)
        
        print(f"\nStep 2: Multi-head用に形状変更")
        print(f"Q shape: {Q.shape} [batch, heads, seq_len, d_k]")
        
        # Step 3: Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        
        print(f"\nStep 3: スコア計算")
        print(f"Scores shape: {scores.shape}")
        print(f"Score[0,0] (第1バッチ、第1ヘッド):")
        print(scores[0, 0])
        
        # Step 4: Softmax
        attn_weights = F.softmax(scores, dim=-1)
        
        print(f"\nStep 4: Softmax")
        print(f"Attention weights[0,0]:")
        print(attn_weights[0, 0])
        print(f"各行の和: {attn_weights[0, 0].sum(dim=-1)}")
        
        # Step 5: Apply attention to values
        context = torch.matmul(attn_weights, V)
        
        print(f"\nStep 5: 値への適用")
        print(f"Context shape: {context.shape}")
        
        # Step 6: Concatenate heads
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        print(f"\nStep 6: ヘッドの結合")
        print(f"Context shape: {context.shape}")
        
        # Step 7: Output projection
        output = context @ W_O
        
        print(f"\nStep 7: 出力投影")
        print(f"Output shape: {output.shape}")

## まとめ：数学は実装で理解する

この章では、Transformerを理解するために必要な数学的概念を、実装を通じて学びました：

1. **線形代数**
   - ベクトル：データの表現
   - 行列：変換と関係性
   - 内積：類似度の計算

2. **確率・統計**
   - 確率分布：不確実性の表現
   - Softmax：スコアから確率へ
   - 期待値と分散：正規化の基礎

3. **微分**
   - 勾配：最適化の方向
   - 連鎖律：複雑な関数の微分
   - 自動微分：効率的な計算

これらの概念は、次章で学ぶPyTorchでの実装に直接つながります。数学は抽象的な理論ではなく、実際に動くコードとして理解することで、Transformerの動作原理がより明確になるはずです。

## 演習問題

1. **ベクトル演算の実装**
   - 2つのベクトル間のコサイン類似度を計算する関数を実装してください
   - 単語埋め込みベクトルで「king - man + woman ≈ queen」を検証してください

2. **行列変換の可視化**
   - 任意の2×2行列による変換を可視化するプログラムを作成してください
   - 固有ベクトルの方向が変換で保存されることを確認してください

3. **Attention の実装**
   - Scaled Dot-Product Attentionを一から実装してください
   - 温度パラメータを追加し、その効果を可視化してください

4. **勾配降下法の実装**
   - 2次元関数の勾配降下法を実装してください
   - 学習率による収束の違いを比較してください

5. **自動微分の仕組み**
   - 簡単な計算グラフを作成し、手動でバックプロパゲーションを計算してください
   - PyTorchの自動微分結果と比較してください

---

次章では、PyTorchを使って実際にこれらの数学的概念を実装していきます。理論と実装を結びつけることで、Transformerの理解がさらに深まるでしょう。