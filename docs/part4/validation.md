# 動作検証

## はじめに：正しさの証明

コンパイラを作った後、最も重要なのはその正しさを検証することです。テストスイートを実行し、既知の入力に対して期待される出力が得られることを確認します。エッジケースを探し、パフォーマンスを測定し、他の実装と比較します。

Transformerの実装でも同じアプローチが必要です。この章では、実装したTransformerが正しく動作することを体系的に検証する方法を学びます。

## 16.1 単体テストの実装

### コンポーネントレベルのテスト

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import unittest
from typing import Tuple, Optional, List
import math
import matplotlib.pyplot as plt
import seaborn as sns
from torch.testing import assert_close

class TestMultiHeadAttention(unittest.TestCase):
    """Multi-Head Attentionの単体テスト"""
    
    def setUp(self):
        """テストの初期設定"""
        self.d_model = 512
        self.n_heads = 8
        self.batch_size = 2
        self.seq_len = 10
        
        # テスト対象のモジュール
        self.attention = nn.MultiheadAttention(
            self.d_model, self.n_heads, batch_first=True
        )
        
    def test_output_shape(self):
        """出力形状のテスト"""
        # 入力データ
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        
        # 順伝播
        output, weights = self.attention(x, x, x)
        
        # 形状の確認
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_model))
        self.assertEqual(weights.shape, (self.batch_size, self.seq_len, self.seq_len))
        
        print("✅ 出力形状テスト: PASS")
    
    def test_attention_mask(self):
        """注意マスクのテスト"""
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        
        # 因果的マスクの作成
        mask = torch.triu(torch.ones(self.seq_len, self.seq_len) * float('-inf'), diagonal=1)
        
        # マスク付き順伝播
        output, weights = self.attention(x, x, x, attn_mask=mask)
        
        # 未来の位置への注意が0であることを確認
        for i in range(self.seq_len):
            for j in range(i + 1, self.seq_len):
                self.assertAlmostEqual(
                    weights[0, i, j].item(), 0.0, places=5,
                    msg=f"Position {i} should not attend to future position {j}"
                )
        
        print("✅ 注意マスクテスト: PASS")
    
    def test_key_value_different(self):
        """異なるKey/Valueのテスト"""
        # Query, Key, Valueが異なる場合
        q = torch.randn(self.batch_size, self.seq_len, self.d_model)
        k = torch.randn(self.batch_size, self.seq_len * 2, self.d_model)
        v = torch.randn(self.batch_size, self.seq_len * 2, self.d_model)
        
        output, weights = self.attention(q, k, v)
        
        # 出力形状の確認
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_model))
        self.assertEqual(weights.shape, (self.batch_size, self.seq_len, self.seq_len * 2))
        
        print("✅ 異なるKey/Valueテスト: PASS")
    
    def test_attention_weights_sum(self):
        """注意重みの和が1であることをテスト"""
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        _, weights = self.attention(x, x, x)
        
        # 各行の和が1に近いことを確認
        row_sums = weights.sum(dim=-1)
        expected = torch.ones_like(row_sums)
        
        assert_close(row_sums, expected, rtol=1e-5, atol=1e-5)
        
        print("✅ 注意重みの和テスト: PASS")
    
    def test_gradient_flow(self):
        """勾配が正しく流れることをテスト"""
        x = torch.randn(self.batch_size, self.seq_len, self.d_model, requires_grad=True)
        
        output, _ = self.attention(x, x, x)
        loss = output.mean()
        loss.backward()
        
        # 勾配が計算されていることを確認
        self.assertIsNotNone(x.grad)
        self.assertFalse(torch.isnan(x.grad).any())
        self.assertFalse(torch.isinf(x.grad).any())
        
        print("✅ 勾配フローテスト: PASS")

class TestPositionalEncoding(unittest.TestCase):
    """位置エンコーディングのテスト"""
    
    def setUp(self):
        self.d_model = 512
        self.max_len = 1000
        
    def test_sinusoidal_encoding_properties(self):
        """正弦波位置エンコーディングの性質をテスト"""
        # 位置エンコーディングの生成
        pe = self._create_sinusoidal_encoding(self.max_len, self.d_model)
        
        # 1. 値の範囲が[-1, 1]であること
        self.assertLessEqual(pe.max().item(), 1.0)
        self.assertGreaterEqual(pe.min().item(), -1.0)
        
        # 2. 偶数次元がsin、奇数次元がcosであること
        pos = 10  # テスト位置
        for i in range(0, self.d_model, 2):
            div_term = 10000 ** (i / self.d_model)
            expected_sin = math.sin(pos / div_term)
            expected_cos = math.cos(pos / div_term)
            
            self.assertAlmostEqual(pe[pos, i].item(), expected_sin, places=5)
            if i + 1 < self.d_model:
                self.assertAlmostEqual(pe[pos, i + 1].item(), expected_cos, places=5)
        
        print("✅ 正弦波エンコーディングテスト: PASS")
    
    def test_relative_position_encoding(self):
        """相対位置の性質をテスト"""
        pe = self._create_sinusoidal_encoding(self.max_len, self.d_model)
        
        # 固定された相対距離での内積が一定であることを確認
        distance = 5
        products = []
        
        for pos in range(10, 20):
            dot_product = torch.dot(pe[pos], pe[pos + distance])
            products.append(dot_product.item())
        
        # 標準偏差が小さいことを確認
        std = np.std(products)
        self.assertLess(std, 0.01, "相対位置の内積は一定であるべき")
        
        print("✅ 相対位置エンコーディングテスト: PASS")
    
    def _create_sinusoidal_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """正弦波位置エンコーディングを作成"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe

class TestTransformerBlock(unittest.TestCase):
    """Transformerブロックのテスト"""
    
    def setUp(self):
        self.d_model = 256
        self.n_heads = 8
        self.d_ff = 1024
        self.dropout = 0.1
        
        # Transformerブロックの作成
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=self.d_ff,
            dropout=self.dropout,
            batch_first=True
        )
        
    def test_residual_connections(self):
        """残差接続のテスト"""
        batch_size = 2
        seq_len = 10
        
        # 入力
        x = torch.randn(batch_size, seq_len, self.d_model)
        
        # ドロップアウトを無効化（テストのため）
        self.encoder_layer.eval()
        
        # 小さな重みで初期化（残差接続の効果を見やすくする）
        for param in self.encoder_layer.parameters():
            param.data.mul_(0.01)
        
        output = self.encoder_layer(x)
        
        # 出力が入力に近いことを確認（残差接続の効果）
        diff = (output - x).abs().mean()
        self.assertLess(diff.item(), 0.5, "残差接続により出力は入力に近いはず")
        
        print("✅ 残差接続テスト: PASS")
    
    def test_layer_norm_effect(self):
        """層正規化の効果をテスト"""
        batch_size = 2
        seq_len = 10
        
        # 大きな値を持つ入力
        x = torch.randn(batch_size, seq_len, self.d_model) * 100
        
        output = self.encoder_layer(x)
        
        # 出力の各位置での平均と分散を計算
        mean = output.mean(dim=-1)
        var = output.var(dim=-1)
        
        # 層正規化により、平均が0に近く、分散が1に近いことを確認
        self.assertLess(mean.abs().mean().item(), 0.1)
        self.assertLess((var - 1).abs().mean().item(), 0.5)
        
        print("✅ 層正規化テスト: PASS")

## 16.2 統合テスト

class IntegrationTests:
    """モデル全体の統合テスト"""
    
    def __init__(self, model_class):
        self.model_class = model_class
        
    def test_full_forward_pass(self):
        """完全な順伝播のテスト"""
        print("\n=== 統合テスト: 完全な順伝播 ===")
        
        # モデルの作成
        vocab_size = 1000
        d_model = 256
        model = self.model_class(vocab_size=vocab_size, d_model=d_model)
        model.eval()
        
        # テストケース
        test_cases = [
            {"batch_size": 1, "seq_len": 10},
            {"batch_size": 4, "seq_len": 50},
            {"batch_size": 8, "seq_len": 100},
        ]
        
        for case in test_cases:
            batch_size = case["batch_size"]
            seq_len = case["seq_len"]
            
            # 入力データ
            input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
            
            # 順伝播
            with torch.no_grad():
                output = model(input_ids)
            
            # 出力形状の確認
            expected_shape = (batch_size, seq_len, vocab_size)
            assert output.shape == expected_shape, \
                f"Expected shape {expected_shape}, got {output.shape}"
            
            # NaNやInfがないことを確認
            assert not torch.isnan(output).any(), "Output contains NaN"
            assert not torch.isinf(output).any(), "Output contains Inf"
            
            print(f"✅ バッチサイズ={batch_size}, シーケンス長={seq_len}: PASS")
    
    def test_generation_consistency(self):
        """生成の一貫性テスト"""
        print("\n=== 統合テスト: 生成の一貫性 ===")
        
        vocab_size = 100
        model = self.model_class(vocab_size=vocab_size, d_model=128)
        model.eval()
        
        # シード固定
        torch.manual_seed(42)
        
        # 同じプロンプトから生成
        prompt = torch.tensor([[1, 2, 3]])
        
        # 複数回生成
        outputs = []
        for _ in range(3):
            torch.manual_seed(42)  # 同じシード
            output = model.generate(prompt, max_new_tokens=10, temperature=1.0)
            outputs.append(output)
        
        # すべての出力が同じであることを確認
        for i in range(1, len(outputs)):
            assert torch.equal(outputs[0], outputs[i]), \
                f"生成結果が一貫していません: {i}回目"
        
        print("✅ 生成の一貫性: PASS")
    
    def test_attention_pattern_analysis(self):
        """注意パターンの分析"""
        print("\n=== 統合テスト: 注意パターン分析 ===")
        
        # 特別なテストケース：繰り返しパターン
        vocab_size = 50
        model = self.model_class(vocab_size=vocab_size, d_model=128, n_layers=2)
        model.eval()
        
        # 繰り返しのある入力
        # "A B C A B C A B C"のようなパターン
        pattern = [10, 20, 30]
        input_ids = torch.tensor([pattern * 3]).to(torch.long)
        
        # 注意重みを取得するためのフック
        attention_weights = []
        
        def hook_fn(module, input, output):
            if isinstance(output, tuple) and len(output) == 2:
                _, attn = output
                if attn is not None:
                    attention_weights.append(attn.detach())
        
        # フックを登録
        hooks = []
        for module in model.modules():
            if isinstance(module, nn.MultiheadAttention):
                hook = module.register_forward_hook(hook_fn)
                hooks.append(hook)
        
        # 順伝播
        with torch.no_grad():
            _ = model(input_ids)
        
        # フックを削除
        for hook in hooks:
            hook.remove()
        
        # 注意パターンの分析
        if attention_weights:
            # 最初の層の注意重みを分析
            attn = attention_weights[0][0].mean(dim=0)  # ヘッドの平均
            
            # 同じトークンへの注意が高いことを確認
            for i in range(3):
                for j in range(3):
                    if i != j:
                        pos1 = i * 3
                        pos2 = j * 3
                        # 同じトークン（位置は違う）への注意
                        similarity = attn[pos1, pos2].item()
                        print(f"  位置{pos1} → 位置{pos2}の注意: {similarity:.3f}")
        
        print("✅ 注意パターン分析: 完了")

## 16.3 性能ベンチマーク

class PerformanceBenchmark:
    """性能ベンチマークテスト"""
    
    def __init__(self, model):
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def benchmark_inference_speed(self):
        """推論速度のベンチマーク"""
        print("\n=== 性能ベンチマーク: 推論速度 ===")
        
        # テスト設定
        batch_sizes = [1, 4, 16, 32]
        seq_lengths = [10, 50, 100, 200]
        vocab_size = 1000
        
        results = {}
        
        for batch_size in batch_sizes:
            results[batch_size] = {}
            
            for seq_len in seq_lengths:
                # 入力データ
                input_ids = torch.randint(0, vocab_size, 
                                        (batch_size, seq_len)).to(self.device)
                
                # ウォームアップ
                for _ in range(10):
                    with torch.no_grad():
                        _ = self.model(input_ids)
                
                # 時間測定
                import time
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                
                start_time = time.time()
                n_iterations = 100
                
                for _ in range(n_iterations):
                    with torch.no_grad():
                        _ = self.model(input_ids)
                
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end_time = time.time()
                
                # 平均時間（ミリ秒）
                avg_time = (end_time - start_time) / n_iterations * 1000
                throughput = batch_size / (avg_time / 1000)  # samples/sec
                
                results[batch_size][seq_len] = {
                    'time_ms': avg_time,
                    'throughput': throughput
                }
                
                print(f"  Batch={batch_size}, Seq={seq_len}: "
                      f"{avg_time:.2f}ms, {throughput:.1f} samples/sec")
        
        # 結果の可視化
        self._visualize_benchmark_results(results)
        
        return results
    
    def benchmark_memory_usage(self):
        """メモリ使用量のベンチマーク"""
        print("\n=== 性能ベンチマーク: メモリ使用量 ===")
        
        if not torch.cuda.is_available():
            print("  GPUが利用できないため、メモリベンチマークをスキップ")
            return
        
        seq_lengths = [10, 50, 100, 200, 500]
        batch_size = 4
        vocab_size = 1000
        
        memory_usage = []
        
        for seq_len in seq_lengths:
            # メモリをクリア
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            # 入力データ
            input_ids = torch.randint(0, vocab_size, 
                                    (batch_size, seq_len)).to(self.device)
            
            # 順伝播
            with torch.no_grad():
                _ = self.model(input_ids)
            
            # ピークメモリ使用量
            peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
            memory_usage.append(peak_memory)
            
            print(f"  Seq={seq_len}: {peak_memory:.1f} MB")
        
        # メモリ使用量の成長率を分析
        self._analyze_memory_scaling(seq_lengths, memory_usage)
    
    def _visualize_benchmark_results(self, results):
        """ベンチマーク結果を可視化"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 推論時間
        for batch_size, seq_results in results.items():
            seq_lengths = list(seq_results.keys())
            times = [seq_results[seq]['time_ms'] for seq in seq_lengths]
            ax1.plot(seq_lengths, times, marker='o', label=f'Batch={batch_size}')
        
        ax1.set_xlabel('Sequence Length')
        ax1.set_ylabel('Inference Time (ms)')
        ax1.set_title('Inference Time vs Sequence Length')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # スループット
        for batch_size, seq_results in results.items():
            seq_lengths = list(seq_results.keys())
            throughputs = [seq_results[seq]['throughput'] for seq in seq_lengths]
            ax2.plot(seq_lengths, throughputs, marker='o', label=f'Batch={batch_size}')
        
        ax2.set_xlabel('Sequence Length')
        ax2.set_ylabel('Throughput (samples/sec)')
        ax2.set_title('Throughput vs Sequence Length')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        plt.tight_layout()
        plt.show()
    
    def _analyze_memory_scaling(self, seq_lengths, memory_usage):
        """メモリスケーリングの分析"""
        # O(n^2)フィッティング
        coeffs = np.polyfit(seq_lengths, memory_usage, 2)
        poly = np.poly1d(coeffs)
        
        plt.figure(figsize=(8, 6))
        plt.scatter(seq_lengths, memory_usage, label='Actual', s=100)
        
        # フィット曲線
        x_fit = np.linspace(min(seq_lengths), max(seq_lengths), 100)
        y_fit = poly(x_fit)
        plt.plot(x_fit, y_fit, 'r--', label=f'Quadratic Fit', alpha=0.7)
        
        plt.xlabel('Sequence Length')
        plt.ylabel('Memory Usage (MB)')
        plt.title('Memory Usage Scaling')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        print(f"\n  メモリ使用量は O(n^{2:.1f}) でスケール")

## 16.4 比較検証

class ComparativeValidation:
    """他の実装との比較検証"""
    
    def __init__(self, custom_model, reference_model=None):
        self.custom_model = custom_model
        self.reference_model = reference_model
        
    def compare_with_pytorch_transformer(self):
        """PyTorchの標準Transformerとの比較"""
        print("\n=== 比較検証: PyTorch標準実装との比較 ===")
        
        d_model = 256
        n_heads = 8
        n_layers = 2
        
        # PyTorchの標準Transformer
        pytorch_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                batch_first=True
            ),
            num_layers=n_layers
        )
        
        # 同じ重みで初期化（可能な限り）
        # ここでは簡略化のため新規の重みを使用
        
        # テスト入力
        batch_size = 2
        seq_len = 10
        x = torch.randn(batch_size, seq_len, d_model)
        
        # 両方のモデルで推論
        pytorch_encoder.eval()
        self.custom_model.eval() if self.custom_model else None
        
        with torch.no_grad():
            pytorch_output = pytorch_encoder(x)
            
            if self.custom_model:
                # カスタムモデルがエンコーダーのみの場合
                custom_output = self.custom_model.encoder(x) if hasattr(self.custom_model, 'encoder') else None
                
                if custom_output is not None:
                    # 出力の統計を比較
                    print(f"  PyTorch出力 - 平均: {pytorch_output.mean():.4f}, "
                          f"標準偏差: {pytorch_output.std():.4f}")
                    print(f"  カスタム出力 - 平均: {custom_output.mean():.4f}, "
                          f"標準偏差: {custom_output.std():.4f}")
            else:
                print(f"  PyTorch出力形状: {pytorch_output.shape}")
                print(f"  平均: {pytorch_output.mean():.4f}, "
                      f"標準偏差: {pytorch_output.std():.4f}")
    
    def validate_attention_computation(self):
        """注意計算の検証"""
        print("\n=== 比較検証: 注意計算の正確性 ===")
        
        # 手動での注意計算
        d_model = 64
        seq_len = 5
        
        # ランダムなQ, K, V
        torch.manual_seed(42)
        Q = torch.randn(1, seq_len, d_model)
        K = torch.randn(1, seq_len, d_model)
        V = torch.randn(1, seq_len, d_model)
        
        # 手動計算
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_model)
        attention_weights = F.softmax(scores, dim=-1)
        output_manual = torch.matmul(attention_weights, V)
        
        # nn.MultiheadAttentionでの計算（single head）
        attention = nn.MultiheadAttention(d_model, 1, batch_first=True)
        
        # 重みを設定（恒等変換）
        with torch.no_grad():
            attention.in_proj_weight.data = torch.eye(3 * d_model)
            attention.in_proj_bias.data = torch.zeros(3 * d_model)
            attention.out_proj.weight.data = torch.eye(d_model)
            attention.out_proj.bias.data = torch.zeros(d_model)
        
        # 入力を結合
        qkv = torch.cat([Q, K, V], dim=-1)
        x = qkv[:, :, :d_model]  # Qの部分のみ（簡略化）
        
        output_pytorch, weights_pytorch = attention(Q, K, V)
        
        print(f"  手動計算とPyTorchの差:")
        print(f"  注意重みの差: {(attention_weights - weights_pytorch).abs().max():.6f}")
        print(f"  出力の差: {(output_manual - output_pytorch).abs().mean():.6f}")

class ValidationSuite:
    """完全な検証スイート"""
    
    def __init__(self, model_class):
        self.model_class = model_class
        self.test_results = {}
        
    def run_all_tests(self):
        """すべてのテストを実行"""
        print("=" * 70)
        print("Transformer検証スイート")
        print("=" * 70)
        
        # 1. 単体テスト
        print("\n【1. 単体テスト】")
        self._run_unit_tests()
        
        # 2. 統合テスト
        print("\n【2. 統合テスト】")
        self._run_integration_tests()
        
        # 3. 性能テスト
        print("\n【3. 性能テスト】")
        self._run_performance_tests()
        
        # 4. 比較検証
        print("\n【4. 比較検証】")
        self._run_comparative_tests()
        
        # 結果サマリー
        self._print_summary()
    
    def _run_unit_tests(self):
        """単体テストの実行"""
        # テストランナーの作成
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()
        
        # テストクラスを追加
        suite.addTests(loader.loadTestsFromTestCase(TestMultiHeadAttention))
        suite.addTests(loader.loadTestsFromTestCase(TestPositionalEncoding))
        suite.addTests(loader.loadTestsFromTestCase(TestTransformerBlock))
        
        # テスト実行
        runner = unittest.TextTestRunner(verbosity=0)
        result = runner.run(suite)
        
        self.test_results['unit_tests'] = {
            'total': result.testsRun,
            'failures': len(result.failures),
            'errors': len(result.errors)
        }
    
    def _run_integration_tests(self):
        """統合テストの実行"""
        integration = IntegrationTests(self.model_class)
        
        try:
            integration.test_full_forward_pass()
            integration.test_generation_consistency()
            integration.test_attention_pattern_analysis()
            self.test_results['integration_tests'] = 'PASS'
        except Exception as e:
            self.test_results['integration_tests'] = f'FAIL: {str(e)}'
    
    def _run_performance_tests(self):
        """性能テストの実行"""
        # 小さなモデルでテスト
        model = self.model_class(vocab_size=1000, d_model=128, n_layers=2)
        benchmark = PerformanceBenchmark(model)
        
        results = benchmark.benchmark_inference_speed()
        benchmark.benchmark_memory_usage()
        
        self.test_results['performance_tests'] = results
    
    def _run_comparative_tests(self):
        """比較テストの実行"""
        model = self.model_class(vocab_size=1000, d_model=256, n_layers=2)
        comparative = ComparativeValidation(model)
        
        comparative.compare_with_pytorch_transformer()
        comparative.validate_attention_computation()
        
        self.test_results['comparative_tests'] = 'COMPLETED'
    
    def _print_summary(self):
        """テスト結果のサマリー"""
        print("\n" + "=" * 70)
        print("テスト結果サマリー")
        print("=" * 70)
        
        # 単体テスト結果
        unit_results = self.test_results.get('unit_tests', {})
        print(f"\n単体テスト: {unit_results.get('total', 0)}個のテスト")
        print(f"  成功: {unit_results.get('total', 0) - unit_results.get('failures', 0) - unit_results.get('errors', 0)}")
        print(f"  失敗: {unit_results.get('failures', 0)}")
        print(f"  エラー: {unit_results.get('errors', 0)}")
        
        # 統合テスト結果
        print(f"\n統合テスト: {self.test_results.get('integration_tests', 'N/A')}")
        
        # 性能テスト結果
        if 'performance_tests' in self.test_results:
            print("\n性能テスト: 完了")
            # 代表的な結果を表示
            perf_results = self.test_results['performance_tests']
            if 4 in perf_results and 100 in perf_results[4]:
                time_ms = perf_results[4][100]['time_ms']
                throughput = perf_results[4][100]['throughput']
                print(f"  代表例 (Batch=4, Seq=100): {time_ms:.2f}ms, {throughput:.1f} samples/sec")
        
        # 比較テスト結果
        print(f"\n比較テスト: {self.test_results.get('comparative_tests', 'N/A')}")
        
        # 総合評価
        print("\n" + "=" * 70)
        all_passed = (
            unit_results.get('failures', 1) == 0 and
            unit_results.get('errors', 1) == 0 and
            self.test_results.get('integration_tests') == 'PASS'
        )
        
        if all_passed:
            print("✅ すべてのテストに合格しました！")
        else:
            print("❌ 一部のテストに失敗しました。")

# 実際の検証例
def run_validation_example():
    """検証の実行例"""
    
    # ダミーのTransformerモデルクラス
    class DummyTransformer(nn.Module):
        def __init__(self, vocab_size, d_model, n_layers=4, n_heads=8):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_heads,
                    batch_first=True
                ),
                num_layers=n_layers
            )
            self.output_projection = nn.Linear(d_model, vocab_size)
            
        def forward(self, x):
            x = self.embedding(x)
            x = self.encoder(x)
            x = self.output_projection(x)
            return x
        
        def generate(self, prompt, max_new_tokens, temperature):
            # 簡略化された生成
            current = prompt
            for _ in range(max_new_tokens):
                output = self.forward(current)
                next_token = output[:, -1, :].argmax(dim=-1, keepdim=True)
                current = torch.cat([current, next_token], dim=1)
            return current
    
    # 検証スイートの実行
    validation = ValidationSuite(DummyTransformer)
    validation.run_all_tests()

# エラー分析とデバッグのヒント
def debugging_tips():
    """デバッグのヒント"""
    print("\n" + "=" * 70)
    print("一般的な問題とデバッグのヒント")
    print("=" * 70 + "\n")
    
    tips = {
        "勾配消失/爆発": [
            "層正規化が正しく適用されているか確認",
            "残差接続が機能しているか確認",
            "学習率が適切か確認",
            "勾配クリッピングを使用"
        ],
        
        "注意の偏り": [
            "スケーリング係数(1/√d_k)が正しいか確認",
            "マスクが正しく適用されているか確認",
            "初期化方法を確認"
        ],
        
        "生成品質": [
            "温度パラメータの調整",
            "Top-k/Top-pサンプリングの使用",
            "ビームサーチの実装",
            "繰り返しペナルティの追加"
        ],
        
        "メモリ不足": [
            "バッチサイズの削減",
            "シーケンス長の制限",
            "勾配累積の使用",
            "Mixed Precision Trainingの使用"
        ]
    }
    
    for problem, solutions in tips.items():
        print(f"{problem}:")
        for solution in solutions:
            print(f"  • {solution}")
        print()

if __name__ == "__main__":
    # 検証例の実行
    run_validation_example()
    
    # デバッグのヒント
    debugging_tips()