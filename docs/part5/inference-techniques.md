# 推論技術

## はじめに：生成の芸術

コンパイラの最適化を考えてみてください。同じプログラムでも、最適化レベルやターゲットアーキテクチャによって全く異なるコードが生成されます。`-O0`では読みやすいが遅いコード、`-O3`では高速だが複雑なコード。目的に応じて適切な戦略を選びます。

言語モデルの推論も同じです。同じモデルでも、サンプリング戦略、ビームサーチ、制約条件などによって全く異なる出力が得られます。この章では、高品質なテキスト生成のための様々な推論技術を学びます。

## 20.1 サンプリング戦略

### 確率分布からの賢い選択

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass
import math
from collections import defaultdict, Counter
import time
from tqdm import tqdm

class SamplingStrategies:
    """様々なサンプリング戦略の実装"""
    
    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        
    def explain_sampling_methods(self):
        """サンプリング手法の説明"""
        print("=== サンプリング戦略 ===\n")
        
        methods = {
            "Greedy Decoding": {
                "説明": "常に最も確率の高いトークンを選択",
                "利点": "決定的、高速",
                "欠点": "繰り返しやすい、多様性なし",
                "用途": "事実的な回答、翻訳"
            },
            
            "Temperature Sampling": {
                "説明": "温度パラメータで分布を調整",
                "利点": "多様性の制御可能",
                "欠点": "品質が不安定",
                "用途": "創造的な生成"
            },
            
            "Top-k Sampling": {
                "説明": "上位k個からサンプリング",
                "利点": "低確率トークンを除外",
                "欠点": "kの選択が難しい",
                "用途": "バランスの取れた生成"
            },
            
            "Top-p (Nucleus) Sampling": {
                "説明": "累積確率p以下でサンプリング",
                "利点": "動的な候補数",
                "欠点": "計算がやや複雑",
                "用途": "高品質な生成"
            },
            
            "Beam Search": {
                "説明": "複数の候補を並列探索",
                "利点": "局所最適を回避",
                "欠点": "計算コスト高、多様性低",
                "用途": "翻訳、要約"
            }
        }
        
        for name, details in methods.items():
            print(f"{name}:")
            for key, value in details.items():
                print(f"  {key}: {value}")
            print()
        
        # 各手法の可視化
        self._visualize_sampling_methods()
    
    def _visualize_sampling_methods(self):
        """サンプリング手法を可視化"""
        # ダミーの確率分布
        torch.manual_seed(42)
        logits = torch.randn(self.vocab_size) * 2
        probs = F.softmax(logits, dim=0)
        
        # ソート
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        # 1. Original Distribution
        ax = axes[0]
        ax.bar(range(50), sorted_probs[:50].numpy(), color='lightblue')
        ax.set_title('Original Probability Distribution')
        ax.set_xlabel('Token Rank')
        ax.set_ylabel('Probability')
        ax.set_ylim(0, max(sorted_probs[:50]) * 1.1)
        
        # 2. Greedy
        ax = axes[1]
        greedy_probs = torch.zeros_like(probs)
        greedy_probs[sorted_indices[0]] = 1.0
        ax.bar(range(50), greedy_probs[sorted_indices[:50]].numpy(), 
               color='green')
        ax.set_title('Greedy Decoding')
        ax.set_xlabel('Token Rank')
        ax.set_ylabel('Probability')
        
        # 3. Temperature
        ax = axes[2]
        for temp in [0.5, 1.0, 2.0]:
            temp_logits = logits / temp
            temp_probs = F.softmax(temp_logits, dim=0)
            sorted_temp_probs = temp_probs[sorted_indices]
            ax.plot(range(50), sorted_temp_probs[:50].numpy(), 
                   label=f'T={temp}', linewidth=2)
        ax.set_title('Temperature Sampling')
        ax.set_xlabel('Token Rank')
        ax.set_ylabel('Probability')
        ax.legend()
        
        # 4. Top-k
        ax = axes[3]
        k = 10
        topk_probs = torch.zeros_like(probs)
        topk_probs[sorted_indices[:k]] = sorted_probs[:k]
        topk_probs = topk_probs / topk_probs.sum()  # 再正規化
        ax.bar(range(50), topk_probs[sorted_indices[:50]].numpy(), 
               color='orange')
        ax.set_title(f'Top-k Sampling (k={k})')
        ax.set_xlabel('Token Rank')
        ax.set_ylabel('Probability')
        ax.axvline(x=k-0.5, color='red', linestyle='--', 
                  label=f'k={k}')
        ax.legend()
        
        # 5. Top-p
        ax = axes[4]
        p = 0.9
        cumsum = torch.cumsum(sorted_probs, dim=0)
        cutoff_idx = (cumsum <= p).sum().item()
        topp_probs = torch.zeros_like(probs)
        topp_probs[sorted_indices[:cutoff_idx]] = sorted_probs[:cutoff_idx]
        topp_probs = topp_probs / topp_probs.sum()  # 再正規化
        ax.bar(range(50), topp_probs[sorted_indices[:50]].numpy(), 
               color='purple')
        ax.set_title(f'Top-p Sampling (p={p})')
        ax.set_xlabel('Token Rank')
        ax.set_ylabel('Probability')
        ax.axvline(x=cutoff_idx-0.5, color='red', linestyle='--',
                  label=f'cutoff={cutoff_idx}')
        ax.legend()
        
        # 6. 比較
        ax = axes[5]
        ax.plot(range(20), sorted_probs[:20].numpy(), 
               'o-', label='Original', linewidth=2)
        
        # 各手法の有効範囲
        ax.axhspan(0, sorted_probs[0], alpha=0.3, color='green', 
                  label='Greedy')
        ax.axvspan(0, k-0.5, alpha=0.3, color='orange', 
                  label=f'Top-k (k={k})')
        ax.axvspan(0, cutoff_idx-0.5, alpha=0.3, color='purple', 
                  label=f'Top-p (p={p})')
        
        ax.set_title('Sampling Methods Comparison')
        ax.set_xlabel('Token Rank')
        ax.set_ylabel('Probability')
        ax.legend()
        
        plt.tight_layout()
        plt.show()

class AdvancedSampling:
    """高度なサンプリング技術"""
    
    def __init__(self, model: Optional[nn.Module] = None):
        self.model = model
        
    def top_k_top_p_filtering(self, logits: torch.Tensor, 
                              top_k: int = 0, top_p: float = 0.0,
                              filter_value: float = -float('Inf')) -> torch.Tensor:
        """Top-kとTop-pフィルタリングの実装"""
        # Top-k
        if top_k > 0:
            # 上位k個以外をフィルタ
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value
        
        # Top-p
        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # 累積確率がpを超える位置を見つける
            sorted_indices_to_remove = cumulative_probs > top_p
            
            # 最初のトークンは保持
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            # 元の順序に戻してフィルタ
            indices_to_remove = sorted_indices_to_remove.scatter(
                -1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = filter_value
        
        return logits
    
    def contrastive_search(self, input_ids: torch.Tensor,
                          max_length: int = 50,
                          alpha: float = 0.6,
                          k: int = 5) -> torch.Tensor:
        """Contrastive Search実装"""
        print("=== Contrastive Search ===\n")
        
        print("アルゴリズム:")
        print("1. 各ステップでk個の候補を生成")
        print("2. スコア = (1-α)×確率 + α×多様性")
        print("3. 最高スコアのトークンを選択\n")
        
        # 簡略化した実装例
        generated = input_ids.clone()
        
        for _ in range(max_length - input_ids.size(1)):
            # モデル予測（ダミー）
            if self.model:
                outputs = self.model(generated)
                logits = outputs.logits[:, -1, :]
            else:
                # ダミーのロジット
                logits = torch.randn(1, 50000)
            
            # Top-k候補
            top_k_probs, top_k_ids = torch.topk(
                F.softmax(logits, dim=-1), k
            )
            
            # 各候補の多様性スコアを計算
            diversity_scores = []
            
            for candidate_id in top_k_ids[0]:
                # 既存トークンとの類似度（簡略版）
                if generated.size(1) > 1:
                    # 実際は埋め込みベクトルの類似度を計算
                    similarity = torch.rand(1).item()
                else:
                    similarity = 0
                
                diversity = 1 - similarity
                diversity_scores.append(diversity)
            
            # 総合スコア
            diversity_scores = torch.tensor(diversity_scores)
            scores = (1 - alpha) * top_k_probs[0] + alpha * diversity_scores
            
            # 最高スコアのトークンを選択
            best_idx = torch.argmax(scores)
            next_token = top_k_ids[0, best_idx].unsqueeze(0).unsqueeze(0)
            
            generated = torch.cat([generated, next_token], dim=1)
        
        return generated
    
    def typical_sampling(self, logits: torch.Tensor, 
                        typical_p: float = 0.95) -> torch.Tensor:
        """Typical Sampling実装"""
        print("=== Typical Sampling ===\n")
        
        print("概念: 情報理論に基づくサンプリング")
        print("「典型的」なトークンを選択\n")
        
        # エントロピーを計算
        normalized = F.log_softmax(logits, dim=-1)
        p = normalized.exp()
        ent = -(normalized * p).sum(-1, keepdim=True)
        
        # 各トークンの負の対数尤度をエントロピーでシフト
        shifted_scores = normalized.abs() - ent
        
        # 典型性でソート
        sorted_scores, sorted_indices = torch.sort(shifted_scores)
        
        # 累積確率
        cumulative_probs = sorted_scores.exp().cumsum(dim=-1)
        
        # typical_p以下のトークンを保持
        mask = cumulative_probs <= typical_p
        
        # フィルタリング
        filtered_logits = logits.clone()
        filtered_logits[~mask] = -float('Inf')
        
        return filtered_logits

## 20.2 ビームサーチと派生手法

class BeamSearchMethods:
    """ビームサーチと派生手法"""
    
    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        
    def explain_beam_search(self):
        """ビームサーチの説明"""
        print("=== ビームサーチ ===\n")
        
        print("基本アルゴリズム:")
        print("1. ビーム幅k個の候補を保持")
        print("2. 各候補から次のトークンを生成")
        print("3. 全候補×語彙サイズから上位k個を選択")
        print("4. 終了条件まで繰り返し\n")
        
        # ビームサーチの過程を可視化
        self._visualize_beam_search()
    
    def _visualize_beam_search(self):
        """ビームサーチの過程を可視化"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # パラメータ
        beam_width = 3
        vocab_size = 5  # 簡略化
        steps = 4
        
        # ノードの位置
        x_spacing = 2
        y_spacing = 1.5
        
        # 仮想的なスコア
        np.random.seed(42)
        
        # ビームを追跡
        beams = [[(0, 0, '<start>', 0)]]  # (x, y, token, score)
        
        for step in range(1, steps):
            new_beams = []
            
            for beam_idx, (prev_x, prev_y, prev_token, prev_score) in enumerate(beams[-1][:beam_width]):
                # 各ビームから展開
                for v in range(min(vocab_size, beam_width + 1)):
                    x = step * x_spacing
                    y = prev_y + (v - vocab_size//2) * y_spacing * 0.3
                    
                    score = prev_score - np.random.exponential(0.5)
                    token = f'T{step}{v}'
                    
                    new_beams.append((x, y, token, score, prev_x, prev_y))
            
            # スコアでソートして上位beam_width個を選択
            new_beams.sort(key=lambda x: x[3], reverse=True)
            selected_beams = new_beams[:beam_width]
            
            # 選択されたビームを描画
            for x, y, token, score, prev_x, prev_y in selected_beams:
                # ノード
                circle = plt.Circle((x, y), 0.3, color='lightblue', 
                                  ec='darkblue', linewidth=2)
                ax.add_patch(circle)
                ax.text(x, y, token, ha='center', va='center', fontsize=8)
                
                # エッジ
                ax.plot([prev_x, x], [prev_y, y], 'b-', alpha=0.5, linewidth=2)
                
                # スコア
                ax.text(x, y-0.5, f'{score:.2f}', ha='center', 
                       fontsize=6, color='red')
            
            # 選択されなかったビームを薄く描画
            for x, y, token, score, prev_x, prev_y in new_beams[beam_width:]:
                circle = plt.Circle((x, y), 0.2, color='lightgray', 
                                  ec='gray', linewidth=1, alpha=0.3)
                ax.add_patch(circle)
                ax.plot([prev_x, x], [prev_y, y], 'gray', 
                       alpha=0.2, linewidth=1)
            
            beams.append([(x, y, token, score) for x, y, token, score, _, _ in selected_beams])
        
        # 開始ノード
        start_circle = plt.Circle((0, 0), 0.3, color='lightgreen',
                                ec='darkgreen', linewidth=2)
        ax.add_patch(start_circle)
        ax.text(0, 0, '<start>', ha='center', va='center', fontsize=8)
        
        ax.set_xlim(-1, steps * x_spacing)
        ax.set_ylim(-4, 4)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f'Beam Search Process (beam_width={beam_width})', 
                    fontsize=14, weight='bold')
        
        # 凡例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='lightblue', edgecolor='darkblue', 
                  label='Selected beam'),
            Patch(facecolor='lightgray', edgecolor='gray', 
                  label='Pruned beam', alpha=0.3)
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.show()
    
    def diverse_beam_search(self, num_beams: int = 4, 
                           num_beam_groups: int = 2,
                           diversity_penalty: float = 0.5):
        """Diverse Beam Searchの説明"""
        print("\n=== Diverse Beam Search ===\n")
        
        print("目的: ビームサーチの多様性を向上")
        print(f"設定: {num_beams}ビーム, {num_beam_groups}グループ\n")
        
        print("アルゴリズム:")
        print("1. ビームをグループに分割")
        print("2. 各グループは異なるペナルティで探索")
        print("3. グループ間の多様性を促進\n")
        
        # 例
        groups = []
        beams_per_group = num_beams // num_beam_groups
        
        for g in range(num_beam_groups):
            print(f"グループ {g+1}:")
            group_beams = []
            
            for b in range(beams_per_group):
                # 仮想的な候補
                candidate = f"Beam_{g}_{b}: " + " ".join([
                    f"token_{np.random.randint(100)}" 
                    for _ in range(5)
                ])
                group_beams.append(candidate)
                print(f"  {candidate}")
            
            groups.append(group_beams)
        
        print(f"\n多様性ペナルティ: {diversity_penalty}")
        print("→ グループ間で異なる文を生成")

## 20.3 制約付き生成

class ConstrainedGeneration:
    """制約付き生成の実装"""
    
    def __init__(self):
        self.constraints = []
        
    def explain_constraints(self):
        """制約の種類を説明"""
        print("=== 制約付き生成 ===\n")
        
        constraints = {
            "長さ制約": {
                "説明": "最小/最大長の指定",
                "例": "min_length=10, max_length=50",
                "実装": "長さに応じてEOSトークンをマスク"
            },
            
            "内容制約": {
                "説明": "特定のキーワードを含む/含まない",
                "例": "must_include=['AI'], must_exclude=['危険']",
                "実装": "制約を満たすまで再サンプリング"
            },
            
            "文法制約": {
                "説明": "文法的に正しい文のみ生成",
                "例": "JSON形式、プログラムコード",
                "実装": "文法に基づくマスキング"
            },
            
            "意味制約": {
                "説明": "特定の意味やトーンを維持",
                "例": "ポジティブな文、専門的な文体",
                "実装": "分類器でフィルタリング"
            }
        }
        
        for name, details in constraints.items():
            print(f"{name}:")
            for key, value in details.items():
                print(f"  {key}: {value}")
            print()
    
    def implement_length_constraint(self, logits: torch.Tensor,
                                  current_length: int,
                                  min_length: int = 10,
                                  max_length: int = 50,
                                  eos_token_id: int = 2) -> torch.Tensor:
        """長さ制約の実装"""
        # 最小長未満ではEOSを禁止
        if current_length < min_length:
            logits[..., eos_token_id] = -float('inf')
        
        # 最大長に達したらEOSを強制
        if current_length >= max_length - 1:
            mask = torch.ones_like(logits) * -float('inf')
            mask[..., eos_token_id] = 0
            logits = logits + mask
        
        return logits
    
    def implement_keyword_constraint(self, generated_text: str,
                                   must_include: List[str],
                                   must_exclude: List[str]) -> bool:
        """キーワード制約のチェック"""
        # 含むべきキーワード
        for keyword in must_include:
            if keyword not in generated_text:
                return False
        
        # 含まないべきキーワード
        for keyword in must_exclude:
            if keyword in generated_text:
                return False
        
        return True
    
    def guided_generation_example(self):
        """ガイド付き生成の例"""
        print("\n=== ガイド付き生成の例 ===\n")
        
        # JSON生成の例
        print("例1: JSON形式の生成")
        
        json_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"},
                "skills": {"type": "array", "items": {"type": "string"}}
            }
        }
        
        print("スキーマ:")
        print(json_schema)
        
        print("\n生成プロセス:")
        print("1. '{' → 必須")
        print("2. '\"name\"' → プロパティ名を提案")
        print("3. ':' → 必須")
        print("4. '\"...\"' → 文字列値のみ許可")
        print("5. ',' or '}' → 文法に従って選択")
        
        # 状態遷移の可視化
        self._visualize_json_generation()
    
    def _visualize_json_generation(self):
        """JSON生成の状態遷移を可視化"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 状態
        states = {
            "start": (2, 4),
            "{": (4, 4),
            "key": (6, 5),
            ":": (8, 5),
            "value": (10, 5),
            ",": (10, 3),
            "}": (12, 4)
        }
        
        # 遷移
        transitions = [
            ("start", "{", "必須"),
            ("{", "key", "プロパティ名"),
            ("key", ":", "必須"),
            (":", "value", "型に応じて"),
            ("value", ",", "続きあり"),
            ("value", "}", "終了"),
            (",", "key", "次のプロパティ")
        ]
        
        # 状態を描画
        for state, (x, y) in states.items():
            if state == "start":
                color = 'lightgreen'
            elif state == "}":
                color = 'lightcoral'
            else:
                color = 'lightblue'
            
            circle = plt.Circle((x, y), 0.6, color=color,
                              ec='black', linewidth=2)
            ax.add_patch(circle)
            ax.text(x, y, state, ha='center', va='center',
                   fontsize=10, weight='bold')
        
        # 遷移を描画
        for from_state, to_state, label in transitions:
            from_pos = states[from_state]
            to_pos = states[to_state]
            
            # 矢印
            if from_state == "," and to_state == "key":
                # 曲線矢印
                ax.annotate('', xy=to_pos, xytext=from_pos,
                          arrowprops=dict(arrowstyle='->', lw=2,
                                        connectionstyle="arc3,rad=-.5"))
            else:
                ax.annotate('', xy=to_pos, xytext=from_pos,
                          arrowprops=dict(arrowstyle='->', lw=2))
            
            # ラベル
            mid_x = (from_pos[0] + to_pos[0]) / 2
            mid_y = (from_pos[1] + to_pos[1]) / 2
            
            if from_state == "," and to_state == "key":
                mid_y -= 0.5
            
            ax.text(mid_x, mid_y + 0.3, label, ha='center',
                   fontsize=8, style='italic',
                   bbox=dict(boxstyle="round,pad=0.3",
                           facecolor='white', alpha=0.8))
        
        ax.set_xlim(0, 14)
        ax.set_ylim(1, 7)
        ax.axis('off')
        ax.set_title('JSON Generation State Machine', 
                    fontsize=14, weight='bold')
        
        plt.tight_layout()
        plt.show()

## 20.4 効率的な推論

class EfficientInference:
    """効率的な推論技術"""
    
    def explain_optimization_techniques(self):
        """最適化技術の説明"""
        print("=== 推論の最適化技術 ===\n")
        
        techniques = {
            "KV Cache": {
                "説明": "Key-Valueの再計算を回避",
                "削減": "計算量をO(n²)→O(n)に",
                "メモリ": "O(n)の追加メモリ必要"
            },
            
            "Flash Decoding": {
                "説明": "効率的なアテンション計算",
                "削減": "メモリ帯域幅の最適化",
                "メモリ": "大幅なメモリ削減"
            },
            
            "Speculative Decoding": {
                "説明": "小モデルで候補生成、大モデルで検証",
                "削減": "レイテンシを2-3倍高速化",
                "メモリ": "追加モデルのメモリ必要"
            },
            
            "Continuous Batching": {
                "説明": "動的なバッチ処理",
                "削減": "GPUの使用率向上",
                "メモリ": "効率的なメモリ管理"
            },
            
            "Quantization": {
                "説明": "低精度での推論",
                "削減": "メモリと計算量を削減",
                "メモリ": "2-4倍のメモリ削減"
            }
        }
        
        for name, details in techniques.items():
            print(f"{name}:")
            for key, value in details.items():
                print(f"  {key}: {value}")
            print()
        
        # パフォーマンス比較
        self._compare_performance()
    
    def _compare_performance(self):
        """パフォーマンス比較"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # レイテンシ比較
        methods = ['Baseline', 'KV Cache', 'Flash', 'Speculative', 'All']
        latencies = [100, 60, 40, 35, 20]  # ms
        colors = ['red', 'orange', 'yellow', 'lightgreen', 'green']
        
        bars = ax1.bar(methods, latencies, color=colors)
        ax1.set_ylabel('Latency (ms)')
        ax1.set_title('Inference Latency Comparison')
        
        # 値を表示
        for bar, latency in zip(bars, latencies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{latency}ms', ha='center', va='bottom')
        
        # スループット比較
        batch_sizes = [1, 4, 16, 64, 256]
        baseline_throughput = [10, 35, 120, 400, 800]
        optimized_throughput = [15, 60, 220, 800, 2000]
        
        ax2.plot(batch_sizes, baseline_throughput, 'ro-', 
                label='Baseline', linewidth=2, markersize=8)
        ax2.plot(batch_sizes, optimized_throughput, 'go-', 
                label='Optimized', linewidth=2, markersize=8)
        
        ax2.set_xlabel('Batch Size')
        ax2.set_ylabel('Throughput (tokens/sec)')
        ax2.set_title('Throughput Scaling')
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def implement_kv_cache(self):
        """KVキャッシュの実装例"""
        print("\n=== KV Cacheの実装 ===\n")
        
        code = '''
class KVCache:
    """Key-Valueキャッシュの実装"""
    
    def __init__(self, batch_size: int, max_seq_len: int,
                 n_heads: int, head_dim: int, n_layers: int):
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.n_layers = n_layers
        
        # キャッシュの初期化
        cache_shape = (batch_size, n_heads, max_seq_len, head_dim)
        self.k_cache = [torch.zeros(cache_shape) for _ in range(n_layers)]
        self.v_cache = [torch.zeros(cache_shape) for _ in range(n_layers)]
        self.cache_len = 0
    
    def update(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor):
        """キャッシュを更新"""
        seq_len = k.size(2)
        
        # 新しいK, Vをキャッシュに追加
        self.k_cache[layer_idx][:, :, self.cache_len:self.cache_len+seq_len] = k
        self.v_cache[layer_idx][:, :, self.cache_len:self.cache_len+seq_len] = v
        
        # 過去のK, Vと結合
        k_full = self.k_cache[layer_idx][:, :, :self.cache_len+seq_len]
        v_full = self.v_cache[layer_idx][:, :, :self.cache_len+seq_len]
        
        self.cache_len += seq_len
        
        return k_full, v_full
    
    def clear(self):
        """キャッシュをクリア"""
        for k, v in zip(self.k_cache, self.v_cache):
            k.zero_()
            v.zero_()
        self.cache_len = 0
'''
        
        print(code)
        
        print("\n効果:")
        print("• 各トークン生成時の計算量: O(n²) → O(n)")
        print("• メモリ使用量: O(n×d×layers×heads)")
        print("• 長いシーケンスほど効果大")

class SpeculativeDecoding:
    """投機的デコーディング"""
    
    def explain_algorithm(self):
        """アルゴリズムの説明"""
        print("=== Speculative Decoding ===\n")
        
        print("概念: 小さなモデルで「推測」、大きなモデルで「検証」\n")
        
        print("アルゴリズム:")
        print("1. ドラフトモデル（小）がK個のトークンを生成")
        print("2. ターゲットモデル（大）が一度に検証")
        print("3. 一致する部分まで採用")
        print("4. 不一致点から再開\n")
        
        # プロセスの可視化
        self._visualize_speculative_process()
    
    def _visualize_speculative_process(self):
        """投機的デコーディングのプロセスを可視化"""
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # ドラフトモデルの生成
        ax = axes[0]
        ax.set_title('Draft Model Generation', fontsize=12)
        
        draft_tokens = ['The', 'cat', 'is', 'sleeping', 'on', 'the', 'mat']
        draft_probs = [0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4]
        
        x_pos = 0
        for token, prob in zip(draft_tokens, draft_probs):
            width = 1.5
            color = plt.cm.Greens(prob)
            rect = plt.Rectangle((x_pos, 0), width, 1,
                               facecolor=color, edgecolor='black')
            ax.add_patch(rect)
            ax.text(x_pos + width/2, 0.5, token, ha='center', va='center')
            ax.text(x_pos + width/2, 1.2, f'{prob:.2f}', ha='center',
                   fontsize=8, color='darkgreen')
            x_pos += width + 0.1
        
        ax.set_xlim(-0.5, x_pos)
        ax.set_ylim(-0.5, 2)
        ax.axis('off')
        ax.text(x_pos/2, -0.3, 'Fast generation (low latency)',
               ha='center', style='italic')
        
        # ターゲットモデルの検証
        ax = axes[1]
        ax.set_title('Target Model Verification', fontsize=12)
        
        # 検証結果（5番目で不一致）
        verification = [True, True, True, True, False, False, False]
        
        x_pos = 0
        for i, (token, verified) in enumerate(zip(draft_tokens, verification)):
            width = 1.5
            if verified:
                color = 'lightgreen'
                edge_color = 'green'
                edge_width = 3
            else:
                color = 'lightcoral'
                edge_color = 'red'
                edge_width = 1
            
            rect = plt.Rectangle((x_pos, 0), width, 1,
                               facecolor=color, edgecolor=edge_color,
                               linewidth=edge_width)
            ax.add_patch(rect)
            ax.text(x_pos + width/2, 0.5, token, ha='center', va='center')
            
            if verified:
                ax.text(x_pos + width/2, 1.2, '✓', ha='center',
                       fontsize=16, color='green')
            else:
                ax.text(x_pos + width/2, 1.2, '✗', ha='center',
                       fontsize=16, color='red')
            
            x_pos += width + 0.1
        
        # 採用範囲を示す
        accept_width = 4 * (1.5 + 0.1) - 0.1
        accept_rect = plt.Rectangle((-0.3, -0.3), accept_width + 0.6, 1.6,
                                  facecolor='none', edgecolor='green',
                                  linewidth=3, linestyle='--')
        ax.add_patch(accept_rect)
        ax.text(accept_width/2, -0.6, 'Accepted tokens',
               ha='center', color='green', weight='bold')
        
        ax.set_xlim(-0.5, x_pos)
        ax.set_ylim(-1, 2)
        ax.axis('off')
        ax.text(x_pos/2, -0.9, 'Parallel verification (high throughput)',
               ha='center', style='italic')
        
        plt.tight_layout()
        plt.show()
        
        print("\n結果: 4トークンが一度に確定（4倍の高速化）")

# デモとまとめ
def run_inference_demo():
    """推論技術のデモを実行"""
    print("=" * 70)
    print("推論技術の詳解")
    print("=" * 70 + "\n")
    
    # 1. サンプリング戦略
    sampling = SamplingStrategies()
    sampling.explain_sampling_methods()
    
    # 2. 高度なサンプリング
    print("\n")
    advanced = AdvancedSampling()
    
    # Top-k/Top-pフィルタリングの例
    logits = torch.randn(100)
    filtered = advanced.top_k_top_p_filtering(logits, top_k=10, top_p=0.9)
    print("Top-k/Top-pフィルタリング適用")
    
    # Contrastive Search
    dummy_input = torch.tensor([[1, 2, 3]])
    advanced.contrastive_search(dummy_input, max_length=10)
    
    # Typical Sampling
    advanced.typical_sampling(logits, typical_p=0.95)
    
    # 3. ビームサーチ
    print("\n")
    beam = BeamSearchMethods()
    beam.explain_beam_search()
    beam.diverse_beam_search()
    
    # 4. 制約付き生成
    print("\n")
    constrained = ConstrainedGeneration()
    constrained.explain_constraints()
    constrained.guided_generation_example()
    
    # 5. 効率的な推論
    print("\n")
    efficient = EfficientInference()
    efficient.explain_optimization_techniques()
    efficient.implement_kv_cache()
    
    # 6. 投機的デコーディング
    print("\n")
    speculative = SpeculativeDecoding()
    speculative.explain_algorithm()
    
    print("\n" + "=" * 70)
    print("まとめ")
    print("=" * 70)
    print("\n推論技術の要点:")
    print("• サンプリング: 品質と多様性のバランス")
    print("• ビームサーチ: 高品質だが計算コスト高")
    print("• 制約付き生成: タスク特有の要求に対応")
    print("• 最適化技術: 実用的な速度を実現")
    print("\n適切な推論技術の選択により、")
    print("モデルの潜在能力を最大限に引き出せます。")

if __name__ == "__main__":
    run_inference_demo()