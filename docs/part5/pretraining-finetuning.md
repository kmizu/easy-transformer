# 事前学習とファインチューニング

## はじめに：知識の転移

コンパイラ開発で、既存のパーサージェネレータやライブラリを活用することを考えてください。ゼロから作るのではなく、汎用的な基盤の上に特定の言語仕様を実装します。深層学習における事前学習とファインチューニングも同じ考え方です。

大規模な汎用コーパスで基礎的な言語理解を学習し（事前学習）、その後特定のタスクに適応させる（ファインチューニング）。この二段階のアプローチが、現代のNLPの成功の鍵となっています。

## 18.1 事前学習の仕組み

### 自己教師あり学習の威力

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import random
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import time
from tqdm import tqdm

class PretrainingObjectives:
    """事前学習の目的関数"""
    
    def explain_objectives(self):
        """主要な事前学習目的を説明"""
        print("=== 事前学習の目的関数 ===\n")
        
        objectives = {
            "Causal Language Modeling (CLM)": {
                "説明": "次のトークンを予測（GPT系）",
                "例": "The cat sat on the [PREDICT]",
                "利点": "自然な生成能力",
                "欠点": "単方向の文脈のみ"
            },
            
            "Masked Language Modeling (MLM)": {
                "説明": "マスクされたトークンを予測（BERT系）",
                "例": "The [MASK] sat on the mat",
                "利点": "双方向の文脈を活用",
                "欠点": "生成タスクに不向き"
            },
            
            "Permutation Language Modeling": {
                "説明": "ランダムな順序で予測（XLNet）",
                "例": "順列: [2,4,1,3] で予測",
                "利点": "双方向性と自己回帰の両立",
                "欠点": "実装が複雑"
            },
            
            "Denoising Objectives": {
                "説明": "破損したテキストを復元（T5, BART）",
                "例": "The <X> on <Y> mat → The cat sat on the mat",
                "利点": "柔軟な事前学習",
                "欠点": "計算コストが高い"
            }
        }
        
        for name, details in objectives.items():
            print(f"{name}:")
            for key, value in details.items():
                print(f"  {key}: {value}")
            print()
        
        # 可視化
        self._visualize_objectives()
    
    def _visualize_objectives(self):
        """目的関数を可視化"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Causal LM
        ax = axes[0, 0]
        tokens = ["The", "cat", "sat", "on", "the", "mat"]
        positions = range(len(tokens))
        
        # トークンを表示
        for i, token in enumerate(tokens):
            rect = plt.Rectangle((i, 0), 1, 1, facecolor='lightblue', 
                               edgecolor='black')
            ax.add_patch(rect)
            ax.text(i+0.5, 0.5, token, ha='center', va='center')
        
        # 予測の矢印
        for i in range(len(tokens)-1):
            ax.arrow(i+0.5, 1.2, 0.5, 0, head_width=0.1, 
                    head_length=0.1, fc='red', ec='red')
            ax.text(i+1, 1.5, "→", ha='center', color='red')
        
        ax.set_xlim(-0.5, len(tokens))
        ax.set_ylim(-0.5, 2)
        ax.set_title('Causal Language Modeling', fontsize=12)
        ax.axis('off')
        
        # 2. Masked LM
        ax = axes[0, 1]
        masked_tokens = ["The", "[MASK]", "sat", "on", "[MASK]", "mat"]
        
        for i, token in enumerate(masked_tokens):
            if token == "[MASK]":
                color = 'lightcoral'
            else:
                color = 'lightblue'
            rect = plt.Rectangle((i, 0), 1, 1, facecolor=color,
                               edgecolor='black')
            ax.add_patch(rect)
            ax.text(i+0.5, 0.5, token, ha='center', va='center',
                   fontsize=8 if token == "[MASK]" else 10)
        
        # 双方向の矢印
        for mask_pos in [1, 4]:
            for j in range(len(masked_tokens)):
                if j != mask_pos:
                    ax.arrow(j+0.5, -0.3, 
                            (mask_pos-j)*0.8, 0, 
                            head_width=0.05, head_length=0.05,
                            fc='blue', ec='blue', alpha=0.3)
        
        ax.set_xlim(-0.5, len(masked_tokens))
        ax.set_ylim(-0.5, 1.5)
        ax.set_title('Masked Language Modeling', fontsize=12)
        ax.axis('off')
        
        # 3. Permutation LM
        ax = axes[1, 0]
        perm_order = [2, 0, 3, 1, 4]  # 例
        perm_tokens = ["The", "cat", "sat", "on", "mat"]
        
        # 元の順序
        for i, token in enumerate(perm_tokens):
            rect = plt.Rectangle((i, 1), 1, 0.5, facecolor='lightgreen',
                               edgecolor='black')
            ax.add_patch(rect)
            ax.text(i+0.5, 1.25, token, ha='center', va='center', fontsize=9)
        
        # 予測順序
        for pred_idx, orig_idx in enumerate(perm_order):
            rect = plt.Rectangle((pred_idx, 0), 1, 0.5, facecolor='lightyellow',
                               edgecolor='black')
            ax.add_patch(rect)
            ax.text(pred_idx+0.5, 0.25, perm_tokens[orig_idx], 
                   ha='center', va='center', fontsize=9)
            
            # 矢印
            ax.arrow(orig_idx+0.5, 0.9, 
                    (pred_idx-orig_idx)*0.9, -0.35,
                    head_width=0.05, head_length=0.05,
                    fc='purple', ec='purple', alpha=0.5)
        
        ax.set_xlim(-0.5, len(perm_tokens))
        ax.set_ylim(-0.2, 2)
        ax.set_title('Permutation Language Modeling', fontsize=12)
        ax.text(2.5, -0.1, 'Prediction Order', ha='center', fontsize=8)
        ax.text(2.5, 1.7, 'Original Order', ha='center', fontsize=8)
        ax.axis('off')
        
        # 4. Denoising
        ax = axes[1, 1]
        corrupted = ["The", "<X>", "on", "<Y>"]
        original = ["The", "cat sat", "on", "the mat"]
        
        # 破損版
        x_pos = 0
        for token in corrupted:
            width = 2 if token in ["<X>", "<Y>"] else 1
            rect = plt.Rectangle((x_pos, 1), width, 0.5, 
                               facecolor='lightcoral' if "<" in token else 'lightblue',
                               edgecolor='black')
            ax.add_patch(rect)
            ax.text(x_pos+width/2, 1.25, token, ha='center', va='center')
            x_pos += width
        
        # 矢印
        ax.arrow(3, 0.9, 0, -0.3, head_width=0.3, head_length=0.1,
                fc='green', ec='green', linewidth=2)
        
        # 復元版
        x_pos = 0
        for token in original:
            width = len(token.split()) * 0.8
            rect = plt.Rectangle((x_pos, 0), width, 0.5,
                               facecolor='lightgreen', edgecolor='black')
            ax.add_patch(rect)
            ax.text(x_pos+width/2, 0.25, token, ha='center', va='center',
                   fontsize=8)
            x_pos += width
        
        ax.set_xlim(-0.5, 6)
        ax.set_ylim(-0.2, 2)
        ax.set_title('Denoising Objective', fontsize=12)
        ax.text(3, 0.65, 'Reconstruct', ha='center', fontsize=8, color='green')
        ax.axis('off')
        
        plt.tight_layout()
        plt.show()

class PretrainingDataset(Dataset):
    """事前学習用データセット"""
    
    def __init__(self, texts: List[str], tokenizer, max_length: int = 512,
                 objective: str = "clm"):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.objective = objective
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # トークン化
        tokens = self.tokenizer.encode(text, max_length=self.max_length,
                                     truncation=True)
        
        if self.objective == "clm":
            # Causal LM: inputとlabelは同じ（シフトは後で）
            return {
                "input_ids": torch.tensor(tokens),
                "labels": torch.tensor(tokens)
            }
        
        elif self.objective == "mlm":
            # Masked LM
            masked_tokens, labels = self._mask_tokens(tokens)
            return {
                "input_ids": torch.tensor(masked_tokens),
                "labels": torch.tensor(labels)
            }
        
        elif self.objective == "denoising":
            # Denoising
            corrupted, original = self._corrupt_tokens(tokens)
            return {
                "input_ids": torch.tensor(corrupted),
                "labels": torch.tensor(original)
            }
    
    def _mask_tokens(self, tokens: List[int], mask_prob: float = 0.15):
        """トークンをマスク"""
        masked_tokens = tokens.copy()
        labels = [-100] * len(tokens)  # -100 = ignore in loss
        
        # マスク位置をランダムに選択
        mask_indices = np.random.binomial(1, mask_prob, size=len(tokens)).astype(bool)
        
        for i, mask in enumerate(mask_indices):
            if mask:
                labels[i] = tokens[i]
                
                # 80%: [MASK]トークンに置換
                if random.random() < 0.8:
                    masked_tokens[i] = self.tokenizer.mask_token_id
                # 10%: ランダムなトークンに置換
                elif random.random() < 0.5:
                    masked_tokens[i] = random.randint(0, len(self.tokenizer) - 1)
                # 10%: そのまま
        
        return masked_tokens, labels
    
    def _corrupt_tokens(self, tokens: List[int]):
        """トークンを破損させる（T5スタイル）"""
        # 簡略版：連続するトークンをマスク
        corrupted = []
        original = tokens.copy()
        
        i = 0
        while i < len(tokens):
            if random.random() < 0.15:  # 15%の確率でスパンをマスク
                span_length = np.random.poisson(3)  # 平均長3
                span_length = min(span_length, len(tokens) - i)
                
                # スパンを特殊トークンで置換
                corrupted.append(self.tokenizer.mask_token_id)
                i += span_length
            else:
                corrupted.append(tokens[i])
                i += 1
        
        return corrupted, original

class PretrainingPipeline:
    """事前学習パイプライン"""
    
    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # 最適化
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # 統計
        self.stats = defaultdict(list)
        
    def _create_optimizer(self):
        """オプティマイザーの作成"""
        # Weight decayを適用するパラメータとしないパラメータを分離
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if "bias" in name or "norm" in name or "embedding" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        optimizer_grouped_parameters = [
            {"params": decay_params, "weight_decay": self.config["weight_decay"]},
            {"params": no_decay_params, "weight_decay": 0.0}
        ]
        
        return torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.config["learning_rate"],
            betas=(0.9, 0.999),
            eps=1e-8
        )
    
    def _create_scheduler(self):
        """学習率スケジューラーの作成"""
        # Linear warmup + Cosine decay
        def lr_lambda(step):
            if step < self.config["warmup_steps"]:
                return step / self.config["warmup_steps"]
            else:
                progress = (step - self.config["warmup_steps"]) / \
                          (self.config["total_steps"] - self.config["warmup_steps"])
                return 0.5 * (1 + np.cos(np.pi * progress))
        
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def train(self, train_dataloader: DataLoader, 
              val_dataloader: Optional[DataLoader] = None):
        """事前学習の実行"""
        print("=== 事前学習開始 ===\n")
        
        global_step = 0
        best_val_loss = float('inf')
        
        for epoch in range(self.config["num_epochs"]):
            print(f"\nEpoch {epoch + 1}/{self.config['num_epochs']}")
            
            # 訓練
            train_loss = self._train_epoch(train_dataloader, global_step)
            self.stats["train_loss"].append(train_loss)
            
            # 検証
            if val_dataloader is not None:
                val_loss = self._validate(val_dataloader)
                self.stats["val_loss"].append(val_loss)
                
                print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                
                # チェックポイント保存
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self._save_checkpoint(epoch, val_loss)
            else:
                print(f"Train Loss: {train_loss:.4f}")
            
            global_step += len(train_dataloader)
        
        # 学習曲線のプロット
        self._plot_training_curves()
    
    def _train_epoch(self, dataloader: DataLoader, global_step: int):
        """1エポックの訓練"""
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(dataloader, desc="Training")
        for batch in progress_bar:
            # デバイスに転送
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            # 順伝播
            outputs = self.model(input_ids, labels=labels)
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            
            # 逆伝播
            loss.backward()
            
            # 勾配クリッピング
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config["max_grad_norm"]
            )
            
            # 最適化ステップ
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            
            # 統計更新
            total_loss += loss.item()
            
            # 進捗表示
            progress_bar.set_postfix({
                "loss": loss.item(),
                "lr": self.optimizer.param_groups[0]["lr"]
            })
            
            # 定期的なログ
            if (global_step + 1) % self.config["log_interval"] == 0:
                self.stats["step_loss"].append(loss.item())
                self.stats["learning_rate"].append(
                    self.optimizer.param_groups[0]["lr"]
                )
            
            global_step += 1
        
        return total_loss / len(dataloader)
    
    def _validate(self, dataloader: DataLoader):
        """検証"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                outputs = self.model(input_ids, labels=labels)
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
                
                total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def _save_checkpoint(self, epoch: int, val_loss: float):
        """チェックポイントの保存"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "val_loss": val_loss,
            "config": self.config
        }
        
        torch.save(checkpoint, f"checkpoint_epoch_{epoch}.pt")
        print(f"Checkpoint saved: checkpoint_epoch_{epoch}.pt")
    
    def _plot_training_curves(self):
        """学習曲線のプロット"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 損失
        epochs = range(1, len(self.stats["train_loss"]) + 1)
        ax1.plot(epochs, self.stats["train_loss"], 'b-', label='Train Loss')
        if self.stats["val_loss"]:
            ax1.plot(epochs, self.stats["val_loss"], 'r-', label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 学習率
        if self.stats["learning_rate"]:
            steps = range(len(self.stats["learning_rate"]))
            ax2.plot(steps, self.stats["learning_rate"], 'g-')
            ax2.set_xlabel('Step')
            ax2.set_ylabel('Learning Rate')
            ax2.set_title('Learning Rate Schedule')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

## 18.2 ファインチューニング戦略

class FineTuningStrategies:
    """ファインチューニング戦略"""
    
    def explain_strategies(self):
        """主要な戦略を説明"""
        print("=== ファインチューニング戦略 ===\n")
        
        strategies = {
            "Full Fine-tuning": {
                "説明": "全パラメータを更新",
                "利点": "最大の表現力",
                "欠点": "計算コストが高い、過学習のリスク",
                "パラメータ数": "100%"
            },
            
            "LoRA (Low-Rank Adaptation)": {
                "説明": "低ランク行列での適応",
                "利点": "パラメータ効率的、複数タスクの同時対応",
                "欠点": "若干の性能低下の可能性",
                "パラメータ数": "~0.1%"
            },
            
            "Prefix Tuning": {
                "説明": "プレフィックスベクトルの学習",
                "利点": "元のモデルを変更しない",
                "欠点": "長いプレフィックスが必要",
                "パラメータ数": "~0.1%"
            },
            
            "Adapter Layers": {
                "説明": "小さなアダプター層を挿入",
                "利点": "モジュラー、タスク特化",
                "欠点": "推論時のオーバーヘッド",
                "パラメータ数": "~1-5%"
            },
            
            "BitFit": {
                "説明": "バイアス項のみを更新",
                "利点": "極めてパラメータ効率的",
                "欠点": "表現力が限定的",
                "パラメータ数": "~0.05%"
            }
        }
        
        # 比較表示
        self._visualize_strategies(strategies)
        
        # 実装例
        self._implement_lora()
    
    def _visualize_strategies(self, strategies: Dict[str, Dict[str, str]]):
        """戦略を可視化"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # データ準備
        names = list(strategies.keys())
        param_percentages = []
        
        for name, info in strategies.items():
            # パーセンテージを抽出
            param_str = info["パラメータ数"]
            if "~" in param_str:
                param_str = param_str.replace("~", "")
            if "-" in param_str:
                # 範囲の場合は平均を取る
                parts = param_str.replace("%", "").split("-")
                param_percentages.append(np.mean([float(p) for p in parts]))
            else:
                param_percentages.append(float(param_str.replace("%", "")))
        
        # 棒グラフ
        colors = plt.cm.viridis(np.linspace(0, 1, len(names)))
        bars = ax.bar(names, param_percentages, color=colors)
        
        # ログスケール
        ax.set_yscale('log')
        ax.set_ylabel('Trainable Parameters (%)', fontsize=12)
        ax.set_title('Parameter Efficiency of Fine-tuning Strategies', fontsize=14)
        
        # 値を表示
        for bar, pct in zip(bars, param_percentages):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{pct:.2f}%', ha='center', va='bottom')
        
        # 回転
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.show()
    
    def _implement_lora(self):
        """LoRAの実装例"""
        print("\n=== LoRA実装例 ===\n")
        
        class LoRALayer(nn.Module):
            """LoRAレイヤー"""
            
            def __init__(self, in_features: int, out_features: int, 
                        rank: int = 16, alpha: float = 16.0):
                super().__init__()
                self.rank = rank
                self.alpha = alpha
                self.scaling = alpha / rank
                
                # 低ランク行列
                self.lora_A = nn.Parameter(torch.randn(in_features, rank))
                self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
                
                # 初期化
                nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
                
            def forward(self, x: torch.Tensor, original_weight: torch.Tensor):
                # 元の重み + LoRA
                lora_weight = (self.lora_A @ self.lora_B) * self.scaling
                return F.linear(x, original_weight + lora_weight.T)
        
        print("LoRAの特徴:")
        print("• W = W₀ + BA (低ランク分解)")
        print("• rank << min(in_features, out_features)")
        print("• 推論時に重みをマージ可能")
        print("• 複数のLoRAを切り替え可能")
        
        # パラメータ削減の計算
        in_features, out_features = 768, 768
        rank = 16
        
        original_params = in_features * out_features
        lora_params = (in_features * rank) + (rank * out_features)
        reduction = 100 * (1 - lora_params / original_params)
        
        print(f"\n例: {in_features}×{out_features}の行列")
        print(f"  元のパラメータ数: {original_params:,}")
        print(f"  LoRAパラメータ数: {lora_params:,}")
        print(f"  削減率: {reduction:.1f}%")

class TaskSpecificFineTuning:
    """タスク特化のファインチューニング"""
    
    def __init__(self, base_model: nn.Module):
        self.base_model = base_model
        
    def create_classification_head(self, num_classes: int):
        """分類ヘッドの作成"""
        
        class ClassificationModel(nn.Module):
            def __init__(self, base_model, num_classes, hidden_size=768):
                super().__init__()
                self.base_model = base_model
                self.classifier = nn.Sequential(
                    nn.Dropout(0.1),
                    nn.Linear(hidden_size, hidden_size),
                    nn.Tanh(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_size, num_classes)
                )
                
            def forward(self, input_ids, attention_mask=None):
                # ベースモデルの出力
                outputs = self.base_model(input_ids, attention_mask=attention_mask)
                
                # [CLS]トークンまたは平均プーリング
                if hasattr(outputs, 'last_hidden_state'):
                    hidden_states = outputs.last_hidden_state
                else:
                    hidden_states = outputs
                
                # プーリング（最初のトークン）
                pooled_output = hidden_states[:, 0]
                
                # 分類
                logits = self.classifier(pooled_output)
                
                return logits
        
        return ClassificationModel(self.base_model, num_classes)
    
    def create_generation_head(self, max_length: int = 512):
        """生成ヘッドの作成"""
        
        class GenerationModel(nn.Module):
            def __init__(self, base_model, max_length):
                super().__init__()
                self.base_model = base_model
                self.max_length = max_length
                
            def forward(self, input_ids, labels=None):
                outputs = self.base_model(input_ids, labels=labels)
                return outputs
            
            @torch.no_grad()
            def generate(self, input_ids, max_new_tokens=50, **kwargs):
                return self.base_model.generate(
                    input_ids, 
                    max_new_tokens=max_new_tokens,
                    **kwargs
                )
        
        return GenerationModel(self.base_model, max_length)

class FineTuningDataCollator:
    """ファインチューニング用データコレーター"""
    
    def __init__(self, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """バッチの作成"""
        # タスクに応じた処理
        if "labels" in examples[0]:
            # 分類タスク
            return self._collate_classification(examples)
        elif "target_text" in examples[0]:
            # 生成タスク
            return self._collate_generation(examples)
        else:
            # デフォルト
            return self._collate_default(examples)
    
    def _collate_classification(self, examples):
        """分類タスク用"""
        texts = [ex["text"] for ex in examples]
        labels = [ex["labels"] for ex in examples]
        
        # トークン化
        encoding = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        encoding["labels"] = torch.tensor(labels)
        return encoding
    
    def _collate_generation(self, examples):
        """生成タスク用"""
        inputs = [ex["input_text"] for ex in examples]
        targets = [ex["target_text"] for ex in examples]
        
        # 入力と出力を結合
        model_inputs = self.tokenizer(
            inputs,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # ラベルの準備
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                targets,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    def _collate_default(self, examples):
        """デフォルト処理"""
        return {key: torch.stack([ex[key] for ex in examples]) 
                for key in examples[0].keys()}

## 18.3 実践的なファインチューニング

class PracticalFineTuning:
    """実践的なファインチューニング例"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def sentiment_analysis_example(self):
        """感情分析のファインチューニング例"""
        print("=== 感情分析ファインチューニング ===\n")
        
        # データセットの例
        train_data = [
            {"text": "This movie was fantastic!", "label": 1},  # Positive
            {"text": "Terrible experience, would not recommend.", "label": 0},  # Negative
            {"text": "The food was amazing and the service excellent.", "label": 1},
            {"text": "Waste of time and money.", "label": 0},
            {"text": "Best purchase I've ever made!", "label": 1},
            {"text": "Completely disappointed with the quality.", "label": 0}
        ]
        
        # データセットクラス
        class SentimentDataset(Dataset):
            def __init__(self, data, tokenizer, max_length=128):
                self.data = data
                self.tokenizer = tokenizer
                self.max_length = max_length
                
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                item = self.data[idx]
                encoding = self.tokenizer(
                    item["text"],
                    truncation=True,
                    padding="max_length",
                    max_length=self.max_length,
                    return_tensors="pt"
                )
                
                return {
                    "input_ids": encoding["input_ids"].squeeze(),
                    "attention_mask": encoding["attention_mask"].squeeze(),
                    "labels": torch.tensor(item["label"])
                }
        
        print("データセット例:")
        for item in train_data[:3]:
            print(f"  Text: '{item['text']}'")
            print(f"  Label: {item['label']} ({'Positive' if item['label'] == 1 else 'Negative'})\n")
        
        # 学習設定
        print("ファインチューニング設定:")
        print("  Learning Rate: 2e-5")
        print("  Batch Size: 16")
        print("  Epochs: 3")
        print("  Warmup Steps: 100")
        
        # 結果の可視化
        self._visualize_finetuning_results()
    
    def _visualize_finetuning_results(self):
        """ファインチューニング結果の可視化"""
        # ダミーの学習曲線
        epochs = np.arange(1, 4)
        train_loss = [0.693, 0.245, 0.089]
        val_loss = [0.672, 0.298, 0.156]
        train_acc = [0.52, 0.91, 0.98]
        val_acc = [0.55, 0.87, 0.92]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 損失
        ax1.plot(epochs, train_loss, 'b-o', label='Train Loss')
        ax1.plot(epochs, val_loss, 'r-o', label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 精度
        ax2.plot(epochs, train_acc, 'b-o', label='Train Accuracy')
        ax2.plot(epochs, val_acc, 'r-o', label='Val Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.show()
    
    def instruction_tuning_example(self):
        """Instruction Tuningの例"""
        print("\n=== Instruction Tuning ===\n")
        
        instruction_examples = [
            {
                "instruction": "Translate the following English text to French:",
                "input": "Hello, how are you today?",
                "output": "Bonjour, comment allez-vous aujourd'hui?"
            },
            {
                "instruction": "Summarize the following text in one sentence:",
                "input": "The quick brown fox jumps over the lazy dog. This pangram sentence contains every letter of the English alphabet at least once.",
                "output": "This is a pangram that includes all 26 letters of the English alphabet."
            },
            {
                "instruction": "Write a Python function that calculates the factorial of a number:",
                "input": "5",
                "output": "def factorial(n):\n    if n == 0 or n == 1:\n        return 1\n    return n * factorial(n - 1)\n\nresult = factorial(5)  # Returns 120"
            }
        ]
        
        print("Instruction Tuning形式:")
        for i, example in enumerate(instruction_examples[:2]):
            print(f"\n例 {i+1}:")
            print(f"Instruction: {example['instruction']}")
            print(f"Input: {example['input']}")
            print(f"Output: {example['output']}")
        
        # プロンプトテンプレート
        print("\n\nプロンプトテンプレート:")
        template = """### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""
        
        print(template)
        
        print("\n効果:")
        print("✓ 明確な指示に従う能力の向上")
        print("✓ ゼロショット汎化の改善")
        print("✓ より自然な対話が可能")

## 18.4 効率的な学習手法

class EfficientTrainingMethods:
    """効率的な学習手法"""
    
    def demonstrate_mixed_precision(self):
        """Mixed Precision Trainingのデモ"""
        print("=== Mixed Precision Training ===\n")
        
        print("通常の学習 (FP32):")
        print("  メモリ使用量: 100%")
        print("  計算速度: 1.0x")
        print("  数値精度: 高い\n")
        
        print("Mixed Precision (FP16 + FP32):")
        print("  メモリ使用量: ~50%")
        print("  計算速度: 2-3x")
        print("  数値精度: 動的ロススケーリングで維持\n")
        
        # 実装例
        print("PyTorch実装例:")
        print("""
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()
    
    # 自動混合精度
    with autocast():
        outputs = model(batch['input_ids'])
        loss = criterion(outputs, batch['labels'])
    
    # スケールされた逆伝播
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
""")
    
    def demonstrate_gradient_accumulation(self):
        """勾配累積のデモ"""
        print("\n=== Gradient Accumulation ===\n")
        
        print("効果的なバッチサイズの増加:")
        print("  実バッチサイズ: 8")
        print("  累積ステップ: 4")
        print("  効果的バッチサイズ: 32\n")
        
        # メモリ使用量の比較
        batch_sizes = [8, 16, 32, 64, 128]
        memory_usage = [2, 4, 8, 16, 32]  # GB
        effective_batch_with_accumulation = [32, 64, 128, 256, 512]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(batch_sizes))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, memory_usage, width, 
                       label='Direct (OOM risk)', color='red', alpha=0.7)
        bars2 = ax.bar(x + width/2, [2] * len(batch_sizes), width,
                       label='With Gradient Accumulation', color='green', alpha=0.7)
        
        # 効果的バッチサイズを表示
        for i, (bar, eff_batch) in enumerate(zip(bars2, effective_batch_with_accumulation)):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                   f'Eff: {eff_batch}', ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel('Actual Batch Size')
        ax.set_ylabel('Memory Usage (GB)')
        ax.set_title('Memory Usage: Direct vs Gradient Accumulation')
        ax.set_xticks(x)
        ax.set_xticklabels(batch_sizes)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # GPUメモリ制限ライン
        ax.axhline(y=16, color='orange', linestyle='--', 
                  label='GPU Memory Limit (16GB)')
        
        plt.tight_layout()
        plt.show()
    
    def demonstrate_data_parallelism(self):
        """データ並列のデモ"""
        print("\n=== Data Parallelism ===\n")
        
        strategies = {
            "Single GPU": {
                "GPUs": 1,
                "Batch/GPU": 8,
                "Total Batch": 8,
                "Speed": "1x"
            },
            "Data Parallel": {
                "GPUs": 4,
                "Batch/GPU": 8,
                "Total Batch": 32,
                "Speed": "~3.8x"
            },
            "Distributed Data Parallel": {
                "GPUs": 4,
                "Batch/GPU": 8,
                "Total Batch": 32,
                "Speed": "~3.95x"
            },
            "FSDP (Fully Sharded)": {
                "GPUs": 4,
                "Batch/GPU": 16,
                "Total Batch": 64,
                "Speed": "~3.9x"
            }
        }
        
        print("並列化戦略の比較:\n")
        for name, details in strategies.items():
            print(f"{name}:")
            for key, value in details.items():
                print(f"  {key}: {value}")
            print()

# 実行とデモ
def run_pretraining_finetuning_demo():
    """事前学習とファインチューニングのデモ"""
    print("=" * 70)
    print("事前学習とファインチューニングの詳解")
    print("=" * 70 + "\n")
    
    # 1. 事前学習目的関数
    objectives = PretrainingObjectives()
    objectives.explain_objectives()
    
    # 2. 事前学習パイプライン（概要のみ）
    print("\n=== 事前学習パイプラインの例 ===")
    config = {
        "learning_rate": 6e-4,
        "weight_decay": 0.01,
        "warmup_steps": 10000,
        "total_steps": 1000000,
        "num_epochs": 1,
        "log_interval": 100,
        "max_grad_norm": 1.0
    }
    
    print("典型的な設定:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # 3. ファインチューニング戦略
    print("\n")
    strategies = FineTuningStrategies()
    strategies.explain_strategies()
    
    # 4. 実践例
    print("\n")
    practical = PracticalFineTuning()
    practical.sentiment_analysis_example()
    practical.instruction_tuning_example()
    
    # 5. 効率的な学習
    print("\n")
    efficient = EfficientTrainingMethods()
    efficient.demonstrate_mixed_precision()
    efficient.demonstrate_gradient_accumulation()
    efficient.demonstrate_data_parallelism()
    
    print("\n" + "=" * 70)
    print("まとめ")
    print("=" * 70)
    print("\n事前学習とファインチューニングのポイント:")
    print("• 事前学習: 大規模データで汎用的な言語理解を獲得")
    print("• ファインチューニング: 特定タスクへの効率的な適応")
    print("• パラメータ効率的手法: 少ないリソースで高性能を実現")
    print("• 最適化技術: Mixed Precision、勾配累積などで効率化")
    print("\nこれらの技術により、限られたリソースでも")
    print("高性能なモデルの開発が可能になりました。")

if __name__ == "__main__":
    run_pretraining_finetuning_demo()