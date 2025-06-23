# トークナイザーの詳細

## はじめに：言語の原子

コンパイラの字句解析器（レキサー）を作ったことを思い出してください。ソースコードを意味のある最小単位（トークン）に分割する作業です。`if (x > 10) { return true; }` を `IF`, `LPAREN`, `IDENT(x)`, `GT`, `NUMBER(10)`, ... に分解します。

自然言語のトークナイザーも同じ役割を果たしますが、プログラミング言語と違って明確な規則がありません。単語？文字？それとも何か別の単位？この章では、現代のトークナイザーがどのように言語を「原子」に分解し、なぜそれが重要なのかを探ります。

## 19.1 トークン化の基礎

### なぜトークン化が重要か

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Set
from collections import Counter, defaultdict
import regex as re  # Unicodeサポートが優れている
import json
from dataclasses import dataclass
import heapq
from tqdm import tqdm

class TokenizationBasics:
    """トークン化の基礎概念"""
    
    def explain_tokenization_challenges(self):
        """トークン化の課題を説明"""
        print("=== トークン化の課題 ===\n")
        
        # 様々な言語での例
        examples = {
            "英語": {
                "text": "I don't think it's working.",
                "word_tokens": ["I", "don't", "think", "it's", "working", "."],
                "char_tokens": list("I don't think it's working."),
                "challenges": "縮約形（don't, it's）の扱い"
            },
            
            "日本語": {
                "text": "私は猫が好きです。",
                "word_tokens": ["私", "は", "猫", "が", "好き", "です", "。"],
                "char_tokens": list("私は猫が好きです。"),
                "challenges": "単語境界が不明確"
            },
            
            "ドイツ語": {
                "text": "Donaudampfschifffahrtsgesellschaft",
                "word_tokens": ["Donaudampfschifffahrtsgesellschaft"],
                "char_tokens": list("Donaudampfschifffahrtsgesellschaft"),
                "challenges": "複合語が非常に長い"
            },
            
            "中国語": {
                "text": "我喜欢吃苹果",
                "word_tokens": ["我", "喜欢", "吃", "苹果"],
                "char_tokens": list("我喜欢吃苹果"),
                "challenges": "スペースがない"
            }
        }
        
        for lang, info in examples.items():
            print(f"{lang}:")
            print(f"  テキスト: {info['text']}")
            print(f"  単語トークン: {info['word_tokens'][:5]}...")
            print(f"  文字トークン: {info['char_tokens'][:10]}...")
            print(f"  課題: {info['challenges']}\n")
        
        # トークン化手法の比較
        self._compare_tokenization_methods()
    
    def _compare_tokenization_methods(self):
        """トークン化手法の比較"""
        print("=== トークン化手法の比較 ===\n")
        
        methods = {
            "Word-level": {
                "vocab_size": "50,000-200,000",
                "OOV_handling": "Poor",
                "morphology": "No",
                "efficiency": "Low"
            },
            "Character-level": {
                "vocab_size": "100-1,000",
                "OOV_handling": "Perfect",
                "morphology": "Implicit",
                "efficiency": "Very Low"
            },
            "Subword (BPE/WordPiece)": {
                "vocab_size": "10,000-100,000",
                "OOV_handling": "Good",
                "morphology": "Partial",
                "efficiency": "High"
            },
            "SentencePiece": {
                "vocab_size": "10,000-100,000",
                "OOV_handling": "Good",
                "morphology": "Partial",
                "efficiency": "High"
            }
        }
        
        # 表形式で表示
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axis('tight')
        ax.axis('off')
        
        # ヘッダー
        headers = ["Method", "Vocab Size", "OOV Handling", "Morphology", "Efficiency"]
        cell_data = []
        
        for method, props in methods.items():
            row = [method] + list(props.values())
            cell_data.append(row)
        
        # テーブル作成
        table = ax.table(cellText=cell_data, colLabels=headers,
                        cellLoc='center', loc='center',
                        colWidths=[0.25, 0.2, 0.15, 0.15, 0.15])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # スタイリング
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # 色分け
        colors = ['#ffebee', '#e8f5e9', '#fff3e0', '#e3f2fd']
        for i, color in enumerate(colors):
            for j in range(len(headers)):
                table[(i+1, j)].set_facecolor(color)
        
        plt.title('Comparison of Tokenization Methods', fontsize=14, weight='bold', pad=20)
        plt.show()

## 19.2 Byte Pair Encoding (BPE)

class BytePairEncoding:
    """BPEトークナイザーの実装"""
    
    def __init__(self):
        self.vocab = {}
        self.merges = []
        
    def train(self, texts: List[str], vocab_size: int = 1000):
        """BPEの学習"""
        print("=== BPE学習プロセス ===\n")
        
        # 初期語彙（文字レベル）
        word_freqs = defaultdict(int)
        for text in texts:
            words = text.split()
            for word in words:
                word_freqs[' '.join(list(word) + ['</w>'])] += 1
        
        # 初期語彙を作成
        self.vocab = {}
        for word, freq in word_freqs.items():
            for char in word.split():
                if char not in self.vocab:
                    self.vocab[char] = len(self.vocab)
        
        print(f"初期語彙サイズ: {len(self.vocab)}")
        print(f"初期語彙例: {list(self.vocab.keys())[:10]}\n")
        
        # マージ操作
        num_merges = vocab_size - len(self.vocab)
        
        for i in tqdm(range(num_merges), desc="Learning merges"):
            # ペアの頻度を計算
            pair_freqs = self._get_pair_frequencies(word_freqs)
            
            if not pair_freqs:
                break
            
            # 最頻出ペアを選択
            best_pair = max(pair_freqs, key=pair_freqs.get)
            self.merges.append(best_pair)
            
            # 語彙を更新
            new_token = ''.join(best_pair)
            self.vocab[new_token] = len(self.vocab)
            
            # コーパスを更新
            word_freqs = self._merge_pair(word_freqs, best_pair)
            
            # 進捗表示
            if (i + 1) % 100 == 0:
                print(f"マージ {i+1}: {best_pair} → {new_token}")
        
        print(f"\n最終語彙サイズ: {len(self.vocab)}")
        
        # 学習結果の可視化
        self._visualize_vocabulary()
    
    def _get_pair_frequencies(self, word_freqs: Dict[str, int]) -> Dict[Tuple[str, str], int]:
        """ペアの頻度を計算"""
        pair_freqs = defaultdict(int)
        
        for word, freq in word_freqs.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pair_freqs[(symbols[i], symbols[i + 1])] += freq
        
        return pair_freqs
    
    def _merge_pair(self, word_freqs: Dict[str, int], 
                    pair: Tuple[str, str]) -> Dict[str, int]:
        """ペアをマージ"""
        new_word_freqs = {}
        bigram = ' '.join(pair)
        replacement = ''.join(pair)
        
        for word, freq in word_freqs.items():
            new_word = word.replace(bigram, replacement)
            new_word_freqs[new_word] = freq
        
        return new_word_freqs
    
    def encode(self, text: str) -> List[int]:
        """テキストをエンコード"""
        words = text.split()
        tokens = []
        
        for word in words:
            # 単語を文字に分割
            word_tokens = list(word) + ['</w>']
            
            # マージを適用
            for merge in self.merges:
                i = 0
                while i < len(word_tokens) - 1:
                    if (word_tokens[i], word_tokens[i + 1]) == merge:
                        word_tokens = word_tokens[:i] + [''.join(merge)] + word_tokens[i + 2:]
                    else:
                        i += 1
            
            # トークンIDに変換
            for token in word_tokens:
                if token in self.vocab:
                    tokens.append(self.vocab[token])
                else:
                    # Unknown token
                    tokens.append(self.vocab.get('<unk>', 0))
        
        return tokens
    
    def decode(self, token_ids: List[int]) -> str:
        """トークンIDをデコード"""
        # 逆引き辞書
        id_to_token = {v: k for k, v in self.vocab.items()}
        
        tokens = [id_to_token.get(id, '<unk>') for id in token_ids]
        text = ' '.join(tokens).replace('</w> ', ' ').replace('</w>', '')
        
        return text
    
    def _visualize_vocabulary(self):
        """語彙の可視化"""
        # トークン長の分布
        token_lengths = [len(token.replace('</w>', '')) for token in self.vocab.keys()]
        
        plt.figure(figsize=(10, 6))
        plt.hist(token_lengths, bins=range(1, max(token_lengths) + 2), 
                alpha=0.7, edgecolor='black')
        plt.xlabel('Token Length (characters)')
        plt.ylabel('Count')
        plt.title('Distribution of Token Lengths in BPE Vocabulary')
        plt.grid(True, alpha=0.3)
        
        # 統計情報
        avg_length = np.mean(token_lengths)
        plt.axvline(avg_length, color='red', linestyle='--', 
                   label=f'Average: {avg_length:.2f}')
        plt.legend()
        
        plt.tight_layout()
        plt.show()

class BPEDemo:
    """BPEのデモンストレーション"""
    
    def demonstrate_bpe_process(self):
        """BPEプロセスのデモ"""
        print("=== BPEプロセスの可視化 ===\n")
        
        # サンプルテキスト
        corpus = [
            "the cat sat on the mat",
            "the dog sat on the log",
            "cats and dogs are pets"
        ]
        
        # 初期状態
        words = []
        for text in corpus:
            words.extend(text.split())
        
        # 単語を文字に分割
        word_splits = {}
        for word in set(words):
            word_splits[word] = list(word) + ['</w>']
        
        print("初期状態（文字分割）:")
        for word, splits in list(word_splits.items())[:5]:
            print(f"  {word}: {' '.join(splits)}")
        
        # マージプロセスのシミュレーション
        merges = [
            ('t', 'h'),      # th
            ('th', 'e'),     # the
            ('a', 't'),      # at
            ('s', 'at'),     # sat
            ('o', 'n'),      # on
        ]
        
        print("\n\nマージプロセス:")
        current_splits = word_splits.copy()
        
        for i, (a, b) in enumerate(merges):
            print(f"\nステップ {i+1}: '{a}' + '{b}' → '{a+b}'")
            
            # マージを適用
            for word, splits in current_splits.items():
                new_splits = []
                j = 0
                while j < len(splits):
                    if j < len(splits) - 1 and splits[j] == a and splits[j+1] == b:
                        new_splits.append(a + b)
                        j += 2
                    else:
                        new_splits.append(splits[j])
                        j += 1
                current_splits[word] = new_splits
            
            # 変更された単語を表示
            for word, splits in list(current_splits.items())[:3]:
                print(f"    {word}: {' '.join(splits)}")
        
        # 最終的なトークン化
        self._visualize_tokenization_result(current_splits)
    
    def _visualize_tokenization_result(self, word_splits: Dict[str, List[str]]):
        """トークン化結果の可視化"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # サンプル文
        sentence = "the cat sat on the mat"
        words = sentence.split()
        
        y_pos = 0.5
        x_pos = 0
        
        colors = plt.cm.Set3(np.linspace(0, 1, 20))
        color_idx = 0
        
        for word in words:
            tokens = word_splits.get(word, list(word) + ['</w>'])
            
            for token in tokens:
                if token == '</w>':
                    # 単語境界マーカー
                    width = 0.3
                    rect = plt.Rectangle((x_pos, y_pos), width, 0.3,
                                       facecolor='lightgray', 
                                       edgecolor='black', linewidth=1)
                    ax.add_patch(rect)
                    ax.text(x_pos + width/2, y_pos + 0.15, '</w>',
                           ha='center', va='center', fontsize=8)
                else:
                    # 通常のトークン
                    width = len(token) * 0.15
                    rect = plt.Rectangle((x_pos, y_pos), width, 0.3,
                                       facecolor=colors[color_idx % len(colors)],
                                       edgecolor='black', linewidth=1)
                    ax.add_patch(rect)
                    ax.text(x_pos + width/2, y_pos + 0.15, token,
                           ha='center', va='center', fontsize=10)
                    color_idx += 1
                
                x_pos += width + 0.05
            
            x_pos += 0.2  # 単語間のスペース
        
        ax.set_xlim(-0.1, x_pos)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('BPE Tokenization Result', fontsize=14, weight='bold')
        
        # 元の文を表示
        ax.text(x_pos/2, 0.9, f'Original: "{sentence}"', 
               ha='center', va='center', fontsize=12, style='italic')
        
        plt.tight_layout()
        plt.show()

## 19.3 WordPieceとSentencePiece

class WordPieceTokenizer:
    """WordPieceトークナイザーの実装"""
    
    def __init__(self):
        self.vocab = {}
        self.unk_token = '[UNK]'
        self.max_input_chars_per_word = 100
        
    def train(self, texts: List[str], vocab_size: int = 1000):
        """WordPieceの学習（簡略版）"""
        print("=== WordPiece学習 ===\n")
        
        # 初期語彙の構築
        char_counts = Counter()
        word_counts = Counter()
        
        for text in texts:
            words = text.lower().split()
            for word in words:
                word_counts[word] += 1
                for char in word:
                    char_counts[char] += 1
        
        # 基本語彙
        self.vocab = {
            '[PAD]': 0,
            '[UNK]': 1,
            '[CLS]': 2,
            '[SEP]': 3,
            '[MASK]': 4
        }
        
        # 文字を追加
        for char, count in char_counts.most_common():
            if len(self.vocab) < 100:  # 最初の100は文字用
                self.vocab[char] = len(self.vocab)
        
        # WordPieceの追加
        print(f"初期語彙サイズ: {len(self.vocab)}")
        
        # サブワード候補の生成と評価
        while len(self.vocab) < vocab_size:
            # 候補を生成
            candidates = self._generate_candidates(word_counts)
            
            if not candidates:
                break
            
            # スコアを計算（簡略版）
            best_candidate = max(candidates, 
                               key=lambda x: self._score_candidate(x, word_counts))
            
            # 語彙に追加
            if best_candidate.startswith('##'):
                self.vocab[best_candidate] = len(self.vocab)
            else:
                self.vocab['##' + best_candidate] = len(self.vocab)
            
            # 進捗
            if len(self.vocab) % 100 == 0:
                print(f"語彙サイズ: {len(self.vocab)}")
        
        print(f"\n最終語彙サイズ: {len(self.vocab)}")
        
        # WordPieceの特徴を可視化
        self._visualize_wordpiece_features()
    
    def _generate_candidates(self, word_counts: Counter) -> List[str]:
        """サブワード候補を生成"""
        candidates = set()
        
        for word in word_counts:
            for i in range(len(word)):
                for j in range(i + 1, min(i + 10, len(word) + 1)):
                    subword = word[i:j]
                    if len(subword) > 1:
                        candidates.add(subword)
        
        return list(candidates)
    
    def _score_candidate(self, candidate: str, word_counts: Counter) -> float:
        """候補のスコアを計算"""
        score = 0
        for word, count in word_counts.items():
            if candidate in word:
                score += count
        return score
    
    def tokenize(self, text: str) -> List[str]:
        """テキストをトークン化"""
        output_tokens = []
        
        for word in text.lower().split():
            if len(word) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue
            
            is_bad = False
            sub_tokens = []
            start = 0
            
            while start < len(word):
                end = len(word)
                cur_substr = None
                
                while start < end:
                    substr = word[start:end]
                    if start > 0:
                        substr = '##' + substr
                    
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    
                    end -= 1
                
                if cur_substr is None:
                    is_bad = True
                    break
                
                sub_tokens.append(cur_substr)
                start = end
            
            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        
        return output_tokens
    
    def _visualize_wordpiece_features(self):
        """WordPieceの特徴を可視化"""
        # ##プレフィックスの統計
        prefix_tokens = [token for token in self.vocab.keys() 
                        if token.startswith('##')]
        
        print(f"\n##プレフィックス付きトークン: {len(prefix_tokens)}")
        print(f"例: {prefix_tokens[:10]}")
        
        # トークン化の例
        examples = [
            "playing",
            "unbelievable",
            "internationalization"
        ]
        
        print("\n\nトークン化の例:")
        for word in examples:
            tokens = self.tokenize(word)
            print(f"  {word} → {tokens}")

class SentencePieceDemo:
    """SentencePieceのデモ"""
    
    def explain_sentencepiece(self):
        """SentencePieceの説明"""
        print("=== SentencePiece ===\n")
        
        print("特徴:")
        print("1. 言語独立:")
        print("   - 前処理不要（トークン化なし）")
        print("   - 生のテキストから直接学習")
        print("   - スペースも通常の文字として扱う\n")
        
        print("2. 可逆的:")
        print("   - デトークン化で元のテキストを完全復元")
        print("   - 情報の損失なし\n")
        
        print("3. サブワードの正規化:")
        print("   - 確率的サンプリング")
        print("   - 複数の分割候補から選択\n")
        
        # アルゴリズムの比較
        self._compare_subword_algorithms()
    
    def _compare_subword_algorithms(self):
        """サブワードアルゴリズムの比較"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 6))
        
        # BPE
        ax = axes[0]
        ax.set_title('BPE', fontsize=12, weight='bold')
        
        # BPEのマージプロセス
        bpe_steps = [
            "c a t s </w>",
            "ca t s </w>",
            "ca ts </w>",
            "cats </w>"
        ]
        
        for i, step in enumerate(bpe_steps):
            y = 0.8 - i * 0.2
            ax.text(0.5, y, step, ha='center', va='center',
                   fontsize=10, family='monospace',
                   bbox=dict(boxstyle="round,pad=0.3", 
                           facecolor='lightblue', alpha=0.7))
            
            if i < len(bpe_steps) - 1:
                ax.arrow(0.5, y - 0.05, 0, -0.1, 
                        head_width=0.05, head_length=0.02,
                        fc='black', ec='black')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.text(0.5, 0.95, 'Bottom-up Merging', ha='center', 
               fontsize=10, style='italic')
        
        # WordPiece
        ax = axes[1]
        ax.set_title('WordPiece', fontsize=12, weight='bold')
        
        # WordPieceの分割
        word = "playing"
        tokens = ["play", "##ing"]
        
        # 単語全体
        rect = plt.Rectangle((0.2, 0.5), 0.6, 0.2,
                           facecolor='lightgreen', edgecolor='black')
        ax.add_patch(rect)
        ax.text(0.5, 0.6, word, ha='center', va='center', fontsize=12)
        
        # 分割後
        x_pos = 0.2
        for token in tokens:
            width = 0.25
            rect = plt.Rectangle((x_pos, 0.2), width, 0.15,
                               facecolor='lightyellow', edgecolor='black')
            ax.add_patch(rect)
            ax.text(x_pos + width/2, 0.275, token, 
                   ha='center', va='center', fontsize=10)
            x_pos += width + 0.1
        
        # 矢印
        ax.arrow(0.5, 0.48, 0, -0.1, head_width=0.05, head_length=0.02,
                fc='black', ec='black')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.text(0.5, 0.95, 'Likelihood Maximization', ha='center',
               fontsize=10, style='italic')
        
        # SentencePiece
        ax = axes[2]
        ax.set_title('SentencePiece', fontsize=12, weight='bold')
        
        # 生テキスト
        text = "▁the▁cat"
        ax.text(0.5, 0.7, text, ha='center', va='center',
               fontsize=12, family='monospace',
               bbox=dict(boxstyle="round,pad=0.3", 
                       facecolor='lightcoral', alpha=0.7))
        
        # 複数の分割候補
        candidates = [
            ["▁the", "▁cat"],
            ["▁th", "e", "▁cat"],
            ["▁the", "▁c", "at"]
        ]
        
        y_start = 0.4
        for i, cand in enumerate(candidates):
            y = y_start - i * 0.15
            text = ' '.join(cand)
            ax.text(0.5, y, text, ha='center', va='center',
                   fontsize=9, family='monospace',
                   bbox=dict(boxstyle="round,pad=0.2",
                           facecolor='lightyellow', alpha=0.5))
        
        ax.text(0.8, 0.25, 'Sample', ha='center', fontsize=8,
               style='italic', color='red')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.text(0.5, 0.95, 'Unigram LM / BPE', ha='center',
               fontsize=10, style='italic')
        
        plt.tight_layout()
        plt.show()

## 19.4 現代的なトークナイザー

class ModernTokenizers:
    """現代的なトークナイザーの実装と比較"""
    
    def compare_modern_tokenizers(self):
        """現代的トークナイザーの比較"""
        print("=== 現代的なトークナイザー ===\n")
        
        tokenizers = {
            "GPT-2/GPT-3": {
                "type": "BPE",
                "vocab_size": "50,257",
                "special": "Byte-level BPE",
                "features": "スペース処理の改善"
            },
            
            "BERT": {
                "type": "WordPiece",
                "vocab_size": "30,522",
                "special": "##プレフィックス",
                "features": "事前トークン化必要"
            },
            
            "T5/mT5": {
                "type": "SentencePiece",
                "vocab_size": "32,000",
                "special": "▁(スペースマーカー)",
                "features": "言語独立"
            },
            
            "LLaMA": {
                "type": "SentencePiece (BPE)",
                "vocab_size": "32,000",
                "special": "Byte fallback",
                "features": "未知文字の処理"
            },
            
            "ChatGPT": {
                "type": "cl100k_base (tiktoken)",
                "vocab_size": "100,277",
                "special": "改良されたBPE",
                "features": "効率的なエンコーディング"
            }
        }
        
        # 比較表示
        self._visualize_tokenizer_comparison(tokenizers)
        
        # エンコーディング効率の比較
        self._compare_encoding_efficiency()
    
    def _visualize_tokenizer_comparison(self, tokenizers: Dict[str, Dict[str, str]]):
        """トークナイザーの比較を可視化"""
        # 語彙サイズの比較
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # 語彙サイズ
        names = list(tokenizers.keys())
        vocab_sizes = []
        for name, info in tokenizers.items():
            size_str = info["vocab_size"].replace(",", "")
            vocab_sizes.append(int(size_str))
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(names)))
        bars = ax1.bar(names, vocab_sizes, color=colors)
        
        # 値を表示
        for bar, size in zip(bars, vocab_sizes):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{size:,}', ha='center', va='bottom')
        
        ax1.set_ylabel('Vocabulary Size')
        ax1.set_title('Vocabulary Sizes of Modern Tokenizers')
        ax1.tick_params(axis='x', rotation=45)
        
        # タイプ別分類
        type_counts = Counter(info["type"].split()[0] for info in tokenizers.values())
        
        ax2.pie(type_counts.values(), labels=type_counts.keys(),
               autopct='%1.1f%%', startangle=90)
        ax2.set_title('Distribution of Tokenizer Types')
        
        plt.tight_layout()
        plt.show()
    
    def _compare_encoding_efficiency(self):
        """エンコーディング効率の比較"""
        print("\n=== エンコーディング効率の比較 ===")
        
        # サンプルテキスト
        samples = {
            "English": "The quick brown fox jumps over the lazy dog.",
            "Code": "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)",
            "Mixed": "Hello世界! 🌍 This is a test → λx.x+1",
            "URL": "https://github.com/openai/gpt-3/blob/main/model.py"
        }
        
        # 仮想的なトークン数（実際の比率に基づく）
        tokenizer_efficiency = {
            "GPT-2": {"English": 11, "Code": 24, "Mixed": 18, "URL": 35},
            "BERT": {"English": 12, "Code": 28, "Mixed": 22, "URL": 40},
            "T5": {"English": 10, "Code": 25, "Mixed": 16, "URL": 38},
            "ChatGPT": {"English": 9, "Code": 20, "Mixed": 14, "URL": 25}
        }
        
        # ヒートマップで表示
        fig, ax = plt.subplots(figsize=(10, 6))
        
        tokenizers = list(tokenizer_efficiency.keys())
        text_types = list(samples.keys())
        
        efficiency_matrix = np.array([
            [tokenizer_efficiency[tok][txt] for txt in text_types]
            for tok in tokenizers
        ])
        
        im = ax.imshow(efficiency_matrix, cmap='RdYlGn_r', aspect='auto')
        
        # ラベル
        ax.set_xticks(np.arange(len(text_types)))
        ax.set_yticks(np.arange(len(tokenizers)))
        ax.set_xticklabels(text_types)
        ax.set_yticklabels(tokenizers)
        
        # 値を表示
        for i in range(len(tokenizers)):
            for j in range(len(text_types)):
                text = ax.text(j, i, efficiency_matrix[i, j],
                             ha="center", va="center", color="black")
        
        ax.set_title('Token Count Comparison (Lower is Better)')
        plt.colorbar(im, ax=ax, label='Number of Tokens')
        
        plt.tight_layout()
        plt.show()

class TokenizerImplementationTips:
    """トークナイザー実装のヒント"""
    
    def share_best_practices(self):
        """ベストプラクティスの共有"""
        print("=== トークナイザー実装のベストプラクティス ===\n")
        
        practices = {
            "1. 前処理": [
                "Unicode正規化（NFKC）",
                "空白文字の統一",
                "特殊文字のエスケープ",
                "大文字小文字の扱いを決定"
            ],
            
            "2. 特殊トークン": [
                "[PAD], [UNK], [CLS], [SEP]の追加",
                "タスク固有トークンの設計",
                "予約領域の確保",
                "トークンIDの固定化"
            ],
            
            "3. 効率化": [
                "トライ木での高速検索",
                "キャッシュの活用",
                "バッチ処理の実装",
                "並列化可能な設計"
            ],
            
            "4. 堅牢性": [
                "未知文字の適切な処理",
                "最大長の制限",
                "エラーハンドリング",
                "デバッグ情報の出力"
            ]
        }
        
        for category, items in practices.items():
            print(f"{category}:")
            for item in items:
                print(f"  • {item}")
            print()
        
        # 実装例
        self._show_implementation_example()
    
    def _show_implementation_example(self):
        """実装例を表示"""
        print("\n=== 効率的なトークナイザーの実装例 ===\n")
        
        code = '''
class EfficientTokenizer:
    """効率的なトークナイザーの実装"""
    
    def __init__(self, vocab: Dict[str, int]):
        self.vocab = vocab
        self.trie = self._build_trie(vocab)
        self.cache = {}
        
    def _build_trie(self, vocab: Dict[str, int]) -> Dict:
        """トライ木の構築"""
        trie = {}
        for token, token_id in vocab.items():
            node = trie
            for char in token:
                if char not in node:
                    node[char] = {}
                node = node[char]
            node['<END>'] = token_id
        return trie
    
    def encode(self, text: str) -> List[int]:
        """高速エンコード"""
        # キャッシュチェック
        if text in self.cache:
            return self.cache[text]
        
        tokens = []
        i = 0
        
        while i < len(text):
            # 最長一致
            node = self.trie
            longest_token_id = None
            longest_end = i
            
            for j in range(i, len(text)):
                if text[j] not in node:
                    break
                node = node[text[j]]
                if '<END>' in node:
                    longest_token_id = node['<END>']
                    longest_end = j + 1
            
            if longest_token_id is not None:
                tokens.append(longest_token_id)
                i = longest_end
            else:
                # Unknown token
                tokens.append(self.vocab.get('<UNK>', 0))
                i += 1
        
        # キャッシュに保存
        self.cache[text] = tokens
        return tokens
'''
        
        print(code)

# 実行とデモ
def run_tokenizer_demo():
    """トークナイザーのデモを実行"""
    print("=" * 70)
    print("トークナイザーの詳細")
    print("=" * 70 + "\n")
    
    # 1. 基礎概念
    basics = TokenizationBasics()
    basics.explain_tokenization_challenges()
    
    # 2. BPE
    print("\n")
    bpe = BytePairEncoding()
    
    # サンプルコーパスでBPEを学習
    sample_corpus = [
        "the cat sat on the mat",
        "the dog sat on the log", 
        "cats and dogs are pets",
        "the quick brown fox jumps"
    ]
    
    bpe.train(sample_corpus, vocab_size=50)
    
    # BPEのデモ
    print("\n")
    bpe_demo = BPEDemo()
    bpe_demo.demonstrate_bpe_process()
    
    # 3. WordPiece
    print("\n")
    wp = WordPieceTokenizer()
    wp.train(sample_corpus, vocab_size=100)
    
    # 4. SentencePiece
    print("\n")
    sp_demo = SentencePieceDemo()
    sp_demo.explain_sentencepiece()
    
    # 5. 現代的なトークナイザー
    print("\n")
    modern = ModernTokenizers()
    modern.compare_modern_tokenizers()
    
    # 6. 実装のヒント
    print("\n")
    tips = TokenizerImplementationTips()
    tips.share_best_practices()
    
    print("\n" + "=" * 70)
    print("まとめ")
    print("=" * 70)
    print("\nトークナイザーの要点:")
    print("• 言語の多様性に対応する柔軟性")
    print("• 計算効率と表現力のバランス")
    print("• サブワード分割による未知語対応")
    print("• タスクとモデルに適したトークナイザー選択")
    print("\nトークナイザーは言語モデルの「目」であり、")
    print("その設計がモデルの性能に大きく影響します。")

if __name__ == "__main__":
    run_tokenizer_demo()