# 単語の数値表現

## はじめに：言語を数値に変換する挑戦

プログラミング言語のコンパイラを作る際、最初に行うのは字句解析（レキシング）です。ソースコードという文字列を、意味のある単位（トークン）に分割します。自然言語処理でも同じことを行いますが、そこには大きな違いがあります。

プログラミング言語は**形式言語**です。厳密な文法規則があり、曖昧さはありません。一方、自然言語は**曖昧さの塊**です。同じ単語が文脈によって全く異なる意味を持ち、新しい単語が日々生まれ、文法規則には無数の例外があります。

この章では、この挑戦的な問題に対するTransformerのアプローチを、プログラミング言語処理の知識を活かしながら理解していきます。

## 5.1 トークン化：言語の原子を見つける

### プログラミング言語 vs 自然言語のトークン化

```python
import re
from typing import List, Dict, Tuple, Optional, Union
from collections import Counter, defaultdict
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, field

class TokenizationComparison:
    """プログラミング言語と自然言語のトークン化の比較"""
    
    def programming_language_tokenizer(self, code: str) -> List[Tuple[str, str]]:
        """プログラミング言語の字句解析器"""
        # トークンパターンの定義
        token_specification = [
            ('NUMBER',    r'\d+(\.\d*)?'),                    # 数値
            ('IDENT',     r'[a-zA-Z_]\w*'),                  # 識別子
            ('STRING',    r'"[^"]*"'),                        # 文字列
            ('COMMENT',   r'//[^\n]*'),                       # コメント
            ('ASSIGN',    r'='),                              # 代入
            ('END',       r';'),                              # 文末
            ('OP',        r'[+\-*/]'),                        # 演算子
            ('LPAREN',    r'\('),                             # 左括弧
            ('RPAREN',    r'\)'),                             # 右括弧
            ('LBRACE',    r'\{'),                             # 左波括弧
            ('RBRACE',    r'\}'),                             # 右波括弧
            ('SKIP',      r'[ \t]+'),                         # スペース
            ('NEWLINE',   r'\n'),                             # 改行
            ('MISMATCH',  r'.'),                              # エラー
        ]
        
        tok_regex = '|'.join(f'(?P<{name}>{pattern})' for name, pattern in token_specification)
        tokens = []
        
        for match in re.finditer(tok_regex, code):
            kind = match.lastgroup
            value = match.group()
            if kind not in ['SKIP', 'NEWLINE', 'COMMENT']:
                tokens.append((kind, value))
        
        return tokens
    
    def natural_language_challenges(self) -> None:
        """自然言語トークン化の課題を実演"""
        print("=== 自然言語トークン化の課題 ===")
        
        # 1. 単語境界の曖昧さ
        examples = {
            "複合語": "New York Times",  # 1トークン？3トークン？
            "縮約": "don't",             # "do not"？1トークン？
            "ハイフン": "state-of-the-art",  # どう分割？
            "数値": "$1,234.56",         # 通貨記号と数値を分ける？
            "絵文字": "Hello 👋 World",   # 絵文字の扱い
            "日本語": "私は学生です",      # スペースがない言語
        }
        
        for category, text in examples.items():
            print(f"\n{category}: '{text}'")
            
            # 単純なスペース分割
            simple_tokens = text.split()
            print(f"  スペース分割: {simple_tokens}")
            
            # より洗練された分割（後述）
    
    def why_subword_tokenization(self) -> None:
        """なぜサブワードトークン化が必要か"""
        
        # 問題1: 語彙爆発
        print("\n=== 語彙爆発の問題 ===")
        
        # 英語の単語バリエーション
        word_variations = [
            "run", "runs", "running", "ran", "runner", "runners",
            "runnable", "rerun", "overrun", "outrun"
        ]
        
        print("'run'の変化形:", word_variations)
        print(f"単語レベルでは{len(word_variations)}個の異なるトークンが必要")
        
        # サブワード分割の例
        subword_splits = {
            "running": ["run", "##ning"],
            "runners": ["run", "##ner", "##s"],
            "unrunnable": ["un", "##run", "##nable"],
        }
        
        print("\nサブワード分割:")
        for word, subwords in subword_splits.items():
            print(f"  {word} → {subwords}")
        
        # 問題2: 未知語（OOV: Out-of-Vocabulary）
        print("\n=== 未知語の問題 ===")
        
        # 訓練時に見たことがない単語
        unknown_words = [
            "COVID-19",      # 新しい用語
            "Pneumonoultramicroscopicsilicovolcanoconiosis",  # 長い専門用語
            "🚀🌟",          # 絵文字の組み合わせ
            "supercalifragilisticexpialidocious",  # 造語
        ]
        
        print("未知語の例:", unknown_words)
        print("単語レベルでは全て[UNK]トークンになってしまう")
```

### Byte-Pair Encoding (BPE) の実装

```python
@dataclass
class BPEToken:
    """BPEトークンのデータ構造"""
    text: str
    frequency: int = 0
    
class BytePairEncoding:
    """BPEアルゴリズムの実装"""
    
    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.word_freq = Counter()
        self.vocab = {}
        self.merges = []
    
    def train(self, corpus: List[str], verbose: bool = True) -> None:
        """コーパスからBPEモデルを学習"""
        # ステップ1: 単語頻度をカウント
        for text in corpus:
            words = text.lower().split()
            for word in words:
                # 単語を文字単位に分割（特殊な終端記号を追加）
                word_tokens = list(word) + ['</w>']
                self.word_freq[tuple(word_tokens)] += 1
        
        # 初期語彙（全ての文字）
        vocab = set()
        for word_tokens, freq in self.word_freq.items():
            for token in word_tokens:
                vocab.add(token)
        
        if verbose:
            print(f"初期語彙サイズ: {len(vocab)}")
            print(f"初期語彙: {sorted(list(vocab))[:20]}...")
        
        # ステップ2: マージを繰り返す
        num_merges = self.vocab_size - len(vocab)
        
        for i in range(num_merges):
            # 最も頻度の高いペアを見つける
            pair_freq = self._get_pair_frequencies()
            
            if not pair_freq:
                break
            
            best_pair = max(pair_freq, key=pair_freq.get)
            self.merges.append(best_pair)
            
            if verbose and i % 100 == 0:
                print(f"\nマージ {i+1}: {best_pair} (頻度: {pair_freq[best_pair]})")
            
            # 語彙を更新
            self.word_freq = self._merge_pair(best_pair)
            
            # 新しいトークンを語彙に追加
            new_token = ''.join(best_pair)
            vocab.add(new_token)
        
        # 最終語彙を作成
        self.vocab = {token: idx for idx, token in enumerate(sorted(vocab))}
        
        if verbose:
            print(f"\n最終語彙サイズ: {len(self.vocab)}")
            print(f"学習されたマージ数: {len(self.merges)}")
    
    def _get_pair_frequencies(self) -> Dict[Tuple[str, str], int]:
        """隣接トークンペアの頻度を計算"""
        pair_freq = defaultdict(int)
        
        for word_tokens, freq in self.word_freq.items():
            for i in range(len(word_tokens) - 1):
                pair = (word_tokens[i], word_tokens[i + 1])
                pair_freq[pair] += freq
        
        return pair_freq
    
    def _merge_pair(self, pair: Tuple[str, str]) -> Counter:
        """指定されたペアをマージ"""
        new_word_freq = Counter()
        
        for word_tokens, freq in self.word_freq.items():
            new_word_tokens = []
            i = 0
            
            while i < len(word_tokens):
                # ペアが見つかったらマージ
                if i < len(word_tokens) - 1 and \
                   word_tokens[i] == pair[0] and word_tokens[i + 1] == pair[1]:
                    new_word_tokens.append(pair[0] + pair[1])
                    i += 2
                else:
                    new_word_tokens.append(word_tokens[i])
                    i += 1
            
            new_word_freq[tuple(new_word_tokens)] = freq
        
        return new_word_freq
    
    def tokenize(self, text: str) -> List[str]:
        """テキストをBPEトークンに分割"""
        words = text.lower().split()
        tokens = []
        
        for word in words:
            word_tokens = list(word) + ['</w>']
            
            # 学習したマージを適用
            for pair in self.merges:
                i = 0
                new_word_tokens = []
                
                while i < len(word_tokens):
                    if i < len(word_tokens) - 1 and \
                       word_tokens[i] == pair[0] and word_tokens[i + 1] == pair[1]:
                        new_word_tokens.append(pair[0] + pair[1])
                        i += 2
                    else:
                        new_word_tokens.append(word_tokens[i])
                        i += 1
                
                word_tokens = new_word_tokens
            
            tokens.extend(word_tokens)
        
        return tokens
    
    def visualize_tokenization(self, text: str) -> None:
        """トークン化の過程を可視化"""
        words = text.lower().split()
        
        fig, axes = plt.subplots(len(words), 1, figsize=(12, 3 * len(words)))
        if len(words) == 1:
            axes = [axes]
        
        for idx, word in enumerate(words):
            ax = axes[idx]
            word_tokens = list(word) + ['</w>']
            
            # 各マージステップを記録
            steps = [word_tokens.copy()]
            
            for pair in self.merges:
                i = 0
                new_word_tokens = []
                
                while i < len(word_tokens):
                    if i < len(word_tokens) - 1 and \
                       word_tokens[i] == pair[0] and word_tokens[i + 1] == pair[1]:
                        new_word_tokens.append(pair[0] + pair[1])
                        i += 2
                    else:
                        new_word_tokens.append(word_tokens[i])
                        i += 1
                
                if new_word_tokens != word_tokens:
                    word_tokens = new_word_tokens
                    steps.append(word_tokens.copy())
            
            # 可視化
            y_labels = [f"Step {i}" for i in range(len(steps))]
            
            # ヒートマップ用のデータ作成
            max_len = max(len(step) for step in steps)
            heatmap_data = np.zeros((len(steps), max_len))
            
            for i, step in enumerate(steps):
                for j, token in enumerate(step):
                    heatmap_data[i, j] = len(token)
            
            im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
            
            # トークンをテキストとして表示
            for i, step in enumerate(steps):
                for j, token in enumerate(step):
                    ax.text(j, i, token, ha='center', va='center', fontsize=10)
            
            ax.set_yticks(range(len(steps)))
            ax.set_yticklabels(y_labels)
            ax.set_xticks([])
            ax.set_title(f"'{word}'のトークン化過程")
        
        plt.tight_layout()
        plt.show()
```

### WordPieceとSentencePieceの比較

```python
class ModernTokenizers:
    """現代的なトークナイザーの比較"""
    
    def __init__(self):
        self.tokenizers = {}
    
    def compare_tokenization_methods(self, text: str) -> None:
        """異なるトークン化手法の比較"""
        print(f"=== トークン化手法の比較 ===")
        print(f"入力テキスト: '{text}'")
        
        # 1. 単語レベル
        word_tokens = text.split()
        print(f"\n1. 単語レベル: {word_tokens}")
        print(f"   トークン数: {len(word_tokens)}")
        
        # 2. 文字レベル
        char_tokens = list(text)
        print(f"\n2. 文字レベル: {char_tokens[:50]}...")
        print(f"   トークン数: {len(char_tokens)}")
        
        # 3. BPE（簡易版）
        bpe_tokens = self._simple_bpe_tokenize(text)
        print(f"\n3. BPE: {bpe_tokens}")
        print(f"   トークン数: {len(bpe_tokens)}")
        
        # 4. WordPiece（簡易版）
        wordpiece_tokens = self._simple_wordpiece_tokenize(text)
        print(f"\n4. WordPiece: {wordpiece_tokens}")
        print(f"   トークン数: {len(wordpiece_tokens)}")
        
        # 可視化
        self._visualize_tokenization_comparison(text, {
            'Word': word_tokens,
            'Character': char_tokens[:30],  # 表示用に制限
            'BPE': bpe_tokens,
            'WordPiece': wordpiece_tokens
        })
    
    def _simple_bpe_tokenize(self, text: str) -> List[str]:
        """簡易BPEトークン化"""
        # 実際のBPEは学習が必要だが、ここでは簡易版
        tokens = []
        for word in text.split():
            if len(word) > 6:
                # 長い単語は分割
                tokens.extend([word[:3], '##' + word[3:]])
            else:
                tokens.append(word)
        return tokens
    
    def _simple_wordpiece_tokenize(self, text: str) -> List[str]:
        """簡易WordPieceトークン化"""
        # 実際のWordPieceも学習が必要だが、ここでは簡易版
        common_prefixes = ['un', 're', 'pre', 'post', 'sub', 'over']
        common_suffixes = ['ing', 'ed', 'er', 'est', 'ly', 'tion', 'ness']
        
        tokens = []
        for word in text.split():
            tokenized = False
            
            # 接頭辞チェック
            for prefix in common_prefixes:
                if word.startswith(prefix) and len(word) > len(prefix) + 2:
                    tokens.extend([prefix, '##' + word[len(prefix):]])
                    tokenized = True
                    break
            
            # 接尾辞チェック
            if not tokenized:
                for suffix in common_suffixes:
                    if word.endswith(suffix) and len(word) > len(suffix) + 2:
                        tokens.extend([word[:-len(suffix)], '##' + suffix])
                        tokenized = True
                        break
            
            if not tokenized:
                tokens.append(word)
        
        return tokens
    
    def _visualize_tokenization_comparison(self, text: str, tokenizations: Dict[str, List[str]]):
        """トークン化手法の比較を可視化"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        methods = list(tokenizations.keys())
        token_counts = [len(tokens) for tokens in tokenizations.values()]
        
        bars = ax.bar(methods, token_counts, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        
        # 各バーの上にトークン数を表示
        for bar, count in zip(bars, token_counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count}', ha='center', va='bottom')
        
        ax.set_xlabel('トークン化手法')
        ax.set_ylabel('トークン数')
        ax.set_title('異なるトークン化手法のトークン数比較')
        
        # 理想的な範囲を表示
        ax.axhspan(10, 30, alpha=0.2, color='green', label='理想的な範囲')
        ax.legend()
        
        plt.tight_layout()
        plt.show()
```

## 5.2 単語埋め込み：意味を捉えるベクトル表現

### ワンホットエンコーディングの限界

```python
class WordEmbeddings:
    """単語埋め込みの概念と実装"""
    
    def __init__(self, vocab_size: int = 10000, embedding_dim: int = 300):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.embeddings = None
    
    def one_hot_problems(self) -> None:
        """ワンホットエンコーディングの問題点"""
        print("=== ワンホットエンコーディングの問題 ===")
        
        # 小さな語彙での例
        words = ["cat", "dog", "animal", "car", "vehicle"]
        vocab_size = len(words)
        
        # ワンホットベクトル作成
        one_hot_vectors = {}
        for idx, word in enumerate(words):
            vector = np.zeros(vocab_size)
            vector[idx] = 1
            one_hot_vectors[word] = vector
        
        print("ワンホットベクトル:")
        for word, vector in one_hot_vectors.items():
            print(f"{word}: {vector}")
        
        # 問題1: スパース性
        print(f"\n問題1: スパース性")
        print(f"語彙サイズ: {vocab_size}")
        print(f"非ゼロ要素: 1/{vocab_size} = {1/vocab_size:.1%}")
        
        # 問題2: 意味的関係の欠如
        print(f"\n問題2: 意味的関係の欠如")
        
        def cosine_similarity(v1, v2):
            return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        
        similarities = {
            "cat vs dog": cosine_similarity(one_hot_vectors["cat"], one_hot_vectors["dog"]),
            "cat vs animal": cosine_similarity(one_hot_vectors["cat"], one_hot_vectors["animal"]),
            "car vs vehicle": cosine_similarity(one_hot_vectors["car"], one_hot_vectors["vehicle"]),
        }
        
        for pair, sim in similarities.items():
            print(f"{pair}: {sim}")
        print("→ 全ての単語間の類似度が0（直交）")
        
        # 問題3: メモリ効率
        print(f"\n問題3: メモリ効率")
        real_vocab_size = 50000
        print(f"実際の語彙サイズ: {real_vocab_size:,}")
        memory_one_hot = real_vocab_size * real_vocab_size * 4  # float32
        print(f"必要メモリ（全単語）: {memory_one_hot / 1e9:.1f} GB")
    
    def dense_embeddings_intuition(self) -> None:
        """密な埋め込みの直感的理解"""
        print("\n=== 密な埋め込みの利点 ===")
        
        # 仮想的な埋め込み（実際は学習で獲得）
        embeddings = {
            "cat": np.array([0.2, 0.8, -0.1, 0.3]),
            "dog": np.array([0.3, 0.7, -0.2, 0.4]),
            "animal": np.array([0.25, 0.75, -0.15, 0.35]),
            "car": np.array([-0.5, -0.3, 0.8, -0.2]),
            "vehicle": np.array([-0.4, -0.35, 0.75, -0.15]),
        }
        
        embedding_dim = 4
        
        print(f"埋め込み次元: {embedding_dim}")
        print("\n埋め込みベクトル:")
        for word, vec in embeddings.items():
            print(f"{word}: {vec}")
        
        # 意味的類似度
        print("\n意味的類似度（コサイン類似度）:")
        
        def cosine_similarity(v1, v2):
            return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        
        pairs = [
            ("cat", "dog"),
            ("cat", "animal"),
            ("car", "vehicle"),
            ("cat", "car"),
        ]
        
        for w1, w2 in pairs:
            sim = cosine_similarity(embeddings[w1], embeddings[w2])
            print(f"{w1} vs {w2}: {sim:.3f}")
        
        # 可視化
        self._visualize_embeddings(embeddings)
    
    def _visualize_embeddings(self, embeddings: Dict[str, np.ndarray]):
        """埋め込みの可視化（2D投影）"""
        from sklearn.decomposition import PCA
        
        words = list(embeddings.keys())
        vectors = np.array(list(embeddings.values()))
        
        # PCAで2次元に削減
        if vectors.shape[1] > 2:
            pca = PCA(n_components=2)
            vectors_2d = pca.fit_transform(vectors)
        else:
            vectors_2d = vectors
        
        plt.figure(figsize=(10, 8))
        
        # 単語をプロット
        for i, word in enumerate(words):
            plt.scatter(vectors_2d[i, 0], vectors_2d[i, 1], s=200)
            plt.annotate(word, 
                        xy=(vectors_2d[i, 0], vectors_2d[i, 1]),
                        xytext=(5, 5), 
                        textcoords='offset points',
                        fontsize=12)
        
        # 類似する単語を線で結ぶ
        similar_pairs = [("cat", "dog"), ("cat", "animal"), ("car", "vehicle")]
        for w1, w2 in similar_pairs:
            idx1 = words.index(w1)
            idx2 = words.index(w2)
            plt.plot([vectors_2d[idx1, 0], vectors_2d[idx2, 0]],
                    [vectors_2d[idx1, 1], vectors_2d[idx2, 1]],
                    'k--', alpha=0.3)
        
        plt.xlabel('第1主成分')
        plt.ylabel('第2主成分')
        plt.title('単語埋め込みの2次元可視化')
        plt.grid(True, alpha=0.3)
        plt.show()
```

### 埋め込み層の実装と学習

```python
class EmbeddingLayer:
    """埋め込み層の実装と学習過程の理解"""
    
    def __init__(self, vocab_size: int, embedding_dim: int):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # 埋め込み行列の初期化
        self.embedding_matrix = torch.nn.Embedding(vocab_size, embedding_dim)
        
        # 初期化方法の比較
        self._compare_initialization_methods()
    
    def _compare_initialization_methods(self):
        """異なる初期化方法の比較"""
        methods = {
            'uniform': lambda: torch.nn.init.uniform_(
                torch.empty(self.vocab_size, self.embedding_dim), -0.1, 0.1
            ),
            'normal': lambda: torch.nn.init.normal_(
                torch.empty(self.vocab_size, self.embedding_dim), 0, 0.02
            ),
            'xavier': lambda: torch.nn.init.xavier_uniform_(
                torch.empty(self.vocab_size, self.embedding_dim)
            ),
        }
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for idx, (name, init_fn) in enumerate(methods.items()):
            matrix = init_fn()
            
            ax = axes[idx]
            im = ax.imshow(matrix[:50, :50], cmap='coolwarm', aspect='auto')
            ax.set_title(f'{name.capitalize()} Initialization')
            ax.set_xlabel('Embedding Dimension')
            ax.set_ylabel('Word Index')
            plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        plt.show()
    
    def forward_pass_explained(self, word_indices: torch.Tensor) -> torch.Tensor:
        """埋め込み層の順伝播を詳しく説明"""
        print("=== 埋め込み層の順伝播 ===")
        
        # 入力の形状
        print(f"入力（単語インデックス）: {word_indices}")
        print(f"入力の形状: {word_indices.shape}")
        
        # 埋め込み行列の形状
        print(f"\n埋め込み行列の形状: {self.embedding_matrix.weight.shape}")
        print(f"  → {self.vocab_size} words × {self.embedding_dim} dimensions")
        
        # ルックアップ操作
        embeddings = self.embedding_matrix(word_indices)
        print(f"\n出力の形状: {embeddings.shape}")
        
        # 可視化
        self._visualize_lookup_operation(word_indices, embeddings)
        
        return embeddings
    
    def _visualize_lookup_operation(self, indices: torch.Tensor, embeddings: torch.Tensor):
        """ルックアップ操作の可視化"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 埋め込み行列全体
        ax = axes[0]
        im = ax.imshow(self.embedding_matrix.weight.data[:20, :20], 
                       cmap='viridis', aspect='auto')
        ax.set_title('埋め込み行列（一部）')
        ax.set_xlabel('Dimension')
        ax.set_ylabel('Word Index')
        
        # 選択された行
        ax = axes[1]
        selected_indices = indices.flatten().tolist()[:5]  # 最初の5つ
        for i, idx in enumerate(selected_indices):
            ax.axhline(y=idx, color='red', linewidth=2, alpha=0.7)
        ax.set_ylim(-0.5, 19.5)
        ax.set_title('選択された単語')
        ax.set_ylabel('Word Index')
        
        # 結果の埋め込み
        ax = axes[2]
        result = embeddings.reshape(-1, self.embedding_dim)[:5]
        im = ax.imshow(result.detach(), cmap='viridis', aspect='auto')
        ax.set_title('取得された埋め込み')
        ax.set_xlabel('Dimension')
        ax.set_ylabel('Selected Words')
        
        plt.tight_layout()
        plt.show()
    
    def gradient_flow_in_embeddings(self):
        """埋め込み層での勾配の流れ"""
        print("\n=== 埋め込み層の勾配更新 ===")
        
        # 簡単な例
        vocab_size = 10
        embedding_dim = 4
        embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        
        # 初期の埋め込み
        print("初期埋め込み（word 3）:")
        print(embedding.weight[3])
        
        # 順伝播
        word_indices = torch.tensor([3, 5, 3])  # word 3が2回出現
        embedded = embedding(word_indices)
        
        # 仮想的な損失
        loss = embedded.sum()
        loss.backward()
        
        # 勾配を確認
        print("\n勾配（word 3）:")
        print(embedding.weight.grad[3])
        print("→ word 3は2回使われたので、勾配も2倍")
        
        # 更新
        with torch.no_grad():
            embedding.weight -= 0.1 * embedding.weight.grad
        
        print("\n更新後の埋め込み（word 3）:")
        print(embedding.weight[3])
```

### 位置を考慮した埋め込み

```python
class PositionalAwareEmbeddings:
    """位置情報を考慮した埋め込み"""
    
    def __init__(self, vocab_size: int, max_length: int, d_model: int):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.d_model = d_model
        
        # トークン埋め込み
        self.token_embedding = torch.nn.Embedding(vocab_size, d_model)
        
        # 位置埋め込み（学習可能）
        self.position_embedding = torch.nn.Embedding(max_length, d_model)
        
        # セグメント埋め込み（BERTスタイル）
        self.segment_embedding = torch.nn.Embedding(2, d_model)
    
    def combined_embeddings(self, token_ids: torch.Tensor, 
                          position_ids: Optional[torch.Tensor] = None,
                          segment_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """複合埋め込みの計算"""
        batch_size, seq_length = token_ids.shape
        
        # トークン埋め込み
        token_embeds = self.token_embedding(token_ids)
        
        # 位置埋め込み
        if position_ids is None:
            position_ids = torch.arange(seq_length).expand(batch_size, -1)
        position_embeds = self.position_embedding(position_ids)
        
        # セグメント埋め込み
        if segment_ids is None:
            segment_ids = torch.zeros_like(token_ids)
        segment_embeds = self.segment_embedding(segment_ids)
        
        # 合計
        embeddings = token_embeds + position_embeds + segment_embeds
        
        # 可視化
        self._visualize_embedding_components(
            token_embeds[0], position_embeds[0], segment_embeds[0], embeddings[0]
        )
        
        return embeddings
    
    def _visualize_embedding_components(self, token_emb, pos_emb, seg_emb, total_emb):
        """埋め込みコンポーネントの可視化"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 各コンポーネント
        components = [
            (token_emb, 'Token Embeddings'),
            (pos_emb, 'Position Embeddings'),
            (seg_emb, 'Segment Embeddings'),
            (total_emb, 'Combined Embeddings')
        ]
        
        for idx, (emb, title) in enumerate(components):
            ax = axes[idx // 2, idx % 2]
            
            # ヒートマップ
            im = ax.imshow(emb[:10, :50].detach(), cmap='coolwarm', aspect='auto')
            ax.set_title(title)
            ax.set_xlabel('Embedding Dimension')
            ax.set_ylabel('Position')
            plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        plt.show()
```

## 5.3 実践的なトークナイザーの実装

### カスタムトークナイザーの構築

```python
class CustomTokenizer:
    """実用的なカスタムトークナイザー"""
    
    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size
        self.word_to_idx = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        self.vocab_built = False
    
    def build_vocab(self, texts: List[str], min_freq: int = 2) -> None:
        """語彙を構築"""
        print("=== 語彙構築 ===")
        
        # 単語頻度をカウント
        word_freq = Counter()
        for text in texts:
            words = self._basic_tokenize(text)
            word_freq.update(words)
        
        print(f"ユニークな単語数: {len(word_freq)}")
        print(f"最頻出単語: {word_freq.most_common(10)}")
        
        # 頻度でフィルタリング
        vocab_words = [word for word, freq in word_freq.items() if freq >= min_freq]
        vocab_words = vocab_words[:self.vocab_size - len(self.word_to_idx)]
        
        # 語彙に追加
        for word in vocab_words:
            if word not in self.word_to_idx:
                idx = len(self.word_to_idx)
                self.word_to_idx[word] = idx
                self.idx_to_word[idx] = word
        
        self.vocab_built = True
        print(f"最終語彙サイズ: {len(self.word_to_idx)}")
    
    def _basic_tokenize(self, text: str) -> List[str]:
        """基本的なトークン化"""
        # 小文字化
        text = text.lower()
        
        # 句読点を分離
        text = re.sub(r'([.!?,;:])', r' \1 ', text)
        
        # 空白で分割
        tokens = text.split()
        
        return tokens
    
    def encode(self, text: str, max_length: Optional[int] = None, 
               padding: bool = True, truncation: bool = True) -> Dict[str, torch.Tensor]:
        """テキストをトークンIDに変換"""
        if not self.vocab_built:
            raise ValueError("語彙が構築されていません。build_vocab()を先に実行してください。")
        
        # トークン化
        tokens = self._basic_tokenize(text)
        
        # CLSとSEPトークンを追加
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        
        # トークンIDに変換
        token_ids = []
        for token in tokens:
            token_ids.append(self.word_to_idx.get(token, self.word_to_idx["[UNK]"]))
        
        # 切り詰めまたはパディング
        if max_length is not None:
            if truncation and len(token_ids) > max_length:
                token_ids = token_ids[:max_length-1] + [self.word_to_idx["[SEP]"]]
            
            if padding and len(token_ids) < max_length:
                padding_length = max_length - len(token_ids)
                token_ids = token_ids + [self.word_to_idx["[PAD]"]] * padding_length
        
        # アテンションマスクの作成
        attention_mask = [1 if token_id != self.word_to_idx["[PAD]"] else 0 
                         for token_id in token_ids]
        
        return {
            "input_ids": torch.tensor(token_ids),
            "attention_mask": torch.tensor(attention_mask),
            "tokens": tokens[:len(token_ids)]
        }
    
    def decode(self, token_ids: torch.Tensor, skip_special_tokens: bool = True) -> str:
        """トークンIDをテキストに変換"""
        tokens = []
        
        for token_id in token_ids:
            token = self.idx_to_word.get(token_id.item(), "[UNK]")
            
            if skip_special_tokens and token in ["[PAD]", "[CLS]", "[SEP]", "[MASK]"]:
                continue
            
            tokens.append(token)
        
        return " ".join(tokens)
    
    def batch_encode(self, texts: List[str], max_length: int = 128) -> Dict[str, torch.Tensor]:
        """バッチエンコーディング"""
        batch_encoding = {
            "input_ids": [],
            "attention_mask": []
        }
        
        for text in texts:
            encoding = self.encode(text, max_length=max_length)
            batch_encoding["input_ids"].append(encoding["input_ids"])
            batch_encoding["attention_mask"].append(encoding["attention_mask"])
        
        # テンソルに変換
        batch_encoding["input_ids"] = torch.stack(batch_encoding["input_ids"])
        batch_encoding["attention_mask"] = torch.stack(batch_encoding["attention_mask"])
        
        return batch_encoding
    
    def visualize_tokenization(self, text: str, max_length: int = 20):
        """トークン化プロセスの可視化"""
        encoding = self.encode(text, max_length=max_length)
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 8))
        
        # トークン
        ax = axes[0]
        tokens = encoding["tokens"]
        y_pos = np.arange(len(tokens))
        colors = ['red' if t.startswith('[') else 'blue' for t in tokens]
        ax.barh(y_pos, [1]*len(tokens), color=colors, alpha=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(tokens)
        ax.set_title('Tokens')
        ax.set_xlim(0, 1)
        
        # トークンID
        ax = axes[1]
        token_ids = encoding["input_ids"].tolist()
        ax.barh(y_pos, [1]*len(token_ids), color='green', alpha=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(token_ids)
        ax.set_title('Token IDs')
        ax.set_xlim(0, 1)
        
        # アテンションマスク
        ax = axes[2]
        attention_mask = encoding["attention_mask"].tolist()
        colors = ['blue' if m == 1 else 'gray' for m in attention_mask]
        ax.barh(y_pos, attention_mask, color=colors, alpha=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(attention_mask)
        ax.set_title('Attention Mask')
        ax.set_xlim(0, 1.1)
        
        plt.tight_layout()
        plt.show()
```

### トークナイザーの性能評価

```python
class TokenizerEvaluation:
    """トークナイザーの性能評価"""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def evaluate_coverage(self, test_texts: List[str]) -> Dict[str, float]:
        """語彙カバー率の評価"""
        total_tokens = 0
        unknown_tokens = 0
        
        for text in test_texts:
            tokens = self.tokenizer._basic_tokenize(text)
            for token in tokens:
                total_tokens += 1
                if token not in self.tokenizer.word_to_idx:
                    unknown_tokens += 1
        
        coverage = 1 - (unknown_tokens / total_tokens)
        
        return {
            "total_tokens": total_tokens,
            "unknown_tokens": unknown_tokens,
            "coverage": coverage,
            "oov_rate": unknown_tokens / total_tokens
        }
    
    def evaluate_compression(self, texts: List[str]) -> Dict[str, float]:
        """圧縮率の評価"""
        original_chars = sum(len(text) for text in texts)
        
        total_tokens = 0
        for text in texts:
            encoding = self.tokenizer.encode(text, padding=False)
            total_tokens += len(encoding["input_ids"])
        
        compression_ratio = original_chars / total_tokens
        
        return {
            "original_chars": original_chars,
            "total_tokens": total_tokens,
            "compression_ratio": compression_ratio,
            "avg_chars_per_token": compression_ratio
        }
    
    def visualize_evaluation(self, test_texts: List[str]):
        """評価結果の可視化"""
        coverage_stats = self.evaluate_coverage(test_texts)
        compression_stats = self.evaluate_compression(test_texts)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # カバー率
        ax = axes[0, 0]
        ax.pie([coverage_stats["coverage"], coverage_stats["oov_rate"]], 
               labels=['Known', 'Unknown'], 
               autopct='%1.1f%%',
               colors=['#2ecc71', '#e74c3c'])
        ax.set_title('Vocabulary Coverage')
        
        # トークン分布
        ax = axes[0, 1]
        token_lengths = []
        for text in test_texts[:100]:  # サンプル
            encoding = self.tokenizer.encode(text, padding=False)
            token_lengths.append(len(encoding["input_ids"]))
        
        ax.hist(token_lengths, bins=20, alpha=0.7, color='#3498db')
        ax.set_xlabel('Number of Tokens')
        ax.set_ylabel('Frequency')
        ax.set_title('Token Length Distribution')
        
        # 圧縮統計
        ax = axes[1, 0]
        metrics = ['Original\nCharacters', 'Total\nTokens', 'Compression\nRatio']
        values = [
            compression_stats["original_chars"] / 1000,  # キロ単位
            compression_stats["total_tokens"] / 1000,
            compression_stats["compression_ratio"]
        ]
        bars = ax.bar(metrics, values, color=['#9b59b6', '#f39c12', '#1abc9c'])
        ax.set_ylabel('Value (K) / Ratio')
        ax.set_title('Compression Statistics')
        
        # 統計サマリー
        ax = axes[1, 1]
        summary_text = f"""
Evaluation Summary:
==================
Coverage: {coverage_stats['coverage']:.1%}
OOV Rate: {coverage_stats['oov_rate']:.1%}
Unknown Tokens: {coverage_stats['unknown_tokens']:,}

Compression Ratio: {compression_stats['compression_ratio']:.2f}
Avg Chars/Token: {compression_stats['avg_chars_per_token']:.2f}
        """
        ax.text(0.1, 0.5, summary_text, transform=ax.transAxes, 
                fontsize=12, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.axis('off')
        
        plt.tight_layout()
        plt.show()
```

## まとめ：言語を数値に変換する技術

この章で学んだことを整理しましょう：

1. **トークン化の重要性**
   - プログラミング言語の字句解析と同様の役割
   - しかし自然言語特有の課題（曖昧性、新語、多言語）への対応が必要
   - サブワードトークン化（BPE、WordPiece）による柔軟な対応

2. **埋め込みの本質**
   - ワンホットエンコーディングの限界を超える
   - 意味的な関係を捉える密なベクトル表現
   - 学習可能なパラメータとしての埋め込み行列

3. **実装上の考慮点**
   - 特殊トークン（[CLS]、[SEP]、[PAD]）の扱い
   - バッチ処理のための統一的な長さ
   - アテンションマスクによる無効領域の管理

次章では、この数値表現された言語データに対して、Transformerの核心である「注意機構」がどのように働くかを見ていきます。

## 演習問題

1. **BPE実装の拡張**: 日本語のような空白で区切られない言語に対応したBPEアルゴリズムを実装してください。

2. **埋め込みの可視化**: Word2Vecスタイルの類推タスク（king - man + woman = queen）を、学習済み埋め込みで検証するコードを書いてください。

3. **トークナイザーの最適化**: 与えられたコーパスに対して、最適な語彙サイズを自動的に決定するアルゴリズムを設計してください。

4. **マルチ言語対応**: 複数言語を同時に扱えるトークナイザーを実装し、言語間でのトークンの共有について分析してください。

---

次章「注意機構の直感的理解」へ続く。