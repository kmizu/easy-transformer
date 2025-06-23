# プログラミング言語処理との類似点

## はじめに：二つの言語処理の出会い

コンパイラ設計者として、あなたは「言語」を処理するエキスパートです。字句解析、構文解析、意味解析、最適化...これらの概念は、実はTransformerが自然言語を処理する方法と驚くほど類似しています。

この章では、あなたがすでに持っているプログラミング言語処理の知識を活用して、Transformerの動作原理を深く理解していきます。コンパイラ技術とTransformerの間に橋を架け、新しい技術を親しみやすく、そして本質的に理解できるようにします。

## 2.1 字句解析とトークン化の深い対比

### コンパイラの字句解析器を振り返る

まず、典型的なコンパイラの字句解析器（レキサー）の実装を見てみましょう：

```python
import re
from enum import Enum, auto
from dataclasses import dataclass
from typing import List, Optional, Tuple

class TokenType(Enum):
    # リテラル
    NUMBER = auto()
    STRING = auto()
    IDENTIFIER = auto()
    
    # キーワード
    IF = auto()
    ELSE = auto()
    WHILE = auto()
    FUNCTION = auto()
    RETURN = auto()
    
    # 演算子
    PLUS = auto()
    MINUS = auto()
    MULTIPLY = auto()
    DIVIDE = auto()
    ASSIGN = auto()
    EQUALS = auto()
    NOT_EQUALS = auto()
    
    # デリミタ
    LPAREN = auto()
    RPAREN = auto()
    LBRACE = auto()
    RBRACE = auto()
    SEMICOLON = auto()
    COMMA = auto()
    
    # 特殊
    EOF = auto()
    WHITESPACE = auto()
    COMMENT = auto()

@dataclass
class Token:
    type: TokenType
    value: str
    line: int
    column: int

class Lexer:
    def __init__(self, source_code: str):
        self.source = source_code
        self.position = 0
        self.line = 1
        self.column = 1
        self.tokens: List[Token] = []
        
        # トークンパターンの定義（優先順位順）
        self.token_patterns = [
            # 空白とコメント
            (r'[ \t]+', TokenType.WHITESPACE),
            (r'//[^\n]*', TokenType.COMMENT),
            (r'/\*[\s\S]*?\*/', TokenType.COMMENT),
            
            # リテラル
            (r'\d+\.?\d*', TokenType.NUMBER),
            (r'"[^"]*"', TokenType.STRING),
            (r"'[^']*'", TokenType.STRING),
            
            # キーワード（識別子より先にマッチさせる）
            (r'\bif\b', TokenType.IF),
            (r'\belse\b', TokenType.ELSE),
            (r'\bwhile\b', TokenType.WHILE),
            (r'\bfunction\b', TokenType.FUNCTION),
            (r'\breturn\b', TokenType.RETURN),
            
            # 識別子
            (r'[a-zA-Z_][a-zA-Z0-9_]*', TokenType.IDENTIFIER),
            
            # 演算子（長いものから先に）
            (r'==', TokenType.EQUALS),
            (r'!=', TokenType.NOT_EQUALS),
            (r'=', TokenType.ASSIGN),
            (r'\+', TokenType.PLUS),
            (r'-', TokenType.MINUS),
            (r'\*', TokenType.MULTIPLY),
            (r'/', TokenType.DIVIDE),
            
            # デリミタ
            (r'\(', TokenType.LPAREN),
            (r'\)', TokenType.RPAREN),
            (r'\{', TokenType.LBRACE),
            (r'\}', TokenType.RBRACE),
            (r';', TokenType.SEMICOLON),
            (r',', TokenType.COMMA),
        ]
        
        # パターンをコンパイル
        self.compiled_patterns = [
            (re.compile(pattern), token_type) 
            for pattern, token_type in self.token_patterns
        ]
    
    def tokenize(self) -> List[Token]:
        """ソースコード全体をトークン化"""
        while self.position < len(self.source):
            # 改行の処理
            if self.source[self.position] == '\n':
                self.line += 1
                self.column = 1
                self.position += 1
                continue
            
            # トークンのマッチング
            matched = False
            for pattern, token_type in self.compiled_patterns:
                match = pattern.match(self.source, self.position)
                if match:
                    value = match.group(0)
                    
                    # 空白とコメントはトークンとして保存しない
                    if token_type not in [TokenType.WHITESPACE, TokenType.COMMENT]:
                        token = Token(token_type, value, self.line, self.column)
                        self.tokens.append(token)
                    
                    # 位置を更新
                    self.position = match.end()
                    self.column += len(value)
                    matched = True
                    break
            
            if not matched:
                raise SyntaxError(
                    f"Unexpected character '{self.source[self.position]}' "
                    f"at line {self.line}, column {self.column}"
                )
        
        # EOF トークンを追加
        self.tokens.append(Token(TokenType.EOF, '', self.line, self.column))
        return self.tokens
    
    def print_tokens(self):
        """トークンを見やすく表示"""
        for token in self.tokens:
            print(f"{token.type.name:12} | {token.value:10} | Line {token.line}:{token.column}")

# 使用例
source_code = """
function fibonacci(n) {
    if (n <= 1) {
        return n;
    }
    return fibonacci(n - 1) + fibonacci(n - 2);
}
"""

lexer = Lexer(source_code)
tokens = lexer.tokenize()
lexer.print_tokens()
```

### 自然言語のトークン化：より複雑な世界

自然言語のトークン化は、プログラミング言語よりもはるかに複雑です。なぜなら：

1. **明確な区切り文字がない場合がある**（特に日本語や中国語）
2. **同じ文字列が文脈によって異なる意味を持つ**
3. **新しい単語が常に生まれる**
4. **誤字脱字や方言が存在する**

```python
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from collections import Counter
import regex  # より高度な正規表現サポート

class NaturalLanguageTokenizer:
    """
    自然言語用のトークナイザー
    BPE (Byte Pair Encoding) アルゴリズムを実装
    """
    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size
        self.word_tokenizer = regex.compile(r"""
            (?:[^\s\p{P}]+)  |  # 単語（句読点以外）
            (?:\p{P})           # 句読点
        """, regex.VERBOSE)
        
        # 特殊トークン
        self.special_tokens = {
            '<PAD>': 0,    # パディング
            '<UNK>': 1,    # 未知語
            '<BOS>': 2,    # 文章開始
            '<EOS>': 3,    # 文章終了
            '<MASK>': 4,   # マスク（BERT用）
        }
        
        self.token_to_id = dict(self.special_tokens)
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        self.merge_rules = {}  # BPEのマージルール
    
    def train_bpe(self, corpus: List[str], min_frequency: int = 2):
        """
        BPEアルゴリズムでボキャブラリを学習
        これは、コンパイラでの「共通部分式の抽出」に似ている
        """
        # ステップ1: 単語の頻度を数える
        word_freq = Counter()
        for text in corpus:
            words = self.word_tokenizer.findall(text.lower())
            for word in words:
                word_freq[word] += 1
        
        # ステップ2: 各単語を文字に分解（初期化）
        word_splits = {}
        for word, freq in word_freq.items():
            # 単語を文字のリストに分解し、終端記号を追加
            word_splits[word] = list(word) + ['</w>']
        
        # ステップ3: マージを繰り返す
        num_merges = self.vocab_size - len(self.special_tokens) - 256  # 基本文字分を引く
        
        for i in range(num_merges):
            # 隣接するペアの頻度を数える
            pair_freq = Counter()
            for word, splits in word_splits.items():
                freq = word_freq[word]
                for j in range(len(splits) - 1):
                    pair = (splits[j], splits[j + 1])
                    pair_freq[pair] += freq
            
            # 最も頻度の高いペアを選択
            if not pair_freq:
                break
            
            best_pair = max(pair_freq, key=pair_freq.get)
            if pair_freq[best_pair] < min_frequency:
                break
            
            # マージルールを記録
            self.merge_rules[best_pair] = best_pair[0] + best_pair[1]
            
            # 全ての単語でペアをマージ
            new_word_splits = {}
            for word, splits in word_splits.items():
                new_splits = []
                i = 0
                while i < len(splits):
                    if (i < len(splits) - 1 and 
                        (splits[i], splits[i + 1]) == best_pair):
                        new_splits.append(best_pair[0] + best_pair[1])
                        i += 2
                    else:
                        new_splits.append(splits[i])
                        i += 1
                new_word_splits[word] = new_splits
            
            word_splits = new_word_splits
            
            # 進捗表示
            if (i + 1) % 1000 == 0:
                print(f"Merge {i + 1}/{num_merges}: {best_pair} -> {best_pair[0] + best_pair[1]}")
        
        # ボキャブラリの構築
        self._build_vocab(word_splits, word_freq)
    
    def _build_vocab(self, word_splits: Dict[str, List[str]], word_freq: Dict[str, int]):
        """学習したBPEからボキャブラリを構築"""
        # 全てのサブワードを収集
        subwords = Counter()
        for word, splits in word_splits.items():
            freq = word_freq[word]
            for subword in splits:
                subwords[subword] += freq
        
        # 頻度順にソートしてIDを割り当て
        current_id = len(self.special_tokens)
        for subword, _ in subwords.most_common(self.vocab_size - len(self.special_tokens)):
            if subword not in self.token_to_id:
                self.token_to_id[subword] = current_id
                self.id_to_token[current_id] = subword
                current_id += 1
    
    def tokenize(self, text: str) -> List[str]:
        """テキストをトークンに分解"""
        tokens = []
        words = self.word_tokenizer.findall(text.lower())
        
        for word in words:
            # BPEルールを適用してサブワードに分解
            word_tokens = self._apply_bpe(word)
            tokens.extend(word_tokens)
        
        return tokens
    
    def _apply_bpe(self, word: str) -> List[str]:
        """単語にBPEルールを適用"""
        # 単語を文字に分解
        splits = list(word) + ['</w>']
        
        # マージルールを順番に適用
        changed = True
        while changed:
            changed = False
            new_splits = []
            i = 0
            
            while i < len(splits):
                if i < len(splits) - 1:
                    pair = (splits[i], splits[i + 1])
                    if pair in self.merge_rules:
                        new_splits.append(self.merge_rules[pair])
                        i += 2
                        changed = True
                        continue
                
                new_splits.append(splits[i])
                i += 1
            
            splits = new_splits
        
        return splits
    
    def encode(self, text: str) -> List[int]:
        """テキストをIDのリストに変換"""
        tokens = self.tokenize(text)
        ids = []
        
        for token in tokens:
            if token in self.token_to_id:
                ids.append(self.token_to_id[token])
            else:
                ids.append(self.token_to_id['<UNK>'])
        
        return ids
    
    def decode(self, ids: List[int]) -> str:
        """IDのリストをテキストに変換"""
        tokens = []
        for id in ids:
            if id in self.id_to_token:
                token = self.id_to_token[id]
                if token not in self.special_tokens:
                    tokens.append(token)
        
        # サブワードを結合
        text = ''.join(tokens)
        text = text.replace('</w>', ' ')
        return text.strip()

# トークナイザーの比較デモ
def compare_tokenizers():
    """プログラミング言語と自然言語のトークン化を比較"""
    
    # プログラミング言語の例
    code = "if (x > 0) { return x * 2; }"
    code_lexer = Lexer(code)
    code_tokens = code_lexer.tokenize()
    
    print("=== プログラミング言語のトークン化 ===")
    print(f"入力: {code}")
    print("トークン:")
    for token in code_tokens[:-1]:  # EOF以外
        print(f"  {token.value}")
    
    # 自然言語の例
    text = "プログラミング言語処理とTransformerの類似点を理解する"
    
    # 簡単な例のため、事前定義されたルールを使用
    nl_tokenizer = NaturalLanguageTokenizer()
    
    # 文字レベルのトークン化（デモ用）
    char_tokens = list(text)
    print("\n=== 自然言語のトークン化（文字レベル）===")
    print(f"入力: {text}")
    print(f"トークン: {char_tokens}")
    
    # サブワードトークン化の例
    print("\n=== サブワードトークン化の例 ===")
    subword_example = "understanding unbelievable preprocessing"
    # 仮想的なBPE結果
    subword_tokens = ["under", "stand", "ing", " ", "un", "believ", "able", " ", "pre", "process", "ing"]
    print(f"入力: {subword_example}")
    print(f"トークン: {subword_tokens}")

compare_tokenizers()
```

### トークン化の本質的な違いと共通点

```python
class TokenizationComparison:
    """プログラミング言語と自然言語のトークン化を詳細に比較"""
    
    def __init__(self):
        self.prog_lang_characteristics = {
            "deterministic": True,      # 決定的
            "context_free": True,       # 文脈自由
            "fixed_vocabulary": True,   # 固定語彙
            "strict_syntax": True,      # 厳密な構文
            "error_handling": "reject", # エラー時は拒否
        }
        
        self.natural_lang_characteristics = {
            "deterministic": False,     # 非決定的（曖昧性あり）
            "context_free": False,      # 文脈依存
            "fixed_vocabulary": False,  # 開放語彙
            "strict_syntax": False,     # 柔軟な構文
            "error_handling": "robust", # エラーに寛容
        }
    
    def demonstrate_ambiguity(self):
        """自然言語の曖昧性を示す例"""
        
        # プログラミング言語：明確
        code_example = "list.append(item)"
        print("プログラミング言語の例:")
        print(f"  {code_example}")
        print("  解釈: 'list'オブジェクトの'append'メソッドを'item'引数で呼び出す")
        print("  曖昧性: なし")
        
        # 自然言語：曖昧
        nl_example = "I saw the man with the telescope"
        print("\n自然言語の例:")
        print(f"  {nl_example}")
        print("  解釈1: 望遠鏡を使って男を見た")
        print("  解釈2: 望遠鏡を持っている男を見た")
        print("  曖昧性: あり（構文的曖昧性）")
        
        # Transformerはこの曖昧性を文脈から解決する
        return self.show_context_resolution()
    
    def show_context_resolution(self):
        """Transformerが文脈から曖昧性を解決する仕組み"""
        
        class ContextAwareTokenizer:
            def __init__(self):
                # 仮想的な文脈認識機構
                self.context_embeddings = {}
            
            def tokenize_with_context(self, sentence: str, context: List[str]) -> List[Tuple[str, str]]:
                """
                文脈を考慮したトークン化
                返り値: [(トークン, 意味タグ), ...]
                """
                # 例: "bank"の曖昧性解決
                if "bank" in sentence:
                    # 文脈から意味を推定
                    financial_words = {"money", "account", "loan", "deposit"}
                    river_words = {"river", "water", "boat", "fishing"}
                    
                    context_text = " ".join(context).lower()
                    
                    if any(word in context_text for word in financial_words):
                        meaning = "financial_institution"
                    elif any(word in context_text for word in river_words):
                        meaning = "river_bank"
                    else:
                        meaning = "unknown"
                    
                    return [("bank", meaning)]
                
                # 簡単のため、他の単語は通常のトークン化
                return [(word, "general") for word in sentence.split()]
        
        # デモ
        tokenizer = ContextAwareTokenizer()
        
        # 文脈1: 金融関連
        context1 = ["I need to deposit money", "The interest rate is high"]
        sentence1 = "I went to the bank"
        result1 = tokenizer.tokenize_with_context(sentence1, context1)
        print("\n文脈を考慮したトークン化の例1:")
        print(f"文脈: {context1}")
        print(f"文: {sentence1}")
        print(f"結果: {result1}")
        
        # 文脈2: 川関連
        context2 = ["The river was flowing fast", "We took our fishing rods"]
        sentence2 = "We sat on the bank"
        result2 = tokenizer.tokenize_with_context(sentence2, context2)
        print("\n文脈を考慮したトークン化の例2:")
        print(f"文脈: {context2}")
        print(f"文: {sentence2}")
        print(f"結果: {result2}")

# 実行
comparison = TokenizationComparison()
comparison.demonstrate_ambiguity()
```

### 統一的な視点：トークン化の本質

プログラミング言語と自然言語のトークン化を統一的に理解するフレームワークを構築しましょう：

```python
from abc import ABC, abstractmethod
import numpy as np

class UniversalTokenizer(ABC):
    """
    プログラミング言語と自然言語の両方に適用可能な
    統一的なトークナイザーインターフェース
    """
    
    @abstractmethod
    def tokenize(self, input_text: str) -> List[Token]:
        """入力をトークンに分解"""
        pass
    
    @abstractmethod
    def embed(self, tokens: List[Token]) -> np.ndarray:
        """トークンをベクトル表現に変換"""
        pass
    
    @abstractmethod
    def handle_unknown(self, unknown_text: str) -> List[Token]:
        """未知の入力の処理"""
        pass

class HybridTokenizer(UniversalTokenizer):
    """
    プログラミング言語と自然言語の両方を扱える
    ハイブリッドトークナイザー
    """
    
    def __init__(self, vocab_size: int = 50000):
        self.vocab_size = vocab_size
        
        # プログラミング言語用の確定的ルール
        self.prog_rules = {
            'keywords': ['if', 'else', 'for', 'while', 'function', 'class', 'return'],
            'operators': ['+', '-', '*', '/', '=', '==', '!=', '<', '>', '<=', '>='],
            'delimiters': ['(', ')', '{', '}', '[', ']', ';', ',', '.'],
        }
        
        # 自然言語用のBPEルール
        self.bpe_rules = {}
        
        # 統一的な語彙
        self.unified_vocab = {}
        self._initialize_vocab()
        
        # 埋め込み行列
        self.embedding_dim = 512
        self.embeddings = nn.Embedding(self.vocab_size, self.embedding_dim)
    
    def _initialize_vocab(self):
        """統一語彙の初期化"""
        current_id = 0
        
        # 特殊トークン
        special_tokens = ['<PAD>', '<UNK>', '<BOS>', '<EOS>', '<CODE>', '<TEXT>']
        for token in special_tokens:
            self.unified_vocab[token] = current_id
            current_id += 1
        
        # プログラミング言語のトークン
        for category in ['keywords', 'operators', 'delimiters']:
            for token in self.prog_rules[category]:
                if token not in self.unified_vocab:
                    self.unified_vocab[token] = current_id
                    current_id += 1
        
        # 基本文字（ASCII）
        for i in range(256):
            char = chr(i)
            if char not in self.unified_vocab and char.isprintable():
                self.unified_vocab[char] = current_id
                current_id += 1
    
    def detect_language_type(self, text: str) -> str:
        """入力がプログラミング言語か自然言語かを判定"""
        prog_indicators = 0
        nl_indicators = 0
        
        # プログラミング言語の指標
        for keyword in self.prog_rules['keywords']:
            if keyword in text:
                prog_indicators += 2
        
        for op in self.prog_rules['operators']:
            if op in text:
                prog_indicators += 1
        
        # セミコロンや中括弧の存在
        if ';' in text or '{' in text or '}' in text:
            prog_indicators += 3
        
        # 自然言語の指標
        # 句読点の使用パターン
        if '。' in text or '、' in text or '. ' in text or ', ' in text:
            nl_indicators += 2
        
        # 単語の長さの分布
        words = text.split()
        if words:
            avg_word_length = sum(len(w) for w in words) / len(words)
            if 3 < avg_word_length < 10:  # 自然言語の典型的な単語長
                nl_indicators += 2
        
        return 'programming' if prog_indicators > nl_indicators else 'natural'
    
    def tokenize(self, input_text: str) -> List[Token]:
        """統一的なトークン化"""
        language_type = self.detect_language_type(input_text)
        
        if language_type == 'programming':
            return self._tokenize_programming(input_text)
        else:
            return self._tokenize_natural(input_text)
    
    def _tokenize_programming(self, code: str) -> List[Token]:
        """プログラミング言語のトークン化"""
        tokens = []
        # 言語タイプマーカー
        tokens.append(Token(TokenType.SPECIAL, '<CODE>', 0, 0))
        
        # 簡略化された実装
        # 実際にはより洗練されたレキサーを使用
        import re
        
        # トークンパターン
        token_pattern = r'(\w+|[^\w\s]|\s+)'
        matches = re.findall(token_pattern, code)
        
        for match in matches:
            if match.strip():  # 空白以外
                if match in self.prog_rules['keywords']:
                    token_type = TokenType.KEYWORD
                elif match in self.prog_rules['operators']:
                    token_type = TokenType.OPERATOR
                elif match in self.prog_rules['delimiters']:
                    token_type = TokenType.DELIMITER
                elif match.isdigit():
                    token_type = TokenType.NUMBER
                elif match.isidentifier():
                    token_type = TokenType.IDENTIFIER
                else:
                    token_type = TokenType.OTHER
                
                tokens.append(Token(token_type, match, 0, 0))
        
        return tokens
    
    def _tokenize_natural(self, text: str) -> List[Token]:
        """自然言語のトークン化"""
        tokens = []
        # 言語タイプマーカー
        tokens.append(Token(TokenType.SPECIAL, '<TEXT>', 0, 0))
        
        # BPEトークン化（簡略版）
        # 実際にはより洗練されたBPEアルゴリズムを使用
        words = text.split()
        
        for word in words:
            # 単語をサブワードに分解
            subwords = self._apply_bpe_rules(word)
            for subword in subwords:
                tokens.append(Token(TokenType.SUBWORD, subword, 0, 0))
        
        return tokens
    
    def _apply_bpe_rules(self, word: str) -> List[str]:
        """BPEルールの適用（簡略版）"""
        # 実際の実装では学習されたマージルールを使用
        # ここでは文字単位の分解を返す
        return list(word) + ['</w>']
    
    def embed(self, tokens: List[Token]) -> torch.Tensor:
        """トークンを埋め込みベクトルに変換"""
        token_ids = []
        
        for token in tokens:
            if token.value in self.unified_vocab:
                token_ids.append(self.unified_vocab[token.value])
            else:
                token_ids.append(self.unified_vocab['<UNK>'])
        
        token_ids = torch.tensor(token_ids)
        return self.embeddings(token_ids)
    
    def handle_unknown(self, unknown_text: str) -> List[Token]:
        """未知の入力の処理"""
        # 文字レベルにフォールバック
        tokens = []
        for char in unknown_text:
            if char in self.unified_vocab:
                tokens.append(Token(TokenType.OTHER, char, 0, 0))
            else:
                tokens.append(Token(TokenType.OTHER, '<UNK>', 0, 0))
        
        return tokens
    
    def visualize_tokenization(self, text: str):
        """トークン化プロセスの可視化"""
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        
        tokens = self.tokenize(text)
        
        # トークンタイプごとの色
        colors = {
            TokenType.KEYWORD: 'blue',
            TokenType.IDENTIFIER: 'green',
            TokenType.OPERATOR: 'red',
            TokenType.NUMBER: 'orange',
            TokenType.DELIMITER: 'purple',
            TokenType.SUBWORD: 'brown',
            TokenType.SPECIAL: 'gray',
            TokenType.OTHER: 'black',
        }
        
        fig, ax = plt.subplots(figsize=(15, 3))
        
        x_position = 0
        for i, token in enumerate(tokens):
            color = colors.get(token.type, 'black')
            
            # トークンを表示
            rect = mpatches.Rectangle(
                (x_position, 0), len(token.value) * 0.1, 1,
                facecolor=color, edgecolor='black', alpha=0.7
            )
            ax.add_patch(rect)
            
            # テキストを追加
            ax.text(
                x_position + len(token.value) * 0.05, 0.5,
                token.value, ha='center', va='center',
                fontsize=8, rotation=0 if len(token.value) < 5 else 45
            )
            
            x_position += len(token.value) * 0.1 + 0.05
        
        ax.set_xlim(0, x_position)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title(f'Tokenization Visualization: "{text[:50]}..."')
        
        # 凡例
        legend_elements = [
            mpatches.Patch(color=color, label=token_type.name)
            for token_type, color in colors.items()
            if any(t.type == token_type for t in tokens)
        ]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.1, 1))
        
        plt.tight_layout()
        plt.show()

# デモ：ハイブリッドトークナイザーの使用
hybrid_tokenizer = HybridTokenizer()

# プログラミング言語の例
code_example = "function add(a, b) { return a + b; }"
print("プログラミング言語のトークン化:")
code_tokens = hybrid_tokenizer.tokenize(code_example)
for token in code_tokens[:10]:  # 最初の10トークン
    print(f"  {token.type.name}: {token.value}")

# 自然言語の例
text_example = "Transformerは自然言語処理に革命をもたらしました。"
print("\n自然言語のトークン化:")
text_tokens = hybrid_tokenizer.tokenize(text_example)
for token in text_tokens[:10]:  # 最初の10トークン
    print(f"  {token.type.name}: {token.value}")

# 可視化
hybrid_tokenizer.visualize_tokenization(code_example)
hybrid_tokenizer.visualize_tokenization(text_example)
```

## 2.2 構文解析と文構造理解の対比

### 抽象構文木（AST）の構築

プログラミング言語の構文解析では、トークンの列から階層的な構造（AST）を構築します：

```python
from dataclasses import dataclass, field
from typing import List, Optional, Union, Any
from abc import ABC, abstractmethod

# ASTノードの基底クラス
class ASTNode(ABC):
    @abstractmethod
    def accept(self, visitor):
        """Visitorパターンの実装"""
        pass
    
    @abstractmethod
    def to_dict(self) -> dict:
        """デバッグ用の辞書表現"""
        pass

# 式ノード
@dataclass
class BinaryOp(ASTNode):
    operator: str
    left: ASTNode
    right: ASTNode
    
    def accept(self, visitor):
        return visitor.visit_binary_op(self)
    
    def to_dict(self) -> dict:
        return {
            'type': 'BinaryOp',
            'operator': self.operator,
            'left': self.left.to_dict(),
            'right': self.right.to_dict()
        }

@dataclass
class UnaryOp(ASTNode):
    operator: str
    operand: ASTNode
    
    def accept(self, visitor):
        return visitor.visit_unary_op(self)
    
    def to_dict(self) -> dict:
        return {
            'type': 'UnaryOp',
            'operator': self.operator,
            'operand': self.operand.to_dict()
        }

@dataclass
class Number(ASTNode):
    value: float
    
    def accept(self, visitor):
        return visitor.visit_number(self)
    
    def to_dict(self) -> dict:
        return {'type': 'Number', 'value': self.value}

@dataclass
class Identifier(ASTNode):
    name: str
    
    def accept(self, visitor):
        return visitor.visit_identifier(self)
    
    def to_dict(self) -> dict:
        return {'type': 'Identifier', 'name': self.name}

# 文ノード
@dataclass
class Assignment(ASTNode):
    target: Identifier
    value: ASTNode
    
    def accept(self, visitor):
        return visitor.visit_assignment(self)
    
    def to_dict(self) -> dict:
        return {
            'type': 'Assignment',
            'target': self.target.to_dict(),
            'value': self.value.to_dict()
        }

@dataclass
class IfStatement(ASTNode):
    condition: ASTNode
    then_branch: List[ASTNode]
    else_branch: Optional[List[ASTNode]] = None
    
    def accept(self, visitor):
        return visitor.visit_if_statement(self)
    
    def to_dict(self) -> dict:
        result = {
            'type': 'IfStatement',
            'condition': self.condition.to_dict(),
            'then': [stmt.to_dict() for stmt in self.then_branch]
        }
        if self.else_branch:
            result['else'] = [stmt.to_dict() for stmt in self.else_branch]
        return result

@dataclass
class FunctionDef(ASTNode):
    name: str
    parameters: List[str]
    body: List[ASTNode]
    
    def accept(self, visitor):
        return visitor.visit_function_def(self)
    
    def to_dict(self) -> dict:
        return {
            'type': 'FunctionDef',
            'name': self.name,
            'parameters': self.parameters,
            'body': [stmt.to_dict() for stmt in self.body]
        }

# パーサーの実装
class Parser:
    """
    再帰下降パーサー
    文法:
        program     -> statement*
        statement   -> assignment | if_stmt | function_def | expression
        assignment  -> IDENTIFIER "=" expression
        if_stmt     -> "if" "(" expression ")" "{" statement* "}" ["else" "{" statement* "}"]
        function_def-> "function" IDENTIFIER "(" params ")" "{" statement* "}"
        expression  -> term (("+"|"-") term)*
        term        -> factor (("*"|"/") factor)*
        factor      -> NUMBER | IDENTIFIER | "(" expression ")" | unary
        unary       -> ("-"|"!") factor
    """
    
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.current = 0
    
    def parse(self) -> List[ASTNode]:
        """プログラム全体をパース"""
        statements = []
        while not self.is_at_end():
            if self.peek().type == TokenType.EOF:
                break
            stmt = self.statement()
            if stmt:
                statements.append(stmt)
        return statements
    
    def statement(self) -> Optional[ASTNode]:
        """文をパース"""
        # 関数定義
        if self.match(TokenType.FUNCTION):
            return self.function_def()
        
        # if文
        if self.match(TokenType.IF):
            return self.if_statement()
        
        # 代入文（先読みが必要）
        if self.peek().type == TokenType.IDENTIFIER and self.peek_next() and self.peek_next().type == TokenType.ASSIGN:
            return self.assignment()
        
        # 式文
        expr = self.expression()
        self.consume(TokenType.SEMICOLON, "Expected ';' after expression")
        return expr
    
    def assignment(self) -> Assignment:
        """代入文をパース"""
        name = self.consume(TokenType.IDENTIFIER, "Expected identifier")
        self.consume(TokenType.ASSIGN, "Expected '='")
        value = self.expression()
        self.consume(TokenType.SEMICOLON, "Expected ';'")
        return Assignment(Identifier(name.value), value)
    
    def if_statement(self) -> IfStatement:
        """if文をパース"""
        self.consume(TokenType.LPAREN, "Expected '(' after 'if'")
        condition = self.expression()
        self.consume(TokenType.RPAREN, "Expected ')' after condition")
        
        self.consume(TokenType.LBRACE, "Expected '{'")
        then_branch = []
        while not self.check(TokenType.RBRACE) and not self.is_at_end():
            then_branch.append(self.statement())
        self.consume(TokenType.RBRACE, "Expected '}'")
        
        else_branch = None
        if self.match(TokenType.ELSE):
            self.consume(TokenType.LBRACE, "Expected '{' after 'else'")
            else_branch = []
            while not self.check(TokenType.RBRACE) and not self.is_at_end():
                else_branch.append(self.statement())
            self.consume(TokenType.RBRACE, "Expected '}'")
        
        return IfStatement(condition, then_branch, else_branch)
    
    def function_def(self) -> FunctionDef:
        """関数定義をパース"""
        name = self.consume(TokenType.IDENTIFIER, "Expected function name")
        self.consume(TokenType.LPAREN, "Expected '('")
        
        parameters = []
        if not self.check(TokenType.RPAREN):
            parameters.append(self.consume(TokenType.IDENTIFIER, "Expected parameter name").value)
            while self.match(TokenType.COMMA):
                parameters.append(self.consume(TokenType.IDENTIFIER, "Expected parameter name").value)
        
        self.consume(TokenType.RPAREN, "Expected ')'")
        self.consume(TokenType.LBRACE, "Expected '{'")
        
        body = []
        while not self.check(TokenType.RBRACE) and not self.is_at_end():
            body.append(self.statement())
        
        self.consume(TokenType.RBRACE, "Expected '}'")
        
        return FunctionDef(name.value, parameters, body)
    
    def expression(self) -> ASTNode:
        """式をパース（加減算）"""
        left = self.term()
        
        while self.match(TokenType.PLUS, TokenType.MINUS):
            op = self.previous()
            right = self.term()
            left = BinaryOp(op.value, left, right)
        
        return left
    
    def term(self) -> ASTNode:
        """項をパース（乗除算）"""
        left = self.factor()
        
        while self.match(TokenType.MULTIPLY, TokenType.DIVIDE):
            op = self.previous()
            right = self.factor()
            left = BinaryOp(op.value, left, right)
        
        return left
    
    def factor(self) -> ASTNode:
        """因子をパース"""
        # 単項演算子
        if self.match(TokenType.MINUS):
            op = self.previous()
            operand = self.factor()
            return UnaryOp(op.value, operand)
        
        # 数値
        if self.match(TokenType.NUMBER):
            return Number(float(self.previous().value))
        
        # 識別子
        if self.match(TokenType.IDENTIFIER):
            return Identifier(self.previous().value)
        
        # 括弧で囲まれた式
        if self.match(TokenType.LPAREN):
            expr = self.expression()
            self.consume(TokenType.RPAREN, "Expected ')' after expression")
            return expr
        
        raise self.error("Expected expression")
    
    # ヘルパーメソッド
    def match(self, *types: TokenType) -> bool:
        """現在のトークンが指定された型のいずれかにマッチするか"""
        for token_type in types:
            if self.check(token_type):
                self.advance()
                return True
        return False
    
    def check(self, token_type: TokenType) -> bool:
        """現在のトークンが指定された型か（進めない）"""
        if self.is_at_end():
            return False
        return self.peek().type == token_type
    
    def advance(self) -> Token:
        """次のトークンに進む"""
        if not self.is_at_end():
            self.current += 1
        return self.previous()
    
    def is_at_end(self) -> bool:
        """トークンの終端に達したか"""
        return self.current >= len(self.tokens)
    
    def peek(self) -> Token:
        """現在のトークンを返す（進めない）"""
        if self.is_at_end():
            return self.tokens[-1]  # EOF
        return self.tokens[self.current]
    
    def peek_next(self) -> Optional[Token]:
        """次のトークンを返す（進めない）"""
        if self.current + 1 >= len(self.tokens):
            return None
        return self.tokens[self.current + 1]
    
    def previous(self) -> Token:
        """前のトークンを返す"""
        return self.tokens[self.current - 1]
    
    def consume(self, token_type: TokenType, message: str) -> Token:
        """指定された型のトークンを消費する"""
        if self.check(token_type):
            return self.advance()
        
        raise self.error(message)
    
    def error(self, message: str):
        """パースエラーを発生させる"""
        token = self.peek()
        return SyntaxError(f"{message} at line {token.line}:{token.column}")

# AST可視化
def visualize_ast(node: ASTNode, level: int = 0):
    """ASTをテキスト形式で可視化"""
    indent = "  " * level
    
    if isinstance(node, Number):
        print(f"{indent}Number({node.value})")
    elif isinstance(node, Identifier):
        print(f"{indent}Identifier({node.name})")
    elif isinstance(node, BinaryOp):
        print(f"{indent}BinaryOp({node.operator})")
        visualize_ast(node.left, level + 1)
        visualize_ast(node.right, level + 1)
    elif isinstance(node, UnaryOp):
        print(f"{indent}UnaryOp({node.operator})")
        visualize_ast(node.operand, level + 1)
    elif isinstance(node, Assignment):
        print(f"{indent}Assignment")
        print(f"{indent}  target:")
        visualize_ast(node.target, level + 2)
        print(f"{indent}  value:")
        visualize_ast(node.value, level + 2)
    elif isinstance(node, IfStatement):
        print(f"{indent}IfStatement")
        print(f"{indent}  condition:")
        visualize_ast(node.condition, level + 2)
        print(f"{indent}  then:")
        for stmt in node.then_branch:
            visualize_ast(stmt, level + 2)
        if node.else_branch:
            print(f"{indent}  else:")
            for stmt in node.else_branch:
                visualize_ast(stmt, level + 2)
    elif isinstance(node, FunctionDef):
        print(f"{indent}FunctionDef({node.name})")
        print(f"{indent}  parameters: {node.parameters}")
        print(f"{indent}  body:")
        for stmt in node.body:
            visualize_ast(stmt, level + 2)

# 使用例
code = """
function calculate(x, y) {
    if (x > 0) {
        result = x + y;
    } else {
        result = x - y;
    }
    return result;
}
"""

lexer = Lexer(code)
tokens = lexer.tokenize()
parser = Parser(tokens)
ast = parser.parse()

print("=== AST ===")
for node in ast:
    visualize_ast(node)
```

### Transformerの階層的理解：暗黙的な構文木

Transformerは明示的なASTを構築しませんが、Attention機構を通じて暗黙的に階層構造を学習します：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional

class SyntacticAttention(nn.Module):
    """
    構文的な関係を学習するAttention機構
    Multi-Head Attentionの各ヘッドが異なる構文関係を捉える
    """
    
    def __init__(self, d_model: int = 512, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # 各ヘッドが異なる構文関係を学習
        self.head_names = [
            "局所依存",      # 隣接する単語間の関係
            "長距離依存",    # 離れた単語間の関係
            "階層構造",      # 句や節の階層
            "主語-動詞",     # 文法的関係
            "修飾関係",      # 形容詞-名詞など
            "並列構造",      # andやorで結ばれた要素
            "照応関係",      # 代名詞と先行詞
            "その他"         # その他の関係
        ]
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch_size, seq_len, d_model]
            mask: [batch_size, seq_len, seq_len]
        Returns:
            output: [batch_size, seq_len, d_model]
            attention_weights: [batch_size, n_heads, seq_len, seq_len]
        """
        batch_size, seq_len, _ = x.size()
        
        # Query, Key, Valueを計算
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attentionスコアを計算
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        
        # マスクを適用
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Attention重みを計算
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 重み付き和を計算
        context = torch.matmul(attention_weights, V)
        
        # 形状を戻す
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # 出力変換
        output = self.W_o(context)
        
        return output, attention_weights
    
    def analyze_syntactic_patterns(self, attention_weights: torch.Tensor, tokens: List[str]):
        """各ヘッドが学習した構文パターンを分析"""
        # attention_weights: [n_heads, seq_len, seq_len]
        
        analysis = {}
        for head_idx in range(self.n_heads):
            head_weights = attention_weights[head_idx].cpu().numpy()
            head_name = self.head_names[head_idx]
            
            # このヘッドが捉えている主要なパターンを特定
            patterns = self._identify_patterns(head_weights, tokens)
            analysis[head_name] = patterns
        
        return analysis
    
    def _identify_patterns(self, weights: np.ndarray, tokens: List[str]) -> dict:
        """Attention重みからパターンを特定"""
        patterns = {
            "diagonal": 0,      # 自己注意
            "previous": 0,      # 直前の単語への注意
            "next": 0,          # 直後の単語への注意
            "first": 0,         # 文頭への注意
            "last": 0,          # 文末への注意
            "periodic": 0,      # 周期的パターン
            "sparse": 0,        # 特定の単語への集中
        }
        
        seq_len = weights.shape[0]
        
        for i in range(seq_len):
            # 対角成分（自己注意）
            patterns["diagonal"] += weights[i, i]
            
            # 前後の単語
            if i > 0:
                patterns["previous"] += weights[i, i-1]
            if i < seq_len - 1:
                patterns["next"] += weights[i, i+1]
            
            # 文頭・文末
            patterns["first"] += weights[i, 0]
            patterns["last"] += weights[i, -1]
            
            # スパース性（エントロピーで測定）
            entropy = -np.sum(weights[i] * np.log(weights[i] + 1e-9))
            if entropy < 1.0:  # 低エントロピー = スパース
                patterns["sparse"] += 1
        
        # 正規化
        for key in patterns:
            patterns[key] /= seq_len
        
        return patterns

class DependencyParser:
    """
    Transformerの Attention重みから依存関係を抽出
    暗黙的な構文木を可視化
    """
    
    def __init__(self, model: SyntacticAttention):
        self.model = model
    
    def extract_dependencies(self, sentence: str, tokenizer) -> dict:
        """文から依存関係を抽出"""
        # トークン化
        tokens = tokenizer.tokenize(sentence)
        token_ids = tokenizer.encode(sentence)
        
        # 埋め込み（仮想的）
        embeddings = torch.randn(1, len(tokens), self.model.d_model)
        
        # Attention計算
        with torch.no_grad():
            output, attention_weights = self.model(embeddings)
        
        # 依存関係の抽出
        dependencies = self._extract_from_attention(attention_weights[0], tokens)
        
        return {
            'tokens': tokens,
            'dependencies': dependencies,
            'attention_weights': attention_weights[0]
        }
    
    def _extract_from_attention(self, attention_weights: torch.Tensor, tokens: List[str]) -> List[Tuple[int, int, str]]:
        """
        Attention重みから依存関係を抽出
        Returns: [(dependent_idx, head_idx, relation_type), ...]
        """
        dependencies = []
        n_heads, seq_len, _ = attention_weights.shape
        
        # 各トークンについて、最も強い依存関係を特定
        for i in range(seq_len):
            # 全ヘッドの平均を取る
            avg_attention = attention_weights.mean(dim=0)
            
            # 自己以外で最も注意を向けている単語を見つける
            attention_to_others = avg_attention[i].clone()
            attention_to_others[i] = 0  # 自己注意を除外
            
            if attention_to_others.sum() > 0:
                head_idx = attention_to_others.argmax().item()
                score = attention_to_others[head_idx].item()
                
                if score > 0.1:  # 閾値
                    # 関係タイプを推定
                    relation = self._infer_relation(i, head_idx, tokens)
                    dependencies.append((i, head_idx, relation))
        
        return dependencies
    
    def _infer_relation(self, dependent: int, head: int, tokens: List[str]) -> str:
        """依存関係のタイプを推定"""
        # 簡単なルールベースの推定
        if head < dependent:
            if dependent - head == 1:
                return "next_to"
            else:
                return "backward_dep"
        else:
            if head - dependent == 1:
                return "forward_next"
            else:
                return "forward_dep"
    
    def visualize_dependencies(self, result: dict):
        """依存関係を可視化"""
        import networkx as nx
        from matplotlib import pyplot as plt
        
        tokens = result['tokens']
        dependencies = result['dependencies']
        
        # グラフを構築
        G = nx.DiGraph()
        
        # ノードを追加
        for i, token in enumerate(tokens):
            G.add_node(i, label=token)
        
        # エッジを追加
        for dep, head, rel in dependencies:
            G.add_edge(head, dep, relation=rel)
        
        # レイアウトを計算
        pos = {}
        for i in range(len(tokens)):
            # 単語を横一列に配置
            pos[i] = (i, 0)
        
        # グラフを描画
        plt.figure(figsize=(15, 8))
        
        # ノードを描画
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=1000)
        
        # ラベルを描画
        labels = {i: tokens[i] for i in range(len(tokens))}
        nx.draw_networkx_labels(G, pos, labels, font_size=10)
        
        # エッジを描画（曲線で）
        for edge in G.edges():
            head, dep = edge
            rel = G[head][dep]['relation']
            
            # 色を関係タイプによって変える
            if 'next' in rel:
                color = 'green'
            elif 'backward' in rel:
                color = 'red'
            else:
                color = 'blue'
            
            # 曲線を描画
            if head != dep:
                connectionstyle = "arc3,rad=0.3" if head < dep else "arc3,rad=-0.3"
                plt.annotate('', xy=pos[dep], xytext=pos[head],
                           arrowprops=dict(arrowstyle='->', color=color,
                                         connectionstyle=connectionstyle,
                                         linewidth=2))
        
        plt.title('Transformer が学習した依存構造（暗黙的な構文木）')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
        # Attention ヒートマップも表示
        self.visualize_attention_heatmap(result['attention_weights'], tokens)
    
    def visualize_attention_heatmap(self, attention_weights: torch.Tensor, tokens: List[str]):
        """Attention重みのヒートマップを表示"""
        n_heads = attention_weights.shape[0]
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        for head_idx in range(n_heads):
            ax = axes[head_idx]
            
            # ヒートマップを描画
            im = ax.imshow(attention_weights[head_idx].cpu().numpy(), cmap='Blues', aspect='auto')
            
            # 軸ラベル
            ax.set_xticks(range(len(tokens)))
            ax.set_yticks(range(len(tokens)))
            ax.set_xticklabels(tokens, rotation=45, ha='right')
            ax.set_yticklabels(tokens)
            
            ax.set_title(f'Head {head_idx + 1}: {self.model.head_names[head_idx]}')
            ax.set_xlabel('Attended to')
            ax.set_ylabel('Attending from')
            
            # カラーバー
            plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        plt.show()

# デモ：プログラミング言語と自然言語の構文解析比較
def compare_syntax_parsing():
    """ASTとTransformerの構文理解を比較"""
    
    # プログラミング言語の例
    code = "if (x > 0) { y = x * 2; } else { y = 0; }"
    
    print("=== プログラミング言語の構文解析 ===")
    lexer = Lexer(code)
    tokens = lexer.tokenize()
    parser = Parser(tokens)
    ast = parser.parse()
    
    print("入力:", code)
    print("\nAST:")
    for node in ast:
        visualize_ast(node)
    
    # 自然言語の例（Transformerによる解析）
    print("\n=== 自然言語の構文解析（Transformer）===")
    sentence = "The quick brown fox jumps over the lazy dog"
    
    # モデルとパーサーを初期化
    model = SyntacticAttention(d_model=128, n_heads=8)
    dep_parser = DependencyParser(model)
    
    # 簡易トークナイザー
    class SimpleTokenizer:
        def tokenize(self, text):
            return text.split()
        def encode(self, text):
            tokens = self.tokenize(text)
            return list(range(len(tokens)))
    
    tokenizer = SimpleTokenizer()
    result = dep_parser.extract_dependencies(sentence, tokenizer)
    
    print("入力:", sentence)
    print("\n抽出された依存関係:")
    for dep, head, rel in result['dependencies']:
        print(f"  {result['tokens'][dep]} <- {result['tokens'][head]} ({rel})")
    
    # 可視化
    dep_parser.visualize_dependencies(result)

# 実行
compare_syntax_parsing()
```

### 構文的曖昧性の解決

プログラミング言語では曖昧性は許されませんが、自然言語では文脈によって解決する必要があります：

```python
class AmbiguityResolution:
    """
    構文的曖昧性をTransformerがどのように解決するかを示すデモ
    """
    
    def __init__(self):
        self.examples = {
            "PP attachment": {
                "sentence": "I saw the man with the telescope",
                "interpretations": [
                    "I used the telescope to see the man",
                    "I saw the man who had the telescope"
                ],
                "context_clues": {
                    "instrumental": ["using", "through", "by means of"],
                    "possessive": ["holding", "carrying", "who had"]
                }
            },
            "coordination": {
                "sentence": "old men and women",
                "interpretations": [
                    "[old men] and [women]",
                    "[old [men and women]]"
                ],
                "context_clues": {
                    "narrow_scope": ["young women", "elderly men"],
                    "wide_scope": ["elderly people", "senior citizens"]
                }
            }
        }
    
    def demonstrate_context_dependency(self):
        """文脈による曖昧性解決のデモ"""
        
        # PP attachment の例
        example = self.examples["PP attachment"]
        
        print("=== 構文的曖昧性の例：PP Attachment ===")
        print(f"曖昧な文: {example['sentence']}")
        print("\n可能な解釈:")
        for i, interp in enumerate(example['interpretations']):
            print(f"  {i+1}. {interp}")
        
        # 文脈を追加して曖昧性を解決
        print("\n文脈による解決:")
        
        # 文脈1：道具として
        context1 = "I had borrowed a telescope from the observatory. " + example['sentence']
        print(f"\n文脈1: {context1}")
        print("→ 解釈: 望遠鏡を使って男を見た")
        
        # 文脈2：所有として
        context2 = "The man was an astronomer. " + example['sentence']
        print(f"\n文脈2: {context2}")
        print("→ 解釈: 望遠鏡を持っている男を見た")
        
        # Transformerがどのように解決するかを可視化
        self.visualize_ambiguity_resolution(example['sentence'], [context1, context2])
    
    def visualize_ambiguity_resolution(self, ambiguous_sentence: str, contexts: List[str]):
        """
        異なる文脈でのAttentionパターンの違いを可視化
        """
        # 仮想的なAttentionパターン
        # 実際にはモデルから取得
        
        fig, axes = plt.subplots(1, len(contexts), figsize=(15, 5))
        
        words = ambiguous_sentence.split()
        
        for i, (context, ax) in enumerate(zip(contexts, axes)):
            # 文脈に応じた仮想的なAttentionパターン
            if i == 0:  # 道具的解釈
                # "with" が "saw" に強く結びつく
                attention_matrix = np.zeros((len(words), len(words)))
                saw_idx = words.index("saw")
                with_idx = words.index("with")
                telescope_idx = words.index("telescope")
                
                attention_matrix[with_idx, saw_idx] = 0.8
                attention_matrix[telescope_idx, with_idx] = 0.7
                
            else:  # 所有的解釈
                # "with" が "man" に強く結びつく
                attention_matrix = np.zeros((len(words), len(words)))
                man_idx = words.index("man")
                with_idx = words.index("with")
                telescope_idx = words.index("telescope")
                
                attention_matrix[with_idx, man_idx] = 0.8
                attention_matrix[telescope_idx, with_idx] = 0.7
            
            # 対角成分（自己注意）を追加
            np.fill_diagonal(attention_matrix, 0.3)
            
            # ヒートマップを描画
            im = ax.imshow(attention_matrix, cmap='Blues', aspect='auto')
            
            # 軸ラベル
            ax.set_xticks(range(len(words)))
            ax.set_yticks(range(len(words)))
            ax.set_xticklabels(words, rotation=45, ha='right')
            ax.set_yticklabels(words)
            
            ax.set_title(f'文脈{i+1}でのAttentionパターン')
            
            # 主要な結合を矢印で強調
            if i == 0:
                ax.annotate('', xy=(saw_idx, with_idx), xytext=(saw_idx, with_idx + 0.3),
                          arrowprops=dict(arrowstyle='->', color='red', lw=2))
            else:
                ax.annotate('', xy=(man_idx, with_idx), xytext=(man_idx, with_idx + 0.3),
                          arrowprops=dict(arrowstyle='->', color='red', lw=2))
        
        plt.tight_layout()
        plt.show()

# デモ実行
ambiguity_demo = AmbiguityResolution()
ambiguity_demo.demonstrate_context_dependency()
```

## 2.3 意味解析と意味理解の対比

### 型推論と文脈理解の類似性

コンパイラの型推論システムとTransformerの文脈理解には、驚くべき類似性があります：

```python
from typing import Dict, List, Set, Union, Optional, Any
from dataclasses import dataclass
import torch
import torch.nn as nn

# 型システムの定義
@dataclass
class Type:
    """基本的な型クラス"""
    pass

@dataclass
class PrimitiveType(Type):
    name: str  # int, float, string, bool

@dataclass
class FunctionType(Type):
    param_types: List[Type]
    return_type: Type

@dataclass
class ArrayType(Type):
    element_type: Type

@dataclass
class ObjectType(Type):
    fields: Dict[str, Type]

class TypeInferenceEngine:
    """
    型推論エンジン
    Hindley-Milner型推論の簡略版
    """
    
    def __init__(self):
        self.type_env: Dict[str, Type] = {}
        self.constraints: List[Tuple[Type, Type]] = []
        
        # 組み込み関数の型
        self.builtin_types = {
            'print': FunctionType([PrimitiveType('any')], PrimitiveType('void')),
            'len': FunctionType([ArrayType(PrimitiveType('any'))], PrimitiveType('int')),
            'str': FunctionType([PrimitiveType('any')], PrimitiveType('string')),
        }
    
    def infer_type(self, ast_node: ASTNode, env: Dict[str, Type] = None) -> Type:
        """ASTノードから型を推論"""
        if env is None:
            env = self.type_env.copy()
        
        if isinstance(ast_node, Number):
            # 数値リテラル
            if '.' in str(ast_node.value):
                return PrimitiveType('float')
            else:
                return PrimitiveType('int')
        
        elif isinstance(ast_node, Identifier):
            # 変数参照
            if ast_node.name in env:
                return env[ast_node.name]
            elif ast_node.name in self.builtin_types:
                return self.builtin_types[ast_node.name]
            else:
                # 未知の変数：型変数を生成
                type_var = self.create_type_variable()
                env[ast_node.name] = type_var
                return type_var
        
        elif isinstance(ast_node, BinaryOp):
            # 二項演算
            left_type = self.infer_type(ast_node.left, env)
            right_type = self.infer_type(ast_node.right, env)
            
            # 演算子に応じた型制約
            if ast_node.operator in ['+', '-', '*', '/']:
                # 数値演算
                self.add_constraint(left_type, right_type)
                if ast_node.operator == '/':
                    return PrimitiveType('float')
                else:
                    return left_type
            
            elif ast_node.operator in ['==', '!=', '<', '>', '<=', '>=']:
                # 比較演算
                self.add_constraint(left_type, right_type)
                return PrimitiveType('bool')
        
        elif isinstance(ast_node, Assignment):
            # 代入文
            value_type = self.infer_type(ast_node.value, env)
            env[ast_node.target.name] = value_type
            return value_type
        
        # ... 他のノードタイプの処理
        
        return PrimitiveType('any')
    
    def create_type_variable(self) -> Type:
        """新しい型変数を生成"""
        # 簡略化のため、ここでは any 型を返す
        return PrimitiveType('any')
    
    def add_constraint(self, type1: Type, type2: Type):
        """型制約を追加"""
        self.constraints.append((type1, type2))
    
    def solve_constraints(self):
        """制約を解いて型を確定"""
        # 簡略化された実装
        # 実際にはunificationアルゴリズムを使用
        pass

class SemanticUnderstanding(nn.Module):
    """
    Transformerによる意味理解
    型推論との類似性を示すモデル
    """
    
    def __init__(self, vocab_size: int, d_model: int = 512, n_heads: int = 8):
        super().__init__()
        self.d_model = d_model
        
        # トークン埋め込み
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # 「型」埋め込み（意味カテゴリ）
        self.semantic_types = [
            "entity",      # エンティティ（人、物、場所）
            "action",      # 動作・行為
            "attribute",   # 属性・性質
            "relation",    # 関係
            "quantity",    # 数量
            "time",        # 時間
            "location",    # 場所
            "abstract"     # 抽象概念
        ]
        self.type_embedding = nn.Embedding(len(self.semantic_types), d_model)
        
        # Transformer層
        self.attention = nn.MultiheadAttention(d_model, n_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        
        # 意味タイプ分類器
        self.type_classifier = nn.Linear(d_model, len(self.semantic_types))
        
        # 関係抽出器
        self.relation_extractor = nn.Bilinear(d_model, d_model, d_model)
    
    def forward(self, input_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        入力から意味表現を計算
        型推論と同様に、各トークンの「意味型」を推論
        """
        # トークン埋め込み
        x = self.token_embedding(input_ids)
        
        # Self-Attention（文脈を考慮）
        attn_output, attention_weights = self.attention(x, x, x)
        x = x + attn_output  # 残差接続
        
        # Feed Forward
        ff_output = self.feed_forward(x)
        x = x + ff_output  # 残差接続
        
        # 各トークンの意味タイプを推論
        type_logits = self.type_classifier(x)
        predicted_types = torch.argmax(type_logits, dim=-1)
        
        # 意味表現
        semantic_representation = x
        
        return {
            'representation': semantic_representation,
            'type_logits': type_logits,
            'predicted_types': predicted_types,
            'attention_weights': attention_weights
        }
    
    def extract_relations(self, representation: torch.Tensor) -> torch.Tensor:
        """
        トークン間の意味的関係を抽出
        型システムの関数型のような関係を学習
        """
        batch_size, seq_len, d_model = representation.shape
        
        # すべてのトークンペア間の関係を計算
        relations = torch.zeros(batch_size, seq_len, seq_len, self.d_model)
        
        for i in range(seq_len):
            for j in range(seq_len):
                if i != j:
                    relations[:, i, j] = self.relation_extractor(
                        representation[:, i],
                        representation[:, j]
                    )
        
        return relations

def compare_type_inference_and_semantic_understanding():
    """
    型推論と意味理解の比較デモ
    """
    
    print("=== 型推論と意味理解の比較 ===\n")
    
    # 1. プログラミング言語の型推論
    print("1. プログラミング言語の型推論")
    code = """
    x = 42
    y = 3.14
    z = x + y
    result = z > 40
    """
    
    # 簡易的な型推論
    type_engine = TypeInferenceEngine()
    print(f"コード:\n{code}")
    print("\n推論された型:")
    print("  x: int")
    print("  y: float")
    print("  z: float (int + float → float)")
    print("  result: bool (float > int → bool)")
    
    # 2. 自然言語の意味理解
    print("\n2. 自然言語の意味理解（Transformer）")
    sentence = "The cat sat on the mat"
    tokens = sentence.split()
    
    # 仮想的な意味タイプ
    semantic_types = {
        "The": "determiner",
        "cat": "entity",
        "sat": "action",
        "on": "relation",
        "the": "determiner",
        "mat": "entity"
    }
    
    print(f"\n文: {sentence}")
    print("\n推論された意味タイプ:")
    for token, sem_type in semantic_types.items():
        print(f"  {token}: {sem_type}")
    
    # 3. 統一的な視点
    print("\n3. 統一的な視点：制約ベースの推論")
    
    class UnifiedInference:
        """型推論と意味推論の統一モデル"""
        
        def __init__(self):
            self.constraints = []
        
        def add_constraint(self, item1, item2, relation):
            """制約を追加"""
            self.constraints.append((item1, item2, relation))
        
        def solve(self):
            """制約を解く"""
            # プログラミング言語の例
            print("\nプログラミング言語の制約:")
            print("  - x + y の型は x と y の型の上限")
            print("  - 比較演算の結果は bool")
            
            # 自然言語の例
            print("\n自然言語の制約:")
            print("  - 'sat' は主語（entity）と場所（location）を要求")
            print("  - 'on' は２つのentityを関係づける")
    
    unified = UnifiedInference()
    unified.solve()
    
    # 可視化
    visualize_inference_process()

def visualize_inference_process():
    """推論プロセスの可視化"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 型推論のグラフ
    ax1.set_title("型推論のプロセス")
    
    # ノード（変数と型）
    variables = ['x', 'y', 'z', 'result']
    types = ['int', 'float', 'float', 'bool']
    
    # グラフの描画
    import networkx as nx
    G1 = nx.DiGraph()
    
    # 変数ノード
    for i, var in enumerate(variables):
        G1.add_node(f"var_{var}", label=var, pos=(0, -i))
    
    # 型ノード
    for i, type_name in enumerate(types):
        G1.add_node(f"type_{type_name}", label=type_name, pos=(2, -i))
    
    # エッジ（推論関係）
    edges = [
        ("var_x", "type_int"),
        ("var_y", "type_float"),
        ("var_z", "type_float"),
        ("var_result", "type_bool")
    ]
    
    G1.add_edges_from(edges)
    
    pos1 = nx.get_node_attributes(G1, 'pos')
    nx.draw(G1, pos1, ax=ax1, with_labels=True, node_color='lightblue',
            node_size=1000, font_size=10, arrows=True)
    
    # 意味理解のグラフ
    ax2.set_title("意味理解のプロセス")
    
    # 文の構造
    G2 = nx.DiGraph()
    
    sentence_structure = {
        "The cat": "entity",
        "sat": "action",
        "on the mat": "location"
    }
    
    # ノードとエッジ
    prev_node = None
    for i, (phrase, sem_type) in enumerate(sentence_structure.items()):
        G2.add_node(phrase, label=f"{phrase}\n({sem_type})", pos=(i, 0))
        if prev_node:
            G2.add_edge(prev_node, phrase)
        prev_node = phrase
    
    pos2 = nx.get_node_attributes(G2, 'pos')
    nx.draw(G2, pos2, ax=ax2, with_labels=True, node_color='lightgreen',
            node_size=2000, font_size=10, arrows=True)
    
    plt.tight_layout()
    plt.show()

# デモ実行
compare_type_inference_and_semantic_understanding()
```

### スコープ解決と文脈窓

変数のスコープ解決と、Transformerの文脈窓には直接的な対応関係があります：

```python
class ScopeAndContext:
    """
    プログラミング言語のスコープとTransformerの文脈窓の比較
    """
    
    def __init__(self):
        self.examples = []
    
    def demonstrate_scope_resolution(self):
        """スコープ解決のデモ"""
        
        print("=== スコープ解決の例 ===\n")
        
        # プログラミング言語のスコープ
        code = '''
        global_var = 100
        
        def outer_function():
            outer_var = 50
            
            def inner_function():
                inner_var = 10
                print(inner_var)    # 10 (ローカル)
                print(outer_var)    # 50 (外側の関数)
                print(global_var)   # 100 (グローバル)
            
            inner_function()
        
        outer_function()
        '''
        
        print("プログラミング言語のスコープ:")
        print(code)
        
        # スコープチェーンの可視化
        self.visualize_scope_chain()
        
        # Transformerの文脈窓
        print("\n\nTransformerの文脈窓:")
        self.demonstrate_context_window()
    
    def visualize_scope_chain(self):
        """スコープチェーンの可視化"""
        
        scopes = [
            {"name": "Global", "vars": ["global_var"], "level": 0},
            {"name": "outer_function", "vars": ["outer_var"], "level": 1},
            {"name": "inner_function", "vars": ["inner_var"], "level": 2}
        ]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # スコープを入れ子の四角形で表現
        colors = ['lightblue', 'lightgreen', 'lightyellow']
        
        for i, scope in enumerate(scopes):
            # 外側から内側へ
            margin = i * 0.1
            rect = plt.Rectangle((margin, margin), 
                               1 - 2*margin, 1 - 2*margin,
                               fill=True, facecolor=colors[i],
                               edgecolor='black', linewidth=2)
            ax.add_patch(rect)
            
            # ラベル
            ax.text(0.5, 1 - margin - 0.05, scope["name"],
                   ha='center', va='top', fontsize=12, weight='bold')
            
            # 変数
            var_text = ", ".join(scope["vars"])
            ax.text(0.5, 0.5 + margin, f"変数: {var_text}",
                   ha='center', va='center', fontsize=10)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('スコープチェーン（内側から外側を参照可能）')
        
        plt.tight_layout()
        plt.show()
    
    def demonstrate_context_window(self):
        """Transformerの文脈窓のデモ"""
        
        class ContextWindow:
            def __init__(self, max_length=512):
                self.max_length = max_length
                self.attention_type = "causal"  # or "bidirectional"
            
            def create_attention_mask(self, seq_length: int) -> torch.Tensor:
                """
                Attention マスクを作成
                文脈窓の制限を実装
                """
                if self.attention_type == "causal":
                    # 因果的マスク（未来を見ない）
                    mask = torch.tril(torch.ones(seq_length, seq_length))
                else:
                    # 双方向マスク（全て見る）
                    mask = torch.ones(seq_length, seq_length)
                
                # 文脈窓の制限を適用
                for i in range(seq_length):
                    # max_length より離れたトークンはマスク
                    mask[i, :max(0, i - self.max_length)] = 0
                    if i + self.max_length < seq_length:
                        mask[i, i + self.max_length:] = 0
                
                return mask
            
            def visualize_context_window(self):
                """文脈窓を可視化"""
                seq_lengths = [10, 50, 100]
                
                fig, axes = plt.subplots(1, len(seq_lengths), figsize=(15, 5))
                
                for ax, seq_len in zip(axes, seq_lengths):
                    mask = self.create_attention_mask(seq_len)
                    
                    im = ax.imshow(mask, cmap='Blues', aspect='auto')
                    ax.set_title(f'文脈窓 (長さ={seq_len})')
                    ax.set_xlabel('参照できる位置')
                    ax.set_ylabel('現在の位置')
                    
                    # 文脈窓の境界を示す線
                    if seq_len > self.max_length:
                        for i in range(seq_len):
                            # 左境界
                            left_bound = max(0, i - self.max_length)
                            ax.axvline(x=left_bound, color='red', linestyle='--', alpha=0.5)
                            
                            # 右境界（因果的マスクの場合は現在位置）
                            right_bound = i if self.attention_type == "causal" else min(seq_len, i + self.max_length)
                            ax.axvline(x=right_bound, color='red', linestyle='--', alpha=0.5)
                
                plt.tight_layout()
                plt.show()
        
        # 異なる文脈窓サイズでのデモ
        for window_size in [8, 16, 512]:
            print(f"\n文脈窓サイズ: {window_size}")
            context = ContextWindow(max_length=window_size)
            context.visualize_context_window()

# 実行
scope_demo = ScopeAndContext()
scope_demo.demonstrate_scope_resolution()
```

### 意味の合成性

プログラミング言語の式の評価と、自然言語の意味の合成には共通の原理があります：

```python
class CompositionalSemantics:
    """
    合成的意味論：部分の意味から全体の意味を構築
    """
    
    def __init__(self):
        self.programming_example = {
            "expression": "f(g(x) + h(y))",
            "evaluation_order": [
                "1. g(x) を評価",
                "2. h(y) を評価",
                "3. g(x) + h(y) を評価",
                "4. f(...) を評価"
            ]
        }
        
        self.natural_language_example = {
            "sentence": "The red car quickly overtook the blue truck",
            "composition": [
                "1. 'red' + 'car' → 'red car' (形容詞修飾)",
                "2. 'blue' + 'truck' → 'blue truck' (形容詞修飾)",
                "3. 'quickly' + 'overtook' → 'quickly overtook' (副詞修飾)",
                "4. 'The red car' + 'quickly overtook' + 'the blue truck' → 完全な文"
            ]
        }
    
    def demonstrate_compositionality(self):
        """合成性のデモンストレーション"""
        
        print("=== 意味の合成性 ===\n")
        
        # プログラミング言語の例
        print("1. プログラミング言語での合成的評価")
        print(f"式: {self.programming_example['expression']}")
        print("評価順序:")
        for step in self.programming_example['evaluation_order']:
            print(f"  {step}")
        
        # 評価木の構築
        self.build_evaluation_tree()
        
        # 自然言語の例
        print("\n2. 自然言語での意味の合成")
        print(f"文: {self.natural_language_example['sentence']}")
        print("合成プロセス:")
        for step in self.natural_language_example['composition']:
            print(f"  {step}")
        
        # Transformerでの合成
        print("\n3. Transformerによる意味の合成")
        self.transformer_composition()
    
    def build_evaluation_tree(self):
        """評価木の構築と可視化"""
        
        # 式: f(g(x) + h(y))
        class ExprNode:
            def __init__(self, value, children=None):
                self.value = value
                self.children = children or []
        
        # 評価木を構築
        tree = ExprNode("f(...)", [
            ExprNode("+", [
                ExprNode("g(x)", [ExprNode("x")]),
                ExprNode("h(y)", [ExprNode("y")])
            ])
        ])
        
        # 可視化
        self.visualize_tree(tree, "プログラミング言語の評価木")
    
    def transformer_composition(self):
        """Transformerによる意味合成の実装"""
        
        class CompositionTransformer(nn.Module):
            def __init__(self, vocab_size, d_model=256):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, d_model)
                self.position_encoding = nn.Embedding(512, d_model)
                
                # 合成のための層
                self.composition_layers = nn.ModuleList([
                    nn.TransformerEncoderLayer(d_model, nhead=8)
                    for _ in range(6)
                ])
                
                # 句構造を予測するヘッド
                self.phrase_predictor = nn.Linear(d_model, 2)  # 句の始まり/終わり
            
            def forward(self, input_ids):
                # 埋め込み
                x = self.embedding(input_ids)
                positions = torch.arange(len(input_ids)).unsqueeze(0)
                x = x + self.position_encoding(positions)
                
                # 各層で徐々に意味を合成
                intermediate_representations = []
                
                for i, layer in enumerate(self.composition_layers):
                    x = layer(x)
                    intermediate_representations.append(x.clone())
                
                # 句構造の予測
                phrase_boundaries = self.phrase_predictor(x)
                
                return {
                    'final_representation': x,
                    'intermediate': intermediate_representations,
                    'phrase_boundaries': phrase_boundaries
                }
        
        # デモ用の可視化
        self.visualize_composition_process()
    
    def visualize_tree(self, root, title):
        """木構造の可視化"""
        import networkx as nx
        
        G = nx.DiGraph()
        
        def add_nodes(node, parent=None, pos_x=0, pos_y=0, layer=1):
            node_id = f"{node.value}_{pos_x}_{pos_y}"
            G.add_node(node_id, label=node.value, pos=(pos_x, -pos_y))
            
            if parent:
                G.add_edge(parent, node_id)
            
            # 子ノードの配置
            num_children = len(node.children)
            if num_children > 0:
                spacing = 2 ** (3 - layer)
                start_x = pos_x - (num_children - 1) * spacing / 2
                
                for i, child in enumerate(node.children):
                    child_x = start_x + i * spacing
                    add_nodes(child, node_id, child_x, pos_y + 1, layer + 1)
            
            return node_id
        
        add_nodes(root)
        
        plt.figure(figsize=(10, 6))
        pos = nx.get_node_attributes(G, 'pos')
        labels = nx.get_node_attributes(G, 'label')
        
        nx.draw(G, pos, labels=labels, with_labels=True,
                node_color='lightblue', node_size=1500,
                font_size=10, arrows=True)
        
        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def visualize_composition_process(self):
        """合成プロセスの可視化"""
        
        sentence = "The red car quickly overtook the blue truck"
        words = sentence.split()
        
        # 各層での表現の変化をシミュレート
        layers = 6
        
        fig, axes = plt.subplots(1, layers, figsize=(20, 4))
        
        for layer in range(layers):
            ax = axes[layer]
            
            # 層が深くなるにつれて、より大きな単位で結合
            if layer == 0:
                # 単語レベル
                groups = [[w] for w in words]
            elif layer == 1:
                # 形容詞と名詞
                groups = [["The", "red", "car"], ["quickly"], ["overtook"], ["the", "blue", "truck"]]
            elif layer == 2:
                # 名詞句
                groups = [["The red car"], ["quickly overtook"], ["the blue truck"]]
            elif layer >= 3:
                # 完全な文
                groups = [[sentence]]
            
            # 可視化
            y_pos = 0
            for group in groups:
                text = " ".join(group)
                ax.text(0.5, y_pos, text, ha='center', va='center',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue'))
                y_pos -= 0.2
            
            ax.set_xlim(0, 1)
            ax.set_ylim(-1, 0.5)
            ax.axis('off')
            ax.set_title(f'Layer {layer + 1}')
        
        plt.suptitle('Transformerによる階層的な意味合成')
        plt.tight_layout()
        plt.show()

# デモ実行
comp_semantics = CompositionalSemantics()
comp_semantics.demonstrate_compositionality()
```

## 2.4 最適化とモデル改善

### コンパイラ最適化技術の応用

コンパイラの最適化技術は、Transformerモデルの最適化にも応用できます：

```python
import torch
import torch.nn as nn
from torch.nn import functional as F
import time
import matplotlib.pyplot as plt

class OptimizationTechniques:
    """
    コンパイラ最適化技術とTransformer最適化の対応
    """
    
    def __init__(self):
        self.compiler_optimizations = {
            "constant_folding": "定数畳み込み",
            "dead_code_elimination": "デッドコード除去",
            "loop_unrolling": "ループ展開",
            "function_inlining": "関数のインライン化",
            "common_subexpression_elimination": "共通部分式の除去"
        }
        
        self.transformer_optimizations = {
            "weight_pruning": "重みの刈り込み",
            "quantization": "量子化",
            "knowledge_distillation": "知識蒸留",
            "attention_optimization": "Attention の最適化",
            "model_fusion": "層の融合"
        }
    
    def demonstrate_constant_folding(self):
        """定数畳み込みとモデルの事前計算"""
        
        print("=== 定数畳み込み ===")
        
        # コンパイラの例
        print("\n1. コンパイラの定数畳み込み:")
        print("  前: x = 2 * 3 * 4")
        print("  後: x = 24")
        
        # Transformerの例
        print("\n2. Transformerでの事前計算:")
        
        class PositionalEncoding(nn.Module):
            def __init__(self, d_model, max_len=5000):
                super().__init__()
                
                # 位置エンコーディングを事前計算（定数畳み込みに相当）
                pe = torch.zeros(max_len, d_model)
                position = torch.arange(0, max_len).unsqueeze(1).float()
                
                div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                                   -(torch.log(torch.tensor(10000.0)) / d_model))
                
                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term)
                
                # バッファとして登録（学習しない定数）
                self.register_buffer('pe', pe)
            
            def forward(self, x):
                # 実行時は単純な加算のみ
                return x + self.pe[:x.size(1)]
        
        # 速度比較
        d_model = 512
        seq_len = 100
        
        # 事前計算なしの場合
        def positional_encoding_naive(x, d_model, seq_len):
            pe = torch.zeros(seq_len, d_model)
            for pos in range(seq_len):
                for i in range(0, d_model, 2):
                    pe[pos, i] = torch.sin(pos / (10000 ** (i / d_model)))
                    pe[pos, i + 1] = torch.cos(pos / (10000 ** (i / d_model)))
            return x + pe
        
        # 事前計算ありの場合
        pe_optimized = PositionalEncoding(d_model)
        
        x = torch.randn(1, seq_len, d_model)
        
        # 時間計測
        import timeit
        
        time_naive = timeit.timeit(
            lambda: positional_encoding_naive(x, d_model, seq_len),
            number=100
        )
        
        time_optimized = timeit.timeit(
            lambda: pe_optimized(x),
            number=100
        )
        
        print(f"\n  事前計算なし: {time_naive:.4f}秒")
        print(f"  事前計算あり: {time_optimized:.4f}秒")
        print(f"  高速化: {time_naive / time_optimized:.2f}倍")
    
    def demonstrate_dead_code_elimination(self):
        """デッドコード除去とモデルプルーニング"""
        
        print("\n=== デッドコード除去 ===")
        
        # コンパイラの例
        print("\n1. コンパイラのデッドコード除去:")
        print("  前: if (false) { expensive_function(); }")
        print("  後: // 削除")
        
        # Transformerの例
        print("\n2. Transformerでの重みプルーニング:")
        
        class PrunableLinear(nn.Module):
            def __init__(self, in_features, out_features):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(out_features, in_features))
                self.bias = nn.Parameter(torch.zeros(out_features))
                self.mask = torch.ones_like(self.weight)
            
            def prune_weights(self, threshold=0.01):
                """小さい重みを除去（ゼロに）"""
                with torch.no_grad():
                    # 重みの絶対値が閾値以下のものをマスク
                    self.mask = (self.weight.abs() > threshold).float()
                    pruned_ratio = 1 - self.mask.sum() / self.mask.numel()
                    print(f"  プルーニング率: {pruned_ratio:.2%}")
                    return pruned_ratio
            
            def forward(self, x):
                # マスクを適用した重みで計算
                masked_weight = self.weight * self.mask
                return F.linear(x, masked_weight, self.bias)
        
        # デモ
        layer = PrunableLinear(512, 256)
        x = torch.randn(10, 512)
        
        # プルーニング前
        output_before = layer(x)
        
        # プルーニング
        pruned_ratio = layer.prune_weights(threshold=0.1)
        
        # プルーニング後
        output_after = layer(x)
        
        # 出力の違いを確認
        diff = (output_before - output_after).abs().mean()
        print(f"  出力の平均差分: {diff:.6f}")
    
    def demonstrate_loop_unrolling(self):
        """ループ展開とAttentionの最適化"""
        
        print("\n=== ループ展開 ===")
        
        # コンパイラの例
        print("\n1. コンパイラのループ展開:")
        print("  前: for i in range(4): sum += arr[i]")
        print("  後: sum += arr[0] + arr[1] + arr[2] + arr[3]")
        
        # Transformerの例
        print("\n2. AttentionのFlash Attention最適化:")
        
        class StandardAttention(nn.Module):
            def __init__(self, d_model):
                super().__init__()
                self.scale = d_model ** -0.5
            
            def forward(self, q, k, v):
                # 標準的な実装
                scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
                attn_weights = torch.softmax(scores, dim=-1)
                output = torch.matmul(attn_weights, v)
                return output
        
        class FlashAttention(nn.Module):
            """
            Flash Attention: メモリ効率的な実装
            ループをタイル化して最適化
            """
            def __init__(self, d_model, block_size=64):
                super().__init__()
                self.scale = d_model ** -0.5
                self.block_size = block_size
            
            def forward(self, q, k, v):
                batch_size, seq_len, d_model = q.shape
                
                # ブロック単位で処理（ループ展開の一種）
                output = torch.zeros_like(q)
                
                for i in range(0, seq_len, self.block_size):
                    i_end = min(i + self.block_size, seq_len)
                    q_block = q[:, i:i_end]
                    
                    # 各ブロックで計算
                    scores_block = torch.matmul(q_block, k.transpose(-2, -1)) * self.scale
                    attn_block = torch.softmax(scores_block, dim=-1)
                    output[:, i:i_end] = torch.matmul(attn_block, v)
                
                return output
        
        # メモリ使用量の比較
        d_model = 512
        seq_len = 1024
        batch_size = 8
        
        q = torch.randn(batch_size, seq_len, d_model)
        k = torch.randn(batch_size, seq_len, d_model)
        v = torch.randn(batch_size, seq_len, d_model)
        
        # 標準実装ではO(seq_len^2)のメモリ
        print(f"\n  標準Attention: メモリ使用量 O({seq_len}²) = O({seq_len**2})")
        print(f"  Flash Attention: メモリ使用量 O({seq_len}) = O({seq_len})")
    
    def demonstrate_common_subexpression_elimination(self):
        """共通部分式の除去と計算の再利用"""
        
        print("\n=== 共通部分式の除去 ===")
        
        # コンパイラの例
        print("\n1. コンパイラの共通部分式除去:")
        print("  前: a = b * c + d; e = b * c - d;")
        print("  後: temp = b * c; a = temp + d; e = temp - d;")
        
        # Transformerの例
        print("\n2. TransformerでのKVキャッシュ:")
        
        class TransformerWithKVCache(nn.Module):
            def __init__(self, d_model, n_heads):
                super().__init__()
                self.attention = nn.MultiheadAttention(d_model, n_heads)
                self.kv_cache = {}
            
            def forward(self, query, key, value, use_cache=True):
                if use_cache and 'key' in self.kv_cache:
                    # キャッシュされたK, Vを再利用
                    cached_key = self.kv_cache['key']
                    cached_value = self.kv_cache['value']
                    
                    # 新しい部分のみ計算
                    key = torch.cat([cached_key, key], dim=0)
                    value = torch.cat([cached_value, value], dim=0)
                
                # Attentionを計算
                output, _ = self.attention(query, key, value)
                
                # キャッシュを更新
                if use_cache:
                    self.kv_cache['key'] = key
                    self.kv_cache['value'] = value
                
                return output
        
        print("\n  利点:")
        print("  - 推論時の計算量削減")
        print("  - 特に自己回帰生成で効果的")
        print("  - メモリと計算のトレードオフ")
    
    def demonstrate_quantization(self):
        """量子化：精度と効率のトレードオフ"""
        
        print("\n=== 量子化 ===")
        
        class QuantizedLinear(nn.Module):
            def __init__(self, in_features, out_features, bits=8):
                super().__init__()
                self.in_features = in_features
                self.out_features = out_features
                self.bits = bits
                
                # 通常の重み（float32）
                self.weight = nn.Parameter(torch.randn(out_features, in_features))
                
                # 量子化パラメータ
                self.register_buffer('scale', torch.tensor(1.0))
                self.register_buffer('zero_point', torch.tensor(0))
            
            def quantize_weights(self):
                """重みを量子化"""
                # 重みの範囲を計算
                w_min = self.weight.min()
                w_max = self.weight.max()
                
                # スケールとゼロ点を計算
                qmin = -(2 ** (self.bits - 1))
                qmax = 2 ** (self.bits - 1) - 1
                self.scale = (w_max - w_min) / (qmax - qmin)
                self.zero_point = qmin - w_min / self.scale
                
                # 量子化
                w_quantized = torch.round(self.weight / self.scale + self.zero_point)
                w_quantized = torch.clamp(w_quantized, qmin, qmax)
                
                return w_quantized
            
            def forward(self, x):
                # 量子化された重みで計算
                w_quantized = self.quantize_weights()
                w_dequantized = (w_quantized - self.zero_point) * self.scale
                return F.linear(x, w_dequantized)
        
        # サイズとメモリの比較
        in_features, out_features = 1024, 512
        
        # 通常の層
        normal_layer = nn.Linear(in_features, out_features)
        normal_size = normal_layer.weight.numel() * 4  # float32 = 4 bytes
        
        # 量子化層（8ビット）
        quantized_layer = QuantizedLinear(in_features, out_features, bits=8)
        quantized_size = quantized_layer.weight.numel() * 1  # int8 = 1 byte
        
        print(f"\n  通常のモデル: {normal_size / 1024:.1f} KB")
        print(f"  量子化モデル: {quantized_size / 1024:.1f} KB")
        print(f"  圧縮率: {normal_size / quantized_size:.1f}倍")
        
        # 精度の比較
        x = torch.randn(10, in_features)
        output_normal = normal_layer(x)
        output_quantized = quantized_layer(x)
        
        mse = F.mse_loss(output_normal, output_quantized)
        print(f"  量子化誤差 (MSE): {mse:.6f}")
    
    def visualize_optimization_impact(self):
        """最適化の影響を可視化"""
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. プルーニングの影響
        ax = axes[0, 0]
        pruning_rates = [0, 0.1, 0.3, 0.5, 0.7, 0.9]
        accuracy = [0.95, 0.94, 0.93, 0.90, 0.85, 0.70]
        model_size = [100, 90, 70, 50, 30, 10]
        
        ax2 = ax.twinx()
        line1 = ax.plot(pruning_rates, accuracy, 'b-o', label='精度')
        line2 = ax2.plot(pruning_rates, model_size, 'r-s', label='モデルサイズ')
        
        ax.set_xlabel('プルーニング率')
        ax.set_ylabel('精度', color='b')
        ax2.set_ylabel('モデルサイズ (%)', color='r')
        ax.set_title('プルーニングの影響')
        
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels)
        
        # 2. 量子化の影響
        ax = axes[0, 1]
        bits = [32, 16, 8, 4, 2]
        accuracy_q = [0.95, 0.94, 0.92, 0.85, 0.70]
        size_ratio = [1.0, 0.5, 0.25, 0.125, 0.0625]
        
        ax2 = ax.twinx()
        ax.plot(bits, accuracy_q, 'b-o', label='精度')
        ax2.plot(bits, size_ratio, 'r-s', label='サイズ比')
        
        ax.set_xlabel('ビット数')
        ax.set_ylabel('精度', color='b')
        ax2.set_ylabel('サイズ比', color='r')
        ax.set_title('量子化の影響')
        ax.invert_xaxis()
        
        # 3. バッチサイズと速度
        ax = axes[1, 0]
        batch_sizes = [1, 2, 4, 8, 16, 32, 64]
        throughput = [10, 19, 36, 68, 120, 200, 280]
        
        ax.plot(batch_sizes, throughput, 'g-o')
        ax.set_xlabel('バッチサイズ')
        ax.set_ylabel('スループット (samples/sec)')
        ax.set_title('バッチサイズと処理速度')
        ax.set_xscale('log', base=2)
        
        # 4. 最適化手法の比較
        ax = axes[1, 1]
        methods = ['ベースライン', 'プルーニング', '量子化', '蒸留', '全て適用']
        speedup = [1.0, 1.5, 2.0, 1.8, 4.5]
        
        bars = ax.bar(methods, speedup, color=['gray', 'blue', 'green', 'orange', 'red'])
        ax.set_ylabel('高速化倍率')
        ax.set_title('最適化手法の組み合わせ効果')
        ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.5)
        
        # 値をバーの上に表示
        for bar, value in zip(bars, speedup):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.1f}x', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()

# デモ実行
opt = OptimizationTechniques()
opt.demonstrate_constant_folding()
opt.demonstrate_dead_code_elimination()
opt.demonstrate_loop_unrolling()
opt.demonstrate_common_subexpression_elimination()
opt.demonstrate_quantization()
opt.visualize_optimization_impact()
```

## まとめ：統一的な理解へ

この章では、プログラミング言語処理とTransformerの深い類似性を探求しました。主要な対応関係をまとめると：

| プログラミング言語処理 | Transformer/自然言語処理 |
|---------------------|----------------------|
| 字句解析（Lexing） | トークン化（Tokenization） |
| 構文解析（Parsing） | 構造理解（Attention） |
| 意味解析（Semantic Analysis） | 文脈理解（Contextual Understanding） |
| 型推論（Type Inference） | 意味推論（Semantic Inference） |
| スコープ解決 | 文脈窓（Context Window） |
| 最適化（Optimization） | モデル圧縮・高速化 |

これらの類似性を理解することで、Transformerの動作原理がより直感的に理解できるようになります。次章では、これらの概念を支える数学的基礎について、プログラマーの視点から解説していきます。

## 演習問題

1. **トークン化の実装**: BPEアルゴリズムを実装し、プログラミング言語のソースコードに適用してみてください。通常の字句解析との違いを分析してください。

2. **Attention可視化**: 簡単なAttention機構を実装し、プログラミング言語の構文（例：関数呼び出し）でどのようなパターンが現れるか可視化してください。

3. **型推論との対応**: Transformerの各層が出力する表現を「型」として解釈し、層を経るごとにどのように「型」が変化するか分析してください。

4. **最適化の実装**: 重みプルーニングを実装し、精度と速度のトレードオフを実験的に確認してください。

5. **ハイブリッドシステム**: プログラミング言語と自然言語の両方を扱えるトークナイザーを設計・実装してください。

---

次章では、Transformerを理解するために必要な数学的基礎を、プログラマーにとって親しみやすい形で解説します。線形代数、確率・統計、微分の基礎を、実装を通じて理解していきましょう。