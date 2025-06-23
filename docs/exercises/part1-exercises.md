# 第1部 演習問題

## 演習 1.1: Transformerの重要性

### 問題 1
以下のタスクのうち、Transformerが特に優れているものはどれですか？その理由も説明してください。

1. 画像のピクセル単位での分類
2. 長文の要約
3. リアルタイムの音声認識
4. 数値計算の最適化

??? 解答
    **答え: 2. 長文の要約**
    
    理由：
    - Transformerの自己注意機構により、文書全体の文脈を効率的に把握できる
    - 長距離依存関係を直接モデル化できる
    - 並列処理により、長文でも高速に処理可能
    
    他の選択肢について：
    - 1: CNNの方が局所的なパターン抽出に適している
    - 3: リアルタイム性を考えるとRNNベースの手法も検討される
    - 4: 数値計算は従来のアルゴリズムの方が効率的

### 問題 2
コンパイラの最適化パスとTransformerの層の類似点を3つ挙げてください。

??? 解答
    1. **段階的な抽象化**
       - コンパイラ: ソースコード → AST → IR → 機械語
       - Transformer: トークン → 埋め込み → 文脈表現 → 出力
    
    2. **情報の保持と変換**
       - コンパイラ: 各パスで必要な情報を保持しつつ変換
       - Transformer: 残差接続により元の情報を保持しつつ変換
    
    3. **並列処理の可能性**
       - コンパイラ: 独立した最適化は並列実行可能
       - Transformer: 自己注意は全位置で並列計算可能

## 演習 1.2: 数学的基礎

### 問題 3
以下の行列の積を計算してください：

```
A = [[1, 2],    B = [[5, 6],
     [3, 4]]         [7, 8]]
```

??? 解答
    ```python
    import numpy as np
    
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    
    C = A @ B
    # C = [[1*5 + 2*7, 1*6 + 2*8],
    #      [3*5 + 4*7, 3*6 + 4*8]]
    #   = [[19, 22],
    #      [43, 50]]
    ```

### 問題 4
ソフトマックス関数を実装し、以下のベクトルに適用してください：
`x = [2.0, 1.0, 0.1]`

??? 解答
    ```python
    import numpy as np
    
    def softmax(x):
        # オーバーフロー対策のため最大値を引く
        x_shifted = x - np.max(x)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x)
    
    x = np.array([2.0, 1.0, 0.1])
    result = softmax(x)
    print(result)
    # [0.6590, 0.2424, 0.0986]
    
    # 確認：合計が1になる
    print(np.sum(result))  # 1.0
    ```

## 演習 1.3: PyTorch実践

### 問題 5
PyTorchで簡単な2層ニューラルネットワークを実装してください。
- 入力次元: 10
- 隠れ層: 20ユニット（ReLU活性化）
- 出力次元: 3

??? 解答
    ```python
    import torch
    import torch.nn as nn
    
    class SimpleNN(nn.Module):
        def __init__(self, input_dim=10, hidden_dim=20, output_dim=3):
            super(SimpleNN, self).__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_dim, output_dim)
            
        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x
    
    # テスト
    model = SimpleNN()
    x = torch.randn(32, 10)  # バッチサイズ32
    output = model(x)
    print(output.shape)  # torch.Size([32, 3])
    ```

### 問題 6
勾配降下法の1ステップを手動で実装してください。

??? 解答
    ```python
    import torch
    
    # パラメータと勾配
    w = torch.tensor([1.0, 2.0], requires_grad=True)
    
    # 簡単な損失関数: L = w[0]^2 + w[1]^2
    loss = w[0]**2 + w[1]**2
    
    # 勾配計算
    loss.backward()
    
    # 手動で勾配降下
    learning_rate = 0.1
    with torch.no_grad():
        # w = w - lr * gradient
        w_new = w - learning_rate * w.grad
        
    print(f"元の重み: {w.data}")
    print(f"勾配: {w.grad}")
    print(f"更新後の重み: {w_new}")
    
    # 期待される結果:
    # 勾配: [2.0, 4.0]
    # 更新: [1.0 - 0.1*2.0, 2.0 - 0.1*4.0] = [0.8, 1.6]
    ```

## 演習 1.4: 総合問題

### 問題 7
簡単な文字レベルの言語モデルの訓練ループを実装してください。
入力: "hello world"
目標: 各文字から次の文字を予測

??? 解答
    ```python
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    # データ準備
    text = "hello world"
    chars = sorted(list(set(text)))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    
    # データセット作成
    data = [char_to_idx[ch] for ch in text]
    x = torch.tensor(data[:-1])
    y = torch.tensor(data[1:])
    
    # 簡単なモデル
    class CharModel(nn.Module):
        def __init__(self, vocab_size, hidden_size=16):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, hidden_size)
            self.fc = nn.Linear(hidden_size, vocab_size)
            
        def forward(self, x):
            x = self.embedding(x)
            x = self.fc(x)
            return x
    
    # 訓練
    model = CharModel(len(chars))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    for epoch in range(100):
        # 順伝播
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        
        # 逆伝播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    # 生成テスト
    with torch.no_grad():
        # "h"から開始
        idx = char_to_idx['h']
        result = 'h'
        
        for _ in range(10):
            x_test = torch.tensor([idx])
            logits = model(x_test)
            probs = F.softmax(logits, dim=-1)
            idx = torch.argmax(probs, dim=-1).item()
            result += idx_to_char[idx]
            
        print(f"生成結果: {result}")
    ```

## チャレンジ問題

### 問題 8 🌟
コンパイラの字句解析器のように、簡単なトークナイザーを実装してください。
以下の規則に従ってテキストをトークン化します：
- 空白で単語を分割
- 句読点は独立したトークンとして扱う
- 数字は1つのトークンとしてまとめる

入力例: "Hello, world! 123 test."
期待される出力: ["Hello", ",", "world", "!", "123", "test", "."]

??? 解答
    ```python
    import re
    
    class SimpleTokenizer:
        def __init__(self):
            # トークン化のパターン
            self.patterns = [
                (r'\d+', 'NUMBER'),           # 数字
                (r'[a-zA-Z]+', 'WORD'),       # 単語
                (r'[.,!?;:]', 'PUNCTUATION'), # 句読点
                (r'\s+', 'SPACE'),            # 空白（スキップ用）
            ]
            self.regex = '|'.join(f'({pattern})' for pattern, _ in self.patterns)
            
        def tokenize(self, text):
            tokens = []
            
            for match in re.finditer(self.regex, text):
                token = match.group()
                
                # トークンタイプを特定
                for i, (pattern, token_type) in enumerate(self.patterns):
                    if match.group(i + 1):  # グループがマッチした
                        if token_type != 'SPACE':  # 空白はスキップ
                            tokens.append(token)
                        break
                        
            return tokens
    
    # テスト
    tokenizer = SimpleTokenizer()
    text = "Hello, world! 123 test."
    tokens = tokenizer.tokenize(text)
    print(tokens)
    # ['Hello', ',', 'world', '!', '123', 'test', '.']
    
    # より複雑な例
    text2 = "The price is $99.99, isn't it?"
    tokens2 = tokenizer.tokenize(text2)
    print(tokens2)
    # ['The', 'price', 'is', '99', '.', '99', ',', 'isn', 't', 'it', '?']
    ```

## 次のステップ

これらの演習を完了したら、第2部に進んでTransformerの核心的な仕組みを学びましょう！

💡 **ヒント**: 解答を見る前に、まず自分で実装してみることをお勧めします。エラーが出ても、それが学習の一部です！