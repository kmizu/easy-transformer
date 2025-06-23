# かんたんTransformer

プログラミング言語実装者のためのTransformer解説と実装

[サイト](https://kmizu.github.io/easy-transformer)

## 概要

このプロジェクトは、コンパイラやインタプリタの実装経験はあるが、機械学習やTransformerについては初心者という方を対象にした、包括的なチュートリアルです。プログラミング言語処理の概念を使いながら、Transformerアーキテクチャを基礎から解説し、実際に動くコードを実装していきます。

## 特徴

- 📚 **100ページ以上の詳細な解説**
- 🎯 **コンパイラ技術との類推を活用**
- 💻 **ステップバイステップの実装**
- 🧪 **豊富な演習問題と解答例**
- 🚀 **最新の最適化技術も網羅**

## 内容

### 第1部：導入と基礎概念
- なぜTransformerが重要なのか
- プログラミング言語処理との類似点
- 必要な数学的基礎
- PyTorchの最小限の使い方

### 第2部：Transformerへの道のり
- 単語の数値表現（トークン化）
- 注意機構の直感的理解
- 位置エンコーディング
- 層の概念と深層学習

### 第3部：Transformerアーキテクチャ詳解
- Multi-Head Attention
- Feed Forward Network
- 残差接続と層正規化
- エンコーダとデコーダ

### 第4部：実装編
- 最小限のTransformer実装
- 各コンポーネントの実装
- デバッグとビジュアライゼーション
- 動作確認

### 第5部：LLMへの拡張
- GPTアーキテクチャ
- 事前学習とファインチューニング
- トークナイザーの詳細
- 推論時の工夫

### 演習問題
- 各部に対応した演習問題
- 実装課題
- チャレンジ問題
- 詳細な解答例

### 発展的なトピック
- 最適化技術とベストプラクティス
- マルチモーダルTransformer
- 最新の研究動向

## セットアップ

### 必要な環境
- Python 3.8以上
- PyTorch 1.9以上
- CUDA対応GPU（推奨）

### インストール

```bash
# リポジトリのクローン
git clone https://github.com/yourusername/easy-transformer.git
cd easy-transformer

# 仮想環境の作成
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 依存関係のインストール
pip install -r requirements.txt
pip install -r requirements-mkdocs.txt  # ドキュメント用
```

### ドキュメントのビルド

```bash
# MkDocsでローカルサーバーを起動
mkdocs serve

# ブラウザで http://127.0.0.1:8000 を開く
```

## 使い方

1. **チュートリアルを読む**: [ドキュメントサイト](https://yourusername.github.io/easy-transformer)にアクセス

2. **コードを実行**: 各章のコード例をJupyter Notebookで実行
   ```bash
   jupyter notebook examples/
   ```

3. **演習問題に挑戦**: `docs/exercises/`ディレクトリの問題を解く

4. **実装プロジェクト**: `src/`ディレクトリの実装を参考に独自のTransformerを作成

## プロジェクト構成

```
easy-transformer/
├── docs/                    # MkDocsドキュメント
│   ├── part1/              # 第1部のコンテンツ
│   ├── part2/              # 第2部のコンテンツ
│   ├── part3/              # 第3部のコンテンツ
│   ├── part4/              # 第4部のコンテンツ
│   ├── part5/              # 第5部のコンテンツ
│   ├── exercises/          # 演習問題
│   ├── advanced/           # 発展的なトピック
│   └── appendix/           # 付録（用語集、参考資料）
├── src/                    # 実装コード
├── examples/               # 使用例
├── tests/                  # テストコード
├── mkdocs.yml             # MkDocs設定
├── requirements.txt       # Python依存関係
└── README.md             # このファイル
```

## 対象読者

- プログラミング言語の実装経験がある方
- コンパイラ、インタプリタ、パーサーを作ったことがある方
- 機械学習の経験は少ないがTransformerを理解したい方
- LLMの仕組みを深く理解したい方

## コントリビューション

プルリクエストや Issue の報告を歓迎します！

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## ライセンス

MITライセンスで公開されています。詳細は[LICENSE](LICENSE)ファイルを参照してください。

## 謝辞

このプロジェクトは以下の素晴らしいリソースに影響を受けています：

- "Attention Is All You Need" (Vaswani et al., 2017)
- The Illustrated Transformer (Jay Alammar)
- Hugging Face Transformers
- PyTorch チュートリアル

## 連絡先

質問や提案がある場合は、[Issues](https://github.com/yourusername/easy-transformer/issues)を作成してください。

---

🚀 **Happy Learning!** Transformerの世界を楽しんでください！
