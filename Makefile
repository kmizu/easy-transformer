# Makefile for easy-transformer project

.PHONY: help clean test docs install dev-install format lint type-check build publish

# デフォルトターゲット
help:
	@echo "使用可能なコマンド:"
	@echo "  make install      - 本番環境用の依存関係をインストール"
	@echo "  make dev-install  - 開発環境用の依存関係をインストール"
	@echo "  make test         - テストを実行"
	@echo "  make docs         - ドキュメントをビルド"
	@echo "  make docs-serve   - ドキュメントサーバーを起動"
	@echo "  make format       - コードをフォーマット"
	@echo "  make lint         - コードをリント"
	@echo "  make type-check   - 型チェックを実行"
	@echo "  make clean        - ビルド成果物をクリーン"
	@echo "  make build        - パッケージをビルド"
	@echo "  make publish      - PyPIに公開"

# 環境構築
install:
	pip install --upgrade pip
	pip install -e .

dev-install:
	pip install --upgrade pip
	pip install -e ".[dev,docs,notebooks]"
	pre-commit install

# テスト
test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=easy_transformer --cov-report=html --cov-report=term

test-slow:
	pytest tests/ -v -m "slow"

test-gpu:
	pytest tests/ -v -m "gpu"

# ドキュメント
docs:
	mkdocs build

docs-serve:
	mkdocs serve

docs-deploy:
	mkdocs gh-deploy

# コード品質
format:
	black src/ tests/ examples/
	isort src/ tests/ examples/

lint:
	flake8 src/ tests/ examples/
	black --check src/ tests/ examples/
	isort --check-only src/ tests/ examples/

type-check:
	mypy src/

# ビルド
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean
	python -m build

# パッケージ公開
publish-test: build
	python -m twine upload --repository testpypi dist/*

publish: build
	python -m twine upload dist/*

# 開発用コマンド
notebook:
	jupyter notebook examples/

# CI/CD
ci-test:
	pytest tests/ -v --cov=easy_transformer --cov-report=xml

ci-lint:
	black --check src/ tests/ examples/
	isort --check-only src/ tests/ examples/
	flake8 src/ tests/ examples/
	mypy src/

# プロジェクト初期化
init-project:
	@echo "プロジェクトを初期化しています..."
	python -m venv venv
	@echo "仮想環境を作成しました。"
	@echo "以下のコマンドで有効化してください:"
	@echo "  source venv/bin/activate  # Linux/Mac"
	@echo "  venv\\Scripts\\activate     # Windows"
	@echo ""
	@echo "その後、以下を実行してください:"
	@echo "  make dev-install"

# 統計情報
stats:
	@echo "=== プロジェクト統計 ==="
	@echo "Pythonファイル数:"
	@find src/ tests/ examples/ -name "*.py" | wc -l
	@echo "コード行数:"
	@find src/ tests/ examples/ -name "*.py" -exec wc -l {} + | tail -1
	@echo "ドキュメント行数:"
	@find docs/ -name "*.md" -exec wc -l {} + | tail -1