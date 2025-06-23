#!/bin/bash
# Build script for easy-transformer project

set -e

# 色付き出力
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_success() { echo -e "${GREEN}✓ $1${NC}"; }
print_error() { echo -e "${RED}✗ $1${NC}"; }
print_info() { echo -e "${YELLOW}→ $1${NC}"; }
print_step() { echo -e "${BLUE}=== $1 ===${NC}"; }

# ビルドモードの確認
BUILD_MODE=${1:-all}

show_help() {
    echo "使用方法: ./scripts/build.sh [MODE]"
    echo ""
    echo "MODE:"
    echo "  all      - すべてをビルド (デフォルト)"
    echo "  code     - Pythonコードのみ"
    echo "  docs     - ドキュメントのみ"
    echo "  dist     - 配布パッケージ"
    echo "  docker   - Dockerイメージ"
    echo "  clean    - ビルド成果物をクリーン"
    echo ""
}

# クリーン処理
clean_build() {
    print_step "ビルド成果物をクリーンアップ"
    
    rm -rf build/ dist/ *.egg-info
    rm -rf .coverage htmlcov/ .pytest_cache/
    rm -rf site/  # MkDocs
    find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete
    
    print_success "クリーンアップ完了"
}

# コードの品質チェック
check_code_quality() {
    print_step "コード品質チェック"
    
    print_info "フォーマットチェック..."
    black --check src/ tests/ examples/ || {
        print_error "コードフォーマットが必要です。'make format'を実行してください。"
        exit 1
    }
    
    print_info "インポート順序チェック..."
    isort --check-only src/ tests/ examples/ || {
        print_error "インポート順序の修正が必要です。'make format'を実行してください。"
        exit 1
    }
    
    print_info "Lintチェック..."
    flake8 src/ tests/ examples/
    
    print_info "型チェック..."
    mypy src/
    
    print_success "コード品質チェック完了"
}

# テストの実行
run_tests() {
    print_step "テスト実行"
    
    print_info "単体テストを実行..."
    pytest tests/unit/ -v
    
    if [ -d "tests/integration" ] && [ "$(ls -A tests/integration/*.py 2>/dev/null)" ]; then
        print_info "統合テストを実行..."
        pytest tests/integration/ -v
    fi
    
    print_info "カバレッジレポート生成..."
    pytest tests/ --cov=easy_transformer --cov-report=html --cov-report=term
    
    print_success "テスト完了"
}

# Pythonパッケージのビルド
build_python_package() {
    print_step "Pythonパッケージのビルド"
    
    print_info "依存関係の確認..."
    pip install --upgrade build twine
    
    print_info "パッケージをビルド..."
    python -m build
    
    print_info "ビルド成果物の確認..."
    ls -la dist/
    
    print_success "パッケージビルド完了"
}

# ドキュメントのビルド
build_docs() {
    print_step "ドキュメントのビルド"
    
    print_info "MkDocsでドキュメントをビルド..."
    mkdocs build
    
    print_info "ビルド成果物の確認..."
    if [ -d "site" ]; then
        doc_count=$(find site -name "*.html" | wc -l)
        print_success "ドキュメントビルド完了 (${doc_count} HTMLファイル)"
    else
        print_error "ドキュメントビルド失敗"
        exit 1
    fi
}

# Dockerイメージのビルド
build_docker() {
    print_step "Dockerイメージのビルド"
    
    # Dockerfileの作成
    if [ ! -f "Dockerfile" ]; then
        print_info "Dockerfileを作成..."
        cat > Dockerfile << 'EOF'
FROM python:3.10-slim

WORKDIR /app

# システム依存関係
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Pythonパッケージのインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションコード
COPY src/ ./src/
COPY setup.py .
COPY pyproject.toml .
COPY README.md .

# パッケージのインストール
RUN pip install -e .

# ドキュメント用
COPY docs/ ./docs/
COPY mkdocs.yml .

EXPOSE 8000

CMD ["mkdocs", "serve", "--dev-addr=0.0.0.0:8000"]
EOF
    fi
    
    print_info "Dockerイメージをビルド..."
    docker build -t easy-transformer:latest .
    
    print_success "Dockerイメージビルド完了"
    docker images | grep easy-transformer
}

# バージョン情報の更新
update_version() {
    local version=$1
    
    print_info "バージョンを更新: $version"
    
    # setup.pyのバージョン更新
    sed -i "s/version=\"[^\"]*\"/version=\"$version\"/" setup.py
    
    # pyproject.tomlのバージョン更新
    sed -i "s/version = \"[^\"]*\"/version = \"$version\"/" pyproject.toml
    
    print_success "バージョン更新完了"
}

# メインビルド処理
main() {
    case $BUILD_MODE in
        all)
            clean_build
            check_code_quality
            run_tests
            build_python_package
            build_docs
            ;;
        code)
            check_code_quality
            run_tests
            build_python_package
            ;;
        docs)
            build_docs
            ;;
        dist)
            clean_build
            check_code_quality
            run_tests
            build_python_package
            ;;
        docker)
            build_docker
            ;;
        clean)
            clean_build
            ;;
        help|--help|-h)
            show_help
            exit 0
            ;;
        *)
            print_error "不明なモード: $BUILD_MODE"
            show_help
            exit 1
            ;;
    esac
    
    print_success "\nビルド完了！"
}

# スクリプト実行
main