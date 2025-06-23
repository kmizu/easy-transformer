#!/bin/bash
# Setup script for easy-transformer project

set -e  # エラーで停止

# 色付き出力
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# ヘルパー関数
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}→ $1${NC}"
}

# プロジェクトルートの確認
if [ ! -f "setup.py" ]; then
    print_error "setup.pyが見つかりません。プロジェクトルートで実行してください。"
    exit 1
fi

print_info "easy-transformer プロジェクトのセットアップを開始します..."

# Python バージョンの確認
print_info "Pythonバージョンを確認しています..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    print_error "Python $required_version 以上が必要です。現在: $python_version"
    exit 1
fi
print_success "Python $python_version"

# 仮想環境の作成
if [ ! -d "venv" ]; then
    print_info "仮想環境を作成しています..."
    python3 -m venv venv
    print_success "仮想環境を作成しました"
else
    print_info "既存の仮想環境を使用します"
fi

# 仮想環境の有効化
print_info "仮想環境を有効化しています..."
source venv/bin/activate

# pipのアップグレード
print_info "pipをアップグレードしています..."
pip install --upgrade pip setuptools wheel

# 依存関係のインストール
print_info "依存関係をインストールしています..."
pip install -e ".[dev,docs,notebooks]"
print_success "依存関係をインストールしました"

# pre-commitのセットアップ
print_info "pre-commitフックをセットアップしています..."
pre-commit install
print_success "pre-commitフックをセットアップしました"

# プロジェクト構造の作成
print_info "プロジェクト構造を確認しています..."

directories=(
    "src/easy_transformer"
    "src/easy_transformer/models"
    "src/easy_transformer/utils"
    "src/easy_transformer/data"
    "src/easy_transformer/configs"
    "tests"
    "tests/unit"
    "tests/integration"
    "examples"
    "notebooks"
    "data"
    "scripts"
)

for dir in "${directories[@]}"; do
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        print_success "作成: $dir"
    fi
done

# __init__.py ファイルの作成
print_info "__init__.pyファイルを作成しています..."
touch src/easy_transformer/__init__.py
touch src/easy_transformer/models/__init__.py
touch src/easy_transformer/utils/__init__.py
touch tests/__init__.py

# .gitignoreの確認
if [ ! -f ".gitignore" ]; then
    print_info ".gitignoreファイルを作成しています..."
    cat > .gitignore << 'EOF'
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
pip-wheel-metadata/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# pyenv
.python-version

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# Pyre type checker
.pyre/

# Project specific
data/raw/
data/processed/
models/
logs/
*.log
.DS_Store
EOF
    print_success ".gitignoreを作成しました"
fi

# 環境情報の表示
print_info "\n=== 環境情報 ==="
echo "Python: $(python --version)"
echo "pip: $(pip --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not installed')"

# GPU情報の確認
if command -v nvidia-smi &> /dev/null; then
    print_info "\n=== GPU情報 ==="
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    print_info "\nGPUは検出されませんでした"
fi

# 完了メッセージ
echo ""
print_success "セットアップが完了しました！"
echo ""
echo "次のステップ:"
echo "1. 仮想環境を有効化: source venv/bin/activate"
echo "2. テストを実行: make test"
echo "3. ドキュメントを表示: make docs-serve"
echo "4. Jupyter Notebookを起動: jupyter notebook examples/"
echo ""
echo "詳細なコマンドは 'make help' を参照してください。"