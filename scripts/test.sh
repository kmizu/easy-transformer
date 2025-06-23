#!/bin/bash
# Test script for easy-transformer project

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

# テストモード
TEST_MODE=${1:-all}

show_help() {
    echo "使用方法: ./scripts/test.sh [MODE]"
    echo ""
    echo "MODE:"
    echo "  all      - すべてのテストを実行 (デフォルト)"
    echo "  unit     - 単体テストのみ"
    echo "  integration - 統合テストのみ"
    echo "  coverage - カバレッジレポート付きテスト"
    echo "  quick    - 高速テスト（slowマーカーをスキップ）"
    echo "  gpu      - GPU必須のテストのみ"
    echo ""
}

# 環境確認
check_environment() {
    print_step "環境確認"
    
    # Python環境
    print_info "Python: $(python --version)"
    print_info "PyTorch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not installed')"
    
    # GPU確認
    if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        gpu_name=$(python -c "import torch; print(torch.cuda.get_device_name(0))")
        print_success "GPU利用可能: $gpu_name"
    else
        print_info "GPUは利用できません"
    fi
}

# 単体テスト
run_unit_tests() {
    print_step "単体テスト実行"
    
    if [ -d "tests/unit" ]; then
        pytest tests/unit/ -v --tb=short
        print_success "単体テスト完了"
    else
        print_info "単体テストディレクトリが見つかりません"
    fi
}

# 統合テスト
run_integration_tests() {
    print_step "統合テスト実行"
    
    if [ -d "tests/integration" ]; then
        pytest tests/integration/ -v --tb=short
        print_success "統合テスト完了"
    else
        print_info "統合テストディレクトリが見つかりません"
    fi
}

# カバレッジ付きテスト
run_coverage_tests() {
    print_step "カバレッジ測定付きテスト実行"
    
    pytest tests/ \
        --cov=easy_transformer \
        --cov-branch \
        --cov-report=term-missing \
        --cov-report=html \
        --cov-report=xml \
        -v
    
    coverage_percent=$(coverage report | grep TOTAL | awk '{print $4}')
    print_success "テストカバレッジ: $coverage_percent"
    
    # カバレッジレポートの場所を表示
    print_info "HTMLレポート: htmlcov/index.html"
}

# 高速テスト（遅いテストをスキップ）
run_quick_tests() {
    print_step "高速テスト実行"
    
    pytest tests/ -v -m "not slow" --tb=short
    print_success "高速テスト完了"
}

# GPUテスト
run_gpu_tests() {
    print_step "GPUテスト実行"
    
    if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        pytest tests/ -v -m "gpu" --tb=short
        print_success "GPUテスト完了"
    else
        print_error "GPUが利用できません。GPUテストをスキップします。"
        exit 1
    fi
}

# 特定のテストファイル/関数を実行
run_specific_test() {
    local test_path=$1
    print_step "特定のテスト実行: $test_path"
    
    pytest "$test_path" -v --tb=short
    print_success "テスト完了"
}

# パフォーマンステスト
run_performance_tests() {
    print_step "パフォーマンステスト実行"
    
    # ベンチマークスクリプトの実行
    if [ -f "tests/benchmarks/test_performance.py" ]; then
        python tests/benchmarks/test_performance.py
        print_success "パフォーマンステスト完了"
    else
        print_info "パフォーマンステストが見つかりません"
    fi
}

# テスト結果のサマリー
show_test_summary() {
    print_step "テスト結果サマリー"
    
    # 最後のテスト結果を解析
    if [ -f ".coverage" ]; then
        coverage_percent=$(coverage report | grep TOTAL | awk '{print $4}')
        print_info "カバレッジ: $coverage_percent"
    fi
    
    # 失敗したテストがあるかチェック
    if [ $? -eq 0 ]; then
        print_success "すべてのテストが成功しました！"
    else
        print_error "一部のテストが失敗しました"
        exit 1
    fi
}

# メイン処理
main() {
    check_environment
    
    case $TEST_MODE in
        all)
            run_unit_tests
            run_integration_tests
            run_coverage_tests
            ;;
        unit)
            run_unit_tests
            ;;
        integration)
            run_integration_tests
            ;;
        coverage)
            run_coverage_tests
            ;;
        quick)
            run_quick_tests
            ;;
        gpu)
            run_gpu_tests
            ;;
        performance)
            run_performance_tests
            ;;
        help|--help|-h)
            show_help
            exit 0
            ;;
        *)
            # 特定のテストファイルとして扱う
            if [ -f "$TEST_MODE" ]; then
                run_specific_test "$TEST_MODE"
            else
                print_error "不明なモードまたはファイル: $TEST_MODE"
                show_help
                exit 1
            fi
            ;;
    esac
    
    show_test_summary
}

# スクリプト実行
main