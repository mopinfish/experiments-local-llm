"""
experiments-local-llm-colab 共通モジュール

Google Colab上でPOI RAGシステムを実行するための共通コンポーネント
"""

__version__ = "1.0.0"

from .test_cases import TestCase, TEST_CASES
from .evaluators import (
    count_keyword_hits,
    has_coordinate,
    has_poi_name,
    calculate_score,
    TestResult
)
from .utils import (
    setup_directories,
    save_results,
    load_results,
    generate_report
)

__all__ = [
    # テストケース
    "TestCase",
    "TEST_CASES",
    # 評価関数
    "count_keyword_hits",
    "has_coordinate", 
    "has_poi_name",
    "calculate_score",
    "TestResult",
    # ユーティリティ
    "setup_directories",
    "save_results",
    "load_results",
    "generate_report",
]
