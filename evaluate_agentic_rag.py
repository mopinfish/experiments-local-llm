#!/usr/bin/env python3
"""
evaluate_agentic_rag.py - Agentic RAGシステムの評価スクリプト

Phase 9: Agentic RAGのテストケース評価

作成日: 2026-02-03
プロジェクト: experiments-local-llm
"""

import json
import time
import sys
from typing import List, Dict, Any
from dataclasses import dataclass

# ローカルモジュール
sys.path.insert(0, 'src')
from test_cases_agentic import ALL_AGENTIC_TEST_CASES, AgenticTestCase
from agentic_rag_system import initialize_system


@dataclass
class EvaluationResult:
    """評価結果"""
    test_id: str
    question: str
    answer: str
    execution_time: float
    iterations: int
    tool_calls: List[str]
    keyword_hit_rate: float
    success: bool
    error: str = None


def evaluate_keyword_hit_rate(answer: str, expected_keywords: List[str]) -> float:
    """
    キーワードヒット率を計算

    Args:
        answer: LLMの回答
        expected_keywords: 期待されるキーワードのリスト

    Returns:
        ヒット率（0.0-1.0）
    """
    if not expected_keywords:
        return 1.0

    answer_lower = answer.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    return hits / len(expected_keywords)


def evaluate_test_case(
    system,
    test_case: AgenticTestCase,
    verbose: bool = True
) -> EvaluationResult:
    """
    単一のテストケースを評価

    Args:
        system: AgenticRAGSystem
        test_case: テストケース
        verbose: デバッグ出力

    Returns:
        EvaluationResult
    """
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Test ID: {test_case.id}")
        print(f"Category: {test_case.subcategory}")
        print(f"Difficulty: {test_case.difficulty}")
        print(f"Question: {test_case.prompt}")
        print(f"{'=' * 60}")

    try:
        # システム実行
        result = system.query(test_case.prompt)

        # ツール呼び出しの抽出
        tool_calls = [tr.get('tool', 'unknown') for tr in result.get('tool_results', [])]

        # キーワードヒット率計算
        keyword_hit_rate = evaluate_keyword_hit_rate(
            result['answer'],
            test_case.expected_keywords
        )

        # 成功判定（キーワードヒット率50%以上）
        success = keyword_hit_rate >= 0.5 and not result.get('error')

        if verbose:
            print(f"\nAnswer: {result['answer'][:200]}...")
            print(f"Execution Time: {result['execution_time']}s")
            print(f"Iterations: {result['iterations']}")
            print(f"Tool Calls: {tool_calls}")
            print(f"Keyword Hit Rate: {keyword_hit_rate:.2%}")
            print(f"Success: {'✓' if success else '✗'}")

        return EvaluationResult(
            test_id=test_case.id,
            question=test_case.prompt,
            answer=result['answer'],
            execution_time=result['execution_time'],
            iterations=result['iterations'],
            tool_calls=tool_calls,
            keyword_hit_rate=keyword_hit_rate,
            success=success,
            error=result.get('error')
        )

    except Exception as e:
        if verbose:
            print(f"\nError: {e}")

        return EvaluationResult(
            test_id=test_case.id,
            question=test_case.prompt,
            answer="",
            execution_time=0.0,
            iterations=0,
            tool_calls=[],
            keyword_hit_rate=0.0,
            success=False,
            error=str(e)
        )


def run_evaluation(
    test_cases: List[AgenticTestCase],
    poi_file: str = "poi_documents.json",
    model_name: str = "qwen2.5:7b-instruct",
    verbose: bool = True
) -> List[EvaluationResult]:
    """
    評価を実行

    Args:
        test_cases: テストケースのリスト
        poi_file: POIデータファイル
        model_name: LLMモデル名
        verbose: デバッグ出力

    Returns:
        評価結果のリスト
    """
    print("=" * 60)
    print("Agentic RAG System Evaluation")
    print("=" * 60)
    print(f"Test Cases: {len(test_cases)}")
    print(f"Model: {model_name}")
    print(f"POI Data: {poi_file}")
    print("=" * 60)

    # システム初期化
    print("\nInitializing system...")
    system = initialize_system(
        poi_file=poi_file,
        model_name=model_name,
        verbose=False  # エージェント実行時の詳細出力は抑制
    )

    # 評価実行
    results = []
    start_time = time.time()

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n[{i}/{len(test_cases)}] Evaluating {test_case.id}...")

        result = evaluate_test_case(system, test_case, verbose=verbose)
        results.append(result)

        # 進捗表示
        if not verbose:
            status = "✓" if result.success else "✗"
            print(f"  {status} {test_case.id}: {result.keyword_hit_rate:.1%} hit rate, {result.execution_time:.1f}s")

    total_time = time.time() - start_time

    # サマリー出力
    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)

    success_count = sum(1 for r in results if r.success)
    avg_keyword_hit = sum(r.keyword_hit_rate for r in results) / len(results)
    avg_execution_time = sum(r.execution_time for r in results) / len(results)
    avg_iterations = sum(r.iterations for r in results) / len(results)

    print(f"Total Test Cases: {len(results)}")
    print(f"Success: {success_count}/{len(results)} ({success_count/len(results):.1%})")
    print(f"Average Keyword Hit Rate: {avg_keyword_hit:.1%}")
    print(f"Average Execution Time: {avg_execution_time:.2f}s")
    print(f"Average Iterations: {avg_iterations:.1f}")
    print(f"Total Evaluation Time: {total_time:.2f}s")

    # 難易度別サマリー
    print("\nBy Difficulty:")
    for difficulty in ["medium", "hard", "expert"]:
        diff_results = [r for r, tc in zip(results, test_cases) if tc.difficulty == difficulty]
        if diff_results:
            diff_success = sum(1 for r in diff_results if r.success)
            diff_hit = sum(r.keyword_hit_rate for r in diff_results) / len(diff_results)
            print(f"  {difficulty}: {diff_success}/{len(diff_results)} ({diff_success/len(diff_results):.1%}), {diff_hit:.1%} hit rate")

    # カテゴリ別サマリー
    print("\nBy Category:")
    for category in ["multi_step_spatial", "conditional_reasoning", "iterative_refinement"]:
        cat_results = [r for r, tc in zip(results, test_cases) if tc.subcategory == category]
        if cat_results:
            cat_success = sum(1 for r in cat_results if r.success)
            cat_hit = sum(r.keyword_hit_rate for r in cat_results) / len(cat_results)
            print(f"  {category}: {cat_success}/{len(cat_results)} ({cat_success/len(cat_results):.1%}), {cat_hit:.1%} hit rate")

    return results


def save_results(results: List[EvaluationResult], output_file: str = "agentic_rag_results.json"):
    """
    評価結果をJSONファイルに保存

    Args:
        results: 評価結果のリスト
        output_file: 出力ファイル名
    """
    output_data = []
    for result in results:
        output_data.append({
            "test_id": result.test_id,
            "question": result.question,
            "answer": result.answer,
            "execution_time": result.execution_time,
            "iterations": result.iterations,
            "tool_calls": result.tool_calls,
            "keyword_hit_rate": result.keyword_hit_rate,
            "success": result.success,
            "error": result.error
        })

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Results saved to {output_file}")


def main():
    """メイン実行"""
    import argparse

    parser = argparse.ArgumentParser(description="Agentic RAG System Evaluation")
    parser.add_argument("--quick", action="store_true", help="Quick test (first 5 cases only)")
    parser.add_argument("--difficulty", type=str, help="Filter by difficulty (medium, hard, expert)")
    parser.add_argument("--category", type=str, help="Filter by category")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--output", type=str, default="agentic_rag_results.json", help="Output file")

    args = parser.parse_args()

    # テストケース選択
    test_cases = ALL_AGENTIC_TEST_CASES

    if args.difficulty:
        test_cases = [tc for tc in test_cases if tc.difficulty == args.difficulty]
        print(f"Filtering by difficulty: {args.difficulty} ({len(test_cases)} cases)")

    if args.category:
        test_cases = [tc for tc in test_cases if tc.subcategory == args.category]
        print(f"Filtering by category: {args.category} ({len(test_cases)} cases)")

    if args.quick:
        test_cases = test_cases[:5]
        print(f"Quick test mode: {len(test_cases)} cases")

    if not test_cases:
        print("No test cases selected!")
        return

    # 評価実行
    results = run_evaluation(
        test_cases,
        verbose=args.verbose
    )

    # 結果保存
    save_results(results, args.output)


if __name__ == "__main__":
    main()
