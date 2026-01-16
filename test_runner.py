#!/usr/bin/env python3
"""
test_runner.py - RAGシステム評価テストランナー

機能:
- RAGあり/なしの比較テスト
- キーワードヒット率の計測
- 具体的データ含有率の計測
- 応答時間の計測
- JSON/Markdownレポート生成
"""
import json
import re
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

from test_cases import TEST_CASES, TestCase, get_all_test_cases
from rag_system import POI_RAG_System


@dataclass
class TestResult:
    """テスト結果"""
    test_id: int
    category: str
    difficulty: str
    prompt: str
    
    # RAGあり結果
    rag_answer: str
    rag_sources: List[dict]
    rag_time_ms: float
    rag_keyword_hits: int
    rag_keyword_total: int
    rag_has_coordinate: bool
    rag_has_poi_name: bool
    
    # RAGなし結果
    no_rag_answer: str
    no_rag_time_ms: float
    no_rag_keyword_hits: int
    no_rag_keyword_total: int
    no_rag_has_coordinate: bool
    no_rag_has_poi_name: bool
    
    # 評価
    rag_score: float
    no_rag_score: float
    improvement: float


@dataclass
class TestSummary:
    """テスト結果サマリー"""
    total_tests: int
    timestamp: str
    
    # RAGあり統計
    rag_avg_keyword_hit_rate: float
    rag_coordinate_rate: float
    rag_poi_name_rate: float
    rag_avg_time_ms: float
    rag_total_score: float
    
    # RAGなし統計
    no_rag_avg_keyword_hit_rate: float
    no_rag_coordinate_rate: float
    no_rag_poi_name_rate: float
    no_rag_avg_time_ms: float
    no_rag_total_score: float
    
    # 改善率
    keyword_improvement: float
    coordinate_improvement: float
    poi_name_improvement: float
    overall_improvement: float


def count_keyword_hits(answer: str, keywords: List[str]) -> int:
    """回答に含まれるキーワード数をカウント"""
    hits = 0
    answer_lower = answer.lower()
    for keyword in keywords:
        if keyword.lower() in answer_lower:
            hits += 1
    return hits


def has_coordinate(answer: str) -> bool:
    """回答に座標情報が含まれるかチェック"""
    # 緯度経度のパターン（35.xxx, 139.xxx など）
    coord_pattern = r'3[45]\.\d{2,}'  # 緯度 34-35度
    lon_pattern = r'13[89]\.\d{2,}'   # 経度 138-139度
    
    has_lat = bool(re.search(coord_pattern, answer))
    has_lon = bool(re.search(lon_pattern, answer))
    
    # または「緯度」「経度」という単語と数字の組み合わせ
    has_explicit = '緯度' in answer and '経度' in answer
    
    return (has_lat and has_lon) or has_explicit


def has_poi_name(answer: str, sources: List[dict]) -> bool:
    """回答にPOI名が含まれるかチェック（RAG検索結果と照合）"""
    if not sources:
        return False
    
    for source in sources:
        name = source.get("name", "")
        if name and name in answer:
            return True
    return False


def has_any_poi_name(answer: str) -> bool:
    """回答に何らかの施設名が含まれるかチェック（RAGなし用）"""
    # 一般的なPOI名のパターン
    patterns = [
        r'[ァ-ヶー]+',  # カタカナ（店舗名に多い）
        r'\d+[-－]\d+[-－]\d+',  # 電話番号パターン
        r'http[s]?://',  # URL
    ]
    
    for pattern in patterns:
        if re.search(pattern, answer):
            return True
    return False


def calculate_score(keyword_hits: int, keyword_total: int, 
                   has_coord: bool, has_name: bool) -> float:
    """総合スコアを計算（0-100）"""
    if keyword_total == 0:
        keyword_score = 0
    else:
        keyword_score = (keyword_hits / keyword_total) * 50  # 最大50点
    
    coord_score = 25 if has_coord else 0  # 25点
    name_score = 25 if has_name else 0    # 25点
    
    return keyword_score + coord_score + name_score


def run_single_test(rag_system: POI_RAG_System, test_case: TestCase, 
                    verbose: bool = True) -> TestResult:
    """単一テストを実行"""
    if verbose:
        print(f"\n[{test_case.id}] {test_case.prompt[:40]}...")
    
    # RAGあり実行
    start_time = time.time()
    rag_result = rag_system.query_with_rag(test_case.prompt)
    rag_time = (time.time() - start_time) * 1000
    
    # RAGなし実行
    start_time = time.time()
    no_rag_result = rag_system.query_without_rag(test_case.prompt)
    no_rag_time = (time.time() - start_time) * 1000
    
    # RAGあり評価
    rag_keyword_hits = count_keyword_hits(rag_result["answer"], test_case.expected_keywords)
    rag_has_coord = has_coordinate(rag_result["answer"])
    rag_has_name = has_poi_name(rag_result["answer"], rag_result["sources"])
    rag_score = calculate_score(
        rag_keyword_hits, len(test_case.expected_keywords),
        rag_has_coord, rag_has_name
    )
    
    # RAGなし評価
    no_rag_keyword_hits = count_keyword_hits(no_rag_result["answer"], test_case.expected_keywords)
    no_rag_has_coord = has_coordinate(no_rag_result["answer"])
    no_rag_has_name = has_any_poi_name(no_rag_result["answer"])
    no_rag_score = calculate_score(
        no_rag_keyword_hits, len(test_case.expected_keywords),
        no_rag_has_coord, no_rag_has_name
    )
    
    # 改善率
    if no_rag_score > 0:
        improvement = ((rag_score - no_rag_score) / no_rag_score) * 100
    else:
        improvement = 100.0 if rag_score > 0 else 0.0
    
    if verbose:
        print(f"    RAGあり: スコア={rag_score:.1f}, キーワード={rag_keyword_hits}/{len(test_case.expected_keywords)}, 時間={rag_time:.0f}ms")
        print(f"    RAGなし: スコア={no_rag_score:.1f}, キーワード={no_rag_keyword_hits}/{len(test_case.expected_keywords)}, 時間={no_rag_time:.0f}ms")
        print(f"    改善率: {improvement:+.1f}%")
    
    return TestResult(
        test_id=test_case.id,
        category=test_case.category,
        difficulty=test_case.difficulty,
        prompt=test_case.prompt,
        
        rag_answer=rag_result["answer"],
        rag_sources=rag_result["sources"],
        rag_time_ms=rag_time,
        rag_keyword_hits=rag_keyword_hits,
        rag_keyword_total=len(test_case.expected_keywords),
        rag_has_coordinate=rag_has_coord,
        rag_has_poi_name=rag_has_name,
        
        no_rag_answer=no_rag_result["answer"],
        no_rag_time_ms=no_rag_time,
        no_rag_keyword_hits=no_rag_keyword_hits,
        no_rag_keyword_total=len(test_case.expected_keywords),
        no_rag_has_coordinate=no_rag_has_coord,
        no_rag_has_poi_name=no_rag_has_name,
        
        rag_score=rag_score,
        no_rag_score=no_rag_score,
        improvement=improvement
    )


def run_all_tests(rag_system: POI_RAG_System, 
                  test_cases: List[TestCase] = None,
                  verbose: bool = True) -> List[TestResult]:
    """全テストを実行"""
    if test_cases is None:
        test_cases = get_all_test_cases()
    
    results = []
    total = len(test_cases)
    
    print("=" * 60)
    print(f"RAGシステム評価テスト開始")
    print(f"テストケース数: {total}")
    print("=" * 60)
    
    for i, tc in enumerate(test_cases, 1):
        if verbose:
            print(f"\n--- テスト {i}/{total} ---")
        
        # デバッグ出力を一時的に無効化
        original_debug = rag_system.debug
        rag_system.debug = False
        
        try:
            result = run_single_test(rag_system, tc, verbose)
            results.append(result)
        except Exception as e:
            print(f"    エラー: {e}")
        finally:
            rag_system.debug = original_debug
    
    return results


def calculate_summary(results: List[TestResult]) -> TestSummary:
    """結果サマリーを計算"""
    total = len(results)
    
    if total == 0:
        return None
    
    # RAGあり統計
    rag_keyword_rates = [r.rag_keyword_hits / r.rag_keyword_total if r.rag_keyword_total > 0 else 0 for r in results]
    rag_coord_count = sum(1 for r in results if r.rag_has_coordinate)
    rag_name_count = sum(1 for r in results if r.rag_has_poi_name)
    rag_times = [r.rag_time_ms for r in results]
    rag_scores = [r.rag_score for r in results]
    
    # RAGなし統計
    no_rag_keyword_rates = [r.no_rag_keyword_hits / r.no_rag_keyword_total if r.no_rag_keyword_total > 0 else 0 for r in results]
    no_rag_coord_count = sum(1 for r in results if r.no_rag_has_coordinate)
    no_rag_name_count = sum(1 for r in results if r.no_rag_has_poi_name)
    no_rag_times = [r.no_rag_time_ms for r in results]
    no_rag_scores = [r.no_rag_score for r in results]
    
    # 改善率計算
    avg_rag_keyword = sum(rag_keyword_rates) / total * 100
    avg_no_rag_keyword = sum(no_rag_keyword_rates) / total * 100
    
    rag_coord_rate = rag_coord_count / total * 100
    no_rag_coord_rate = no_rag_coord_count / total * 100
    
    rag_name_rate = rag_name_count / total * 100
    no_rag_name_rate = no_rag_name_count / total * 100
    
    avg_rag_score = sum(rag_scores) / total
    avg_no_rag_score = sum(no_rag_scores) / total
    
    # 改善率
    def calc_improvement(rag_val, no_rag_val):
        if no_rag_val > 0:
            return ((rag_val - no_rag_val) / no_rag_val) * 100
        return 100.0 if rag_val > 0 else 0.0
    
    return TestSummary(
        total_tests=total,
        timestamp=datetime.now().isoformat(),
        
        rag_avg_keyword_hit_rate=avg_rag_keyword,
        rag_coordinate_rate=rag_coord_rate,
        rag_poi_name_rate=rag_name_rate,
        rag_avg_time_ms=sum(rag_times) / total,
        rag_total_score=avg_rag_score,
        
        no_rag_avg_keyword_hit_rate=avg_no_rag_keyword,
        no_rag_coordinate_rate=no_rag_coord_rate,
        no_rag_poi_name_rate=no_rag_name_rate,
        no_rag_avg_time_ms=sum(no_rag_times) / total,
        no_rag_total_score=avg_no_rag_score,
        
        keyword_improvement=calc_improvement(avg_rag_keyword, avg_no_rag_keyword),
        coordinate_improvement=calc_improvement(rag_coord_rate, no_rag_coord_rate),
        poi_name_improvement=calc_improvement(rag_name_rate, no_rag_name_rate),
        overall_improvement=calc_improvement(avg_rag_score, avg_no_rag_score)
    )


def generate_json_report(results: List[TestResult], summary: TestSummary, 
                         output_path: str) -> None:
    """JSONレポートを生成"""
    report = {
        "summary": asdict(summary),
        "results": [asdict(r) for r in results]
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\nJSONレポート出力: {output_path}")


def generate_markdown_report(results: List[TestResult], summary: TestSummary,
                              output_path: str) -> None:
    """Markdownレポートを生成"""
    lines = []
    
    # ヘッダー
    lines.append("# RAGシステム評価レポート")
    lines.append("")
    lines.append(f"**生成日時**: {summary.timestamp}")
    lines.append(f"**テスト総数**: {summary.total_tests}")
    lines.append("")
    
    # サマリーテーブル
    lines.append("## 📊 サマリー")
    lines.append("")
    lines.append("| 指標 | RAGあり | RAGなし | 改善率 |")
    lines.append("|------|---------|---------|--------|")
    lines.append(f"| キーワードヒット率 | {summary.rag_avg_keyword_hit_rate:.1f}% | {summary.no_rag_avg_keyword_hit_rate:.1f}% | {summary.keyword_improvement:+.1f}% |")
    lines.append(f"| 座標情報含有率 | {summary.rag_coordinate_rate:.1f}% | {summary.no_rag_coordinate_rate:.1f}% | {summary.coordinate_improvement:+.1f}% |")
    lines.append(f"| POI名含有率 | {summary.rag_poi_name_rate:.1f}% | {summary.no_rag_poi_name_rate:.1f}% | {summary.poi_name_improvement:+.1f}% |")
    lines.append(f"| 平均応答時間 | {summary.rag_avg_time_ms:.0f}ms | {summary.no_rag_avg_time_ms:.0f}ms | - |")
    lines.append(f"| **総合スコア** | **{summary.rag_total_score:.1f}** | **{summary.no_rag_total_score:.1f}** | **{summary.overall_improvement:+.1f}%** |")
    lines.append("")
    
    # カテゴリ別結果
    lines.append("## 📁 カテゴリ別結果")
    lines.append("")
    
    categories = {}
    for r in results:
        if r.category not in categories:
            categories[r.category] = []
        categories[r.category].append(r)
    
    for cat, cat_results in categories.items():
        avg_rag = sum(r.rag_score for r in cat_results) / len(cat_results)
        avg_no_rag = sum(r.no_rag_score for r in cat_results) / len(cat_results)
        avg_improvement = sum(r.improvement for r in cat_results) / len(cat_results)
        
        lines.append(f"### {cat}")
        lines.append("")
        lines.append(f"- テスト数: {len(cat_results)}")
        lines.append(f"- RAGありスコア: {avg_rag:.1f}")
        lines.append(f"- RAGなしスコア: {avg_no_rag:.1f}")
        lines.append(f"- 平均改善率: {avg_improvement:+.1f}%")
        lines.append("")
    
    # 詳細結果
    lines.append("## 📝 詳細結果")
    lines.append("")
    lines.append("| ID | 質問 | RAGあり | RAGなし | 改善率 |")
    lines.append("|----|------|---------|---------|--------|")
    
    for r in results:
        prompt_short = r.prompt[:30] + "..." if len(r.prompt) > 30 else r.prompt
        lines.append(f"| {r.test_id} | {prompt_short} | {r.rag_score:.1f} | {r.no_rag_score:.1f} | {r.improvement:+.1f}% |")
    
    lines.append("")
    
    # 結論
    lines.append("## 🎯 結論")
    lines.append("")
    if summary.overall_improvement > 50:
        lines.append("RAGシステムにより**大幅な改善**が見られました。")
    elif summary.overall_improvement > 20:
        lines.append("RAGシステムにより**有意な改善**が見られました。")
    elif summary.overall_improvement > 0:
        lines.append("RAGシステムにより**若干の改善**が見られました。")
    else:
        lines.append("RAGシステムによる改善は見られませんでした。")
    
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*このレポートは自動生成されました*")
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    
    print(f"Markdownレポート出力: {output_path}")


def print_summary(summary: TestSummary) -> None:
    """サマリーをコンソール出力"""
    print("\n" + "=" * 60)
    print("【評価結果サマリー】")
    print("=" * 60)
    
    print(f"\nテスト総数: {summary.total_tests}")
    
    print("\n■ キーワードヒット率")
    print(f"  RAGあり: {summary.rag_avg_keyword_hit_rate:.1f}%")
    print(f"  RAGなし: {summary.no_rag_avg_keyword_hit_rate:.1f}%")
    print(f"  改善率:  {summary.keyword_improvement:+.1f}%")
    
    print("\n■ 座標情報含有率")
    print(f"  RAGあり: {summary.rag_coordinate_rate:.1f}%")
    print(f"  RAGなし: {summary.no_rag_coordinate_rate:.1f}%")
    print(f"  改善率:  {summary.coordinate_improvement:+.1f}%")
    
    print("\n■ POI名含有率")
    print(f"  RAGあり: {summary.rag_poi_name_rate:.1f}%")
    print(f"  RAGなし: {summary.no_rag_poi_name_rate:.1f}%")
    print(f"  改善率:  {summary.poi_name_improvement:+.1f}%")
    
    print("\n■ 平均応答時間")
    print(f"  RAGあり: {summary.rag_avg_time_ms:.0f}ms")
    print(f"  RAGなし: {summary.no_rag_avg_time_ms:.0f}ms")
    
    print("\n■ 総合スコア (0-100)")
    print(f"  RAGあり: {summary.rag_total_score:.1f}")
    print(f"  RAGなし: {summary.no_rag_total_score:.1f}")
    print(f"  改善率:  {summary.overall_improvement:+.1f}%")
    
    print("\n" + "=" * 60)


def main():
    """メイン関数"""
    # 引数解析
    verbose = "--quiet" not in sys.argv
    quick = "--quick" in sys.argv  # クイックテスト（最初の5件のみ）
    
    # RAGシステム初期化
    print("RAGシステムを初期化中...")
    rag_system = POI_RAG_System(rebuild=False, debug=False)
    
    # テストケース取得
    if quick:
        test_cases = get_all_test_cases()[:5]
        print(f"\nクイックモード: 最初の{len(test_cases)}件のみテスト")
    else:
        test_cases = get_all_test_cases()
    
    # テスト実行
    results = run_all_tests(rag_system, test_cases, verbose)
    
    # サマリー計算
    summary = calculate_summary(results)
    
    # コンソール出力
    print_summary(summary)
    
    # レポート生成
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = f"test_report_{timestamp}.json"
    md_path = f"test_report_{timestamp}.md"
    
    generate_json_report(results, summary, json_path)
    generate_markdown_report(results, summary, md_path)
    
    # 最新版へのシンボリックリンク
    Path("test_report_latest.json").unlink(missing_ok=True)
    Path("test_report_latest.md").unlink(missing_ok=True)
    Path("test_report_latest.json").symlink_to(json_path)
    Path("test_report_latest.md").symlink_to(md_path)
    
    print(f"\n最新レポート: test_report_latest.json, test_report_latest.md")
    
    return 0 if summary.overall_improvement > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
