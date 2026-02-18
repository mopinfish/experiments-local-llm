#!/usr/bin/env python3
"""
evaluators_multi_area.py - Phase 9-B: 複数エリア対応評価モジュール

評価指標:
- keyword_hit_rate: キーワードヒット率（0.0-1.0）
- success: keyword_hit_rate >= 0.5
- area_detection_correct: エリア特定の正確性
- language_issue: 中国語混入検出

チェックポイント機能:
- 10件ごとにJSON保存
- 再開時はスキップ

system_fn仕様:
- Callable[[str], Dict]
- 入力: 質問文(str)
- 出力: {"answer": str, "detected_area": Optional[str]}
"""

import json
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional


# =============================================================================
# 評価結果データクラス
# =============================================================================

@dataclass
class MultiAreaEvalResult:
    """複数エリア対応評価結果"""
    test_id: str
    system_name: str
    target_area: Optional[str]
    query_type: str
    subcategory: str
    level: int
    answer: str
    time_sec: float
    keyword_hit_rate: float
    success: bool                    # keyword_hit_rate >= 0.5
    area_detected: Optional[str]
    area_detection_correct: bool
    error: Optional[str] = None
    language_issue: bool = False

    def to_dict(self) -> dict:
        """辞書に変換"""
        return asdict(self)


# =============================================================================
# 評価クラス
# =============================================================================

class MultiAreaEvaluator:
    """複数エリア対応評価クラス"""

    def __init__(self, areas_config: Optional[Dict] = None):
        """
        Args:
            areas_config: エリア設定辞書（osm_poi_fetcher.AREASパターン）
        """
        self.areas_config = areas_config or {}

    def evaluate_keyword_hit_rate(self, answer: str, keywords: List[str]) -> float:
        """キーワードヒット率を計算 (0.0-1.0)

        Case-insensitive matching を行い、ヒット数 / 全キーワード数 を返す。
        """
        if not answer or not keywords:
            return 0.0

        answer_lower = answer.lower()
        hit_count = sum(1 for kw in keywords if kw.lower() in answer_lower)
        return hit_count / len(keywords)

    def detect_language_issue(self, answer: str) -> bool:
        """中国語混入を検出する

        中国語特有の文法パターン（語気助詞、代名詞、構文パターンなど）を
        正規表現でチェックし、いずれかにマッチすれば True を返す。
        """
        if not answer:
            return False

        # Chinese-specific patterns
        chinese_patterns = [
            r'[\u4e00-\u9fff]{2,}[的了吗呢吧]',  # Chinese sentence-ending particles
            r'这[个是]',     # 这个, 这是
            r'那[个是]',     # 那个, 那是
            r'我们|他们|她们',  # Chinese pronouns
            r'没有',         # 没有 (not commonly used in Japanese context this way)
            r'什么',         # 什么
            r'怎么',         # 怎么
            r'可以',         # 可以 (Chinese usage)
            r'非常',         # 非常 (could be Japanese but check context)
        ]

        for pattern in chinese_patterns:
            if re.search(pattern, answer):
                return True

        return False

    def evaluate_single_case(
        self,
        system_name: str,
        system_fn: Callable[[str], Dict],
        test_case,  # MultiAreaTestCase
    ) -> MultiAreaEvalResult:
        """1テストケースを評価

        Args:
            system_name: システム識別名
            system_fn: Callable[[str], Dict] - {"answer": str, "detected_area": Optional[str]}
            test_case: MultiAreaTestCase インスタンス

        Returns:
            MultiAreaEvalResult
        """
        start = time.time()
        try:
            result = system_fn(test_case.prompt)
            elapsed = time.time() - start
            answer = result.get("answer", "")
            detected_area = result.get("detected_area")

            keyword_hit_rate = self.evaluate_keyword_hit_rate(
                answer, test_case.expected_keywords
            )
            language_issue = self.detect_language_issue(answer)

            # Area detection correctness
            area_detection_correct = True
            if test_case.target_area is not None:
                area_detection_correct = (detected_area == test_case.target_area)

            return MultiAreaEvalResult(
                test_id=test_case.id,
                system_name=system_name,
                target_area=test_case.target_area,
                query_type=test_case.query_type,
                subcategory=test_case.subcategory,
                level=test_case.level,
                answer=answer,
                time_sec=round(elapsed, 2),
                keyword_hit_rate=round(keyword_hit_rate, 4),
                success=keyword_hit_rate >= 0.5,
                area_detected=detected_area,
                area_detection_correct=area_detection_correct,
                language_issue=language_issue,
            )
        except Exception as e:
            elapsed = time.time() - start
            return MultiAreaEvalResult(
                test_id=test_case.id,
                system_name=system_name,
                target_area=test_case.target_area,
                query_type=test_case.query_type,
                subcategory=test_case.subcategory,
                level=test_case.level,
                answer="",
                time_sec=round(elapsed, 2),
                keyword_hit_rate=0.0,
                success=False,
                area_detected=None,
                area_detection_correct=False,
                error=str(e),
            )

    def evaluate_all(
        self,
        system_name: str,
        system_fn: Callable[[str], Dict],
        test_cases: List,
        checkpoint_file: Optional[str] = None,
    ) -> List[MultiAreaEvalResult]:
        """全テストケースを評価（チェックポイント付き）

        - 10件ごとにcheckpoint_fileにJSON保存
        - 再開時は既に完了したtest_idをスキップ
        """
        results: List[MultiAreaEvalResult] = []
        completed_ids: set = set()

        # Load checkpoint if exists
        if checkpoint_file:
            cp_path = Path(checkpoint_file)
            if cp_path.exists():
                with open(cp_path, encoding="utf-8") as f:
                    saved = json.load(f)
                for item in saved:
                    result = MultiAreaEvalResult(**item)
                    results.append(result)
                    completed_ids.add(result.test_id)
                print(f"チェックポイントから{len(completed_ids)}件復元しました")

        for i, tc in enumerate(test_cases):
            if tc.id in completed_ids:
                continue

            prompt_preview = tc.prompt[:40]
            print(f"  [{i+1}/{len(test_cases)}] {tc.id}: {prompt_preview}...")
            result = self.evaluate_single_case(system_name, system_fn, tc)
            results.append(result)

            # Checkpoint every 10 cases
            if checkpoint_file and len(results) % 10 == 0:
                self._save_checkpoint(checkpoint_file, results)

        # Final save
        if checkpoint_file:
            self._save_checkpoint(checkpoint_file, results)

        return results

    def _save_checkpoint(
        self, filepath: str, results: List[MultiAreaEvalResult]
    ) -> None:
        """チェックポイントをJSON保存"""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(
                [r.to_dict() for r in results], f, ensure_ascii=False, indent=2
            )

    def generate_summary(self, results: List[MultiAreaEvalResult]) -> Dict:
        """評価結果のサマリーを生成

        Returns dict with keys:
        - overall: {success_rate, avg_keyword_hit_rate, avg_time_sec, total, error_count, language_issue_count}
        - by_area: {area_key: {success_rate, avg_keyword_hit_rate, count, avg_time_sec}}
        - by_level: {level: {success_rate, avg_keyword_hit_rate, count}}
        - by_query_type: {query_type: {success_rate, count}}
        - by_subcategory: {subcategory: {success_rate, avg_keyword_hit_rate, count}}
        - area_detection: {total, correct, accuracy}
        - cross_area: {success_rate, count, avg_time_sec}
        """
        if not results:
            return {}

        total = len(results)

        # --- overall ---
        success_count = sum(1 for r in results if r.success)
        error_count = sum(1 for r in results if r.error is not None)
        language_issue_count = sum(1 for r in results if r.language_issue)
        avg_khr = sum(r.keyword_hit_rate for r in results) / total
        avg_time = sum(r.time_sec for r in results) / total

        overall = {
            "success_rate": round(success_count / total, 4),
            "avg_keyword_hit_rate": round(avg_khr, 4),
            "avg_time_sec": round(avg_time, 2),
            "total": total,
            "error_count": error_count,
            "language_issue_count": language_issue_count,
        }

        # --- by_area ---
        by_area: Dict[str, Dict] = {}
        area_groups: Dict[str, List[MultiAreaEvalResult]] = {}
        for r in results:
            area_key = r.target_area or "cross_area"
            area_groups.setdefault(area_key, []).append(r)

        for area_key, group in area_groups.items():
            count = len(group)
            s_rate = sum(1 for r in group if r.success) / count
            a_khr = sum(r.keyword_hit_rate for r in group) / count
            a_time = sum(r.time_sec for r in group) / count
            by_area[area_key] = {
                "success_rate": round(s_rate, 4),
                "avg_keyword_hit_rate": round(a_khr, 4),
                "count": count,
                "avg_time_sec": round(a_time, 2),
            }

        # --- by_level ---
        by_level: Dict[int, Dict] = {}
        level_groups: Dict[int, List[MultiAreaEvalResult]] = {}
        for r in results:
            level_groups.setdefault(r.level, []).append(r)

        for level, group in sorted(level_groups.items()):
            count = len(group)
            s_rate = sum(1 for r in group if r.success) / count
            a_khr = sum(r.keyword_hit_rate for r in group) / count
            by_level[level] = {
                "success_rate": round(s_rate, 4),
                "avg_keyword_hit_rate": round(a_khr, 4),
                "count": count,
            }

        # --- by_query_type ---
        by_query_type: Dict[str, Dict] = {}
        qt_groups: Dict[str, List[MultiAreaEvalResult]] = {}
        for r in results:
            qt_groups.setdefault(r.query_type, []).append(r)

        for qt, group in qt_groups.items():
            count = len(group)
            s_rate = sum(1 for r in group if r.success) / count
            by_query_type[qt] = {
                "success_rate": round(s_rate, 4),
                "count": count,
            }

        # --- by_subcategory ---
        by_subcategory: Dict[str, Dict] = {}
        subcat_groups: Dict[str, List[MultiAreaEvalResult]] = {}
        for r in results:
            subcat_groups.setdefault(r.subcategory, []).append(r)

        for subcat, group in subcat_groups.items():
            count = len(group)
            s_rate = sum(1 for r in group if r.success) / count
            a_khr = sum(r.keyword_hit_rate for r in group) / count
            by_subcategory[subcat] = {
                "success_rate": round(s_rate, 4),
                "avg_keyword_hit_rate": round(a_khr, 4),
                "count": count,
            }

        # --- area_detection ---
        results_with_area = [r for r in results if r.target_area is not None]
        ad_total = len(results_with_area)
        ad_correct = sum(1 for r in results_with_area if r.area_detection_correct)
        area_detection = {
            "total": ad_total,
            "correct": ad_correct,
            "accuracy": round(ad_correct / ad_total, 4) if ad_total > 0 else 0.0,
        }

        # --- cross_area ---
        cross_results = [r for r in results if r.query_type == "cross_area"]
        if cross_results:
            ca_count = len(cross_results)
            ca_success = sum(1 for r in cross_results if r.success) / ca_count
            ca_time = sum(r.time_sec for r in cross_results) / ca_count
            cross_area = {
                "success_rate": round(ca_success, 4),
                "count": ca_count,
                "avg_time_sec": round(ca_time, 2),
            }
        else:
            cross_area = {
                "success_rate": 0.0,
                "count": 0,
                "avg_time_sec": 0.0,
            }

        return {
            "overall": overall,
            "by_area": by_area,
            "by_level": by_level,
            "by_query_type": by_query_type,
            "by_subcategory": by_subcategory,
            "area_detection": area_detection,
            "cross_area": cross_area,
        }

    def compare_systems(
        self, all_results: Dict[str, List[MultiAreaEvalResult]]
    ) -> Dict:
        """複数システムの比較分析

        Args:
            all_results: {system_name: [MultiAreaEvalResult, ...]}

        Returns dict with:
        - summaries: {system_name: summary_dict}
        - rankings: sorted list of (system_name, overall_success_rate)
        - area_consistency: {system_name: score_variance_across_areas}
        """
        summaries: Dict[str, Dict] = {}
        rankings_data: List[tuple] = []

        for system_name, results in all_results.items():
            summary = self.generate_summary(results)
            summaries[system_name] = summary
            overall_sr = summary.get("overall", {}).get("success_rate", 0.0)
            rankings_data.append((system_name, overall_sr))

        # Sort rankings by success_rate descending
        rankings = sorted(rankings_data, key=lambda x: x[1], reverse=True)

        # Area consistency: variance of success_rate across areas
        area_consistency: Dict[str, float] = {}
        for system_name, results in all_results.items():
            summary = summaries[system_name]
            by_area = summary.get("by_area", {})
            if len(by_area) >= 2:
                rates = [v["success_rate"] for v in by_area.values()]
                mean_rate = sum(rates) / len(rates)
                variance = sum((r - mean_rate) ** 2 for r in rates) / len(rates)
                area_consistency[system_name] = round(variance, 6)
            else:
                area_consistency[system_name] = 0.0

        return {
            "summaries": summaries,
            "rankings": rankings,
            "area_consistency": area_consistency,
        }


# =============================================================================
# セルフテスト
# =============================================================================

if __name__ == "__main__":
    print("evaluators_multi_area.py セルフテスト")
    print("=" * 60)

    evaluator = MultiAreaEvaluator()

    # ------------------------------------------------------------------
    # 1. evaluate_keyword_hit_rate テスト
    # ------------------------------------------------------------------
    print("\n[1] evaluate_keyword_hit_rate テスト")
    rate = evaluator.evaluate_keyword_hit_rate(
        "渋谷駅周辺にはスターバックスやドトールがあります。",
        ["スターバックス", "ドトール", "タリーズ"],
    )
    print(f"  ヒット率: {rate:.4f} (期待: 0.6667)")
    assert abs(rate - 2 / 3) < 0.001, f"Expected ~0.6667, got {rate}"

    rate_empty = evaluator.evaluate_keyword_hit_rate("", ["a", "b"])
    assert rate_empty == 0.0, "Empty answer should return 0.0"

    rate_no_kw = evaluator.evaluate_keyword_hit_rate("hello", [])
    assert rate_no_kw == 0.0, "Empty keywords should return 0.0"
    print("  OK")

    # ------------------------------------------------------------------
    # 2. detect_language_issue テスト
    # ------------------------------------------------------------------
    print("\n[2] detect_language_issue テスト")
    assert evaluator.detect_language_issue("这是一个很好的地方") is True, \
        "Chinese text should be detected"
    assert evaluator.detect_language_issue("渋谷駅の近くにカフェがあります") is False, \
        "Japanese text should not be flagged"
    assert evaluator.detect_language_issue("我们可以去那个地方") is True, \
        "Chinese pronouns should be detected"
    assert evaluator.detect_language_issue("") is False, \
        "Empty string should return False"
    print("  OK")

    # ------------------------------------------------------------------
    # 3. generate_summary テスト (ダミーデータ)
    # ------------------------------------------------------------------
    print("\n[3] generate_summary テスト")

    dummy_results = [
        MultiAreaEvalResult(
            test_id="MA-SBY-L1-01",
            system_name="system_a",
            target_area="shibuya",
            query_type="single_area",
            subcategory="basic_location",
            level=1,
            answer="渋谷駅周辺にスターバックスがあります",
            time_sec=1.5,
            keyword_hit_rate=0.8,
            success=True,
            area_detected="shibuya",
            area_detection_correct=True,
        ),
        MultiAreaEvalResult(
            test_id="MA-SBY-L2-01",
            system_name="system_a",
            target_area="shibuya",
            query_type="single_area",
            subcategory="proximity",
            level=2,
            answer="最も近いコンビニはローソンです",
            time_sec=2.1,
            keyword_hit_rate=0.6,
            success=True,
            area_detected="shibuya",
            area_detection_correct=True,
        ),
        MultiAreaEvalResult(
            test_id="MA-SJK-L1-01",
            system_name="system_a",
            target_area="shinjuku",
            query_type="single_area",
            subcategory="basic_location",
            level=1,
            answer="新宿駅の東口にカフェがあります",
            time_sec=1.8,
            keyword_hit_rate=0.5,
            success=True,
            area_detected="shinjuku",
            area_detection_correct=True,
        ),
        MultiAreaEvalResult(
            test_id="MA-SJK-L3-01",
            system_name="system_a",
            target_area="shinjuku",
            query_type="single_area",
            subcategory="aggregation",
            level=3,
            answer="カフェは全部で5件です",
            time_sec=3.0,
            keyword_hit_rate=0.3,
            success=False,
            area_detected="ikebukuro",
            area_detection_correct=False,
        ),
        MultiAreaEvalResult(
            test_id="MA-CROSS-L2-01",
            system_name="system_a",
            target_area=None,
            query_type="cross_area",
            subcategory="comparison",
            level=2,
            answer="渋谷と新宿を比較すると新宿の方がカフェが多い",
            time_sec=4.5,
            keyword_hit_rate=0.7,
            success=True,
            area_detected=None,
            area_detection_correct=True,
        ),
        MultiAreaEvalResult(
            test_id="MA-IKB-L1-01",
            system_name="system_a",
            target_area="ikebukuro",
            query_type="single_area",
            subcategory="basic_location",
            level=1,
            answer="池袋駅周辺にレストランがあります",
            time_sec=1.2,
            keyword_hit_rate=1.0,
            success=True,
            area_detected="ikebukuro",
            area_detection_correct=True,
        ),
        MultiAreaEvalResult(
            test_id="MA-IKB-L2-01",
            system_name="system_a",
            target_area="ikebukuro",
            query_type="single_area",
            subcategory="proximity",
            level=2,
            answer="",
            time_sec=0.5,
            keyword_hit_rate=0.0,
            success=False,
            area_detected=None,
            area_detection_correct=False,
            error="Timeout",
        ),
        MultiAreaEvalResult(
            test_id="MA-CROSS-L3-01",
            system_name="system_a",
            target_area=None,
            query_type="cross_area",
            subcategory="comparison",
            level=3,
            answer="这是渋谷和新宿的比较结果",
            time_sec=5.0,
            keyword_hit_rate=0.4,
            success=False,
            area_detected=None,
            area_detection_correct=True,
            language_issue=True,
        ),
    ]

    summary = evaluator.generate_summary(dummy_results)

    print(f"  overall: {summary['overall']}")
    print(f"  by_area keys: {list(summary['by_area'].keys())}")
    print(f"  by_level keys: {list(summary['by_level'].keys())}")
    print(f"  by_query_type keys: {list(summary['by_query_type'].keys())}")
    print(f"  by_subcategory keys: {list(summary['by_subcategory'].keys())}")
    print(f"  area_detection: {summary['area_detection']}")
    print(f"  cross_area: {summary['cross_area']}")

    assert summary["overall"]["total"] == 8
    assert summary["overall"]["error_count"] == 1
    assert summary["overall"]["language_issue_count"] == 1
    assert summary["overall"]["success_rate"] == round(5 / 8, 4)
    assert "shibuya" in summary["by_area"]
    assert "shinjuku" in summary["by_area"]
    assert "ikebukuro" in summary["by_area"]
    assert "cross_area" in summary["by_area"]
    assert 1 in summary["by_level"]
    assert 2 in summary["by_level"]
    assert 3 in summary["by_level"]
    assert summary["area_detection"]["total"] == 6  # results with target_area != None
    assert summary["cross_area"]["count"] == 2
    print("  OK")

    # ------------------------------------------------------------------
    # 4. compare_systems テスト (ダミーデータ)
    # ------------------------------------------------------------------
    print("\n[4] compare_systems テスト")

    # system_b: slightly different results
    dummy_results_b = [
        MultiAreaEvalResult(
            test_id="MA-SBY-L1-01",
            system_name="system_b",
            target_area="shibuya",
            query_type="single_area",
            subcategory="basic_location",
            level=1,
            answer="渋谷エリアのスポット情報",
            time_sec=2.0,
            keyword_hit_rate=0.4,
            success=False,
            area_detected="shibuya",
            area_detection_correct=True,
        ),
        MultiAreaEvalResult(
            test_id="MA-SJK-L1-01",
            system_name="system_b",
            target_area="shinjuku",
            query_type="single_area",
            subcategory="basic_location",
            level=1,
            answer="新宿駅周辺の店舗情報",
            time_sec=1.5,
            keyword_hit_rate=0.9,
            success=True,
            area_detected="shinjuku",
            area_detection_correct=True,
        ),
        MultiAreaEvalResult(
            test_id="MA-CROSS-L2-01",
            system_name="system_b",
            target_area=None,
            query_type="cross_area",
            subcategory="comparison",
            level=2,
            answer="比較結果です",
            time_sec=3.0,
            keyword_hit_rate=0.5,
            success=True,
            area_detected=None,
            area_detection_correct=True,
        ),
    ]

    comparison = evaluator.compare_systems({
        "system_a": dummy_results,
        "system_b": dummy_results_b,
    })

    print(f"  rankings: {comparison['rankings']}")
    print(f"  area_consistency: {comparison['area_consistency']}")
    assert len(comparison["summaries"]) == 2
    assert len(comparison["rankings"]) == 2
    # system_a has 5/8 = 0.625, system_b has 2/3 ~ 0.6667
    assert comparison["rankings"][0][0] == "system_b", \
        f"Expected system_b first, got {comparison['rankings'][0][0]}"
    assert "system_a" in comparison["area_consistency"]
    assert "system_b" in comparison["area_consistency"]
    print("  OK")

    # ------------------------------------------------------------------
    # 5. evaluate_single_case テスト (ダミー system_fn)
    # ------------------------------------------------------------------
    print("\n[5] evaluate_single_case テスト")

    @dataclass
    class _DummyTestCase:
        id: str = "MA-TEST-01"
        prompt: str = "渋谷駅周辺のカフェを教えてください"
        expected_keywords: List[str] = None
        target_area: Optional[str] = "shibuya"
        query_type: str = "single_area"
        subcategory: str = "basic_location"
        level: int = 1

        def __post_init__(self):
            if self.expected_keywords is None:
                self.expected_keywords = ["カフェ", "渋谷"]

    def dummy_system_fn(question: str) -> Dict:
        return {
            "answer": "渋谷駅近くにはスターバックスなどのカフェがあります。",
            "detected_area": "shibuya",
        }

    tc = _DummyTestCase()
    eval_result = evaluator.evaluate_single_case("dummy_system", dummy_system_fn, tc)
    print(f"  test_id: {eval_result.test_id}")
    print(f"  keyword_hit_rate: {eval_result.keyword_hit_rate}")
    print(f"  success: {eval_result.success}")
    print(f"  area_detection_correct: {eval_result.area_detection_correct}")
    print(f"  language_issue: {eval_result.language_issue}")
    assert eval_result.success is True
    assert eval_result.area_detection_correct is True
    assert eval_result.language_issue is False
    print("  OK")

    # ------------------------------------------------------------------
    # 6. to_dict テスト
    # ------------------------------------------------------------------
    print("\n[6] to_dict テスト")
    d = eval_result.to_dict()
    assert isinstance(d, dict)
    assert d["test_id"] == "MA-TEST-01"
    assert d["system_name"] == "dummy_system"
    print(f"  keys: {list(d.keys())}")
    print("  OK")

    # ------------------------------------------------------------------
    # 7. エラーハンドリング テスト
    # ------------------------------------------------------------------
    print("\n[7] エラーハンドリング テスト")

    def error_system_fn(question: str) -> Dict:
        raise ValueError("テスト用エラー")

    tc_err = _DummyTestCase(id="MA-ERR-01")
    err_result = evaluator.evaluate_single_case("error_system", error_system_fn, tc_err)
    assert err_result.success is False
    assert err_result.error == "テスト用エラー"
    assert err_result.keyword_hit_rate == 0.0
    print(f"  error captured: {err_result.error}")
    print("  OK")

    print("\n" + "=" * 60)
    print("全セルフテスト完了")
