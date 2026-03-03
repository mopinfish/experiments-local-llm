"""
test_cases_multi_area_v2.py - Variant B: データソース非依存テストケース

既存 test_cases_multi_area.py の 130 テストケースに対し、
OSM 固有の POI 名（特定店舗名等）を汎用キーワードに置換した Variant B を生成。
MapFan API など異なるデータソースでも公平に評価できるようにする。

Phase 10-A: MCP サーバー構造化ツール拡張実験
作成日: 2026-03-03
"""

import copy
from typing import Dict, List, Optional

try:
    from .test_cases_multi_area import (
        MultiAreaTestCase,
        get_all_test_cases,
        get_quick_test_cases,
        get_area_tests,
        get_cross_area_tests,
        get_landmark_tests,
        get_detection_tests,
        get_tests_by_level,
    )
except ImportError:
    from test_cases_multi_area import (
        MultiAreaTestCase,
        get_all_test_cases,
        get_quick_test_cases,
        get_area_tests,
        get_cross_area_tests,
        get_landmark_tests,
        get_detection_tests,
        get_tests_by_level,
    )


# =============================================================================
# キーワード置換ルール
# =============================================================================

# OSM 固有の店舗名 → 汎用キーワードへの置換マッピング
# 形式: { "元のキーワード": "置換後のキーワード" }
# None の場合はキーワードリストから削除
KEYWORD_REPLACEMENTS: Dict[str, Optional[str]] = {
    # コンビニ固有名 → 汎用
    "ローソン": "コンビニ",
    "ファミリーマート": "コンビニ",
    "セブンイレブン": "コンビニ",
    "セブン-イレブン": "コンビニ",
    "セブン": None,  # 「コンビニ」で十分な場合は削除

    # カフェ固有名
    "スターバックス": "カフェ",
    "ドトール": "カフェ",
    "タリーズ": "カフェ",

    # ファストフード固有名
    "マクドナルド": "ファストフード",
    "マック": None,
    "モスバーガー": "ファストフード",
    "吉野家": "ファストフード",
    "松屋": "ファストフード",

    # ラーメン店固有名
    "一風堂": "ラーメン",
    "天下一品": "ラーメン",

    # スーパー固有名
    "まいばすけっと": "スーパー",
    "成城石井": "スーパー",
    "オオゼキ": "スーパー",
    "ライフ": "スーパー",
    "東急ストア": "スーパー",

    # ドラッグストア固有名
    "マツモトキヨシ": "薬局",
    "ツルハ": "薬局",
    "サンドラッグ": "薬局",
    "ウエルシア": "薬局",
    "トモズ": "薬局",

    # 銀行固有名
    "三菱UFJ": "銀行",
    "みずほ": "銀行",
    "三井住友": "銀行",
    "りそな": "銀行",

    # 商業施設固有名
    "東急ハンズ": "雑貨店",
    "ドン・キホーテ": "ディスカウントストア",
    "ヨドバシカメラ": "家電店",
    "ビックカメラ": "家電店",
    "紀伊國屋書店": "書店",
    "ジュンク堂": "書店",
    "TSUTAYA": "書店",
    "蔦屋書店": "書店",

    # 渋谷固有ランドマーク（残す: 質問文で使われるため）
    # "渋谷109", "ハチ公像" 等は置換しない（質問文中でも使用）
}


def _replace_keywords(keywords: List[str]) -> List[str]:
    """
    expected_keywords のリストを Variant B 用に変換。

    - KEYWORD_REPLACEMENTS に該当するものを置換
    - 置換後の重複を除去
    - None 指定のキーワードは削除
    """
    result = []
    seen = set()

    for kw in keywords:
        replaced = False
        for original, replacement in KEYWORD_REPLACEMENTS.items():
            if original in kw:
                if replacement is not None and replacement not in seen:
                    result.append(replacement)
                    seen.add(replacement)
                replaced = True
                break

        if not replaced:
            if kw not in seen:
                result.append(kw)
                seen.add(kw)

    # キーワードが空になった場合は元のリストの一部を残す
    if not result and keywords:
        result = keywords[:2]

    return result


def convert_to_variant_b(test_case: MultiAreaTestCase) -> MultiAreaTestCase:
    """
    単一テストケースを Variant B に変換。

    - expected_keywords を汎用化
    - id に "-VB" サフィックスを追加
    """
    new_case = copy.deepcopy(test_case)
    new_case.id = test_case.id + "-VB"
    new_case.expected_keywords = _replace_keywords(test_case.expected_keywords)
    return new_case


# =============================================================================
# Variant B テストケース取得関数
# =============================================================================

def get_all_test_cases_v2() -> List[MultiAreaTestCase]:
    """全 130 テストケースの Variant B 版を返す。"""
    return [convert_to_variant_b(tc) for tc in get_all_test_cases()]


def get_quick_test_cases_v2() -> List[MultiAreaTestCase]:
    """クイックテスト（13 件）の Variant B 版を返す。"""
    return [convert_to_variant_b(tc) for tc in get_quick_test_cases()]


def get_area_tests_v2(area: str) -> List[MultiAreaTestCase]:
    """エリア別テストの Variant B 版。"""
    return [convert_to_variant_b(tc) for tc in get_area_tests(area)]


def get_cross_area_tests_v2() -> List[MultiAreaTestCase]:
    """クロスエリアテストの Variant B 版。"""
    return [convert_to_variant_b(tc) for tc in get_cross_area_tests()]


def get_tests_by_level_v2(level: int) -> List[MultiAreaTestCase]:
    """レベル別テストの Variant B 版。"""
    return [convert_to_variant_b(tc) for tc in get_tests_by_level(level)]


# =============================================================================
# 変換統計
# =============================================================================

def get_variant_b_stats() -> Dict[str, int]:
    """
    Variant B への変換で何件のキーワードが変更されたかの統計。
    """
    original = get_all_test_cases()
    v2 = get_all_test_cases_v2()

    changed = 0
    total_kw_original = 0
    total_kw_v2 = 0

    for orig, converted in zip(original, v2):
        total_kw_original += len(orig.expected_keywords)
        total_kw_v2 += len(converted.expected_keywords)
        if orig.expected_keywords != converted.expected_keywords:
            changed += 1

    return {
        "total_cases": len(original),
        "cases_with_keyword_changes": changed,
        "cases_unchanged": len(original) - changed,
        "total_keywords_original": total_kw_original,
        "total_keywords_v2": total_kw_v2,
    }
