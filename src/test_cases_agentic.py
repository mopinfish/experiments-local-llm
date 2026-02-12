#!/usr/bin/env python3
"""
test_cases_agentic.py - Phase 9: Agentic RAG向けテストケース

Agentic RAGの特徴を活かす複雑な質問パターン:
- multi_step_spatial: 複数ステップの空間推論（5件）
- conditional_reasoning: 条件付き推論（5件）
- iterative_refinement: 反復的な絞り込み（5件）

作成日: 2026-02-03
プロジェクト: experiments-local-llm
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict


@dataclass
class AgenticTestCase:
    """Agentic RAG向けテストケース"""
    id: str  # "A-01" 形式
    category: str  # テストカテゴリ
    subcategory: str  # サブカテゴリ
    prompt: str  # 質問文
    expected_keywords: List[str]  # 期待されるキーワード
    expected_tools: List[str]  # 使用が期待されるツール
    difficulty: str  # medium, hard, expert
    description: str  # テストの説明
    evaluation_points: List[str]  # 評価ポイント
    requires_multi_step: bool = False  # 複数ステップ推論が必要か
    requires_tool_chaining: bool = False  # ツール連鎖が必要か
    requires_self_correction: bool = False  # 自己修正が必要か
    min_tool_calls: int = 1  # 最小ツール呼び出し数
    max_tool_calls: int = 5  # 最大ツール呼び出し数


# =============================================================================
# Category 1: multi_step_spatial（複数ステップの空間推論）- 5件
# =============================================================================

MULTI_STEP_SPATIAL_CASES = [
    AgenticTestCase(
        id="A-01",
        category="agentic",
        subcategory="multi_step_spatial",
        prompt="渋谷駅から最も近いカフェを見つけて、そのカフェから300m以内にある他のカフェは何件ありますか？",
        expected_keywords=["カフェ", "最も近い", "300m", "件"],
        expected_tools=["tool_get_nearest_pois", "tool_count_pois_in_radius"],
        difficulty="hard",
        description="最寄りPOIを見つけた後、そこを起点にした別の検索を実行",
        evaluation_points=[
            "2段階の空間検索を実行できるか",
            "第1段階の結果を第2段階で使用できるか",
            "最終的な件数を正確に回答できるか"
        ],
        requires_multi_step=True,
        requires_tool_chaining=True,
        min_tool_calls=2,
        max_tool_calls=4
    ),
    AgenticTestCase(
        id="A-02",
        category="agentic",
        subcategory="multi_step_spatial",
        prompt="渋谷駅の東側で最も近いコンビニと西側で最も近いコンビニ、どちらが駅に近いですか？",
        expected_keywords=["東側", "西側", "コンビニ", "近い", "距離"],
        expected_tools=["tool_get_nearest_pois", "tool_compare_east_west"],
        difficulty="hard",
        description="方向別の最寄り検索と比較",
        evaluation_points=[
            "東西それぞれで最寄りPOIを検索できるか",
            "2つの距離を比較できるか",
            "どちらが近いか明確に回答できるか"
        ],
        requires_multi_step=True,
        requires_tool_chaining=True,
        min_tool_calls=2,
        max_tool_calls=4
    ),
    AgenticTestCase(
        id="A-03",
        category="agentic",
        subcategory="multi_step_spatial",
        prompt="500m以内にカフェが何件あるか調べてから、その半径を700mに広げると何件増えますか？",
        expected_keywords=["500m", "700m", "カフェ", "増える", "件"],
        expected_tools=["tool_count_pois_in_radius", "tool_compare_radius"],
        difficulty="medium",
        description="段階的な半径拡大と件数差分計算",
        evaluation_points=[
            "2つの半径で件数を取得できるか",
            "増加件数を正確に計算できるか",
            "比較ツールを適切に使用できるか"
        ],
        requires_multi_step=True,
        requires_tool_chaining=True,
        min_tool_calls=1,
        max_tool_calls=3
    ),
    AgenticTestCase(
        id="A-04",
        category="agentic",
        subcategory="multi_step_spatial",
        prompt="カフェの件数上位3カテゴリを調べて、そのうち東側に多いカテゴリはどれですか？",
        expected_keywords=["上位", "カテゴリ", "東側", "多い"],
        expected_tools=["tool_get_top_categories", "tool_compare_east_west"],
        difficulty="expert",
        description="ランキング取得後の方向別分析",
        evaluation_points=[
            "カテゴリランキングを取得できるか",
            "各カテゴリの東西分布を分析できるか",
            "東側優位のカテゴリを特定できるか"
        ],
        requires_multi_step=True,
        requires_tool_chaining=True,
        min_tool_calls=2,
        max_tool_calls=5
    ),
    AgenticTestCase(
        id="A-05",
        category="agentic",
        subcategory="multi_step_spatial",
        prompt="渋谷駅から300m以内のレストランの中で、最も東にあるのはどこですか？",
        expected_keywords=["300m", "レストラン", "東", "最も"],
        expected_tools=["tool_filter_by_area", "tool_get_nearest_pois"],
        difficulty="hard",
        description="エリアフィルタ後の方向別ソート",
        evaluation_points=[
            "半径フィルタを適用できるか",
            "結果から最東のPOIを特定できるか",
            "POI名と位置情報を提示できるか"
        ],
        requires_multi_step=True,
        requires_tool_chaining=True,
        min_tool_calls=2,
        max_tool_calls=4
    ),
]


# =============================================================================
# Category 2: conditional_reasoning（条件付き推論）- 5件
# =============================================================================

CONDITIONAL_REASONING_CASES = [
    AgenticTestCase(
        id="A-06",
        category="agentic",
        subcategory="conditional_reasoning",
        prompt="もし500m以内にカフェが50件未満なら「少ない」、50件以上なら「多い」と答えてください。実際はどちらですか？",
        expected_keywords=["500m", "カフェ", "件", "多い", "少ない"],
        expected_tools=["tool_count_pois_in_radius"],
        difficulty="medium",
        description="条件分岐による判定",
        evaluation_points=[
            "件数を取得できるか",
            "条件に基づいて判定できるか",
            "根拠となる実数値を提示できるか"
        ],
        requires_multi_step=False,
        requires_tool_chaining=False,
        min_tool_calls=1,
        max_tool_calls=2
    ),
    AgenticTestCase(
        id="A-07",
        category="agentic",
        subcategory="conditional_reasoning",
        prompt="カフェは東側と西側でどちらが多いですか？多い方が50件以上なら「人気エリア」と判定してください。",
        expected_keywords=["東側", "西側", "多い", "人気エリア"],
        expected_tools=["tool_compare_east_west"],
        difficulty="medium",
        description="比較と条件判定の組み合わせ",
        evaluation_points=[
            "東西比較を実行できるか",
            "多い方を特定できるか",
            "50件以上の条件判定ができるか"
        ],
        requires_multi_step=True,
        requires_tool_chaining=False,
        min_tool_calls=1,
        max_tool_calls=3
    ),
    AgenticTestCase(
        id="A-08",
        category="agentic",
        subcategory="conditional_reasoning",
        prompt="300mと500mでカフェの件数を比較して、もし1.5倍以上増えるなら「集中度が低い」、それ以外なら「集中度が高い」と答えてください。",
        expected_keywords=["300m", "500m", "カフェ", "倍", "集中度"],
        expected_tools=["tool_compare_radius"],
        difficulty="hard",
        description="増加率に基づく条件判定",
        evaluation_points=[
            "半径比較を実行できるか",
            "増加率を計算できるか",
            "条件に基づいて適切に判定できるか"
        ],
        requires_multi_step=True,
        requires_tool_chaining=False,
        min_tool_calls=1,
        max_tool_calls=2
    ),
    AgenticTestCase(
        id="A-09",
        category="agentic",
        subcategory="conditional_reasoning",
        prompt="渋谷駅から最も近いカフェが100m以内にあれば「駅直結」、100-300mなら「駅近」、それ以上なら「駅から離れている」と分類してください。",
        expected_keywords=["最も近い", "カフェ", "距離", "分類"],
        expected_tools=["tool_get_nearest_pois"],
        difficulty="medium",
        description="距離に基づく多段階分類",
        evaluation_points=[
            "最寄りPOIを検索できるか",
            "距離を取得できるか",
            "3段階の条件分岐で正しく分類できるか"
        ],
        requires_multi_step=False,
        requires_tool_chaining=False,
        min_tool_calls=1,
        max_tool_calls=2
    ),
    AgenticTestCase(
        id="A-10",
        category="agentic",
        subcategory="conditional_reasoning",
        prompt="カフェとレストランの件数を比較して、カフェの方が多ければカフェの東西分布を、レストランの方が多ければレストランの東西分布を教えてください。",
        expected_keywords=["カフェ", "レストラン", "件数", "東西"],
        expected_tools=["tool_count_by_category", "tool_compare_east_west"],
        difficulty="expert",
        description="比較結果に基づく動的な分析対象選択",
        evaluation_points=[
            "2つのカテゴリの件数を比較できるか",
            "多い方のカテゴリを特定できるか",
            "選択したカテゴリの東西分布を分析できるか"
        ],
        requires_multi_step=True,
        requires_tool_chaining=True,
        min_tool_calls=2,
        max_tool_calls=4
    ),
]


# =============================================================================
# Category 3: iterative_refinement（反復的な絞り込み）- 5件
# =============================================================================

ITERATIVE_REFINEMENT_CASES = [
    AgenticTestCase(
        id="A-11",
        category="agentic",
        subcategory="iterative_refinement",
        prompt="カフェが最も多い半径を見つけてください。100m、200m、300m、500mで比較して最適な半径を教えてください。",
        expected_keywords=["カフェ", "多い", "半径", "最適"],
        expected_tools=["tool_analyze_sensitivity", "tool_count_pois_in_radius"],
        difficulty="hard",
        description="複数の半径を試行して最適解を発見",
        evaluation_points=[
            "複数の半径で件数を取得できるか",
            "最も多い半径を特定できるか",
            "根拠となる数値を全て提示できるか"
        ],
        requires_multi_step=True,
        requires_tool_chaining=False,
        min_tool_calls=1,
        max_tool_calls=3
    ),
    AgenticTestCase(
        id="A-12",
        category="agentic",
        subcategory="iterative_refinement",
        prompt="東側のカフェが西側より多いか確認してください。もし西側が多ければ、西側で最も多いサブカテゴリを教えてください。",
        expected_keywords=["東側", "西側", "カフェ", "多い", "サブカテゴリ"],
        expected_tools=["tool_compare_east_west", "tool_analyze_category_by_direction"],
        difficulty="hard",
        description="比較結果に基づく追加分析",
        evaluation_points=[
            "東西比較を実行できるか",
            "結果に基づいて追加分析を判断できるか",
            "動的に分析を深掘りできるか"
        ],
        requires_multi_step=True,
        requires_tool_chaining=True,
        requires_self_correction=True,
        min_tool_calls=2,
        max_tool_calls=4
    ),
    AgenticTestCase(
        id="A-13",
        category="agentic",
        subcategory="iterative_refinement",
        prompt="300m以内のカフェ件数を確認し、もし20件未満なら500mで再検索、20件以上ならそのまま最寄りの3件を教えてください。",
        expected_keywords=["300m", "500m", "カフェ", "件", "最寄り"],
        expected_tools=["tool_count_pois_in_radius", "tool_get_nearest_pois"],
        difficulty="expert",
        description="条件に基づく動的な半径調整",
        evaluation_points=[
            "最初の件数チェックができるか",
            "条件に基づいて半径を調整できるか",
            "最終的に適切な結果を提示できるか"
        ],
        requires_multi_step=True,
        requires_tool_chaining=True,
        requires_self_correction=True,
        min_tool_calls=2,
        max_tool_calls=4
    ),
    AgenticTestCase(
        id="A-14",
        category="agentic",
        subcategory="iterative_refinement",
        prompt="レストランの件数が多い方向（東西南北）を特定し、その方向で最も近いレストランを3件教えてください。",
        expected_keywords=["レストラン", "方向", "多い", "最も近い"],
        expected_tools=["tool_analyze_category_by_direction", "tool_get_nearest_pois"],
        difficulty="expert",
        description="方向別分析後の最寄り検索",
        evaluation_points=[
            "方向別の件数分析ができるか",
            "最多方向を特定できるか",
            "その方向での最寄り検索ができるか"
        ],
        requires_multi_step=True,
        requires_tool_chaining=True,
        min_tool_calls=2,
        max_tool_calls=4
    ),
    AgenticTestCase(
        id="A-15",
        category="agentic",
        subcategory="iterative_refinement",
        prompt="カフェの件数が東西で同程度（差が10件以内）か確認し、同程度なら南北の比較も行って分布の傾向を教えてください。",
        expected_keywords=["カフェ", "東西", "南北", "同程度", "分布"],
        expected_tools=["tool_compare_east_west", "tool_compare_north_south"],
        difficulty="hard",
        description="複数軸での段階的な比較分析",
        evaluation_points=[
            "東西比較を実行できるか",
            "差が10件以内の条件判定ができるか",
            "条件に基づいて南北比較も実行できるか"
        ],
        requires_multi_step=True,
        requires_tool_chaining=True,
        requires_self_correction=True,
        min_tool_calls=2,
        max_tool_calls=4
    ),
]


# =============================================================================
# 全テストケースの統合
# =============================================================================

ALL_AGENTIC_TEST_CASES = (
    MULTI_STEP_SPATIAL_CASES +
    CONDITIONAL_REASONING_CASES +
    ITERATIVE_REFINEMENT_CASES
)


def get_test_cases_by_category(category: str) -> List[AgenticTestCase]:
    """カテゴリ別にテストケースを取得"""
    return [tc for tc in ALL_AGENTIC_TEST_CASES if tc.subcategory == category]


def get_test_cases_by_difficulty(difficulty: str) -> List[AgenticTestCase]:
    """難易度別にテストケースを取得"""
    return [tc for tc in ALL_AGENTIC_TEST_CASES if tc.difficulty == difficulty]


def get_test_case_by_id(test_id: str) -> Optional[AgenticTestCase]:
    """IDでテストケースを取得"""
    for tc in ALL_AGENTIC_TEST_CASES:
        if tc.id == test_id:
            return tc
    return None


def print_test_case_summary():
    """テストケースのサマリーを出力"""
    print("=" * 60)
    print("Agentic RAG Test Cases Summary")
    print("=" * 60)

    print(f"\n総テストケース数: {len(ALL_AGENTIC_TEST_CASES)}")

    # カテゴリ別
    categories = {}
    for tc in ALL_AGENTIC_TEST_CASES:
        categories[tc.subcategory] = categories.get(tc.subcategory, 0) + 1

    print("\nカテゴリ別:")
    for cat, count in categories.items():
        print(f"  - {cat}: {count}件")

    # 難易度別
    difficulties = {}
    for tc in ALL_AGENTIC_TEST_CASES:
        difficulties[tc.difficulty] = difficulties.get(tc.difficulty, 0) + 1

    print("\n難易度別:")
    for diff, count in difficulties.items():
        print(f"  - {diff}: {count}件")

    # ツール使用統計
    all_tools = set()
    for tc in ALL_AGENTIC_TEST_CASES:
        all_tools.update(tc.expected_tools)

    print(f"\n期待されるツール数: {len(all_tools)}")
    print("使用ツール:")
    for tool in sorted(all_tools):
        count = sum(1 for tc in ALL_AGENTIC_TEST_CASES if tool in tc.expected_tools)
        print(f"  - {tool}: {count}ケース")


if __name__ == "__main__":
    print_test_case_summary()

    print("\n" + "=" * 60)
    print("サンプルテストケース")
    print("=" * 60)

    sample = ALL_AGENTIC_TEST_CASES[0]
    print(f"\nID: {sample.id}")
    print(f"カテゴリ: {sample.subcategory}")
    print(f"難易度: {sample.difficulty}")
    print(f"質問: {sample.prompt}")
    print(f"期待ツール: {', '.join(sample.expected_tools)}")
    print(f"評価ポイント:")
    for point in sample.evaluation_points:
        print(f"  - {point}")
