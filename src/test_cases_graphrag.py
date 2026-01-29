#!/usr/bin/env python3
"""
test_cases_graphrag.py - GraphRAG評価用テストケース（15件）

グラフRAGの強みを測定するために設計されたテストケース。
構造化RAGとの比較評価で使用。

カテゴリ:
- relation: 関係性クエリ（5件） - SAME_AREA, NEAR_TOエッジを活用
- multi_hop: マルチホップクエリ（3件） - グラフトラバーサル
- aggregation: 集計クエリ（3件） - ノード集計
- comparison: 比較クエリ（2件） - 方向別比較
- proximity: 近接性クエリ（2件） - 距離ソート
"""
from dataclasses import dataclass
from typing import List


@dataclass
class GraphRAGTestCase:
    """グラフRAG評価用テストケース"""
    id: str
    category: str  # relation, multi_hop, aggregation, comparison, proximity
    question: str
    expected_keywords: List[str]
    description: str
    graph_advantage: str  # グラフRAGの想定優位性


GRAPHRAG_TEST_CASES: List[GraphRAGTestCase] = [
    # ==========================================================================
    # 関係性クエリ（5件）- SAME_AREA, NEAR_TOエッジを活用
    # ==========================================================================
    GraphRAGTestCase(
        id="GR-01",
        category="relation",
        question="渋谷駅の東側にあるカフェで、同じエリアにコンビニもある場所はどこですか？",
        expected_keywords=["カフェ", "コンビニ", "東"],
        description="2カテゴリの空間的共起を問う",
        graph_advantage="SAME_AREAエッジで効率的に共起を検出"
    ),
    GraphRAGTestCase(
        id="GR-02",
        category="relation",
        question="銀行とカフェが両方あるエリアを教えてください",
        expected_keywords=["銀行", "カフェ", "エリア"],
        description="2カテゴリのエリア共起を問う",
        graph_advantage="エリアノードを介した効率的な検索"
    ),
    GraphRAGTestCase(
        id="GR-03",
        category="relation",
        question="ホテルの近くにあるレストランを教えてください",
        expected_keywords=["ホテル", "レストラン", "近く"],
        description="POI間の近接関係を問う",
        graph_advantage="NEAR_TOエッジで直接検索"
    ),
    GraphRAGTestCase(
        id="GR-04",
        category="relation",
        question="駅周辺で、薬局とコンビニが同じ場所にあるところはありますか？",
        expected_keywords=["薬局", "コンビニ", "駅"],
        description="駅周辺エリアでの共起を問う",
        graph_advantage="エリアフィルタ + 共起検索"
    ),
    GraphRAGTestCase(
        id="GR-05",
        category="relation",
        question="映画館の近くにあるカフェはどこですか？",
        expected_keywords=["映画館", "カフェ"],
        description="娯楽施設と飲食店の関係を問う",
        graph_advantage="NEAR_TOエッジで直接検索"
    ),

    # ==========================================================================
    # マルチホップクエリ（3件）- グラフトラバーサル
    # ==========================================================================
    GraphRAGTestCase(
        id="GR-06",
        category="multi_hop",
        question="カフェを起点に、そこから50m以内にある書店を教えてください",
        expected_keywords=["カフェ", "書店", "50m"],
        description="2ホップの経路探索",
        graph_advantage="グラフトラバーサルによる経路探索"
    ),
    GraphRAGTestCase(
        id="GR-07",
        category="multi_hop",
        question="渋谷駅から100m以内のコンビニと、そこから近いカフェを教えてください",
        expected_keywords=["コンビニ", "カフェ", "100m"],
        description="距離制約付き2ホップ検索",
        graph_advantage="DISTANCE_FROM + NEAR_TOの連鎖"
    ),
    GraphRAGTestCase(
        id="GR-08",
        category="multi_hop",
        question="ホテルから徒歩で行ける範囲にあるレストランとカフェを教えてください",
        expected_keywords=["ホテル", "レストラン", "カフェ"],
        description="ホテルを起点とした飲食店探索",
        graph_advantage="複数カテゴリへの同時トラバーサル"
    ),

    # ==========================================================================
    # 集計クエリ（3件）- ノード集計
    # ==========================================================================
    GraphRAGTestCase(
        id="GR-09",
        category="aggregation",
        question="飲食店が最も多いエリアはどこですか？",
        expected_keywords=["飲食店", "エリア", "多い"],
        description="エリア別カテゴリ集計",
        graph_advantage="LOCATED_INエッジのカウント"
    ),
    GraphRAGTestCase(
        id="GR-10",
        category="aggregation",
        question="北側と南側でPOIの数が多いのはどちらですか？",
        expected_keywords=["北", "南", "数", "多い"],
        description="方向別POI集計",
        graph_advantage="方向属性によるフィルタ集計"
    ),
    GraphRAGTestCase(
        id="GR-11",
        category="aggregation",
        question="渋谷で最も多いカテゴリのPOIは何ですか？上位3つを教えてください",
        expected_keywords=["カテゴリ", "多い", "上位"],
        description="カテゴリ別POIランキング",
        graph_advantage="カテゴリノードへのエッジ集計"
    ),

    # ==========================================================================
    # 比較クエリ（2件）- 方向別比較
    # ==========================================================================
    GraphRAGTestCase(
        id="GR-12",
        category="comparison",
        question="東側と西側で、飲食店のカテゴリ多様性が高いのはどちらですか？",
        expected_keywords=["東", "西", "飲食店", "多様性"],
        description="方向別カテゴリ多様性比較",
        graph_advantage="サブカテゴリノードの数で多様性を計算"
    ),
    GraphRAGTestCase(
        id="GR-13",
        category="comparison",
        question="駅の近く（200m以内）と遠く（500m以上）で、どちらにホテルが多いですか？",
        expected_keywords=["駅", "近く", "遠く", "ホテル"],
        description="距離帯別POI比較",
        graph_advantage="distance_zoneによるフィルタ比較"
    ),

    # ==========================================================================
    # 近接性クエリ（2件）- 距離ソート
    # ==========================================================================
    GraphRAGTestCase(
        id="GR-14",
        category="proximity",
        question="渋谷駅に最も近いホテルはどこですか？",
        expected_keywords=["駅", "近い", "ホテル"],
        description="最寄りPOI検索",
        graph_advantage="DISTANCE_FROMエッジによる距離ソート"
    ),
    GraphRAGTestCase(
        id="GR-15",
        category="proximity",
        question="渋谷駅から300m以内にある映画館を距離順に教えてください",
        expected_keywords=["映画館", "300m", "距離"],
        description="距離制約付き近接検索",
        graph_advantage="距離フィルタ + ソート"
    ),
]


def get_graphrag_test_cases() -> List[GraphRAGTestCase]:
    """GraphRAGテストケースを取得"""
    return GRAPHRAG_TEST_CASES


def get_graphrag_test_cases_by_category(category: str) -> List[GraphRAGTestCase]:
    """カテゴリ別にテストケースを取得"""
    return [tc for tc in GRAPHRAG_TEST_CASES if tc.category == category]


def get_graphrag_test_case_stats() -> dict:
    """テストケースの統計情報を取得"""
    from collections import Counter
    categories = Counter(tc.category for tc in GRAPHRAG_TEST_CASES)
    return {
        "total": len(GRAPHRAG_TEST_CASES),
        "by_category": dict(categories)
    }


if __name__ == "__main__":
    stats = get_graphrag_test_case_stats()
    print(f"GraphRAGテストケース総数: {stats['total']}件")
    print(f"\nカテゴリ別:")
    for cat, count in stats["by_category"].items():
        print(f"  {cat}: {count}件")
