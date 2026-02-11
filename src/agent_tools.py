"""
agent_tools.py - Agentic RAG用ツール定義

Phase 9: Agentic RAGシステムで使用するツール群
LangGraphエージェントから呼び出される関数とツール定義

作成日: 2026-02-03
プロジェクト: experiments-local-llm
"""

import json
from typing import List, Dict, Any, Optional, Tuple
from langchain_core.tools import tool

# ローカルモジュール（パッケージ/スクリプト両対応）
try:
    from .geo_utils import (
        get_nearest_pois,
        filter_by_radius,
        count_by_radius,
        compare_by_radius,
        haversine_distance,
        SHIBUYA_STATION,
        filter_east_side,
        filter_west_side,
        analyze_radius_sensitivity
    )
    from .aggregator import (
        count_by_category,
        compare_east_west,
        compare_north_south,
        get_top_categories,
        filter_by_category,
        analyze_category_by_direction
    )
except ImportError:
    from geo_utils import (
        get_nearest_pois,
        filter_by_radius,
        count_by_radius,
        compare_by_radius,
        haversine_distance,
        SHIBUYA_STATION,
        filter_east_side,
        filter_west_side,
        analyze_radius_sensitivity
    )
    from aggregator import (
        count_by_category,
        compare_east_west,
        compare_north_south,
        get_top_categories,
        filter_by_category,
        analyze_category_by_direction
    )


# =============================================================================
# グローバルPOIデータ（初期化時にセット）
# =============================================================================

_GLOBAL_POIS: List[Dict[str, Any]] = []


def set_global_pois(pois: List[Dict[str, Any]]) -> None:
    """
    グローバルPOIデータを設定

    Args:
        pois: POIデータのリスト（空間情報付き）
    """
    global _GLOBAL_POIS
    _GLOBAL_POIS = pois


def get_global_pois() -> List[Dict[str, Any]]:
    """グローバルPOIデータを取得"""
    return _GLOBAL_POIS


# =============================================================================
# 空間計算ツール群
# =============================================================================

@tool
def tool_get_nearest_pois(
    category: Optional[str] = None,
    top_n: int = 3
) -> str:
    """
    渋谷駅から最も近いPOIを検索します。

    Args:
        category: 検索するカテゴリ（例: "カフェ", "飲食店", "コンビニ"）。Noneの場合は全カテゴリ対象。
        top_n: 取得する件数（デフォルト: 3）

    Returns:
        最寄りPOI情報のJSON文字列

    Examples:
        - 最寄りのカフェを探す: tool_get_nearest_pois(category="カフェ", top_n=3)
        - 最寄りのPOIを探す: tool_get_nearest_pois(top_n=5)
    """
    pois = get_global_pois()
    nearest = get_nearest_pois(pois, category, top_n)

    result = {
        "category": category or "全て",
        "count": len(nearest),
        "pois": [
            {
                "name": poi.get("name", "不明"),
                "category": poi.get("category", "不明"),
                "distance_m": poi.get("distance_from_station", "不明"),
                "direction": poi.get("direction_from_station_jp", "不明")
            }
            for poi in nearest
        ]
    }

    return json.dumps(result, ensure_ascii=False, indent=2)


@tool
def tool_count_pois_in_radius(
    radius_m: float,
    category: Optional[str] = None
) -> str:
    """
    指定した半径内のPOI件数を集計します。

    Args:
        radius_m: 半径（メートル）
        category: フィルタするカテゴリ（部分一致）。Noneの場合は全カテゴリ対象。

    Returns:
        集計結果のJSON文字列

    Examples:
        - 500m以内のカフェを数える: tool_count_pois_in_radius(radius_m=500, category="カフェ")
        - 300m以内の全POIを数える: tool_count_pois_in_radius(radius_m=300)
    """
    pois = get_global_pois()
    count = count_by_radius(pois, radius_m, category)

    result = {
        "radius_m": radius_m,
        "category": category or "全て",
        "count": count
    }

    return json.dumps(result, ensure_ascii=False, indent=2)


@tool
def tool_compare_radius(
    radius1_m: float,
    radius2_m: float,
    category: Optional[str] = None
) -> str:
    """
    異なる半径での件数を比較します（感度分析）。
    半径を変えた時にPOI件数がどう変化するかを分析できます。

    Args:
        radius1_m: 半径1（メートル）
        radius2_m: 半径2（メートル）
        category: フィルタするカテゴリ。Noneの場合は全カテゴリ対象。

    Returns:
        比較結果のJSON文字列

    Examples:
        - 300mと500mでカフェの件数を比較: tool_compare_radius(radius1_m=300, radius2_m=500, category="カフェ")
    """
    pois = get_global_pois()
    comparison = compare_by_radius(pois, radius1_m, radius2_m, category)

    result = {
        "radius1_m": comparison.radius1_m,
        "radius2_m": comparison.radius2_m,
        "count1": comparison.count1,
        "count2": comparison.count2,
        "difference": comparison.difference,
        "ratio": comparison.ratio,
        "category": comparison.category or "全て",
        "summary_jp": comparison.to_japanese()
    }

    return json.dumps(result, ensure_ascii=False, indent=2)


@tool
def tool_analyze_sensitivity(
    category: Optional[str] = None,
    radii: Optional[List[float]] = None
) -> str:
    """
    複数の半径でのPOI件数変化を分析します（詳細な感度分析）。

    Args:
        category: フィルタするカテゴリ
        radii: 分析する半径のリスト。Noneの場合は[100, 200, 300, 500, 800, 1000]

    Returns:
        感度分析結果のJSON文字列

    Examples:
        - カフェの半径別件数を分析: tool_analyze_sensitivity(category="カフェ")
    """
    pois = get_global_pois()
    result = analyze_radius_sensitivity(pois, category, radii)

    return json.dumps(result, ensure_ascii=False, indent=2)


@tool
def tool_filter_by_area(
    radius_m: float,
    category: Optional[str] = None
) -> str:
    """
    指定した半径内のPOIリストを取得します。

    Args:
        radius_m: 半径（メートル）
        category: フィルタするカテゴリ

    Returns:
        POIリストのJSON文字列

    Examples:
        - 500m以内のカフェをリスト化: tool_filter_by_area(radius_m=500, category="カフェ")
    """
    pois = get_global_pois()
    filtered = filter_by_radius(pois, radius_m, category)

    result = {
        "radius_m": radius_m,
        "category": category or "全て",
        "count": len(filtered),
        "pois": [
            {
                "name": poi.get("name", "不明"),
                "category": poi.get("category", "不明"),
                "distance_m": poi.get("distance_from_station", "不明"),
                "direction": poi.get("direction_from_station_jp", "不明")
            }
            for poi in filtered[:20]  # 最大20件に制限
        ]
    }

    return json.dumps(result, ensure_ascii=False, indent=2)


@tool
def tool_calculate_distance(
    poi_name1: str,
    poi_name2: str
) -> str:
    """
    2つのPOI間の距離を計算します。

    Args:
        poi_name1: POI名1（部分一致で検索）
        poi_name2: POI名2（部分一致で検索）

    Returns:
        距離計算結果のJSON文字列

    Examples:
        - 2つの店舗間の距離: tool_calculate_distance(poi_name1="スターバックス", poi_name2="タリーズ")
    """
    pois = get_global_pois()

    # POI検索
    poi1 = None
    poi2 = None
    for poi in pois:
        if poi_name1.lower() in poi.get("name", "").lower() and poi1 is None:
            poi1 = poi
        if poi_name2.lower() in poi.get("name", "").lower() and poi2 is None:
            poi2 = poi
        if poi1 and poi2:
            break

    if not poi1:
        return json.dumps({"error": f"POI '{poi_name1}' が見つかりません"}, ensure_ascii=False)
    if not poi2:
        return json.dumps({"error": f"POI '{poi_name2}' が見つかりません"}, ensure_ascii=False)

    distance = haversine_distance(
        poi1["lat"], poi1["lon"],
        poi2["lat"], poi2["lon"]
    )

    result = {
        "poi1": {"name": poi1["name"], "category": poi1.get("category", "不明")},
        "poi2": {"name": poi2["name"], "category": poi2.get("category", "不明")},
        "distance_m": round(distance, 2)
    }

    return json.dumps(result, ensure_ascii=False, indent=2)


# =============================================================================
# 比較・集計ツール群
# =============================================================================

@tool
def tool_compare_east_west(
    category: Optional[str] = None
) -> str:
    """
    渋谷駅の東側と西側でPOI件数を比較します。

    Args:
        category: 比較するカテゴリ。Noneの場合は全カテゴリ対象。

    Returns:
        東西比較結果のJSON文字列

    Examples:
        - カフェの東西比較: tool_compare_east_west(category="カフェ")
        - 全POIの東西比較: tool_compare_east_west()
    """
    pois = get_global_pois()
    comparison = compare_east_west(pois, category)

    result = {
        "category": category or "全て",
        "east_count": comparison.item1_count,
        "west_count": comparison.item2_count,
        "winner": comparison.winner,
        "difference": comparison.difference,
        "summary_jp": comparison.to_japanese()
    }

    return json.dumps(result, ensure_ascii=False, indent=2)


@tool
def tool_compare_north_south(
    category: Optional[str] = None
) -> str:
    """
    渋谷駅の北側と南側でPOI件数を比較します。

    Args:
        category: 比較するカテゴリ。Noneの場合は全カテゴリ対象。

    Returns:
        南北比較結果のJSON文字列
    """
    pois = get_global_pois()
    comparison = compare_north_south(pois, category)

    result = {
        "category": category or "全て",
        "north_count": comparison.item1_count,
        "south_count": comparison.item2_count,
        "winner": comparison.winner,
        "difference": comparison.difference,
        "summary_jp": comparison.to_japanese()
    }

    return json.dumps(result, ensure_ascii=False, indent=2)


@tool
def tool_count_by_category(
    categories: Optional[List[str]] = None
) -> str:
    """
    カテゴリ別にPOI件数を集計します。

    Args:
        categories: 集計するカテゴリのリスト。Noneの場合は全カテゴリを集計。

    Returns:
        カテゴリ別集計結果のJSON文字列

    Examples:
        - 全カテゴリを集計: tool_count_by_category()
        - 特定カテゴリのみ集計: tool_count_by_category(categories=["カフェ", "レストラン"])
    """
    pois = get_global_pois()

    if categories:
        # 特定カテゴリのみ
        result = {}
        for cat in categories:
            filtered = filter_by_category(pois, cat)
            result[cat] = len(filtered)
    else:
        # 全カテゴリ
        result = count_by_category(pois)

    return json.dumps(result, ensure_ascii=False, indent=2)


@tool
def tool_get_top_categories(
    top_n: int = 5
) -> str:
    """
    件数上位のカテゴリランキングを取得します。

    Args:
        top_n: 取得する件数（デフォルト: 5）

    Returns:
        ランキング結果のJSON文字列

    Examples:
        - トップ5カテゴリ: tool_get_top_categories(top_n=5)
        - トップ10カテゴリ: tool_get_top_categories(top_n=10)
    """
    pois = get_global_pois()
    top_cats = get_top_categories(pois, top_n, include_examples=True, example_count=3)

    result = {
        "top_n": top_n,
        "categories": [
            {
                "rank": i + 1,
                "category": cat.category,
                "count": cat.count,
                "examples": cat.examples
            }
            for i, cat in enumerate(top_cats)
        ]
    }

    return json.dumps(result, ensure_ascii=False, indent=2)


@tool
def tool_analyze_category_by_direction(
    category: str
) -> str:
    """
    特定カテゴリのPOIを方向別に詳細分析します。

    Args:
        category: 分析するカテゴリ

    Returns:
        方向別分析結果のJSON文字列

    Examples:
        - カフェの方向別分析: tool_analyze_category_by_direction(category="カフェ")
    """
    pois = get_global_pois()
    analysis = analyze_category_by_direction(pois, category)

    return json.dumps(analysis, ensure_ascii=False, indent=2)


# =============================================================================
# 検索ツール群
# =============================================================================

@tool
def tool_vector_search(
    query: str,
    k: int = 5
) -> str:
    """
    セマンティック検索でPOIを検索します。
    質問の意図に近いPOIを意味的に検索します。

    Args:
        query: 検索クエリ
        k: 取得件数

    Returns:
        検索結果のJSON文字列

    Examples:
        - "コーヒーが飲める場所": tool_vector_search(query="コーヒーが飲める場所", k=5)
    """
    # TODO: ベクトル検索の実装（chromadbとの連携）
    # 現在は簡易実装
    pois = get_global_pois()

    # キーワードベースの簡易検索
    query_lower = query.lower()
    results = []
    for poi in pois:
        name = poi.get("name", "").lower()
        category = poi.get("category", "").lower()
        if query_lower in name or query_lower in category:
            results.append(poi)

    result = {
        "query": query,
        "count": len(results[:k]),
        "pois": [
            {
                "name": poi.get("name", "不明"),
                "category": poi.get("category", "不明"),
                "distance_m": poi.get("distance_from_station", "不明"),
                "direction": poi.get("direction_from_station_jp", "不明")
            }
            for poi in results[:k]
        ]
    }

    return json.dumps(result, ensure_ascii=False, indent=2)


@tool
def tool_find_pois_by_keyword(
    keyword: str
) -> str:
    """
    キーワードでPOIを検索します（名前またはカテゴリに含まれるもの）。

    Args:
        keyword: 検索キーワード

    Returns:
        検索結果のJSON文字列

    Examples:
        - "スターバックス"を検索: tool_find_pois_by_keyword(keyword="スターバックス")
        - "カフェ"を検索: tool_find_pois_by_keyword(keyword="カフェ")
    """
    pois = get_global_pois()
    keyword_lower = keyword.lower()

    results = [
        poi for poi in pois
        if keyword_lower in poi.get("name", "").lower() or
           keyword_lower in poi.get("category", "").lower()
    ]

    result = {
        "keyword": keyword,
        "count": len(results),
        "pois": [
            {
                "name": poi.get("name", "不明"),
                "category": poi.get("category", "不明"),
                "distance_m": poi.get("distance_from_station", "不明"),
                "direction": poi.get("direction_from_station_jp", "不明")
            }
            for poi in results[:20]  # 最大20件
        ]
    }

    return json.dumps(result, ensure_ascii=False, indent=2)


# =============================================================================
# グラフトラバーサル風ツール群
# =============================================================================

@tool
def tool_find_nearby_similar_pois(
    poi_name: str,
    max_distance_m: float = 200
) -> str:
    """
    特定のPOIの近くにある同じカテゴリのPOIを検索します（競合店検索）。

    Args:
        poi_name: 起点となるPOI名（部分一致で検索）
        max_distance_m: 最大距離（メートル、デフォルト: 200m）

    Returns:
        近隣の同カテゴリPOIのJSON文字列

    Examples:
        - スターバックスの近くにある他のカフェ: tool_find_nearby_similar_pois(poi_name="スターバックス", max_distance_m=200)
    """
    pois = get_global_pois()

    # 起点POIを検索
    origin_poi = None
    for poi in pois:
        if poi_name.lower() in poi.get("name", "").lower():
            origin_poi = poi
            break

    if not origin_poi:
        return json.dumps({"error": f"POI '{poi_name}' が見つかりません"}, ensure_ascii=False)

    # 同カテゴリで近隣のPOIを検索
    origin_category = origin_poi.get("category", "")
    origin_lat = origin_poi["lat"]
    origin_lon = origin_poi["lon"]

    nearby_pois = []
    for poi in pois:
        if poi.get("name") == origin_poi.get("name"):
            continue  # 自分自身は除外

        if origin_category in poi.get("category", ""):
            distance = haversine_distance(
                origin_lat, origin_lon,
                poi["lat"], poi["lon"]
            )
            if distance <= max_distance_m:
                nearby_pois.append({
                    "name": poi.get("name", "不明"),
                    "category": poi.get("category", "不明"),
                    "distance_m": round(distance, 2),
                    "direction": poi.get("direction_from_station_jp", "不明")
                })

    # 距離でソート
    nearby_pois.sort(key=lambda x: x["distance_m"])

    result = {
        "origin_poi": origin_poi.get("name"),
        "origin_category": origin_category,
        "max_distance_m": max_distance_m,
        "count": len(nearby_pois),
        "nearby_pois": nearby_pois[:10]  # 最大10件
    }

    return json.dumps(result, ensure_ascii=False, indent=2)


@tool
def tool_find_complementary_pois(
    poi_name: str,
    target_category: str,
    max_distance_m: float = 200
) -> str:
    """
    特定のPOIの近くにある異なるカテゴリのPOIを検索します（相補的POI検索）。
    例: ホテルの近くのレストラン、映画館の近くのカフェなど

    Args:
        poi_name: 起点となるPOI名（部分一致で検索）
        target_category: 検索したいカテゴリ（部分一致）
        max_distance_m: 最大距離（メートル、デフォルト: 200m）

    Returns:
        近隣の異カテゴリPOIのJSON文字列

    Examples:
        - ホテル近くのレストラン: tool_find_complementary_pois(poi_name="ホテル", target_category="レストラン", max_distance_m=300)
        - 映画館近くのカフェ: tool_find_complementary_pois(poi_name="シネマ", target_category="カフェ")
    """
    pois = get_global_pois()

    # 起点POIを検索
    origin_poi = None
    for poi in pois:
        if poi_name.lower() in poi.get("name", "").lower():
            origin_poi = poi
            break

    if not origin_poi:
        return json.dumps({"error": f"POI '{poi_name}' が見つかりません"}, ensure_ascii=False)

    # 指定カテゴリで近隣のPOIを検索
    origin_lat = origin_poi["lat"]
    origin_lon = origin_poi["lon"]

    nearby_pois = []
    for poi in pois:
        if target_category.lower() in poi.get("category", "").lower():
            distance = haversine_distance(
                origin_lat, origin_lon,
                poi["lat"], poi["lon"]
            )
            if distance <= max_distance_m:
                nearby_pois.append({
                    "name": poi.get("name", "不明"),
                    "category": poi.get("category", "不明"),
                    "distance_m": round(distance, 2),
                    "direction": poi.get("direction_from_station_jp", "不明")
                })

    # 距離でソート
    nearby_pois.sort(key=lambda x: x["distance_m"])

    result = {
        "origin_poi": origin_poi.get("name"),
        "origin_category": origin_poi.get("category", "不明"),
        "target_category": target_category,
        "max_distance_m": max_distance_m,
        "count": len(nearby_pois),
        "complementary_pois": nearby_pois[:10]  # 最大10件
    }

    return json.dumps(result, ensure_ascii=False, indent=2)


@tool
def tool_find_pois_in_same_area(
    poi_name: str,
    target_category: Optional[str] = None
) -> str:
    """
    特定のPOIと同じエリアクラスタにあるPOIを検索します。

    Args:
        poi_name: 起点となるPOI名（部分一致で検索）
        target_category: 検索したいカテゴリ（Noneの場合は全カテゴリ）

    Returns:
        同エリアのPOIのJSON文字列

    Examples:
        - 同じエリアのカフェ: tool_find_pois_in_same_area(poi_name="スターバックス", target_category="カフェ")
    """
    pois = get_global_pois()

    # 起点POIを検索
    origin_poi = None
    for poi in pois:
        if poi_name.lower() in poi.get("name", "").lower():
            origin_poi = poi
            break

    if not origin_poi:
        return json.dumps({"error": f"POI '{poi_name}' が見つかりません"}, ensure_ascii=False)

    # 同じエリアクラスタのPOIを検索
    origin_area = origin_poi.get("area_cluster", "")
    if not origin_area:
        return json.dumps({"error": "起点POIのエリア情報がありません"}, ensure_ascii=False)

    same_area_pois = []
    for poi in pois:
        if poi.get("name") == origin_poi.get("name"):
            continue  # 自分自身は除外

        if poi.get("area_cluster") == origin_area:
            # カテゴリフィルタ
            if target_category and target_category.lower() not in poi.get("category", "").lower():
                continue

            same_area_pois.append({
                "name": poi.get("name", "不明"),
                "category": poi.get("category", "不明"),
                "distance_m": poi.get("distance_from_station", "不明"),
                "direction": poi.get("direction_from_station_jp", "不明")
            })

    result = {
        "origin_poi": origin_poi.get("name"),
        "area_cluster": origin_area,
        "target_category": target_category or "全て",
        "count": len(same_area_pois),
        "same_area_pois": same_area_pois[:20]  # 最大20件
    }

    return json.dumps(result, ensure_ascii=False, indent=2)


# =============================================================================
# ツールリスト取得
# =============================================================================

def get_all_tools() -> List:
    """全ツールのリストを取得"""
    return [
        # 空間計算ツール
        tool_get_nearest_pois,
        tool_count_pois_in_radius,
        tool_compare_radius,
        tool_analyze_sensitivity,
        tool_filter_by_area,
        tool_calculate_distance,
        # 比較・集計ツール
        tool_compare_east_west,
        tool_compare_north_south,
        tool_count_by_category,
        tool_get_top_categories,
        tool_analyze_category_by_direction,
        # 検索ツール
        tool_vector_search,
        tool_find_pois_by_keyword,
        # グラフトラバーサルツール
        tool_find_nearby_similar_pois,
        tool_find_complementary_pois,
        tool_find_pois_in_same_area,
    ]


# =============================================================================
# テスト・デバッグ用
# =============================================================================

def test_tools():
    """ツール動作確認"""
    print("=" * 60)
    print("agent_tools.py 動作確認")
    print("=" * 60)

    # テスト用POIデータ読み込み
    try:
        with open("poi_documents.json", "r", encoding="utf-8") as f:
            raw_pois = json.load(f)
    except FileNotFoundError:
        print("poi_documents.json が見つかりません")
        return

    # メタデータをフラット化
    flat_pois = []
    for poi in raw_pois:
        if "metadata" in poi:
            flat_poi = poi["metadata"].copy()
            flat_pois.append(flat_poi)
        else:
            flat_pois.append(poi)

    # 空間情報付加
    from geo_utils import enrich_all_pois
    pois = enrich_all_pois(flat_pois)
    set_global_pois(pois)

    print(f"\nPOIデータ読み込み完了: {len(pois)}件")

    # ツールテスト
    print("\n--- 最寄りカフェ検索 ---")
    print(tool_get_nearest_pois.invoke({"category": "カフェ", "top_n": 3}))

    print("\n--- 500m以内のカフェ件数 ---")
    print(tool_count_pois_in_radius.invoke({"radius_m": 500, "category": "カフェ"}))

    print("\n--- 東西比較（カフェ） ---")
    print(tool_compare_east_west.invoke({"category": "カフェ"}))

    print("\n✅ agent_tools.py 動作確認完了")


if __name__ == "__main__":
    test_tools()
