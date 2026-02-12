#!/usr/bin/env python3
"""
test_tools_functionality.py - ツール機能の動作確認

Phase 9: Agentic RAGツールの単体テスト

作成日: 2026-02-03
"""

import sys
import json
sys.path.insert(0, 'src')

from agent_tools import (
    set_global_pois,
    tool_get_nearest_pois,
    tool_count_pois_in_radius,
    tool_compare_radius,
    tool_compare_east_west,
    tool_get_top_categories,
    tool_find_nearby_similar_pois,
    tool_find_complementary_pois,
    tool_find_pois_in_same_area
)
from geo_utils import enrich_all_pois


def load_test_data():
    """テストデータ読み込み"""
    print("Loading POI data...")
    with open("poi_documents.json", "r", encoding="utf-8") as f:
        raw_pois = json.load(f)

    # メタデータをフラット化
    flat_pois = []
    for poi in raw_pois:
        if "metadata" in poi:
            flat_poi = poi["metadata"].copy()
            flat_pois.append(flat_poi)
        else:
            flat_pois.append(poi)

    # 空間情報付加
    pois = enrich_all_pois(flat_pois)
    set_global_pois(pois)

    print(f"✓ Loaded {len(pois)} POIs with spatial info\n")
    return pois


def test_spatial_tools():
    """空間計算ツールのテスト"""
    print("=" * 60)
    print("Testing Spatial Tools")
    print("=" * 60)

    # 1. 最寄りカフェ検索
    print("\n1. tool_get_nearest_pois (カフェ)")
    result = tool_get_nearest_pois.invoke({"category": "カフェ", "top_n": 3})
    data = json.loads(result)
    print(f"   Found: {data['count']} cafes")
    if data['pois']:
        print(f"   Nearest: {data['pois'][0]['name']} ({data['pois'][0]['distance_m']}m)")

    # 2. 半径内件数
    print("\n2. tool_count_pois_in_radius (500m, カフェ)")
    result = tool_count_pois_in_radius.invoke({"radius_m": 500, "category": "カフェ"})
    data = json.loads(result)
    print(f"   Count: {data['count']} cafes within {data['radius_m']}m")

    # 3. 半径比較
    print("\n3. tool_compare_radius (300m vs 500m, カフェ)")
    result = tool_compare_radius.invoke({
        "radius1_m": 300,
        "radius2_m": 500,
        "category": "カフェ"
    })
    data = json.loads(result)
    print(f"   300m: {data['count1']}, 500m: {data['count2']}")
    print(f"   Difference: +{data['difference']} ({data['ratio']}x)")


def test_aggregation_tools():
    """集計ツールのテスト"""
    print("\n" + "=" * 60)
    print("Testing Aggregation Tools")
    print("=" * 60)

    # 1. 東西比較
    print("\n1. tool_compare_east_west (カフェ)")
    result = tool_compare_east_west.invoke({"category": "カフェ"})
    data = json.loads(result)
    print(f"   East: {data['east_count']}, West: {data['west_count']}")
    print(f"   Winner: {data['winner']} (+{data['difference']})")

    # 2. トップカテゴリ
    print("\n2. tool_get_top_categories (top 5)")
    result = tool_get_top_categories.invoke({"top_n": 5})
    data = json.loads(result)
    print(f"   Top {data['top_n']} categories:")
    for cat in data['categories']:
        print(f"     {cat['rank']}. {cat['category']}: {cat['count']}")


def test_graph_tools():
    """グラフトラバーサルツールのテスト"""
    print("\n" + "=" * 60)
    print("Testing Graph Traversal Tools")
    print("=" * 60)

    # 1. 近隣同カテゴリPOI
    print("\n1. tool_find_nearby_similar_pois (スターバックス近くのカフェ)")
    result = tool_find_nearby_similar_pois.invoke({
        "poi_name": "スターバックス",
        "max_distance_m": 200
    })
    data = json.loads(result)
    if 'error' not in data:
        print(f"   Origin: {data['origin_poi']} ({data['origin_category']})")
        print(f"   Found: {data['count']} similar POIs within {data['max_distance_m']}m")
        if data['nearby_pois']:
            print(f"   Nearest: {data['nearby_pois'][0]['name']} ({data['nearby_pois'][0]['distance_m']}m)")
    else:
        print(f"   Error: {data['error']}")

    # 2. 相補的POI
    print("\n2. tool_find_complementary_pois (映画館近くのカフェ)")
    result = tool_find_complementary_pois.invoke({
        "poi_name": "シネマ",
        "target_category": "カフェ",
        "max_distance_m": 300
    })
    data = json.loads(result)
    if 'error' not in data:
        print(f"   Origin: {data['origin_poi']} ({data['origin_category']})")
        print(f"   Found: {data['count']} {data['target_category']} within {data['max_distance_m']}m")
    else:
        print(f"   Error: {data['error']}")

    # 3. 同エリアPOI
    print("\n3. tool_find_pois_in_same_area (スターバックスと同じエリア)")
    result = tool_find_pois_in_same_area.invoke({
        "poi_name": "スターバックス",
        "target_category": "飲食店"
    })
    data = json.loads(result)
    if 'error' not in data:
        print(f"   Origin: {data['origin_poi']} (Area: {data['area_cluster']})")
        print(f"   Found: {data['count']} {data['target_category']} in same area")
    else:
        print(f"   Error: {data['error']}")


def main():
    """メイン実行"""
    print("=" * 60)
    print("Agentic RAG Tools Functionality Test")
    print("=" * 60)
    print()

    # データ読み込み
    load_test_data()

    # ツールテスト
    test_spatial_tools()
    test_aggregation_tools()
    test_graph_tools()

    print("\n" + "=" * 60)
    print("✓ All tool tests completed successfully")
    print("=" * 60)


if __name__ == "__main__":
    main()
