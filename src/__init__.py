"""
experiments-local-llm/src パッケージ

Phase 6: RAG改善モジュール群

モジュール一覧:
- geo_utils: 座標計算ユーティリティ
- aggregator: 集計・比較ユーティリティ
- structured_rag_system: 構造化RAGシステム
"""

# Phase 6 新規モジュール
from .geo_utils import (
    # 定数
    SHIBUYA_STATION,
    STATIONS,
    AREA_STATION_MAP,
    LANDMARKS,
    DIRECTION_JP,

    # データクラス
    GeoPoint,
    SpatialInfo,

    # 距離計算
    haversine_distance,
    distance_from_station,
    get_distance_zone,

    # 方向判定
    get_direction,
    direction_from_station,
    get_direction_jp,
    simplify_direction,
    is_east_side,
    is_west_side,
    is_north_side,
    is_south_side,

    # エリアクラスタリング
    get_area_cluster,

    # 空間情報計算
    compute_spatial_info,
    enrich_poi_with_spatial_info,
    enrich_all_pois,

    # フィルタリング
    filter_by_distance,
    filter_by_direction,
    filter_east_side,
    filter_west_side,
    filter_by_area_cluster,

    # Phase 6.2追加: 近接性検索・感度分析
    get_nearest_pois,
    get_farthest_pois,
    filter_by_radius,
    count_by_radius,
    compare_by_radius,
    analyze_radius_sensitivity,
    get_poi_distance_stats,
    generate_proximity_context,
    generate_sensitivity_context,
    RadiusComparisonResult,

    # Phase 9-B: 駅・ランドマーク解決
    resolve_station,
    resolve_landmark,

    # Phase 9-B: エリア特定
    detect_target_area,
    detect_cross_area_query,

    # Phase 9-B: 広域エンリッチメント
    enrich_pois_for_area,
    enrich_all_areas,
)

from .aggregator import (
    # データクラス
    CategoryCount,
    DirectionCount,
    ComparisonResult,
    AggregationResult,

    # 基本集計
    count_total,
    count_by_category,
    count_by_subcategory,
    count_by_main_category,
    count_by_direction,
    count_by_area_cluster,
    count_by_distance_zone,

    # ランキング
    get_top_categories,
    get_top_subcategories,

    # フィルタリング
    filter_by_category,
    filter_by_keyword,

    # 比較
    compare_counts,
    compare_categories,
    compare_east_west,
    compare_north_south,

    # 詳細分析
    analyze_category_by_direction,
    analyze_direction,
    get_full_aggregation,

    # コンテキスト生成
    generate_aggregation_context,
    generate_comparison_context,

    # Phase 9-B: エリア間比較
    aggregate_by_area,
    compare_across_areas,
    generate_cross_area_context,
)

from .structured_rag_system import (
    # 定数
    CATEGORY_KEYWORDS,
    PROXIMITY_KEYWORDS,
    SENSITIVITY_KEYWORDS,
    
    # クラス
    QuestionAnalysis,
    StructuredRAGSystem,
    
    # 関数
    analyze_question,
    create_structured_rag_system,
)

# バージョン情報
__version__ = "0.9.2"
__phase__ = "Phase 9-B: Multi-Area Core Modules"
