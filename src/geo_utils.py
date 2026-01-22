"""
geo_utils.py - 地理座標計算ユーティリティ

Phase 6: RAG改善のための座標計算・方向判定・エリアクラスタリング機能

作成日: 2026-01-22
プロジェクト: experiments-local-llm
"""

import math
from typing import Tuple, List, Dict, Any, Optional
from dataclasses import dataclass


# =============================================================================
# 定数定義
# =============================================================================

# 渋谷駅の座標（基準点）
SHIBUYA_STATION = {
    "name": "渋谷駅",
    "lat": 35.658034,
    "lon": 139.701636
}

# 地球の半径（メートル）
EARTH_RADIUS_M = 6371000

# 方向の日本語マッピング
DIRECTION_JP = {
    "north": "北",
    "northeast": "北東",
    "east": "東",
    "southeast": "南東",
    "south": "南",
    "southwest": "南西",
    "west": "西",
    "northwest": "北西",
    "center": "中心"
}

# 簡略化方向（4方位）
SIMPLIFIED_DIRECTION = {
    "north": "north",
    "northeast": "east",
    "east": "east",
    "southeast": "east",
    "south": "south",
    "southwest": "west",
    "west": "west",
    "northwest": "north"
}


# =============================================================================
# データクラス
# =============================================================================

@dataclass
class GeoPoint:
    """地理座標を表すデータクラス"""
    lat: float
    lon: float
    name: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "lat": self.lat,
            "lon": self.lon,
            "name": self.name
        }


@dataclass
class SpatialInfo:
    """空間情報を表すデータクラス"""
    distance_m: float
    direction: str
    direction_jp: str
    area_cluster: str
    distance_zone: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "distance_m": self.distance_m,
            "direction": self.direction,
            "direction_jp": self.direction_jp,
            "area_cluster": self.area_cluster,
            "distance_zone": self.distance_zone
        }


# =============================================================================
# 距離計算関数
# =============================================================================

def haversine_distance(
    lat1: float, lon1: float,
    lat2: float, lon2: float
) -> float:
    """
    2点間の距離をHaversine公式で計算（メートル単位）
    
    Args:
        lat1, lon1: 点1の緯度・経度
        lat2, lon2: 点2の緯度・経度
    
    Returns:
        距離（メートル）
    
    Example:
        >>> haversine_distance(35.658034, 139.701636, 35.6595, 139.7004)
        179.23  # 約179m
    """
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    
    a = (math.sin(delta_phi / 2) ** 2 + 
         math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return EARTH_RADIUS_M * c


def distance_from_station(
    lat: float, lon: float,
    station: Dict[str, float] = None
) -> float:
    """
    指定した駅からの距離を計算
    
    Args:
        lat, lon: 対象地点の緯度・経度
        station: 基準駅の座標辞書（デフォルトは渋谷駅）
    
    Returns:
        距離（メートル）
    """
    if station is None:
        station = SHIBUYA_STATION
    
    return haversine_distance(
        station["lat"], station["lon"],
        lat, lon
    )


def get_distance_zone(distance_m: float) -> str:
    """
    距離に基づいてゾーンを判定
    
    Args:
        distance_m: 距離（メートル）
    
    Returns:
        ゾーン名（"station", "near", "mid", "far"）
    """
    if distance_m < 200:
        return "station"
    elif distance_m < 500:
        return "near"
    elif distance_m < 800:
        return "mid"
    else:
        return "far"


# =============================================================================
# 方向判定関数
# =============================================================================

def get_direction(
    base_lat: float, base_lon: float,
    target_lat: float, target_lon: float
) -> str:
    """
    基準点から見た対象点の方向を判定（8方位）
    
    Args:
        base_lat, base_lon: 基準点の緯度・経度
        target_lat, target_lon: 対象点の緯度・経度
    
    Returns:
        方向（"north", "south", "east", "west", 
              "northeast", "northwest", "southeast", "southwest"）
    
    Example:
        >>> get_direction(35.658034, 139.701636, 35.660, 139.705)
        'northeast'
    """
    delta_lat = target_lat - base_lat
    delta_lon = target_lon - base_lon
    
    # 非常に近い場合は中心
    if abs(delta_lat) < 0.0001 and abs(delta_lon) < 0.0001:
        return "center"
    
    # 角度を計算（北を0度、時計回りで正）
    angle = math.degrees(math.atan2(delta_lon, delta_lat))
    
    # 8方位の判定
    if -22.5 <= angle < 22.5:
        return "north"
    elif 22.5 <= angle < 67.5:
        return "northeast"
    elif 67.5 <= angle < 112.5:
        return "east"
    elif 112.5 <= angle < 157.5:
        return "southeast"
    elif angle >= 157.5 or angle < -157.5:
        return "south"
    elif -157.5 <= angle < -112.5:
        return "southwest"
    elif -112.5 <= angle < -67.5:
        return "west"
    else:
        return "northwest"


def direction_from_station(
    lat: float, lon: float,
    station: Dict[str, float] = None
) -> str:
    """
    指定した駅から見た方向を判定
    
    Args:
        lat, lon: 対象地点の緯度・経度
        station: 基準駅の座標辞書（デフォルトは渋谷駅）
    
    Returns:
        方向（8方位）
    """
    if station is None:
        station = SHIBUYA_STATION
    
    return get_direction(
        station["lat"], station["lon"],
        lat, lon
    )


def get_direction_jp(direction: str) -> str:
    """
    方向を日本語に変換
    
    Args:
        direction: 英語の方向
    
    Returns:
        日本語の方向
    """
    return DIRECTION_JP.get(direction, direction)


def simplify_direction(direction: str) -> str:
    """
    8方位を4方位に簡略化
    
    Args:
        direction: 8方位の方向
    
    Returns:
        4方位の方向
    """
    return SIMPLIFIED_DIRECTION.get(direction, direction)


def is_east_side(direction: str) -> bool:
    """東側かどうかを判定"""
    return direction in ["east", "northeast", "southeast"]


def is_west_side(direction: str) -> bool:
    """西側かどうかを判定"""
    return direction in ["west", "northwest", "southwest"]


def is_north_side(direction: str) -> bool:
    """北側かどうかを判定"""
    return direction in ["north", "northeast", "northwest"]


def is_south_side(direction: str) -> bool:
    """南側かどうかを判定"""
    return direction in ["south", "southeast", "southwest"]


# =============================================================================
# エリアクラスタリング関数
# =============================================================================

def get_area_cluster(
    lat: float, lon: float,
    station: Dict[str, float] = None
) -> str:
    """
    座標からエリアクラスタを判定
    
    Args:
        lat, lon: 対象地点の緯度・経度
        station: 基準駅の座標辞書（デフォルトは渋谷駅）
    
    Returns:
        エリアクラスタ名（例: "east_near", "west_station"）
    
    Example:
        >>> get_area_cluster(35.660, 139.705)
        'east_near'
    """
    if station is None:
        station = SHIBUYA_STATION
    
    direction = direction_from_station(lat, lon, station)
    distance = distance_from_station(lat, lon, station)
    distance_zone = get_distance_zone(distance)
    
    # 方向を簡略化（4方位）
    simple_direction = simplify_direction(direction)
    
    # 中心の場合は特別扱い
    if direction == "center":
        return "station_center"
    
    return f"{simple_direction}_{distance_zone}"


# =============================================================================
# 空間情報計算関数
# =============================================================================

def compute_spatial_info(
    lat: float, lon: float,
    station: Dict[str, float] = None
) -> SpatialInfo:
    """
    POIの空間情報を一括計算
    
    Args:
        lat, lon: 対象地点の緯度・経度
        station: 基準駅の座標辞書（デフォルトは渋谷駅）
    
    Returns:
        SpatialInfoオブジェクト
    """
    if station is None:
        station = SHIBUYA_STATION
    
    distance = distance_from_station(lat, lon, station)
    direction = direction_from_station(lat, lon, station)
    
    return SpatialInfo(
        distance_m=round(distance, 2),
        direction=direction,
        direction_jp=get_direction_jp(direction),
        area_cluster=get_area_cluster(lat, lon, station),
        distance_zone=get_distance_zone(distance)
    )


def enrich_poi_with_spatial_info(
    poi: Dict[str, Any],
    station: Dict[str, float] = None
) -> Dict[str, Any]:
    """
    POIデータに空間情報を追加
    
    Args:
        poi: POIデータ辞書（lat, lonを含む）
        station: 基準駅の座標辞書
    
    Returns:
        空間情報を追加したPOIデータ
    """
    if poi.get("lat") is None or poi.get("lon") is None:
        return poi
    
    spatial_info = compute_spatial_info(poi["lat"], poi["lon"], station)
    
    enriched = poi.copy()
    enriched["distance_from_station"] = spatial_info.distance_m
    enriched["direction_from_station"] = spatial_info.direction
    enriched["direction_from_station_jp"] = spatial_info.direction_jp
    enriched["area_cluster"] = spatial_info.area_cluster
    enriched["distance_zone"] = spatial_info.distance_zone
    
    return enriched


def enrich_all_pois(
    pois: List[Dict[str, Any]],
    station: Dict[str, float] = None
) -> List[Dict[str, Any]]:
    """
    全POIデータに空間情報を追加
    
    Args:
        pois: POIデータのリスト
        station: 基準駅の座標辞書
    
    Returns:
        空間情報を追加したPOIデータのリスト
    """
    return [enrich_poi_with_spatial_info(poi, station) for poi in pois]


# =============================================================================
# フィルタリング関数
# =============================================================================

def filter_by_distance(
    pois: List[Dict[str, Any]],
    max_distance_m: float,
    station: Dict[str, float] = None
) -> List[Dict[str, Any]]:
    """
    距離でPOIをフィルタリング
    
    Args:
        pois: POIデータのリスト
        max_distance_m: 最大距離（メートル）
        station: 基準駅の座標辞書
    
    Returns:
        フィルタリングされたPOIリスト
    """
    if station is None:
        station = SHIBUYA_STATION
    
    result = []
    for poi in pois:
        if poi.get("lat") and poi.get("lon"):
            distance = distance_from_station(poi["lat"], poi["lon"], station)
            if distance <= max_distance_m:
                result.append(poi)
    
    return result


def filter_by_direction(
    pois: List[Dict[str, Any]],
    directions: List[str],
    station: Dict[str, float] = None
) -> List[Dict[str, Any]]:
    """
    方向でPOIをフィルタリング
    
    Args:
        pois: POIデータのリスト
        directions: 含める方向のリスト（例: ["east", "northeast"]）
        station: 基準駅の座標辞書
    
    Returns:
        フィルタリングされたPOIリスト
    """
    if station is None:
        station = SHIBUYA_STATION
    
    result = []
    for poi in pois:
        if poi.get("lat") and poi.get("lon"):
            direction = direction_from_station(poi["lat"], poi["lon"], station)
            if direction in directions:
                result.append(poi)
    
    return result


def filter_east_side(
    pois: List[Dict[str, Any]],
    station: Dict[str, float] = None
) -> List[Dict[str, Any]]:
    """東側のPOIをフィルタリング"""
    return filter_by_direction(pois, ["east", "northeast", "southeast"], station)


def filter_west_side(
    pois: List[Dict[str, Any]],
    station: Dict[str, float] = None
) -> List[Dict[str, Any]]:
    """西側のPOIをフィルタリング"""
    return filter_by_direction(pois, ["west", "northwest", "southwest"], station)


def filter_by_area_cluster(
    pois: List[Dict[str, Any]],
    clusters: List[str]
) -> List[Dict[str, Any]]:
    """
    エリアクラスタでPOIをフィルタリング
    
    Args:
        pois: 空間情報を持つPOIデータのリスト
        clusters: 含めるクラスタのリスト（例: ["east_near", "east_station"]）
    
    Returns:
        フィルタリングされたPOIリスト
    """
    return [poi for poi in pois if poi.get("area_cluster") in clusters]


# =============================================================================
# テスト・デバッグ用関数
# =============================================================================

def test_geo_utils():
    """
    geo_utilsモジュールの動作確認
    """
    print("=" * 60)
    print("geo_utils.py 動作確認")
    print("=" * 60)
    
    # テスト用の座標（渋谷駅周辺）
    test_points = [
        {"name": "東口方面", "lat": 35.659, "lon": 139.705},
        {"name": "西口方面", "lat": 35.658, "lon": 139.698},
        {"name": "北方面", "lat": 35.662, "lon": 139.701},
        {"name": "南方面", "lat": 35.654, "lon": 139.701},
        {"name": "駅近く", "lat": 35.6582, "lon": 139.7018},
    ]
    
    print(f"\n基準点: {SHIBUYA_STATION['name']}")
    print(f"座標: ({SHIBUYA_STATION['lat']}, {SHIBUYA_STATION['lon']})")
    
    print("\n--- 各地点の空間情報 ---")
    for point in test_points:
        info = compute_spatial_info(point["lat"], point["lon"])
        print(f"\n{point['name']}:")
        print(f"  距離: {info.distance_m:.1f}m")
        print(f"  方向: {info.direction} ({info.direction_jp})")
        print(f"  エリア: {info.area_cluster}")
        print(f"  ゾーン: {info.distance_zone}")
    
    # 東西判定テスト
    print("\n--- 東西判定テスト ---")
    for point in test_points:
        direction = direction_from_station(point["lat"], point["lon"])
        east = "東側" if is_east_side(direction) else ""
        west = "西側" if is_west_side(direction) else ""
        print(f"{point['name']}: {direction} → {east}{west}")
    
    print("\n✅ geo_utils.py 動作確認完了")


if __name__ == "__main__":
    test_geo_utils()
