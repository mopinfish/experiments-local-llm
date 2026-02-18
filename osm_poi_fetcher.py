#!/usr/bin/env python3
"""
osm_poi_fetcher.py - OpenStreetMapからPOIデータを取得

拡張版: ブランド、営業時間、料理ジャンル等の追加タグを取得
Phase 9-B: 複数エリア対応（渋谷・新宿・池袋・東京）
"""
import requests
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]

# 対象エリア定義 (south,west,north,east)
# Phase 9-B: 4エリア対応
AREAS = {
    "shibuya": {
        "name": "渋谷駅周辺",
        "station": {"name": "渋谷駅", "lat": 35.658034, "lon": 139.701636},
        "bbox": "35.655,139.695,35.665,139.710"
    },
    "shinjuku": {
        "name": "新宿駅周辺",
        "station": {"name": "新宿駅", "lat": 35.689607, "lon": 139.700571},
        "bbox": "35.685,139.693,35.697,139.710"
    },
    "ikebukuro": {
        "name": "池袋駅周辺",
        "station": {"name": "池袋駅", "lat": 35.729503, "lon": 139.710999},
        "bbox": "35.725,139.704,35.736,139.718"
    },
    "tokyo": {
        "name": "東京駅周辺",
        "station": {"name": "東京駅", "lat": 35.681236, "lon": 139.767125},
        "bbox": "35.676,139.760,35.687,139.775"
    },
}

# Overpass APIレートリミット対策: エリア間の待機秒数
API_REQUEST_INTERVAL = 10

# 既知のブランド/チェーン店（名前からの抽出用）
KNOWN_BRANDS = {
    # コンビニ
    "セブン-イレブン": "7-Eleven",
    "セブンイレブン": "7-Eleven",
    "ローソン": "Lawson",
    "ファミリーマート": "FamilyMart",
    "ファミマ": "FamilyMart",
    "ミニストップ": "Ministop",
    # カフェ
    "スターバックス": "Starbucks",
    "STARBUCKS": "Starbucks",
    "Starbucks": "Starbucks",
    "ドトール": "Doutor",
    "DOUTOR": "Doutor",
    "タリーズ": "Tully's",
    "TULLY'S": "Tully's",
    "カフェ・ベローチェ": "Veloce",
    "ベローチェ": "Veloce",
    "コメダ珈琲": "Komeda",
    "エクセルシオール": "Excelsior",
    "プロント": "Pronto",
    "サンマルク": "Saint Marc",
    # ファストフード
    "マクドナルド": "McDonald's",
    "McDonald's": "McDonald's",
    "モスバーガー": "Mos Burger",
    "バーガーキング": "Burger King",
    "ケンタッキー": "KFC",
    "KFC": "KFC",
    "松屋": "Matsuya",
    "吉野家": "Yoshinoya",
    "すき家": "Sukiya",
    "なか卯": "Nakau",
    "CoCo壱番屋": "CoCo Ichibanya",
    "ココイチ": "CoCo Ichibanya",
    # 銀行
    "三菱UFJ": "MUFG",
    "みずほ": "Mizuho",
    "三井住友": "SMBC",
    "りそな": "Resona",
    # ドラッグストア
    "マツモトキヨシ": "Matsumoto Kiyoshi",
    "ウエルシア": "Welcia",
    "ツルハ": "Tsuruha",
    # ホテルチェーン
    "東横イン": "Toyoko Inn",
    "アパホテル": "APA Hotel",
    "ドーミーイン": "Dormy Inn",
}

def build_query(bbox: str) -> str:
    """Overpass APIクエリを構築"""
    return f"""
[out:json][timeout:180];
(
  // 飲食店
  node["amenity"~"restaurant|cafe|fast_food|bar|pub"]({bbox});
  way["amenity"~"restaurant|cafe|fast_food|bar|pub"]({bbox});
  
  // コンビニ・商店
  node["shop"~"convenience|supermarket|bakery|books"]({bbox});
  way["shop"~"convenience|supermarket|bakery|books"]({bbox});
  
  // 観光・娯楽
  node["tourism"~"attraction|museum|hotel|information"]({bbox});
  way["tourism"~"attraction|museum|hotel|information"]({bbox});
  node["amenity"~"cinema|theatre"]({bbox});
  
  // 交通
  node["railway"="station"]({bbox});
  node["amenity"="parking"]({bbox});
  
  // 公共施設
  node["amenity"~"hospital|clinic|pharmacy|bank|post_office|police"]({bbox});
  way["amenity"~"hospital|clinic|pharmacy|bank|post_office|police"]({bbox});
);
out center tags;
"""

def fetch_osm_pois(area_key: str) -> dict:
    """Overpass APIからPOIを取得（フェイルオーバー付き）"""
    area = AREAS[area_key]
    query = build_query(area["bbox"])

    for url in OVERPASS_URLS:
        try:
            print(f"  クエリ実行中: {area['name']} ({url.split('//')[1].split('/')[0]})...")
            response = requests.post(url, data={"data": query}, timeout=240)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"  失敗 ({url.split('//')[1].split('/')[0]}): {e}")
            continue

    raise RuntimeError(f"全てのOverpass APIエンドポイントで {area['name']} の取得に失敗しました")

def get_category(tags: dict) -> str:
    """タグからカテゴリを判定"""
    if "amenity" in tags:
        amenity = tags["amenity"]
        category_map = {
            "restaurant": "飲食店/レストラン",
            "cafe": "飲食店/カフェ",
            "fast_food": "飲食店/ファストフード",
            "bar": "飲食店/バー",
            "pub": "飲食店/パブ",
            "cinema": "娯楽/映画館",
            "theatre": "娯楽/劇場",
            "hospital": "医療/病院",
            "clinic": "医療/クリニック",
            "pharmacy": "医療/薬局",
            "bank": "金融/銀行",
            "post_office": "公共/郵便局",
            "police": "公共/警察",
            "parking": "交通/駐車場",
        }
        return category_map.get(amenity, f"施設/{amenity}")
    
    if "shop" in tags:
        shop = tags["shop"]
        shop_map = {
            "convenience": "商店/コンビニ",
            "supermarket": "商店/スーパー",
            "bakery": "商店/パン屋",
            "books": "商店/書店",
        }
        return shop_map.get(shop, f"商店/{shop}")
    
    if "tourism" in tags:
        tourism = tags["tourism"]
        tourism_map = {
            "attraction": "観光/名所",
            "museum": "観光/博物館",
            "hotel": "宿泊/ホテル",
            "information": "観光/案内所",
        }
        return tourism_map.get(tourism, f"観光/{tourism}")
    
    if "railway" in tags:
        return "交通/鉄道駅"
    
    return "その他"

def extract_brand(name: str, tags: dict) -> Optional[str]:
    """POI名またはタグからブランドを抽出"""
    # OSMタグから直接取得
    if "brand" in tags:
        return tags["brand"]
    if "operator" in tags:
        return tags["operator"]

    # 名前からブランドを推定
    for brand_jp, brand_en in KNOWN_BRANDS.items():
        if brand_jp in name:
            return brand_en

    return None


def parse_opening_hours(opening_hours: str) -> Dict[str, Any]:
    """営業時間を解析して構造化"""
    if not opening_hours or opening_hours == "営業時間情報なし":
        return {"raw": None, "is_24h": False, "late_night": False, "early_morning": False}

    result = {
        "raw": opening_hours,
        "is_24h": "24/7" in opening_hours or "24時間" in opening_hours,
        "late_night": False,
        "early_morning": False
    }

    # 深夜営業（22:00以降）の検出
    late_patterns = [r"2[2-3]:\d{2}", r"0[0-4]:\d{2}"]
    for pattern in late_patterns:
        if re.search(pattern, opening_hours):
            result["late_night"] = True
            break

    # 早朝営業（6:00以前）の検出
    early_patterns = [r"0[5-6]:\d{2}", r"05:", r"06:"]
    for pattern in early_patterns:
        if re.search(pattern, opening_hours):
            result["early_morning"] = True
            break

    return result


def extract_cuisine(tags: dict) -> Optional[str]:
    """料理ジャンルを抽出"""
    if "cuisine" in tags:
        return tags["cuisine"]
    return None


def convert_to_documents(osm_data: dict, area_key: str, area_info: dict) -> list:
    """OSMデータをRAG用ドキュメントに変換（拡張版）

    Args:
        osm_data: Overpass APIのレスポンスJSON
        area_key: エリアキー（"shibuya", "shinjuku" 等）
        area_info: エリア情報辞書（name, station, bbox）
    """
    area_name = area_info["name"]
    station = area_info["station"]
    documents = []

    for element in osm_data.get("elements", []):
        tags = element.get("tags", {})
        if not tags:
            continue

        # 名前がないPOIはスキップ
        name = tags.get("name", tags.get("name:ja", ""))
        if not name:
            continue

        # 座標を取得（wayの場合はcenterを使用）
        if element.get("type") == "node":
            lat = element.get("lat")
            lon = element.get("lon")
        else:
            center = element.get("center", {})
            lat = center.get("lat")
            lon = center.get("lon")

        if not lat or not lon:
            continue

        name_en = tags.get("name:en", "")
        category = get_category(tags)

        # 拡張メタデータの抽出
        brand = extract_brand(name, tags)
        opening_hours_raw = tags.get('opening_hours', '')
        opening_hours_parsed = parse_opening_hours(opening_hours_raw)
        cuisine = extract_cuisine(tags)

        # 住所情報を構築
        addr_parts = []
        for key in ["addr:province", "addr:city", "addr:district", "addr:street", "addr:housenumber"]:
            if key in tags:
                addr_parts.append(tags[key])
        address = "".join(addr_parts) if addr_parts else tags.get("addr:full", "住所情報なし")

        # 住所キー（同一建物検出用）
        addr_key = None
        if "addr:street" in tags and "addr:housenumber" in tags:
            addr_key = f"{tags.get('addr:street', '')}_{tags.get('addr:housenumber', '')}"

        # ドキュメント生成
        doc_text = f"""【POI名称】{name}
【英語名】{name_en if name_en else "なし"}
【カテゴリ】{category}
【ブランド】{brand if brand else "なし"}
【エリア】{area_name}
【座標】緯度 {lat:.6f}, 経度 {lon:.6f}
【住所】{address}
【営業時間】{opening_hours_raw if opening_hours_raw else '営業時間情報なし'}
【料理ジャンル】{cuisine if cuisine else "なし"}
【電話番号】{tags.get('phone', tags.get('contact:phone', '電話番号情報なし'))}
【ウェブサイト】{tags.get('website', tags.get('contact:website', 'なし'))}
【バリアフリー】{tags.get('wheelchair', 'unknown')}
【Wi-Fi】{tags.get('internet_access', 'unknown')}
【説明】{tags.get('description', tags.get('description:ja', ''))}"""

        documents.append({
            "id": f"osm_{element['type']}_{element['id']}",
            "content": doc_text.strip(),
            "metadata": {
                "osm_id": element["id"],
                "osm_type": element["type"],
                "name": name,
                "name_en": name_en,
                "category": category,
                "area": area_name,
                "area_key": area_key,
                "station_name": station["name"],
                "station_lat": station["lat"],
                "station_lon": station["lon"],
                "lat": lat,
                "lon": lon,
                "source": "openstreetmap",
                # 拡張メタデータ
                "brand": brand,
                "brand_wikidata": tags.get("brand:wikidata"),
                "operator": tags.get("operator"),
                "opening_hours": opening_hours_raw if opening_hours_raw else None,
                "is_24h": opening_hours_parsed["is_24h"],
                "late_night": opening_hours_parsed["late_night"],
                "early_morning": opening_hours_parsed["early_morning"],
                "cuisine": cuisine,
                "wheelchair": tags.get("wheelchair"),
                "internet_access": tags.get("internet_access"),
                "level": tags.get("level"),
                "addr_key": addr_key,
            }
        })

    return documents

def save_json(documents: list, path: Path) -> None:
    """ドキュメントリストをJSONファイルに保存"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)
    print(f"  保存完了: {path} ({len(documents)}件)")


def print_stats(documents: list, label: str = "") -> None:
    """ドキュメントの統計情報を表示"""
    if label:
        print(f"\n{'=' * 50}")
        print(f"統計情報: {label}")
        print(f"{'=' * 50}")

    total = len(documents)
    if total == 0:
        print("  ドキュメントなし")
        return

    # カテゴリ別集計
    print(f"\n【カテゴリ別集計】(合計 {total}件)")
    categories = {}
    for doc in documents:
        cat = doc["metadata"]["category"]
        categories[cat] = categories.get(cat, 0) + 1

    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}件")

    # ブランド別集計
    print("\n【ブランド別集計】")
    brands = {}
    for doc in documents:
        brand = doc["metadata"].get("brand")
        if brand:
            brands[brand] = brands.get(brand, 0) + 1

    if brands:
        for brand, count in sorted(brands.items(), key=lambda x: -x[1])[:15]:
            print(f"  {brand}: {count}件")
        print(f"  ... 合計 {len(brands)} ブランド検出")
    else:
        print("  ブランド情報なし")

    # 拡張メタデータ統計
    print("\n【拡張メタデータ統計】")
    meta_keys = ["brand", "opening_hours", "is_24h", "late_night",
                 "cuisine", "wheelchair", "internet_access"]
    for key in meta_keys:
        count = sum(1 for doc in documents if doc["metadata"].get(key))
        pct = count / total * 100
        print(f"  {key}: {count}件 ({pct:.1f}%)")


def main():
    print("=" * 50)
    print("OSM POIデータ取得ツール（複数エリア対応版）")
    print("=" * 50)

    data_dir = Path("./data")
    all_documents = []
    documents_by_area = {}

    area_keys = list(AREAS.keys())
    for i, area_key in enumerate(area_keys):
        area_info = AREAS[area_key]
        print(f"\n[{i+1}/{len(area_keys)}] [{area_info['name']}] データ取得中...")

        # レートリミット対策: 2番目以降のエリアは待機
        if i > 0:
            print(f"  Overpass API待機中 ({API_REQUEST_INTERVAL}秒)...")
            time.sleep(API_REQUEST_INTERVAL)

        try:
            osm_data = fetch_osm_pois(area_key)
            elements_count = len(osm_data.get("elements", []))
            print(f"  取得した要素数: {elements_count}")

            documents = convert_to_documents(osm_data, area_key, area_info)
            print(f"  有効なPOI数: {len(documents)}")

            documents_by_area[area_key] = documents
            all_documents.extend(documents)
        except Exception as e:
            print(f"  エラー: {e}")
            documents_by_area[area_key] = []

    # --- 保存 ---
    print(f"\n{'=' * 50}")
    print(f"保存処理")
    print(f"{'=' * 50}")
    print(f"合計ドキュメント数: {len(all_documents)}")

    # エリア別ファイル
    for area_key, docs in documents_by_area.items():
        save_json(docs, data_dir / f"poi_{area_key}.json")

    # 全エリア統合ファイル
    save_json(all_documents, data_dir / "poi_all_areas.json")

    # 後方互換: 渋谷のみのpoi_documents.json
    shibuya_docs = documents_by_area.get("shibuya", [])
    save_json(shibuya_docs, Path("./poi_documents.json"))

    # --- ID重複チェック ---
    print(f"\n【ID重複チェック】")
    ids = [doc["id"] for doc in all_documents]
    unique_ids = set(ids)
    if len(ids) == len(unique_ids):
        print(f"  重複なし (ユニークID数: {len(unique_ids)})")
    else:
        duplicates = len(ids) - len(unique_ids)
        print(f"  ⚠️ 重複あり: {duplicates}件")

    # --- エリア別サマリー ---
    print(f"\n{'=' * 50}")
    print(f"エリア別サマリー")
    print(f"{'=' * 50}")
    for area_key, docs in documents_by_area.items():
        area_name = AREAS[area_key]["name"]
        print(f"  {area_name}: {len(docs)}件")

    # --- 統計情報 ---
    for area_key, docs in documents_by_area.items():
        print_stats(docs, label=AREAS[area_key]["name"])

    print_stats(all_documents, label="全エリア統合")


if __name__ == "__main__":
    main()
