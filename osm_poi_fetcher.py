#!/usr/bin/env python3
"""
osm_poi_fetcher.py - OpenStreetMapからPOIデータを取得
"""
import requests
import json
from pathlib import Path

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# 対象エリア: 渋谷駅周辺 (south,west,north,east)
AREAS = {
    "shibuya": {
        "name": "渋谷駅周辺",
        "bbox": "35.655,139.695,35.665,139.710"
    }
}

def build_query(bbox: str) -> str:
    """Overpass APIクエリを構築"""
    return f"""
[out:json][timeout:60];
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
    """Overpass APIからPOIを取得"""
    area = AREAS[area_key]
    query = build_query(area["bbox"])
    
    print(f"  クエリ実行中: {area['name']}...")
    response = requests.post(OVERPASS_URL, data={"data": query}, timeout=120)
    response.raise_for_status()
    return response.json()

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

def convert_to_documents(osm_data: dict, area_name: str) -> list:
    """OSMデータをRAG用ドキュメントに変換"""
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
        
        # 住所情報を構築
        addr_parts = []
        for key in ["addr:province", "addr:city", "addr:district", "addr:street", "addr:housenumber"]:
            if key in tags:
                addr_parts.append(tags[key])
        address = "".join(addr_parts) if addr_parts else tags.get("addr:full", "住所情報なし")
        
        # ドキュメント生成
        doc_text = f"""【POI名称】{name}
【英語名】{name_en if name_en else "なし"}
【カテゴリ】{category}
【エリア】{area_name}
【座標】緯度 {lat:.6f}, 経度 {lon:.6f}
【住所】{address}
【営業時間】{tags.get('opening_hours', '営業時間情報なし')}
【電話番号】{tags.get('phone', tags.get('contact:phone', '電話番号情報なし'))}
【ウェブサイト】{tags.get('website', tags.get('contact:website', 'なし'))}
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
                "lat": lat,
                "lon": lon,
                "source": "openstreetmap"
            }
        })
    
    return documents

def main():
    print("=" * 50)
    print("OSM POIデータ取得ツール")
    print("=" * 50)
    
    all_documents = []
    
    for area_key, area_info in AREAS.items():
        print(f"\n[{area_info['name']}] データ取得中...")
        
        try:
            osm_data = fetch_osm_pois(area_key)
            elements_count = len(osm_data.get("elements", []))
            print(f"  取得した要素数: {elements_count}")
            
            documents = convert_to_documents(osm_data, area_info["name"])
            print(f"  有効なPOI数: {len(documents)}")
            
            all_documents.extend(documents)
        except Exception as e:
            print(f"  エラー: {e}")
    
    print(f"\n合計ドキュメント数: {len(all_documents)}")
    
    # 保存
    output_path = Path("./poi_documents.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_documents, f, ensure_ascii=False, indent=2)
    
    print(f"保存完了: {output_path}")
    
    # カテゴリ別集計
    print("\n【カテゴリ別集計】")
    categories = {}
    for doc in all_documents:
        cat = doc["metadata"]["category"]
        categories[cat] = categories.get(cat, 0) + 1
    
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}件")

if __name__ == "__main__":
    main()
