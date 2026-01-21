#!/usr/bin/env python3
"""
utils.py - ユーティリティ関数

機能:
- ディレクトリ管理
- 結果の保存・読込
- レポート生成
- POIデータ取得
"""
import os
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import asdict


# =============================================================================
# ディレクトリ管理
# =============================================================================

def setup_directories(base_dir: str = "/content/drive/MyDrive/experiments-local-llm") -> Dict[str, str]:
    """
    プロジェクトディレクトリを作成
    
    Args:
        base_dir: ベースディレクトリのパス
        
    Returns:
        各ディレクトリパスの辞書
    """
    dirs = {
        "base": base_dir,
        "data": f"{base_dir}/data",
        "results": f"{base_dir}/results",
        "src": f"{base_dir}/src",
        "models": f"{base_dir}/models",
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs


def get_timestamp() -> str:
    """タイムスタンプ文字列を取得"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# =============================================================================
# 結果の保存・読込
# =============================================================================

def save_results(
    results: List[Any],
    output_dir: str,
    prefix: str = "result",
    metadata: Optional[Dict] = None
) -> str:
    """
    テスト結果をJSONファイルに保存
    
    Args:
        results: テスト結果のリスト
        output_dir: 出力ディレクトリ
        prefix: ファイル名プレフィックス
        metadata: 追加メタデータ
        
    Returns:
        保存したファイルパス
    """
    timestamp = get_timestamp()
    filename = f"{prefix}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    # 結果を辞書に変換
    results_data = []
    for r in results:
        if hasattr(r, "to_dict"):
            results_data.append(r.to_dict())
        elif hasattr(r, "__dict__"):
            results_data.append(asdict(r) if hasattr(r, "__dataclass_fields__") else r.__dict__)
        else:
            results_data.append(r)
    
    # 保存データ構造
    data = {
        "timestamp": timestamp,
        "test_count": len(results),
        "results": results_data
    }
    
    # メタデータを追加
    if metadata:
        data["metadata"] = metadata
    
    # 保存
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    return filepath


def load_results(filepath: str) -> Dict:
    """
    テスト結果をJSONファイルから読み込み
    
    Args:
        filepath: 読み込むファイルパス
        
    Returns:
        結果データの辞書
    """
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_poi_data(poi_documents: List[Dict], output_path: str) -> str:
    """
    POIデータを保存
    
    Args:
        poi_documents: POIドキュメントのリスト
        output_path: 出力ファイルパス
        
    Returns:
        保存したファイルパス
    """
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(poi_documents, f, ensure_ascii=False, indent=2)
    return output_path


def load_poi_data(filepath: str) -> List[Dict]:
    """
    POIデータを読み込み
    
    Args:
        filepath: 読み込むファイルパス
        
    Returns:
        POIドキュメントのリスト
    """
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# レポート生成
# =============================================================================

def generate_report(
    results_summary: Dict,
    model_info: Dict,
    raspi_baseline: Optional[Dict] = None,
    output_path: Optional[str] = None
) -> str:
    """
    Markdownレポートを生成
    
    Args:
        results_summary: 結果サマリー
        model_info: モデル情報
        raspi_baseline: Raspberry Piベースライン結果（比較用）
        output_path: 出力ファイルパス（省略時はレポート文字列のみ返す）
        
    Returns:
        レポートのMarkdown文字列
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"""# POI RAG System - Test Report

**実行日時**: {timestamp}
**環境**: Google Colab (GPU)

---

## 1. 実験設定

| 項目 | 値 |
|------|-----|
| LLMモデル | {model_info.get('llm_model', 'N/A')} |
| Embeddingモデル | {model_info.get('embedding_model', 'N/A')} |
| POIデータ件数 | {model_info.get('poi_count', 'N/A')}件 |
| テストケース数 | {results_summary.get('test_count', 'N/A')}件 |

---

## 2. 全体結果

| 指標 | RAGあり | RAGなし | 差分 |
|------|---------|---------|------|
| 平均スコア | {results_summary.get('avg_rag_score', 'N/A')} | {results_summary.get('avg_no_rag_score', 'N/A')} | {results_summary.get('avg_improvement', 'N/A'):+.1f} |
| キーワードヒット率 | {results_summary.get('rag_keyword_rate', 'N/A')}% | {results_summary.get('no_rag_keyword_rate', 'N/A')}% | {results_summary.get('rag_keyword_rate', 0) - results_summary.get('no_rag_keyword_rate', 0):+.1f}% |
| 座標含有率 | {results_summary.get('rag_coordinate_rate', 'N/A')}% | {results_summary.get('no_rag_coordinate_rate', 'N/A')}% | {results_summary.get('rag_coordinate_rate', 0) - results_summary.get('no_rag_coordinate_rate', 0):+.1f}% |
| POI名含有率 | {results_summary.get('rag_poi_name_rate', 'N/A')}% | {results_summary.get('no_rag_poi_name_rate', 'N/A')}% | {results_summary.get('rag_poi_name_rate', 0) - results_summary.get('no_rag_poi_name_rate', 0):+.1f}% |
| 平均処理時間 | {results_summary.get('avg_rag_time_ms', 'N/A'):.0f}ms | {results_summary.get('avg_no_rag_time_ms', 'N/A'):.0f}ms | - |

"""

    # Raspberry Pi比較（ベースラインがある場合）
    if raspi_baseline:
        speedup = raspi_baseline.get('avg_rag_time_ms', 0) / max(results_summary.get('avg_rag_time_ms', 1), 1)
        report += f"""---

## 3. Raspberry Pi 4B との比較

| 指標 | Raspberry Pi | Google Colab | 改善 |
|------|--------------|--------------|------|
| RAGスコア | {raspi_baseline.get('avg_rag_score', 'N/A')} | {results_summary.get('avg_rag_score', 'N/A')} | {results_summary.get('avg_rag_score', 0) - raspi_baseline.get('avg_rag_score', 0):+.1f} |
| 処理時間 | {raspi_baseline.get('avg_rag_time_ms', 0)/1000:.0f}秒 | {results_summary.get('avg_rag_time_ms', 0)/1000:.1f}秒 | **{speedup:.1f}倍高速** |

"""

    report += f"""---

## 4. 考察

### RAGの効果
- 全体改善率: {results_summary.get('avg_improvement', 0):+.1f}%
- 座標情報の提供: RAGにより座標含有率が向上
- POI名の精度: RAGにより実在する施設名を回答に含める率が向上

### 今後の課題
- 複合検索（complex）カテゴリの改善
- より高度な空間推論への対応
- 応答速度のさらなる最適化

---

**レポート生成日時**: {timestamp}
"""

    # ファイルに保存（指定された場合）
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
    
    return report


# =============================================================================
# POIデータ取得
# =============================================================================

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

def build_overpass_query(bbox: str) -> str:
    """
    Overpass APIクエリを構築
    
    Args:
        bbox: バウンディングボックス (south,west,north,east)
        
    Returns:
        Overpassクエリ文字列
    """
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


def get_category(tags: dict) -> str:
    """
    OSMタグからカテゴリを判定
    
    Args:
        tags: OSMタグの辞書
        
    Returns:
        カテゴリ文字列
    """
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


def fetch_poi_from_osm(area_config: dict) -> List[Dict]:
    """
    OpenStreetMapからPOIデータを取得
    
    Args:
        area_config: エリア設定 {"name": "渋谷駅周辺", "bbox": "35.655,139.695,35.665,139.710"}
        
    Returns:
        POIドキュメントのリスト
    """
    import requests
    
    query = build_overpass_query(area_config["bbox"])
    
    print(f"📡 Overpass APIにクエリ送信中: {area_config['name']}...")
    response = requests.post(OVERPASS_URL, data={"data": query}, timeout=120)
    response.raise_for_status()
    osm_data = response.json()
    
    elements = osm_data.get("elements", [])
    print(f"   取得した要素数: {len(elements)}")
    
    documents = []
    for element in elements:
        tags = element.get("tags", {})
        if not tags:
            continue
        
        # 名前がないPOIはスキップ
        name = tags.get("name", tags.get("name:ja", ""))
        if not name:
            continue
        
        # 座標を取得
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
        
        # 住所情報
        addr_parts = []
        for key in ["addr:province", "addr:city", "addr:district", "addr:street", "addr:housenumber"]:
            if key in tags:
                addr_parts.append(tags[key])
        address = "".join(addr_parts) if addr_parts else tags.get("addr:full", "住所情報なし")
        
        # ドキュメント生成
        doc_text = f"""【POI名称】{name}
【英語名】{name_en if name_en else "なし"}
【カテゴリ】{category}
【エリア】{area_config['name']}
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
                "area": area_config['name'],
                "lat": lat,
                "lon": lon,
                "source": "openstreetmap"
            }
        })
    
    print(f"   有効なPOI数: {len(documents)}")
    return documents


# =============================================================================
# カテゴリキーワードマッピング
# =============================================================================

CATEGORY_KEYWORDS = {
    "レストラン": ["飲食店/レストラン"],
    "カフェ": ["飲食店/カフェ"],
    "コーヒー": ["飲食店/カフェ"],
    "バー": ["飲食店/バー"],
    "居酒屋": ["飲食店/バー", "飲食店/パブ"],
    "ファストフード": ["飲食店/ファストフード"],
    "マクドナルド": ["飲食店/ファストフード"],
    "コンビニ": ["商店/コンビニ"],
    "ローソン": ["商店/コンビニ"],
    "セブン": ["商店/コンビニ"],
    "映画館": ["娯楽/映画館"],
    "シネマ": ["娯楽/映画館"],
    "ホテル": ["宿泊/ホテル"],
    "駅": ["交通/鉄道駅"],
    "銀行": ["金融/銀行"],
    "郵便局": ["公共/郵便局"],
    "病院": ["医療/病院"],
    "薬局": ["医療/薬局"],
    "交番": ["公共/警察"],
}


def detect_category(question: str) -> tuple:
    """
    質問文からカテゴリを検出
    
    Args:
        question: 質問文
        
    Returns:
        (検出カテゴリリスト, マッチしたキーワードリスト)
    """
    detected = []
    matched_keywords = []
    
    for keyword, categories in CATEGORY_KEYWORDS.items():
        if keyword in question:
            detected.extend(categories)
            matched_keywords.append(keyword)
    
    return list(set(detected)), matched_keywords


if __name__ == "__main__":
    # テスト
    print("ユーティリティ関数テスト")
    print("-" * 40)
    
    # ディレクトリ設定テスト
    dirs = setup_directories("/tmp/test_experiment")
    print(f"作成したディレクトリ: {list(dirs.keys())}")
    
    # タイムスタンプテスト
    ts = get_timestamp()
    print(f"タイムスタンプ: {ts}")
    
    # カテゴリ検出テスト
    question = "渋谷駅周辺のカフェを教えて"
    cats, kws = detect_category(question)
    print(f"質問: {question}")
    print(f"検出カテゴリ: {cats}")
    print(f"マッチキーワード: {kws}")
