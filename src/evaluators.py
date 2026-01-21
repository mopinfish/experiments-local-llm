#!/usr/bin/env python3
"""
evaluators.py - RAGシステム評価関数群

評価指標:
- キーワードヒット率: 期待キーワードの含有率
- 座標含有率: 座標情報の有無
- POI名含有率: 実在POI名の含有率
- 総合スコア: 上記を重み付けした総合評価
"""
import re
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Any


# =============================================================================
# テスト結果データクラス
# =============================================================================

@dataclass
class TestResult:
    """テスト結果"""
    test_id: int
    test_category: str
    prompt: str
    # RAGあり結果
    rag_answer: str
    rag_time_ms: int
    rag_keyword_hits: int
    rag_keyword_total: int
    rag_has_coordinate: bool
    rag_has_poi_name: bool
    # RAGなし結果
    no_rag_answer: str
    no_rag_time_ms: int
    no_rag_keyword_hits: int
    no_rag_has_coordinate: bool
    no_rag_has_poi_name: bool
    # スコア
    rag_score: float
    no_rag_score: float
    improvement: float
    # 追加情報（オプション）
    difficulty: str = ""
    rag_sources: Optional[List[Dict]] = None
    
    def to_dict(self) -> dict:
        """辞書に変換"""
        return asdict(self)


# =============================================================================
# 評価関数
# =============================================================================

def count_keyword_hits(answer: str, keywords: List[str]) -> int:
    """
    回答に含まれるキーワードの数をカウント
    
    Args:
        answer: LLMの回答テキスト
        keywords: 期待されるキーワードのリスト
        
    Returns:
        ヒットしたキーワードの数
    """
    if not answer or not keywords:
        return 0
    
    hits = 0
    answer_lower = answer.lower()
    
    for keyword in keywords:
        if keyword.lower() in answer_lower:
            hits += 1
    
    return hits


def has_coordinate(answer: str) -> bool:
    """
    回答に座標情報が含まれるかチェック
    
    Args:
        answer: LLMの回答テキスト
        
    Returns:
        座標情報が含まれる場合True
    """
    if not answer:
        return False
    
    # 緯度・経度のパターン（東京周辺: 緯度35.x, 経度139.x）
    patterns = [
        r'35\.\d{3,}',  # 緯度（小数点以下3桁以上）
        r'139\.\d{3,}',  # 経度（小数点以下3桁以上）
        r'緯度\s*[:：]?\s*\d+\.\d+',  # 「緯度: 35.xxx」形式
        r'経度\s*[:：]?\s*\d+\.\d+',  # 「経度: 139.xxx」形式
        r'lat[itude]*\s*[:=]?\s*\d+\.\d+',  # 英語形式
        r'lon[gitude]*\s*[:=]?\s*\d+\.\d+',  # 英語形式
        r'\d+°\d+[\'′]\d+[\"″]?[NS]',  # 度分秒形式（緯度）
        r'\d+°\d+[\'′]\d+[\"″]?[EW]',  # 度分秒形式（経度）
    ]
    
    for pattern in patterns:
        if re.search(pattern, answer, re.IGNORECASE):
            return True
    
    return False


def has_poi_name(answer: str, poi_documents: List[Dict]) -> bool:
    """
    回答にPOI名が含まれるかチェック
    
    Args:
        answer: LLMの回答テキスト
        poi_documents: POIドキュメントのリスト
        
    Returns:
        POI名が含まれる場合True
    """
    if not answer or not poi_documents:
        return False
    
    for poi in poi_documents:
        # metadataからnameを取得
        if isinstance(poi, dict):
            name = poi.get("metadata", {}).get("name", "") or poi.get("name", "")
        else:
            name = getattr(poi, "name", "")
        
        # 2文字以下の名前はスキップ（誤検出防止）
        if name and len(name) > 2 and name in answer:
            return True
    
    return False


def has_any_poi_name(answer: str, known_poi_names: Optional[List[str]] = None) -> bool:
    """
    回答に既知のPOI名が含まれるかチェック（RAGなし評価用）
    
    Args:
        answer: LLMの回答テキスト
        known_poi_names: 既知のPOI名リスト（Noneの場合はデフォルトリスト使用）
        
    Returns:
        POI名が含まれる場合True
    """
    if not answer:
        return False
    
    # デフォルトの既知POI名（渋谷エリアの主要施設）
    if known_poi_names is None:
        known_poi_names = [
            "渋谷駅", "渋谷109", "ハチ公", "スクランブル交差点",
            "渋谷ヒカリエ", "渋谷パルコ", "東急", "マークシティ",
            "センター街", "道玄坂", "宮益坂", "公園通り",
            "マクドナルド", "スターバックス", "ローソン", "セブンイレブン",
            "ファミリーマート", "東武ホテル", "エクセルホテル"
        ]
    
    for name in known_poi_names:
        if name in answer:
            return True
    
    return False


def calculate_score(
    keyword_hits: int,
    keyword_total: int,
    has_coord: bool,
    has_name: bool,
    expected_data_type: str = "name",
    weights: Optional[Dict[str, float]] = None
) -> float:
    """
    総合スコアを計算（0-100）
    
    Args:
        keyword_hits: ヒットしたキーワード数
        keyword_total: 期待キーワード総数
        has_coord: 座標情報の有無
        has_name: POI名の有無
        expected_data_type: 期待されるデータタイプ（coordinate, name, time, address）
        weights: カスタム重み（省略時はデフォルト使用）
        
    Returns:
        総合スコア（0-100）
    """
    # デフォルト重み
    if weights is None:
        weights = {
            "keyword": 0.4,  # キーワードヒット率 40%
            "coordinate": 0.3,  # 座標情報 30%
            "poi_name": 0.3  # POI名含有 30%
        }
    
    # キーワードヒット率スコア (0-100)
    if keyword_total > 0:
        keyword_score = (keyword_hits / keyword_total) * 100
    else:
        keyword_score = 0
    
    # 座標情報スコア (0-100)
    if expected_data_type == "coordinate":
        # 座標が期待される場合、座標の有無で0/100
        coord_score = 100 if has_coord else 0
    else:
        # 座標が期待されない場合、あってもなくても50点
        coord_score = 50
    
    # POI名含有スコア (0-100)
    name_score = 100 if has_name else 0
    
    # 重み付け総合スコア
    total_score = (
        keyword_score * weights["keyword"] +
        coord_score * weights["coordinate"] +
        name_score * weights["poi_name"]
    )
    
    return round(total_score, 1)


def calculate_improvement(rag_score: float, no_rag_score: float) -> float:
    """
    RAGによる改善率を計算
    
    Args:
        rag_score: RAGありスコア
        no_rag_score: RAGなしスコア
        
    Returns:
        改善率（スコア差分）
    """
    return round(rag_score - no_rag_score, 1)


def calculate_improvement_percentage(rag_score: float, no_rag_score: float) -> float:
    """
    RAGによる改善率をパーセンテージで計算
    
    Args:
        rag_score: RAGありスコア
        no_rag_score: RAGなしスコア
        
    Returns:
        改善率（パーセンテージ）
    """
    if no_rag_score > 0:
        return round(((rag_score - no_rag_score) / no_rag_score) * 100, 1)
    elif rag_score > 0:
        return 100.0
    else:
        return 0.0


# =============================================================================
# 集計関数
# =============================================================================

def aggregate_results(results: List[TestResult]) -> Dict[str, Any]:
    """
    テスト結果を集計
    
    Args:
        results: テスト結果のリスト
        
    Returns:
        集計結果の辞書
    """
    if not results:
        return {}
    
    n = len(results)
    
    # 基本統計
    avg_rag_score = sum(r.rag_score for r in results) / n
    avg_no_rag_score = sum(r.no_rag_score for r in results) / n
    avg_improvement = sum(r.improvement for r in results) / n
    avg_rag_time = sum(r.rag_time_ms for r in results) / n
    avg_no_rag_time = sum(r.no_rag_time_ms for r in results) / n
    
    # キーワードヒット率
    total_rag_hits = sum(r.rag_keyword_hits for r in results)
    total_no_rag_hits = sum(r.no_rag_keyword_hits for r in results)
    total_keywords = sum(r.rag_keyword_total for r in results)
    
    rag_keyword_rate = (total_rag_hits / total_keywords * 100) if total_keywords > 0 else 0
    no_rag_keyword_rate = (total_no_rag_hits / total_keywords * 100) if total_keywords > 0 else 0
    
    # 座標含有率
    rag_coord_rate = sum(1 for r in results if r.rag_has_coordinate) / n * 100
    no_rag_coord_rate = sum(1 for r in results if r.no_rag_has_coordinate) / n * 100
    
    # POI名含有率
    rag_name_rate = sum(1 for r in results if r.rag_has_poi_name) / n * 100
    no_rag_name_rate = sum(1 for r in results if r.no_rag_has_poi_name) / n * 100
    
    return {
        "test_count": n,
        "avg_rag_score": round(avg_rag_score, 1),
        "avg_no_rag_score": round(avg_no_rag_score, 1),
        "avg_improvement": round(avg_improvement, 1),
        "avg_rag_time_ms": round(avg_rag_time, 0),
        "avg_no_rag_time_ms": round(avg_no_rag_time, 0),
        "rag_keyword_rate": round(rag_keyword_rate, 1),
        "no_rag_keyword_rate": round(no_rag_keyword_rate, 1),
        "rag_coordinate_rate": round(rag_coord_rate, 1),
        "no_rag_coordinate_rate": round(no_rag_coord_rate, 1),
        "rag_poi_name_rate": round(rag_name_rate, 1),
        "no_rag_poi_name_rate": round(no_rag_name_rate, 1),
    }


def aggregate_by_category(results: List[TestResult]) -> Dict[str, Dict[str, Any]]:
    """
    カテゴリ別にテスト結果を集計
    
    Args:
        results: テスト結果のリスト
        
    Returns:
        カテゴリ別集計結果
    """
    categories = {}
    
    for r in results:
        cat = r.test_category
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(r)
    
    return {cat: aggregate_results(cat_results) for cat, cat_results in categories.items()}


if __name__ == "__main__":
    # テスト
    print("評価関数テスト")
    print("-" * 40)
    
    # キーワードヒットテスト
    answer = "渋谷駅は緯度35.658、経度139.702にあります。"
    keywords = ["渋谷", "駅", "35.", "139."]
    hits = count_keyword_hits(answer, keywords)
    print(f"キーワードヒット: {hits}/{len(keywords)}")
    
    # 座標検出テスト
    print(f"座標含有: {has_coordinate(answer)}")
    
    # スコア計算テスト
    score = calculate_score(hits, len(keywords), True, True, "coordinate")
    print(f"総合スコア: {score}")
