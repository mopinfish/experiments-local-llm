#!/usr/bin/env python3
"""
evaluators_v2.py - Phase 5: 拡張評価関数群

追加評価指標:
- reasoning_accuracy: 推論の正確性
- evidence_grounding: 根拠の明示度
- constraint_satisfaction: 制約充足度
- uncertainty_handling: 不確実性への言及

レベル別スコア計算式:
- L1: keyword*0.4 + coordinate*0.3 + poi_name*0.3
- L2: keyword*0.3 + coordinate*0.2 + reasoning*0.5
- L3: keyword*0.2 + poi_name*0.2 + constraint*0.4 + evidence*0.2
- L4: reasoning*0.3 + evidence*0.3 + constraint*0.2 + uncertainty*0.2
- L5: reasoning*0.4 + evidence*0.3 + uncertainty*0.3
"""
import re
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Any


# =============================================================================
# テスト結果データクラス（拡張版）
# =============================================================================

@dataclass
class TestResultV2:
    """拡張テスト結果（Phase 5用）"""
    test_id: str  # "L1-01" 形式
    level: int
    category: str
    subcategory: str
    prompt: str
    difficulty: str
    
    # RAGあり結果
    rag_answer: str
    rag_time_ms: int
    rag_keyword_hits: int
    rag_keyword_total: int
    rag_has_coordinate: bool
    rag_has_poi_name: bool
    rag_reasoning_score: float  # 0-5
    rag_evidence_score: float  # 0-5
    rag_constraint_score: float  # 0-5
    rag_uncertainty_score: float  # 0-5
    
    # RAGなし結果
    no_rag_answer: str
    no_rag_time_ms: int
    no_rag_keyword_hits: int
    no_rag_has_coordinate: bool
    no_rag_has_poi_name: bool
    no_rag_reasoning_score: float
    no_rag_evidence_score: float
    no_rag_constraint_score: float
    no_rag_uncertainty_score: float
    
    # スコア
    rag_score: float
    no_rag_score: float
    improvement: float
    
    # 追加情報
    rag_sources: Optional[List[Dict]] = None
    evaluation_notes: str = ""
    
    def to_dict(self) -> dict:
        """辞書に変換"""
        return asdict(self)


# =============================================================================
# 基本評価関数（evaluators.pyから継承）
# =============================================================================

def count_keyword_hits(answer: str, keywords: List[str]) -> int:
    """キーワードヒット数をカウント"""
    if not answer or not keywords:
        return 0
    
    hits = 0
    answer_lower = answer.lower()
    
    for keyword in keywords:
        if keyword.lower() in answer_lower:
            hits += 1
    
    return hits


def has_coordinate(answer: str) -> bool:
    """座標情報が含まれるかチェック"""
    if not answer:
        return False
    
    patterns = [
        r'35\.\d{3,}',
        r'139\.\d{3,}',
        r'緯度\s*[:：]?\s*\d+\.\d+',
        r'経度\s*[:：]?\s*\d+\.\d+',
    ]
    
    for pattern in patterns:
        if re.search(pattern, answer, re.IGNORECASE):
            return True
    
    return False


def has_poi_name(answer: str, poi_documents: List[Dict]) -> bool:
    """POI名が含まれるかチェック"""
    if not answer or not poi_documents:
        return False
    
    for poi in poi_documents:
        if isinstance(poi, dict):
            name = poi.get("metadata", {}).get("name", "") or poi.get("name", "")
        else:
            name = getattr(poi, "name", "")
        
        if name and len(name) > 2 and name in answer:
            return True
    
    return False


# =============================================================================
# 拡張評価関数（Phase 5用）
# =============================================================================

def evaluate_reasoning_accuracy(
    answer: str,
    test_case: Any,
    poi_documents: Optional[List[Dict]] = None
) -> float:
    """
    推論の正確性を評価（0-5スコア）
    
    評価基準:
    - 5: 完全に正確な推論、論理的に一貫
    - 4: 概ね正確、軽微な誤りのみ
    - 3: 部分的に正確、一部論理に問題
    - 2: 推論の方向性は正しいが誤りが多い
    - 1: 推論を試みているが不正確
    - 0: 推論なし、または完全に誤り
    """
    if not answer:
        return 0.0
    
    score = 0.0
    
    # 推論を示すキーワードの存在チェック
    reasoning_indicators = [
        "なぜなら", "理由は", "考えられる", "推測", "判断",
        "したがって", "よって", "ため", "から", "ので",
        "比較すると", "分析すると", "評価すると",
        "多い", "少ない", "近い", "遠い", "集中", "分散",
    ]
    
    indicator_count = sum(1 for ind in reasoning_indicators if ind in answer)
    
    # 基本スコア（推論の試み）
    if indicator_count >= 5:
        score = 3.0
    elif indicator_count >= 3:
        score = 2.0
    elif indicator_count >= 1:
        score = 1.0
    
    # 数値的根拠の提示でボーナス
    if re.search(r'\d+\s*(件|軒|店|か所|箇所|m|メートル|分)', answer):
        score += 1.0
    
    # 比較表現の使用でボーナス
    comparison_patterns = [
        r'より(多|少|近|遠)',
        r'(最も|一番)(多|少|近|遠)',
        r'(高|低|大|小)い',
    ]
    for pattern in comparison_patterns:
        if re.search(pattern, answer):
            score += 0.5
            break
    
    return min(score, 5.0)


def evaluate_evidence_grounding(
    answer: str,
    poi_documents: Optional[List[Dict]] = None
) -> float:
    """
    根拠の明示度を評価（0-5スコア）
    
    評価基準:
    - 5: 全ての結論に具体的なPOIデータを引用
    - 4: 主要な結論に根拠を提示
    - 3: 一部の結論に根拠あり
    - 2: 根拠が曖昧または不十分
    - 1: 根拠の言及があるが具体性なし
    - 0: 根拠なし
    """
    if not answer:
        return 0.0
    
    score = 0.0
    
    # 具体的なPOI名の引用をチェック
    poi_name_count = 0
    if poi_documents:
        for poi in poi_documents:
            if isinstance(poi, dict):
                name = poi.get("metadata", {}).get("name", "") or poi.get("name", "")
            else:
                name = getattr(poi, "name", "")
            
            if name and len(name) > 2 and name in answer:
                poi_name_count += 1
    
    if poi_name_count >= 5:
        score = 4.0
    elif poi_name_count >= 3:
        score = 3.0
    elif poi_name_count >= 1:
        score = 2.0
    
    # 座標情報の引用でボーナス
    if has_coordinate(answer):
        score += 0.5
    
    # 数値データの引用でボーナス
    if re.search(r'(約|およそ)?\d+\s*(件|軒|店|か所)', answer):
        score += 0.5
    
    # 「〜によると」「〜に基づくと」などの引用表現
    citation_patterns = [
        r'データ(から|によると|に基づ)',
        r'情報(から|によると|に基づ)',
        r'検索結果',
        r'POI',
    ]
    for pattern in citation_patterns:
        if re.search(pattern, answer):
            score += 0.5
            break
    
    return min(score, 5.0)


def evaluate_constraint_satisfaction(
    answer: str,
    constraints: Optional[List[str]] = None
) -> float:
    """
    制約充足度を評価（0-5スコア）
    
    Args:
        answer: LLMの回答
        constraints: 制約条件のリスト（例: ["距離500m以内", "電話番号あり"]）
    
    評価基準:
    - 5: 全ての制約を完全に充足
    - 4: ほぼ全ての制約を充足
    - 3: 主要な制約を充足
    - 2: 一部の制約のみ充足
    - 1: 制約への言及はあるが充足していない
    - 0: 制約を無視
    """
    if not answer:
        return 0.0
    
    if not constraints:
        return 3.0  # 制約なしの場合は中立スコア
    
    satisfied_count = 0
    
    for constraint in constraints:
        # 距離制約のチェック
        if "m以内" in constraint or "メートル以内" in constraint:
            distance_match = re.search(r'(\d+)\s*m', constraint)
            if distance_match:
                distance = distance_match.group(1)
                if distance in answer or "近い" in answer or "近く" in answer:
                    satisfied_count += 1
        
        # 属性制約のチェック
        elif "電話番号あり" in constraint:
            if re.search(r'電話|TEL|\d{2,4}-\d{2,4}-\d{4}', answer, re.IGNORECASE):
                satisfied_count += 1
        
        elif "ウェブサイトあり" in constraint:
            if re.search(r'(ウェブサイト|サイト|URL|http|www)', answer, re.IGNORECASE):
                satisfied_count += 1
        
        elif "24時間営業" in constraint:
            if "24時間" in answer or "終日" in answer:
                satisfied_count += 1
        
        # カテゴリ制約のチェック
        elif "を含む" in constraint or "を除外" in constraint:
            # より詳細な分析が必要だが、言及があれば部分点
            keyword = constraint.replace("を含む", "").replace("を除外", "").strip()
            if keyword in answer:
                satisfied_count += 0.5
    
    # スコア計算
    if len(constraints) > 0:
        ratio = satisfied_count / len(constraints)
        score = ratio * 5.0
    else:
        score = 3.0
    
    return min(score, 5.0)


def evaluate_uncertainty_handling(answer: str) -> float:
    """
    不確実性への言及を評価（0-5スコア）
    
    評価基準:
    - 5: 不確実性を適切に認識し、明確に区別
    - 4: 主要な不確実性に言及
    - 3: 一部の不確実性に言及
    - 2: 不確実性への言及はあるが不十分
    - 1: 曖昧な留保のみ
    - 0: 不確実性への言及なし
    """
    if not answer:
        return 0.0
    
    score = 0.0
    
    # 不確実性を示すキーワード
    uncertainty_indicators = [
        "可能性があ", "かもしれ", "推測", "推定",
        "不明", "わかりません", "確認できません",
        "データの限界", "情報がない", "情報が不足",
        "確実ではない", "断定できない",
        "おそらく", "恐らく", "思われ",
        "注意", "留意", "ご注意",
    ]
    
    # 区別を示すキーワード
    distinction_indicators = [
        "データからは", "情報からは",
        "推測できる", "推測できない",
        "判断できる", "判断できない",
        "一方で", "ただし", "しかし",
    ]
    
    # 不確実性への言及をカウント
    uncertainty_count = sum(1 for ind in uncertainty_indicators if ind in answer)
    distinction_count = sum(1 for ind in distinction_indicators if ind in answer)
    
    # スコア計算
    if uncertainty_count >= 3 and distinction_count >= 2:
        score = 5.0
    elif uncertainty_count >= 2 and distinction_count >= 1:
        score = 4.0
    elif uncertainty_count >= 2:
        score = 3.0
    elif uncertainty_count >= 1:
        score = 2.0
    elif distinction_count >= 1:
        score = 1.5
    
    return min(score, 5.0)


# =============================================================================
# レベル別スコア計算
# =============================================================================

def calculate_level_score(
    level: int,
    keyword_hits: int,
    keyword_total: int,
    has_coord: bool,
    has_name: bool,
    reasoning_score: float,
    evidence_score: float,
    constraint_score: float,
    uncertainty_score: float
) -> float:
    """
    レベル別の総合スコアを計算（0-100）
    
    スコア計算式:
    - L1: keyword*0.4 + coordinate*0.3 + poi_name*0.3
    - L2: keyword*0.3 + coordinate*0.2 + reasoning*0.5
    - L3: keyword*0.2 + poi_name*0.2 + constraint*0.4 + evidence*0.2
    - L4: reasoning*0.3 + evidence*0.3 + constraint*0.2 + uncertainty*0.2
    - L5: reasoning*0.4 + evidence*0.3 + uncertainty*0.3
    """
    # 基本スコアの正規化（0-100）
    keyword_score = (keyword_hits / keyword_total * 100) if keyword_total > 0 else 0
    coord_score = 100 if has_coord else 0
    name_score = 100 if has_name else 0
    reasoning_norm = reasoning_score * 20  # 0-5 → 0-100
    evidence_norm = evidence_score * 20
    constraint_norm = constraint_score * 20
    uncertainty_norm = uncertainty_score * 20
    
    if level == 1:
        # L1: 基礎検索
        score = keyword_score * 0.4 + coord_score * 0.3 + name_score * 0.3
    
    elif level == 2:
        # L2: 空間推論
        score = keyword_score * 0.3 + coord_score * 0.2 + reasoning_norm * 0.5
    
    elif level == 3:
        # L3: 制約充足
        score = keyword_score * 0.2 + name_score * 0.2 + constraint_norm * 0.4 + evidence_norm * 0.2
    
    elif level == 4:
        # L4: 意思決定支援
        score = reasoning_norm * 0.3 + evidence_norm * 0.3 + constraint_norm * 0.2 + uncertainty_norm * 0.2
    
    elif level == 5:
        # L5: 高度推論
        score = reasoning_norm * 0.4 + evidence_norm * 0.3 + uncertainty_norm * 0.3
    
    else:
        # デフォルト（L1と同じ）
        score = keyword_score * 0.4 + coord_score * 0.3 + name_score * 0.3
    
    return round(score, 1)


# =============================================================================
# 集計関数
# =============================================================================

def aggregate_results_v2(results: List[TestResultV2]) -> Dict[str, Any]:
    """テスト結果を集計（拡張版）"""
    if not results:
        return {}
    
    n = len(results)
    
    # 基本統計
    stats = {
        "test_count": n,
        "avg_rag_score": round(sum(r.rag_score for r in results) / n, 1),
        "avg_no_rag_score": round(sum(r.no_rag_score for r in results) / n, 1),
        "avg_improvement": round(sum(r.improvement for r in results) / n, 1),
        "avg_rag_time_ms": round(sum(r.rag_time_ms for r in results) / n, 0),
        "avg_no_rag_time_ms": round(sum(r.no_rag_time_ms for r in results) / n, 0),
    }
    
    # 拡張評価指標の平均
    stats["avg_rag_reasoning"] = round(sum(r.rag_reasoning_score for r in results) / n, 2)
    stats["avg_rag_evidence"] = round(sum(r.rag_evidence_score for r in results) / n, 2)
    stats["avg_rag_constraint"] = round(sum(r.rag_constraint_score for r in results) / n, 2)
    stats["avg_rag_uncertainty"] = round(sum(r.rag_uncertainty_score for r in results) / n, 2)
    
    stats["avg_no_rag_reasoning"] = round(sum(r.no_rag_reasoning_score for r in results) / n, 2)
    stats["avg_no_rag_evidence"] = round(sum(r.no_rag_evidence_score for r in results) / n, 2)
    stats["avg_no_rag_constraint"] = round(sum(r.no_rag_constraint_score for r in results) / n, 2)
    stats["avg_no_rag_uncertainty"] = round(sum(r.no_rag_uncertainty_score for r in results) / n, 2)
    
    return stats


def aggregate_by_level(results: List[TestResultV2]) -> Dict[int, Dict[str, Any]]:
    """レベル別にテスト結果を集計"""
    levels = {}
    
    for r in results:
        level = r.level
        if level not in levels:
            levels[level] = []
        levels[level].append(r)
    
    return {level: aggregate_results_v2(level_results) for level, level_results in sorted(levels.items())}


def aggregate_by_subcategory(results: List[TestResultV2]) -> Dict[str, Dict[str, Any]]:
    """サブカテゴリ別にテスト結果を集計"""
    subcategories = {}
    
    for r in results:
        subcat = r.subcategory
        if subcat not in subcategories:
            subcategories[subcat] = []
        subcategories[subcat].append(r)
    
    return {subcat: aggregate_results_v2(subcat_results) for subcat, subcat_results in subcategories.items()}


if __name__ == "__main__":
    # テスト
    print("拡張評価関数テスト")
    print("-" * 40)
    
    test_answer = """
    渋谷駅周辺で最も近いコンビニはローソン渋谷神南店です。
    渋谷駅から約150mの距離にあり、徒歩で約2分です。
    座標は緯度35.6620、経度139.7005です。
    
    なぜなら、データを分析すると、渋谷駅（35.658, 139.702）から
    最も距離が短いコンビニがこの店舗だからです。
    
    ただし、営業時間などの情報は確認できませんでした。
    """
    
    print(f"推論スコア: {evaluate_reasoning_accuracy(test_answer, None)}")
    print(f"根拠スコア: {evaluate_evidence_grounding(test_answer)}")
    print(f"制約スコア: {evaluate_constraint_satisfaction(test_answer, ['距離200m以内'])}")
    print(f"不確実性スコア: {evaluate_uncertainty_handling(test_answer)}")
