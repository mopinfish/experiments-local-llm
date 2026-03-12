# Phase 5-6 RAG改善プロジェクト 進捗レポート

**作成日**: 2026-01-23  
**プロジェクト**: experiments-local-llm  
**フェーズ**: Phase 5完了 → Phase 6.1完了

---

## 1. エグゼクティブサマリー

### 1.1 主要成果

| 指標 | Phase 5 | Phase 6.1 | 改善 |
|------|---------|-----------|------|
| **全体平均スコア** | 60.3pt | **69.6pt** | **+9.3pt (+15.4%)** |
| **処理時間/質問** | 22.7秒 | **12.9秒** | **-43.2%** |
| **spatial_comparison** | 51.4pt | **76.0pt** | **+24.6pt** |
| **advanced_comparison** | 59.5pt | **64.4pt** | **+4.9pt** |

### 1.2 達成状況

| 目標 | 目標値 | 実績 | 達成状況 |
|------|--------|------|----------|
| spatial_comparison改善 | +5pt以上 | **+24.6pt** | ✅ **大幅達成** |
| advanced_comparison改善 | +10pt以上 | +4.9pt | △ 部分達成 |
| 処理時間維持 | <30秒 | 12.9秒 | ✅ 達成 |

---

## 2. Phase 5 完了時点の状況

### 2.1 評価結果（55件階層化テストケース）

Phase 5では、Raspberry Pi 4BからGoogle Colabへの移行を完了し、55件の階層化テストケース（L1-L5）による評価を実施しました。

**レベル別スコア:**

| レベル | カテゴリ | 平均スコア | 件数 |
|--------|---------|-----------|------|
| L1 | 基礎検索 | 65.0pt | 10件 |
| L2 | 空間推論 | 59.4pt | 15件 |
| L3 | 制約充足 | 53.3pt | 10件 |
| L4 | 意思決定支援 | 63.3pt | 10件 |
| L5 | 高度推論 | 62.1pt | 10件 |

### 2.2 特定された弱点

Phase 5の詳細分析により、以下の弱点が明確になりました：

| サブカテゴリ | スコア | RAG効果 | 問題点 |
|-------------|--------|---------|--------|
| **spatial_comparison** | 51.4pt | **-8.9pt** | 方向概念なし、POI件数集計不可 |
| **advanced_comparison** | 59.5pt | **-0.8pt** | 複数検索結果の比較不可 |
| constraint_single | 53.3pt | -6.7pt | 距離制約のフィルタリング不可 |
| constraint_multi | 53.3pt | -6.7pt | 複数条件の論理処理が不十分 |

### 2.3 根本原因分析

従来のRAGシステムの限界：

1. **ベクトル検索のみ**: 類似度検索は得意だが、集計・比較は不可能
2. **座標情報の未活用**: POIに座標があるが、距離計算や方向判定に使われていない
3. **構造化クエリ不可**: 「東側のカフェ」「500m以内」などの条件処理ができない

---

## 3. Phase 6.1 構造化RAG実装

### 3.1 設計方針

Phase 5で特定された弱点を解消するため、以下の3つのモジュールを新規実装しました：

```
experiments-local-llm/src/
├── geo_utils.py           # 座標計算・方向判定
├── aggregator.py          # 集計・比較機能
└── structured_rag_system.py  # 統合RAGシステム
```

### 3.2 実装モジュール詳細

#### 3.2.1 geo_utils.py（座標計算モジュール）

**主要機能:**

| 関数 | 機能 | 用途 |
|------|------|------|
| `haversine_distance()` | 2点間距離計算 | 駅からの距離測定 |
| `get_direction()` | 8方位判定 | 東西南北の方向分類 |
| `get_area_cluster()` | エリアクラスタリング | "east_near"等の分類 |
| `enrich_all_pois()` | 全POIに空間情報追加 | メタデータ拡張 |

**追加されるメタデータ:**

```python
{
    "distance_from_station": 150,      # メートル
    "direction_from_station": "east",  # 8方位
    "direction_from_station_jp": "東", # 日本語
    "area_cluster": "east_near",       # エリア分類
    "distance_zone": "near"            # station/near/mid/far
}
```

#### 3.2.2 aggregator.py（集計・比較モジュール）

**主要機能:**

| 関数 | 機能 | 出力例 |
|------|------|--------|
| `compare_east_west()` | 東西比較 | "西側が268件多い（東287件 vs 西555件）" |
| `compare_categories()` | カテゴリ間比較 | "カフェ149件 vs バー147件" |
| `get_top_categories()` | ランキング生成 | "1位: レストラン367件" |
| `analyze_category_by_direction()` | 方向別分析 | カフェの東西分布 |

#### 3.2.3 structured_rag_system.py（統合システム）

**処理フロー:**

```
質問入力
    ↓
質問分析（analyze_question）
    ↓ タイプ判定: simple/comparison/aggregation/spatial
検索戦略選択
    ↓
┌─────────────────────────────────────┐
│ ベクトル検索 │ 集計処理 │ 比較処理 │
└─────────────────────────────────────┘
    ↓
コンテキスト構築
    ↓
LLM回答生成
```

### 3.3 POIデータの空間分布

構造化RAGにより、POIデータの空間分布が明確になりました：

**方向別分布（1,047件）:**

| 方向 | 件数 | 割合 |
|------|------|------|
| northwest（北西） | 278件 | 26.6% |
| west（西） | 241件 | 23.0% |
| north（北） | 156件 | 14.9% |
| northeast（北東） | 143件 | 13.7% |
| east（東） | 82件 | 7.8% |
| southeast（南東） | 62件 | 5.9% |
| south（南） | 49件 | 4.7% |
| southwest（南西） | 36件 | 3.4% |

**東西比較:**
- 東側（east + northeast + southeast）: **287件**
- 西側（west + northwest + southwest）: **555件**
- **西側が268件（約2倍）多い**

---

## 4. Phase 6.1 評価結果

### 4.1 全体結果

| 指標 | Phase 5 | Phase 6.1 | 改善 |
|------|---------|-----------|------|
| 全体平均スコア | 60.3pt | 69.6pt | +9.3pt |
| 最小スコア | - | 5.7pt | - |
| 最大スコア | - | 100.0pt | - |
| 平均処理時間 | 22.7秒 | 12.9秒 | -43% |

### 4.2 サブカテゴリ別比較

#### 大幅改善（+20pt以上）🎉

| サブカテゴリ | Phase 5 | Phase 6.1 | 改善 | 要因 |
|-------------|---------|-----------|------|------|
| constraint_multi | 53.3pt | 80.0pt | **+26.7pt** | 複数条件の論理処理向上 |
| basic_location | 71.7pt | 96.8pt | **+25.1pt** | 基本検索の精度向上 |
| spatial_comparison | 51.4pt | 76.0pt | **+24.6pt** | 東西比較の数値回答が可能に |
| basic_category | 58.3pt | 81.3pt | **+23.0pt** | カテゴリ検索の精度向上 |
| decision_business | 63.3pt | 83.5pt | **+20.2pt** | 集計データに基づく判断 |

#### 改善（+1pt〜+19pt）✅

| サブカテゴリ | Phase 5 | Phase 6.1 | 改善 |
|-------------|---------|-----------|------|
| advanced_comparison | 59.5pt | 64.4pt | +4.9pt |
| advanced_uncertainty | 66.7pt | 67.3pt | +0.6pt |

#### 悪化（-1pt以下）⚠️

| サブカテゴリ | Phase 5 | Phase 6.1 | 変化 | 原因分析 |
|-------------|---------|-----------|------|----------|
| advanced_sensitivity | 60.0pt | 48.7pt | **-11.3pt** | 感度分析に動的フィルタリング未対応 |
| spatial_proximity | 61.7pt | 54.8pt | **-6.9pt** | 「最も近い」判定に距離ソート未実装 |
| decision_location | 63.3pt | 57.6pt | **-5.7pt** | 複合的空間分析が不足 |
| spatial_density | 65.0pt | 62.8pt | -2.2pt | 密度計算の精緻化が必要 |
| constraint_single | 53.3pt | 51.3pt | -2.0pt | 単一制約のフィルタリング不十分 |

### 4.3 成功・失敗の分析

**成功パターン（構造化RAGが効果的）:**

1. **東西比較質問**: 「渋谷駅の東側と西側、どちらにカフェが多いですか？」
   - Phase 5: 回答不可（-8.9pt）
   - Phase 6.1: 「西側にカフェが多いです。西側には58件のカフェがあり、東側の51件よりも7件多い」

2. **カテゴリランキング**: 「渋谷駅周辺で最も多いPOIカテゴリは？」
   - Phase 5: 曖昧な回答
   - Phase 6.1: 「1位: レストラン367件、2位: カフェ149件、3位: バー147件」

**失敗パターン（追加機能が必要）:**

1. **近接性判断**: 「渋谷駅に最も近いコンビニは？」
   - 必要機能: 距離でソートして最近傍を取得する機能

2. **感度分析**: 「半径を500mから300mに変えても結論は成立しますか？」
   - 必要機能: 動的な半径フィルタリングと再計算

---

## 5. 残された課題

### 5.1 技術的課題

| 課題 | 影響サブカテゴリ | 必要な実装 |
|------|-----------------|-----------|
| **距離ソート未実装** | spatial_proximity | `get_nearest_pois()` 関数 |
| **動的半径フィルタ未実装** | advanced_sensitivity | `filter_by_radius()` 関数 |
| **複合空間分析不足** | decision_location | 複数カテゴリの空間分布統合 |
| **密度計算未実装** | spatial_density | エリア別密度計算 |

### 5.2 評価上の課題

| 課題 | 説明 |
|------|------|
| **評価関数の限界** | キーワードマッチングでは推論品質を十分に評価できない |
| **LLM-as-Judge未導入** | 高度な推論の評価には別LLMによる評価が必要 |
| **テストケースの網羅性** | 悪化したサブカテゴリの追加テストが必要 |

---

## 6. 次の改善計画

### 6.1 Phase 6.2: 近接性・距離機能の強化（推奨）

**目的**: spatial_proximity、advanced_sensitivityの改善

**実装内容:**

```python
# geo_utils.py に追加
def get_nearest_pois(pois, category=None, top_n=3):
    """駅に最も近いPOIを取得"""
    filtered = filter_by_category(pois, category) if category else pois
    sorted_pois = sorted(filtered, key=lambda p: p.get('distance_from_station', float('inf')))
    return sorted_pois[:top_n]

def filter_by_radius(pois, radius_m, category=None):
    """指定半径内のPOIをフィルタ"""
    filtered = filter_by_category(pois, category) if category else pois
    return [p for p in filtered if p.get('distance_from_station', float('inf')) <= radius_m]

def compare_by_radius(pois, category, radius1, radius2):
    """異なる半径での件数比較（感度分析用）"""
    count1 = len(filter_by_radius(pois, radius1, category))
    count2 = len(filter_by_radius(pois, radius2, category))
    return {"radius1": radius1, "count1": count1, "radius2": radius2, "count2": count2}
```

**期待改善:**
- spatial_proximity: -6.9pt → +5pt以上
- advanced_sensitivity: -11.3pt → +5pt以上

### 6.2 Phase 6.3: グラフRAG（オプション）

**目的**: POI間の関係性を活用した検索

**実装内容:**
- POI間の近接関係をグラフ化
- 「○○の近くにある△△」クエリへの対応
- エリア単位でのクラスタリング

### 6.3 Phase 6.4: ハイブリッド統合

**目的**: 各手法の最適な組み合わせ

**実装内容:**
- 質問タイプに応じた手法の自動選択
- ベクトル検索 + 構造化検索 + グラフ検索の統合
- 検索結果のランキング最適化

---

## 7. ロードマップ

```
Phase 6.1 [完了] 構造化RAG基盤実装
    │
    ├── geo_utils.py: 座標計算・方向判定 ✅
    ├── aggregator.py: 集計・比較機能 ✅
    └── structured_rag_system.py: 統合システム ✅
    │
    ▼
Phase 6.2 [次回] 近接性・距離機能強化
    │
    ├── get_nearest_pois(): 最近傍検索
    ├── filter_by_radius(): 半径フィルタ
    └── compare_by_radius(): 感度分析
    │
    ▼
Phase 6.3 [予定] グラフRAG実装
    │
    ▼
Phase 6.4 [予定] ハイブリッド統合
    │
    ▼
Phase 7 [予定] ファインチューニング
```

---

## 8. 結論

### 8.1 Phase 6.1の成果

1. **主要目標達成**: spatial_comparison を +24.6pt 改善（目標+5ptを大幅超過）
2. **全体性能向上**: 平均スコア +9.3pt、処理時間 -43%
3. **基盤構築完了**: 座標計算・集計・比較の基盤モジュールを実装

### 8.2 残された課題

1. **一部サブカテゴリの悪化**: spatial_proximity (-6.9pt)、advanced_sensitivity (-11.3pt)
2. **距離ソート・半径フィルタ未実装**: Phase 6.2で対応予定

### 8.3 次のアクション

1. **即時**: Phase 6.2の実装（距離ソート・半径フィルタ）
2. **短期**: 悪化したサブカテゴリの再評価
3. **中期**: グラフRAG・ハイブリッド統合の検討

---

**作成者**: Claude  
**プロジェクト**: experiments-local-llm  
**ドキュメントバージョン**: 1.0
