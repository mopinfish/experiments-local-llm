# Phase 6 構造化RAG改善レポート

**プロジェクト**: experiments-local-llm  
**期間**: 2026年1月  
**モデル**: Qwen/Qwen2.5-7B-Instruct  
**評価データ**: 渋谷駅周辺POI 1,047件、テストケース 55件（L1-L5）

---

## エグゼクティブサマリー

Phase 5のベースラインRAGシステム（60.3pt）から、構造化RAGアプローチの導入により**Phase 6.2.1で91.6pt（+31.3pt、52%向上）**を達成しました。全12サブカテゴリで改善を実現し、特に感度分析（advanced_sensitivity）では100.0ptの完全スコアを記録しました。

```
Phase 5:    60.3pt ─────────────────────────────────────
Phase 6.1:  69.6pt ████████████████ (+9.3pt)
Phase 6.2:  64.1pt ████████████ (-5.5pt) ※一時的な悪化
Phase 6.2.1: 91.6pt ████████████████████████████████████████████████ (+27.5pt)
```

---

## 1. Phase 5 ベースライン（60.3pt）

### システム構成

- **アーキテクチャ**: 単純なベクトル検索RAG
- **Embedding**: multilingual-e5-base
- **LLM**: Qwen2.5-7B-Instruct（4bit量子化）
- **ベクトルストア**: ChromaDB

### 課題

| サブカテゴリ | スコア | 主な問題点 |
|-------------|--------|-----------|
| spatial_comparison | 51.4pt | 東西比較ができない |
| constraint_single | 53.3pt | 距離制約の処理が不正確 |
| constraint_multi | 53.3pt | 複合条件の処理が困難 |
| basic_category | 58.3pt | カテゴリ集計が不正確 |
| advanced_comparison | 59.5pt | 複雑な比較推論が困難 |
| advanced_sensitivity | 60.0pt | 条件変更の影響分析ができない |

### 根本原因

1. **ベクトル検索の限界**: 意味的類似性のみで検索するため、空間的・数値的な質問に対応できない
2. **構造化データの未活用**: POIの座標、カテゴリ、距離情報が効果的に使用されていない
3. **集計機能の欠如**: 「東側と西側どちらが多い？」のような質問に回答できない

---

## 2. Phase 6.1 構造化RAG導入（69.6pt、+9.3pt）

### 実装した機能

#### 2.1 空間情報エンリッチメント（geo_utils.py）

```python
def enrich_poi_with_spatial_info(poi, station_coords):
    """POIに空間情報を追加"""
    # 駅からの距離を計算
    distance = calculate_distance(station_coords, (poi['lat'], poi['lon']))
    # 駅からの方角を計算（8方位）
    direction = calculate_direction(station_coords, (poi['lat'], poi['lon']))
    
    poi['distance_from_station'] = distance
    poi['direction_from_station'] = direction  # east, west, north, south, etc.
    return poi
```

**効果**: 全POIに`distance_from_station`と`direction_from_station`属性を付与し、空間クエリの基盤を構築。

#### 2.2 東西比較機能（aggregator.py）

```python
def compare_east_west(pois, category=None):
    """東側と西側のPOI数を比較"""
    east_dirs = ['east', 'northeast', 'southeast']
    west_dirs = ['west', 'northwest', 'southwest']
    
    east_count = count_by_directions(pois, east_dirs, category)
    west_count = count_by_directions(pois, west_dirs, category)
    
    return ComparisonResult(
        east_count=east_count,
        west_count=west_count,
        winner='east' if east_count > west_count else 'west',
        difference=abs(east_count - west_count)
    )
```

**効果**: 「渋谷駅の東側と西側、どちらにカフェが多いですか？」のような質問に正確に回答可能に。

#### 2.3 カテゴリ集計機能（aggregator.py）

```python
def get_top_categories(pois, n=5):
    """上位カテゴリをランキング"""
    category_counts = Counter(poi['category'] for poi in pois)
    return [CategoryCount(cat, count) for cat, count in category_counts.most_common(n)]

def filter_by_category(pois, category):
    """カテゴリでフィルタリング"""
    return [poi for poi in pois if category in poi.get('category', '')]
```

**効果**: カテゴリ別の集計と、特定カテゴリのPOI抽出が可能に。

#### 2.4 質問分析システム（structured_rag_system.py）

```python
@dataclass
class QuestionAnalysis:
    question_type: str  # simple, comparison, aggregation, spatial
    categories: List[str]
    subcategories: List[str]
    directions: List[str]
    requires_aggregation: bool
    requires_comparison: bool
    requires_spatial: bool
    distance_constraint: Optional[float]
```

**効果**: 質問の意図を解析し、適切な処理パスを選択。

### Phase 6.1 結果

| サブカテゴリ | Phase 5 | Phase 6.1 | 変化 | 改善要因 |
|-------------|---------|-----------|------|----------|
| **spatial_comparison** | 51.4pt | **76.0pt** | **+24.6pt** | 東西比較機能 |
| basic_location | 71.7pt | **96.8pt** | +25.1pt | 空間情報エンリッチメント |
| basic_category | 58.3pt | **81.3pt** | +23.0pt | カテゴリ集計機能 |
| constraint_multi | 53.3pt | **80.0pt** | +26.7pt | 複合条件処理 |
| decision_business | 63.3pt | **83.5pt** | +20.2pt | 集計データの活用 |

### Phase 6.1 で悪化したサブカテゴリ

| サブカテゴリ | Phase 5 | Phase 6.1 | 変化 | 原因 |
|-------------|---------|-----------|------|------|
| spatial_proximity | 61.7pt | 54.8pt | **-6.9pt** | 距離ソート機能の欠如 |
| advanced_sensitivity | 60.0pt | 48.7pt | **-11.3pt** | 感度分析機能の欠如 |

---

## 3. Phase 6.2 近接性・感度分析（64.1pt、-5.5pt）

### 実装した機能

#### 3.1 最近傍検索（geo_utils.py）

```python
def get_nearest_pois(pois, category=None, top_n=3, station=None):
    """駅に最も近いPOIを距離順で取得"""
    filtered = filter_by_category(pois, category) if category else pois
    sorted_pois = sorted(filtered, key=lambda p: p.get('distance_from_station', float('inf')))
    return sorted_pois[:top_n]

def generate_proximity_context(pois, category, top_n=5, station=None):
    """最近傍POI情報をLLMコンテキスト用に整形"""
    nearest = get_nearest_pois(pois, category, top_n, station)
    lines = [f"【{category}の最寄り{top_n}件】"]
    for i, poi in enumerate(nearest, 1):
        lines.append(f"{i}. {poi['name']} - {poi['distance_from_station']:.0f}m ({poi['direction_from_station']})")
    return "\n".join(lines)
```

**効果**: 「渋谷駅に最も近いコンビニは？」のような近接性クエリに正確に回答可能に。

#### 3.2 半径比較・感度分析（geo_utils.py）

```python
@dataclass
class RadiusComparisonResult:
    radius1_m: float
    radius2_m: float
    count1: int
    count2: int
    category: str
    difference: int
    ratio: float
    
    def to_japanese(self) -> str:
        return f"半径{self.radius1_m}m: {self.count1}件 → 半径{self.radius2_m}m: {self.count2}件 ({self.difference:+d}件、{self.ratio:.2f}倍)"

def compare_by_radius(pois, radius1_m, radius2_m, category=None, station=None):
    """2つの半径での件数を比較"""
    count1 = len(filter_by_radius(pois, radius1_m, category, station))
    count2 = len(filter_by_radius(pois, radius2_m, category, station))
    return RadiusComparisonResult(
        radius1_m=radius1_m,
        radius2_m=radius2_m,
        count1=count1,
        count2=count2,
        category=category,
        difference=count2 - count1,
        ratio=count2 / count1 if count1 > 0 else 0
    )
```

**効果**: 「半径を500mから300mに変えても結論は成立する？」のような感度分析クエリに対応。

#### 3.3 質問分析の拡張（structured_rag_system.py）

```python
PROXIMITY_KEYWORDS = ["最も近い", "一番近い", "最寄り", "近い順", "最短"]
SENSITIVITY_KEYWORDS = ["変えても", "変更しても", "広げても", "狭めても", "範囲を", "半径を", "成立"]

@dataclass
class QuestionAnalysis:
    # ... 既存フィールド ...
    requires_proximity: bool      # Phase 6.2追加
    requires_sensitivity: bool    # Phase 6.2追加
    sensitivity_radii: Optional[Tuple[float, float]]  # Phase 6.2追加
```

### Phase 6.2 の問題点

**コンテキスト構築ロジックの排他的分岐が原因で、基本機能が悪化**:

```python
# Phase 6.2の問題のあるコード
def _build_context(self, question, analysis):
    parts = []
    
    if analysis.requires_proximity and cat:
        parts.append(proximity_context)  # 近接性のみ
    elif analysis.requires_sensitivity and cat:
        parts.append(sensitivity_context)  # 感度分析のみ
    elif analysis.requires_comparison:
        parts.append(comparison_context)  # 比較のみ
    # ...
    
    # ベクトル検索はフォールバック（他の処理がない場合のみ）
    if not parts:
        parts.append(vector_search_context)
```

**問題**: 近接性や感度分析が優先された結果、基本的なベクトル検索が実行されず、座標情報や具体的なPOI情報が欠落。

### Phase 6.2 結果

| サブカテゴリ | Phase 6.1 | Phase 6.2 | 変化 | 状況 |
|-------------|-----------|-----------|------|------|
| **spatial_proximity** | 54.8pt | **68.4pt** | **+13.6pt** | ✅ 目標達成 |
| **advanced_sensitivity** | 48.7pt | **67.3pt** | **+18.6pt** | ✅ 目標達成 |
| basic_location | 96.8pt | 62.8pt | **-34.0pt** | ❌ 大幅悪化 |
| basic_category | 81.3pt | 51.3pt | **-30.0pt** | ❌ 大幅悪化 |
| decision_business | 83.5pt | 65.5pt | **-18.0pt** | ❌ 悪化 |

---

## 4. Phase 6.2.1 統合最適化（91.6pt、+27.5pt）

### 根本的な修正

**コンテキスト構築ロジックの変更**: ベクトル検索を「フォールバック」から「常に補完」に変更。

```python
# Phase 6.2.1の修正後コード
def _build_context(self, question, analysis):
    structured_parts = []  # 構造化データからのコンテキスト
    
    # === 構造化コンテキストの構築（複数が同時に追加可能） ===
    
    # Phase 6.2: 近接性検索
    if analysis.requires_proximity and cat:
        structured_parts.append(generate_proximity_context(...))
    
    # Phase 6.2: 感度分析
    if analysis.requires_sensitivity and cat:
        structured_parts.append(generate_sensitivity_context(...))
        # 結論も追加
        if comparison.ratio >= 1.5:
            structured_parts.append("結論は条件に依存します。")
        else:
            structured_parts.append("結論は大きく変わりません。")
    
    # Phase 6.1: 東西比較
    if analysis.requires_comparison and "東" in question and "西" in question:
        structured_parts.append(compare_east_west(...))
    
    # Phase 6.1: 集計
    if analysis.requires_aggregation:
        structured_parts.append(aggregation_context)
    
    # === ベクトル検索は常に追加（補完として） ===
    vector_context = self._get_vector_search_context(question, k=5)
    
    # 統合
    all_parts = structured_parts + [vector_context]
    return "\n".join(all_parts)
```

### 修正のポイント

1. **`elif`を`if`に変更**: 複数の構造化処理が同時に実行可能に
2. **ベクトル検索を常に実行**: 具体的なPOI情報と座標が常にコンテキストに含まれる
3. **プロンプトの改善**: 「具体的な数値や座標を含めて回答」を明示

### Phase 6.2.1 最終結果

| サブカテゴリ | Phase 5 | Phase 6.2 | Phase 6.2.1 | 総改善 |
|-------------|---------|-----------|-------------|--------|
| advanced_sensitivity | 60.0pt | 67.3pt | **100.0pt** | **+40.0pt** 🏆 |
| decision_location | 63.3pt | 59.2pt | **98.4pt** | **+35.1pt** |
| constraint_multi | 53.3pt | 70.0pt | **98.0pt** | **+44.7pt** |
| decision_business | 63.3pt | 65.5pt | **97.3pt** | **+34.0pt** |
| basic_location | 71.7pt | 62.8pt | **96.8pt** | **+25.1pt** |
| spatial_proximity | 61.7pt | 68.4pt | **95.7pt** | **+34.0pt** |
| spatial_comparison | 51.4pt | 64.0pt | **92.0pt** | **+40.6pt** |
| advanced_comparison | 59.5pt | 68.3pt | **90.8pt** | **+31.3pt** |
| constraint_single | 53.3pt | 70.0pt | **85.3pt** | **+32.0pt** |
| advanced_uncertainty | 66.7pt | 54.7pt | **82.0pt** | **+15.3pt** |
| basic_category | 58.3pt | 51.3pt | **81.3pt** | **+23.0pt** |
| spatial_density | 65.0pt | 66.4pt | **80.8pt** | **+15.8pt** |

---

## 5. 改善の因果関係マトリクス

### 各機能がどのサブカテゴリに寄与したか

| 実装機能 | 主な効果 | 寄与したサブカテゴリ |
|---------|---------|---------------------|
| **空間情報エンリッチメント** | POIに距離・方角を付与 | basic_location, spatial_* |
| **東西比較機能** | 方向別集計 | spatial_comparison (+40.6pt) |
| **カテゴリ集計機能** | カテゴリ別カウント | basic_category, decision_* |
| **最近傍検索** | 距離ソートPOI取得 | spatial_proximity (+34.0pt) |
| **感度分析** | 半径比較・結論生成 | advanced_sensitivity (+40.0pt) |
| **ベクトル検索の常時実行** | 座標・具体例の補完 | 全サブカテゴリ |

### 相乗効果の発生

```
構造化コンテキスト（集計・比較・分析）
        ↓
    + ベクトル検索結果（具体例・座標）
        ↓
    = 高品質な統合コンテキスト
        ↓
    → LLMが正確かつ具体的な回答を生成
```

---

## 6. 処理時間の推移

| Phase | 平均処理時間 | 変化 | 主な要因 |
|-------|------------|------|---------|
| Phase 6.1 | 12.9秒 | - | ベースライン |
| Phase 6.2 | 15.8秒 | +22% | 近接性・感度分析の追加 |
| Phase 6.2.1 | 21.9秒 | +70% | ベクトル検索の常時実行 |

**トレードオフ**: 処理時間は増加したが、スコアの大幅改善（+27.5pt）を考慮すると許容範囲。必要に応じて最適化可能。

---

## 7. 学んだ教訓

### 7.1 アーキテクチャ設計

1. **排他的分岐（elif）は危険**: 新機能追加時に既存機能を壊す可能性がある
2. **補完的アプローチが有効**: 構造化処理とベクトル検索は排他ではなく補完関係
3. **段階的な評価が重要**: 各フェーズで全サブカテゴリを評価し、悪化を早期発見

### 7.2 RAGシステム設計

1. **コンテキストは多層的に**: 集計データ + 具体例 + 座標情報 = 最適なコンテキスト
2. **質問分析の精度が重要**: 質問タイプの正確な識別が適切な処理パス選択の鍵
3. **プロンプトエンジニアリング**: 「座標を含めて回答」など、出力形式の明示が効果的

### 7.3 評価手法

1. **階層化テストケースが有効**: L1-L5の難易度別評価で問題の特定が容易
2. **サブカテゴリ別分析**: 全体スコアだけでなく、カテゴリ別の変化を追跡
3. **回帰テストの重要性**: 新機能が既存機能を壊していないか常に確認

---

## 8. 今後の展望

### 短期（Phase 7）

- **ファインチューニング**: 91.6ptをベースラインとして、モデル調整で95pt+を目指す
- **処理時間最適化**: キャッシュ導入、不要なベクトル検索の削減

### 中期

- **グラフRAG導入**: POI間の関係性（近接、同一建物内など）を活用
- **ハイブリッド検索**: キーワード検索とベクトル検索の組み合わせ

### 長期

- **MapFan MCP統合**: 本改善をMCPサーバーに適用
- **全国展開**: 渋谷以外のエリアへの適用

---

## 付録：ファイル構成

```
experiments-local-llm/
├── src/
│   ├── geo_utils.py           # 空間処理（距離計算、方角、近接性、感度分析）
│   ├── aggregator.py          # 集計処理（東西比較、カテゴリ集計）
│   ├── structured_rag_system.py  # 質問分析、コンテキスト構築
│   ├── test_cases_v2.py       # 55件のテストケース定義
│   └── __init__.py            # バージョン管理
├── notebooks/
│   └── phase6_full_evaluation.ipynb  # 評価用Notebook
├── data/
│   └── poi_documents.json     # POIデータ（1,047件）
└── docs/
    └── PHASE6_IMPROVEMENT_REPORT.md  # 本ドキュメント
```

---

**作成日**: 2026年1月23日  
**バージョン**: Phase 6.2.1  
**最終スコア**: 91.6pt（Phase 5比 +31.3pt、52%向上）
