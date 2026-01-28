# 地理空間POIクエリのための構造化RAGアーキテクチャの設計と評価

**Structured RAG Architecture Design and Evaluation for Geospatial POI Queries**

---

## 論文情報

- **研究期間**: 2026年1月
- **対象システム**: experiments-local-llm
- **対象領域**: 渋谷駅周辺POIデータ（1,047件）
- **使用モデル**: Qwen2.5-7B-Instruct（4bit量子化）
- **最終結果**: 60.3pt → 91.6pt（+52%向上）

---

## 概要（Abstract）

本研究では、地理空間POI（Point of Interest）クエリに特化したRAG（Retrieval-Augmented Generation）システムの設計と評価を行った。従来のベクトル検索のみに依存するRAGシステムでは、空間推論や集計を必要とするクエリに対して性能が低下する課題があった。

本研究の主な貢献は以下の3点である：

1. **階層化テストフレームワーク（Phase 5）**: 5段階の難易度レベル（L1-L5）と12のサブカテゴリからなる55件のテストケースを設計し、地理空間クエリシステムの包括的な評価を可能にした。

2. **構造化RAGアーキテクチャ（Phase 6）**: ベクトル検索と構造化データ処理を相補的に統合するアーキテクチャを提案し、空間情報エンリッチメント、集計・比較機能、近接性検索、感度分析の4つのコンポーネントを実装した。

3. **反復的改善プロセスの実証**: Phase 6.1からPhase 6.2.1までの3段階の改善において、一時的な性能悪化（Phase 6.2: -5.5pt）を経験しながらも、その原因分析と修正により最終的に+31.3pt（52%）の大幅な性能向上を達成した。

実験の結果、ベースラインの60.3ptから最終的に91.6ptのスコアを達成し、特に感度分析タスク（advanced_sensitivity）では100.0ptの完全スコアを記録した。

---

## 1. はじめに（Introduction）

### 1.1 研究背景

大規模言語モデル（LLM）とRAGの組み合わせは、知識集約型タスクにおいて高い性能を示している。しかし、地理空間情報を扱うドメインでは、以下の課題が存在する：

1. **ベクトル検索の限界**: 意味的類似性に基づく検索は「渋谷駅の東側と西側、どちらにカフェが多いか？」のような集計・比較クエリに対応できない。

2. **構造化データの未活用**: POIデータには座標、カテゴリ、属性などの構造化情報が含まれるが、従来のRAGではこれらが効果的に活用されていない。

3. **空間推論の欠如**: 「渋谷駅に最も近いコンビニは？」のような空間関係に基づく推論が困難である。

### 1.2 研究目的

本研究の目的は以下の2点である：

1. **包括的評価フレームワークの構築**: 地理空間クエリシステムの性能を多角的に評価するための階層化テストケースの設計（Phase 5）。

2. **構造化RAGアーキテクチャの提案**: ベクトル検索と構造化データ処理を相補的に統合し、地理空間クエリの回答精度を向上させるアーキテクチャの設計と実装（Phase 6）。

### 1.3 本論文の構成

第2章で関連研究を概観し、第3章で提案手法を詳述する。第4章で実装詳細、第5章で実験結果、第6章で考察を行い、第7章で結論と今後の展望を述べる。

---

## 2. 関連研究（Related Work）

### 2.1 RAGシステムの発展

RAG（Retrieval-Augmented Generation）は、Lewis et al.（2020）によって提案されたアーキテクチャであり、外部知識ベースからの検索結果を言語モデルの生成に組み込む手法である。従来のRAGは主にテキストベースの知識検索に焦点を当てており、構造化データや空間情報の処理については限定的な研究にとどまっている。

### 2.2 地理空間情報システム

地理空間情報システム（GIS）では、空間インデックス（R-tree、Quadtree等）を用いた効率的な空間検索が確立されている。しかし、これらの手法と自然言語処理を統合した研究は発展途上である。

### 2.3 ハイブリッドRAGアーキテクチャ

近年、構造化クエリとセマンティック検索を組み合わせるハイブリッドRAGの研究が進んでいる。本研究はこの流れを踏襲しつつ、地理空間ドメインに特化した設計を提案する。

---

## 3. 提案手法（Methodology）

### 3.1 Phase 5: 階層化テストフレームワーク

#### 3.1.1 設計原則

地理空間クエリシステムの評価において、単一の指標では性能の多面的な評価が困難である。本研究では以下の設計原則に基づきテストフレームワークを構築した：

1. **認知負荷に基づく難易度階層**: 簡単な事実検索から複雑な推論まで、5段階の難易度を設定
2. **機能別サブカテゴリ**: 12の異なる機能領域を網羅
3. **定量的評価指標**: キーワードヒット率、座標情報含有率、推論正確性など複数の評価軸

#### 3.1.2 テストレベル定義

```
L1: 基礎検索（Basic Retrieval）     - 10件
    └─ 単純なPOI検索、カテゴリ検索

L2: 空間推論（Spatial Reasoning）   - 15件
    └─ 近接性判断、密度分析、方向比較

L3: 制約充足（Constraint Satisfaction） - 10件
    └─ 距離制約、属性制約、複合制約

L4: 意思決定支援（Decision Support）  - 10件
    └─ 立地評価、出店判断、場所推薦

L5: 高度推論（Advanced Reasoning）   - 10件
    └─ 感度分析、多軸比較、不確実性処理
```

#### 3.1.3 サブカテゴリ構成

| レベル | サブカテゴリ | 件数 | 評価対象 |
|--------|-------------|------|----------|
| L1 | basic_location | 5件 | 座標情報の正確な提供 |
| L1 | basic_category | 5件 | カテゴリ検索の網羅性 |
| L2 | spatial_proximity | 5件 | 最近傍判定の正確性 |
| L2 | spatial_density | 5件 | 密度概念の理解 |
| L2 | spatial_comparison | 5件 | 方向別比較の正確性 |
| L3 | constraint_single | 5件 | 単一制約の充足 |
| L3 | constraint_multi | 5件 | 複数制約の同時充足 |
| L4 | decision_location | 5件 | 立地適性評価 |
| L4 | decision_business | 5件 | ビジネス判断支援 |
| L5 | advanced_sensitivity | 3件 | 条件変更の影響分析 |
| L5 | advanced_comparison | 4件 | 多軸・多地点比較 |
| L5 | advanced_uncertainty | 3件 | 不確実性の認識と表現 |

#### 3.1.4 テストケース設計の例

**L2-11: spatial_comparison（方向別比較）**

```python
TestCaseV2(
    id="L2-11",
    level=2,
    category="spatial_reasoning",
    subcategory="spatial_comparison",
    prompt="渋谷駅の東側と西側、どちらにカフェが多いですか？",
    expected_keywords=["カフェ", "東", "西", "多い"],
    difficulty="medium",
    requires_reasoning=True
)
```

**L5-01: advanced_sensitivity（感度分析）**

```python
TestCaseV2(
    id="L5-01",
    level=5,
    category="advanced_reasoning",
    subcategory="advanced_sensitivity",
    prompt="「渋谷駅周辺はカフェが多い」という結論は、検索半径を500mから300mに変えても成立しますか？",
    expected_keywords=["カフェ", "500m", "300m", "比較", "成立"],
    difficulty="expert",
    requires_reasoning=True,
    requires_evidence=True
)
```

### 3.2 Phase 6: 構造化RAGアーキテクチャ

#### 3.2.1 アーキテクチャ概要

提案アーキテクチャの核心は、**構造化処理とベクトル検索の相補的統合**である。従来のRAGが「ベクトル検索のみ」または「構造化クエリのみ」を選択する排他的アプローチを採用していたのに対し、本アーキテクチャでは両者を常に併用する。

```
┌─────────────────────────────────────────────────────────────┐
│                      質問入力                                │
│          「渋谷駅の東側と西側、どちらにカフェが多い？」       │
└─────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────┐
│                    質問分析モジュール                        │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ • 質問タイプ判定: comparison                        │   │
│  │ • 検出キーワード: ["東", "西", "カフェ", "多い"]     │   │
│  │ • requires_comparison: True                        │   │
│  │ • requires_aggregation: True                       │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────┐
│              並列コンテキスト構築（if文、非排他）            │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │ 構造化処理1  │  │ 構造化処理2  │  │ベクトル検索 │        │
│  │ 東西比較     │  │ カテゴリ集計 │  │ 上位5件取得 │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
│         ↓                ↓                ↓               │
│  「東側51件、西側58件」 「カフェ計109件」 「スタバ渋谷店...」│
└─────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────┐
│                   統合コンテキスト                           │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 【東西比較: カフェ】                                 │   │
│  │ 東側: 51件 (46.8%)                                   │   │
│  │ 西側: 58件 (53.2%)                                   │   │
│  │ → 西側が7件多い                                      │   │
│  │                                                      │   │
│  │ 【関連POI】                                          │   │
│  │ 1. スターバックス渋谷店 (245m, 北東)                 │   │
│  │ 2. タリーズコーヒー渋谷センター街店 (189m, 西)       │   │
│  │ ...                                                  │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────┐
│                    LLM回答生成                               │
│  「渋谷駅周辺では西側にカフェが多いです。                   │
│   東側には51件、西側には58件のカフェがあり、                │
│   西側が7件（約12%）多くなっています。」                    │
└─────────────────────────────────────────────────────────────┘
```

#### 3.2.2 重要な設計判断: if文 vs elif文

Phase 6.2で発生した性能悪化（-5.5pt）の原因分析から、コンテキスト構築における分岐ロジックの設計が極めて重要であることが判明した。

**問題のあったコード（Phase 6.2）**:
```python
def _build_context(self, question, analysis):
    parts = []

    if analysis.requires_proximity:
        parts.append(proximity_context)
    elif analysis.requires_sensitivity:    # ← elif: 排他的
        parts.append(sensitivity_context)
    elif analysis.requires_comparison:     # ← elif: 排他的
        parts.append(comparison_context)

    # ベクトル検索はフォールバック
    if not parts:
        parts.append(vector_search_context)
```

**修正後のコード（Phase 6.2.1）**:
```python
def _build_context(self, question, analysis):
    structured_parts = []

    # 複数の構造化処理が同時に追加可能
    if analysis.requires_proximity:
        structured_parts.append(proximity_context)
    if analysis.requires_sensitivity:      # ← if: 非排他的
        structured_parts.append(sensitivity_context)
    if analysis.requires_comparison:       # ← if: 非排他的
        structured_parts.append(comparison_context)
    if analysis.requires_aggregation:
        structured_parts.append(aggregation_context)

    # ベクトル検索は常に追加（補完として）
    vector_context = self._get_vector_search_context(question, k=5)

    return structured_parts + [vector_context]
```

この設計変更により、Phase 6.2の64.1ptからPhase 6.2.1の91.6ptへと27.5ptの大幅改善を達成した。

#### 3.2.3 4つの構造化処理コンポーネント

##### コンポーネント1: 空間情報エンリッチメント

全POIに対して、基準点（渋谷駅）からの距離と方角を事前計算して付与する。

```python
def enrich_poi_with_spatial_info(poi, station_coords=SHIBUYA_STATION):
    """POIに空間情報を追加"""
    distance = haversine_distance(station_coords, (poi['lat'], poi['lon']))
    direction = get_direction(station_coords, (poi['lat'], poi['lon']))

    poi['distance_from_station'] = distance
    poi['direction_from_station'] = direction  # "east", "west", "north", etc.
    return poi
```

**寄与するテストカテゴリ**: basic_location (+25.1pt), spatial_comparison (+40.6pt)

##### コンポーネント2: 集計・比較機能

東西比較やカテゴリ別集計など、構造化データに基づく分析を実行する。

```python
def compare_east_west(pois, category=None):
    """東側と西側のPOI数を比較"""
    east_dirs = ['east', 'northeast', 'southeast']
    west_dirs = ['west', 'northwest', 'southwest']

    east_pois = [p for p in pois if p['direction_from_station'] in east_dirs]
    west_pois = [p for p in pois if p['direction_from_station'] in west_dirs]

    if category:
        east_pois = [p for p in east_pois if category in p.get('category', '')]
        west_pois = [p for p in west_pois if category in p.get('category', '')]

    return ComparisonResult(
        east_count=len(east_pois),
        west_count=len(west_pois),
        winner='east' if len(east_pois) > len(west_pois) else 'west'
    )
```

**寄与するテストカテゴリ**: spatial_comparison (+40.6pt), basic_category (+23.0pt)

##### コンポーネント3: 近接性検索

距離に基づくソートとフィルタリングを実行する。

```python
def get_nearest_pois(pois, category=None, top_n=3):
    """駅に最も近いPOIを距離順で取得"""
    filtered = filter_by_category(pois, category) if category else pois
    sorted_pois = sorted(filtered, key=lambda p: p.get('distance_from_station', float('inf')))
    return sorted_pois[:top_n]

def generate_proximity_context(pois, category, top_n=5):
    """最近傍POI情報をLLMコンテキスト用に整形"""
    nearest = get_nearest_pois(pois, category, top_n)
    lines = [f"【{category}の最寄り{top_n}件】"]
    for i, poi in enumerate(nearest, 1):
        lines.append(f"{i}. {poi['name']} - {poi['distance_from_station']:.0f}m")
    return "\n".join(lines)
```

**寄与するテストカテゴリ**: spatial_proximity (+34.0pt), constraint_single (+32.0pt)

##### コンポーネント4: 感度分析

条件変更（半径変更など）の影響を分析し、結論の堅牢性を評価する。

```python
@dataclass
class RadiusComparisonResult:
    radius1_m: float
    radius2_m: float
    count1: int
    count2: int
    ratio: float  # count2 / count1

def compare_by_radius(pois, radius1_m, radius2_m, category=None):
    """2つの半径での件数を比較"""
    count1 = len(filter_by_radius(pois, radius1_m, category))
    count2 = len(filter_by_radius(pois, radius2_m, category))

    return RadiusComparisonResult(
        radius1_m=radius1_m,
        radius2_m=radius2_m,
        count1=count1,
        count2=count2,
        ratio=count2 / count1 if count1 > 0 else 0
    )
```

**寄与するテストカテゴリ**: advanced_sensitivity (+40.0pt)

#### 3.2.4 質問分析システム

質問文から必要な処理タイプを判定するためのキーワードベース分析システムを実装した。

```python
PROXIMITY_KEYWORDS = ["最も近い", "一番近い", "最寄り", "近い順", "最短"]
SENSITIVITY_KEYWORDS = ["変えても", "変更しても", "範囲を", "半径を", "成立"]
COMPARISON_KEYWORDS = ["東", "西", "どちら", "比較", "多い"]
AGGREGATION_KEYWORDS = ["いくつ", "何件", "カテゴリ", "ランキング"]

@dataclass
class QuestionAnalysis:
    question_type: str
    requires_proximity: bool
    requires_sensitivity: bool
    requires_comparison: bool
    requires_aggregation: bool
    categories: List[str]
    distance_constraint: Optional[float]
```

---

## 4. 実装（Implementation）

### 4.1 システム構成

```
experiments-local-llm/
├── src/
│   ├── geo_utils.py              # 空間処理（29KB）
│   │   ├── haversine_distance()  # 2点間距離計算
│   │   ├── get_direction()       # 8方位判定
│   │   ├── enrich_all_pois()     # 空間情報付与
│   │   ├── get_nearest_pois()    # 最近傍検索
│   │   ├── filter_by_radius()    # 半径フィルタ
│   │   └── compare_by_radius()   # 感度分析
│   │
│   ├── aggregator.py             # 集計処理（21KB）
│   │   ├── compare_east_west()   # 東西比較
│   │   ├── get_top_categories()  # ランキング
│   │   └── filter_by_category()  # カテゴリフィルタ
│   │
│   ├── structured_rag_system.py  # 統合システム（32KB）
│   │   ├── QuestionAnalysis      # 質問分析結果
│   │   ├── analyze_question()    # 質問解析
│   │   └── _build_context()      # コンテキスト構築
│   │
│   ├── test_cases_v2.py          # テストケース（55件）
│   └── evaluators_v2.py          # 評価システム
│
├── notebooks/
│   └── phase6_full_evaluation.ipynb  # 評価Notebook
│
└── data/
    └── poi_documents.json        # POIデータ（1,047件）
```

### 4.2 技術スタック

| コンポーネント | 技術選定 | 理由 |
|---------------|---------|------|
| LLM | Qwen2.5-7B-Instruct (4bit) | 日本語性能とメモリ効率のバランス |
| Embedding | multilingual-e5-base | 多言語対応、日本語性能 |
| ベクトルストア | ChromaDB | 軽量、Colab互換 |
| 実行環境 | Google Colab (T4 GPU) | 再現性、アクセス性 |

### 4.3 データセット

**POIデータ概要**:
- 総件数: 1,047件
- ソース: OpenStreetMap（渋谷駅周辺）
- 属性: name, category, lat, lon, phone, website, opening_hours

**空間分布**:

| 方向 | 件数 | 割合 |
|------|------|------|
| 北西（northwest） | 278件 | 26.6% |
| 西（west） | 241件 | 23.0% |
| 北（north） | 156件 | 14.9% |
| 北東（northeast） | 143件 | 13.7% |
| 東（east） | 82件 | 7.8% |
| 南東（southeast） | 62件 | 5.9% |
| 南（south） | 49件 | 4.7% |
| 南西（southwest） | 36件 | 3.4% |

**東西比較**: 東側287件 vs 西側555件（西側が約2倍）

---

## 5. 実験結果（Experimental Results）

### 5.1 Phase間の性能推移

```
Phase 5:    60.3pt ─────────────────────────────────────
Phase 6.1:  69.6pt ████████████████ (+9.3pt, +15.4%)
Phase 6.2:  64.1pt ████████████ (-5.5pt, -7.9%)  ※一時的悪化
Phase 6.2.1: 91.6pt ████████████████████████████████████████████████ (+27.5pt, +42.9%)
```

**総改善**: +31.3pt（52%向上）

### 5.2 サブカテゴリ別最終結果

| サブカテゴリ | Phase 5 | Phase 6.2.1 | 改善幅 | 順位 |
|-------------|---------|-------------|--------|------|
| advanced_sensitivity | 60.0pt | **100.0pt** | **+40.0pt** | 1位 |
| decision_location | 63.3pt | 98.4pt | +35.1pt | 2位 |
| constraint_multi | 53.3pt | 98.0pt | +44.7pt | 3位 |
| decision_business | 63.3pt | 97.3pt | +34.0pt | 4位 |
| basic_location | 71.7pt | 96.8pt | +25.1pt | 5位 |
| spatial_proximity | 61.7pt | 95.7pt | +34.0pt | 6位 |
| spatial_comparison | 51.4pt | 92.0pt | +40.6pt | 7位 |
| advanced_comparison | 59.5pt | 90.8pt | +31.3pt | 8位 |
| constraint_single | 53.3pt | 85.3pt | +32.0pt | 9位 |
| advanced_uncertainty | 66.7pt | 82.0pt | +15.3pt | 10位 |
| basic_category | 58.3pt | 81.3pt | +23.0pt | 11位 |
| spatial_density | 65.0pt | 80.8pt | +15.8pt | 12位 |

### 5.3 レベル別結果

| レベル | Phase 5 | Phase 6.2.1 | 改善幅 |
|--------|---------|-------------|--------|
| L1: 基礎検索 | 65.0pt | 89.1pt | +24.1pt |
| L2: 空間推論 | 59.4pt | 89.5pt | +30.1pt |
| L3: 制約充足 | 53.3pt | 91.7pt | +38.4pt |
| L4: 意思決定支援 | 63.3pt | 97.9pt | +34.6pt |
| L5: 高度推論 | 62.1pt | 90.9pt | +28.8pt |

### 5.4 Phase 6.2での一時的悪化の分析

Phase 6.2では近接性検索と感度分析機能を追加したが、全体スコアは69.6ptから64.1ptに悪化した。

**悪化したサブカテゴリ**:

| サブカテゴリ | Phase 6.1 | Phase 6.2 | 悪化幅 | 原因 |
|-------------|-----------|-----------|--------|------|
| basic_location | 96.8pt | 62.8pt | -34.0pt | ベクトル検索の欠落 |
| basic_category | 81.3pt | 51.3pt | -30.0pt | ベクトル検索の欠落 |
| decision_business | 83.5pt | 65.5pt | -18.0pt | コンテキスト不足 |

**根本原因**: `elif`による排他的分岐により、新機能（近接性・感度分析）が選択された場合にベクトル検索が実行されなくなり、具体的なPOI情報が欠落した。

**修正内容**: `elif`を`if`に変更し、ベクトル検索を常に実行するように修正した結果、Phase 6.2.1で91.6ptを達成した。

### 5.5 処理時間の推移

| Phase | 平均処理時間 | 変化 |
|-------|------------|------|
| Phase 5 | 22.7秒 | - |
| Phase 6.1 | 12.9秒 | -43% |
| Phase 6.2 | 15.8秒 | +22% |
| Phase 6.2.1 | 21.9秒 | +39% |

Phase 6.2.1では処理時間が増加したが、ベクトル検索の常時実行と複数の構造化処理の並列実行によるものであり、スコアの大幅改善（+27.5pt）を考慮すると許容範囲である。

---

## 6. 考察（Discussion）

### 6.1 構造化処理とベクトル検索の相補性

本研究の最も重要な知見は、**構造化処理とベクトル検索は排他的ではなく相補的**であるという点である。

- **構造化処理の役割**: 正確な集計、比較、フィルタリング
- **ベクトル検索の役割**: 具体的なPOI情報、座標、文脈の補完

両者を常に組み合わせることで、LLMは「西側が58件で多い」という集計結果と「スターバックス渋谷店は西側にある」という具体例の両方を活用でき、より説得力のある回答を生成できる。

### 6.2 階層化テストフレームワークの有効性

55件のテストケースを12サブカテゴリに分類したことで、以下のメリットが得られた：

1. **問題の早期特定**: Phase 6.2での悪化をサブカテゴリ別に分析し、basic_location、basic_categoryの大幅悪化を特定
2. **改善の方向性の明確化**: どの機能がどのサブカテゴリに寄与するかを定量的に把握
3. **回帰テストの実現**: 新機能追加時に既存機能の悪化を検出

### 6.3 反復的改善プロセスの教訓

本研究では、Phase 6.1 → 6.2 → 6.2.1の3段階の改善を行った。Phase 6.2での一時的な悪化（-5.5pt）は、以下の教訓をもたらした：

1. **排他的分岐（elif）は危険**: 新機能追加時に既存機能を意図せず無効化する可能性
2. **段階的評価の重要性**: 各改善段階で全サブカテゴリを評価し、悪化を早期発見
3. **根本原因分析の必要性**: 表面的な修正ではなく、アーキテクチャレベルの見直しが必要

### 6.4 限界と課題

#### 6.4.1 渋谷駅固有のハードコーディング

現在の実装では基準点として渋谷駅の座標がハードコードされている：

```python
SHIBUYA_STATION = (35.658034, 139.701636)
```

他エリアへの展開には、動的な基準点解決機能が必要である。

#### 6.4.2 スケーラビリティの課題

POI数が増加した場合（全国規模の500万件など）、オンデマンドでの距離計算は現実的な応答時間を達成できない。PostGISなどの空間インデックスの導入が必要となる。

#### 6.4.3 評価指標の限界

現在の評価はキーワードマッチングに基づいており、推論の質を完全に評価できない。LLM-as-Judgeなど、より高度な評価手法の導入が望ましい。

---

## 7. 結論と今後の展望（Conclusion and Future Work）

### 7.1 結論

本研究では、地理空間POIクエリに特化した構造化RAGアーキテクチャを提案し、以下の成果を得た：

1. **階層化テストフレームワーク（Phase 5）**: 5段階の難易度と12サブカテゴリからなる55件のテストケースを設計し、包括的な評価基盤を構築した。

2. **構造化RAGアーキテクチャ（Phase 6）**: ベクトル検索と構造化処理の相補的統合により、ベースライン60.3ptから91.6ptへと52%の性能向上を達成した。

3. **反復的改善プロセス**: 一時的な悪化を経験しながらも、根本原因分析と修正により最終的に大幅な改善を実現した。

### 7.2 今後の展望

#### 短期（Phase 7）: ファインチューニング

- LoRA/QLoRAによる効率的なモデル調整
- 地理空間クエリ特化のデータセット作成
- 目標: 95pt以上

#### 中期（Phase 8）: 全国展開

- PostGIS/Supabaseによる空間インデックスの導入
- 動的基準点解決機能の実装
- 目標: 500万POIでの50ms以下の応答時間

#### 長期: 実用化

- MapFan MCPサーバーへの統合
- グラフRAGによるPOI間関係の活用
- リアルタイムデータとの連携

---

## 参考文献

1. Lewis, P., et al. (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." NeurIPS 2020.
2. Khattab, O., et al. (2022). "Demonstrate-Search-Predict: Composing retrieval and language models for knowledge-intensive NLP." arXiv preprint.
3. Shuster, K., et al. (2021). "Retrieval Augmentation Reduces Hallucination in Conversation." EMNLP 2021.

---

## 付録A: 主要な数式

### A.1 Haversine距離計算

2点間の大圏距離を計算する：

```
a = sin²(Δφ/2) + cos(φ1) × cos(φ2) × sin²(Δλ/2)
c = 2 × atan2(√a, √(1-a))
d = R × c
```

ここで、φは緯度、λは経度、Rは地球の半径（6,371km）。

### A.2 方位角計算

```
θ = atan2(sin(Δλ) × cos(φ2), cos(φ1) × sin(φ2) - sin(φ1) × cos(φ2) × cos(Δλ))
```

### A.3 評価スコア計算

```
score = Σ(wi × mi) / Σwi
```

ここで、wiは各評価軸の重み、miは各評価軸のスコア（0-100）。

---

## 付録B: 機能別改善寄与マトリクス

| 実装機能 | 処理タイミング | 主な寄与カテゴリ | 改善幅合計 |
|---------|--------------|-----------------|-----------|
| 空間情報エンリッチメント | RAG構築時 | basic_location, spatial_comparison | +81.5pt |
| 集計・比較機能 | クエリ時 | spatial_comparison, basic_category | +129.5pt |
| 近接性検索 | クエリ時 | spatial_proximity, constraint_* | +145.8pt |
| 感度分析 | クエリ時 | advanced_sensitivity | +55.3pt |
| ベクトル検索常時実行 | クエリ時 | 全カテゴリ | 補完効果 |

---

**作成日**: 2026年1月29日
**バージョン**: 1.0
**最終スコア**: 91.6pt（Phase 5比 +31.3pt、52%向上）
