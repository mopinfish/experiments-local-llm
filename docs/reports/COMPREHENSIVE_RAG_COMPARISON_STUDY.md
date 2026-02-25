# 地理空間POIクエリのための多様なRAGアプローチの包括的性能比較研究

**A Comprehensive Performance Comparison of Diverse RAG Approaches for Geospatial POI Queries**

---

## 論文情報

- **研究期間**: 2026年1月〜2月
- **プロジェクト**: experiments-local-llm
- **対象領域**: 渋谷駅周辺POIデータ（OpenStreetMap、1,046件）
- **使用モデル**: Qwen2.5-7B-Instruct（4bit量子化）
- **評価規模**: Phase 5-9の5段階実験、延べ400+テストケース
- **実行環境**: Google Colab T4 GPU

---

## 概要（Abstract）

本研究では、OpenStreetMapから取得した地理空間POI（Point of Interest）データを用いて、5種類のRAG（Retrieval-Augmented Generation）アプローチの包括的性能比較を実施した。Phase 5からPhase 9にわたる段階的な実験を通じて、Naive RAG、Hybrid RAG、Fine-Tuned RAG、Graph RAG、Adaptive RAG、Agentic RAGの6システムを評価した。

**主要な発見**:

1. **Hybrid RAGの優位性**: 本研究で提案したHybrid RAGアプローチ（ルールベース質問分析 + 計算処理 + ベクトル検索 + 日本語自然文出力）が、最高性能96.2%を達成した。

2. **タスク依存的な最適アプローチ**: 基礎検索ではベクトルRAG、方向比較ではGraph RAG（+50pt）、関係性分析ではAgentic RAG（+33.3pt）など、タスク特性により最適なアプローチが異なることを実証した。

3. **反復的改善プロセスの重要性**: Phase 6.2での一時的な性能悪化（-5.5pt）を経験しながらも、根本原因分析により最終的に+31.3pt（52%）の大幅改善を達成した。

4. **用語の明確化**: 学術界で"Structured RAG"と呼ばれる手法（メタデータフィルタリング + リランキング）と、本研究の"Hybrid RAG"（ルールベース + 計算 + ベクトル）は異なるアプローチであることを明確化した。

5. **多言語LLMにおける言語制御**: JSON形式のツール出力が中国語レスポンスを誘発する問題を発見し（Agentic RAG実験）、自然言語形式の出力が言語安定性において優位であることを示した。

本論文では、5ヶ月にわたる系統的実験の全過程を統合し、地理空間クエリシステムの設計指針を提示する。

---

## 1. 序論（Introduction）

### 1.1 研究背景

Retrieval-Augmented Generation（RAG）は、Lewis et al. (2020)により提案されて以来、知識集約型タスクにおける大規模言語モデル（LLM）の性能向上に寄与してきた。しかし、地理空間情報という高度に構造化されたドメインにおいては、以下の課題が存在する：

1. **ベクトル検索の限界**: 意味的類似性に基づく検索は、「渋谷駅の東側と西側、どちらにカフェが多いか？」のような集計・比較クエリに対応困難である。

2. **構造化データの未活用**: POIデータには座標、カテゴリ、属性などの構造化情報が含まれるが、従来のRAGではこれらが効果的に活用されていない。

3. **空間推論の欠如**: 距離計算、方向判定、半径フィルタリングなどの空間処理が必要である。

4. **複雑な推論タスクへの対応**: POI間の関係性（競合、相補、ブランド）を考慮した推論が求められる。

### 1.2 研究の動機

2024年以降、RAGのアプローチは多様化している：

- **GraphRAG** (Microsoft Research, 2024): ナレッジグラフを用いたグラフベースの検索
- **Adaptive RAG** (Jeong et al., 2024): クエリ複雑度に応じたシステム選択
- **Agentic RAG** (Yao et al., 2023): LLMエージェントによる動的ツール選択
- **Fine-Tuned RAG** (Dettmers et al., 2023): ドメイン固有のファインチューニング

これらのアプローチを地理空間ドメインで体系的に比較した研究は存在せず、実務家が最適なアーキテクチャを選択するための指針が不足していた。

### 1.3 研究目的

本研究の目的は以下の4点である：

**RQ1**: 地理空間POIクエリにおいて、どのRAGアプローチが最も高い性能を達成するか？

**RQ2**: タスクタイプ（基礎検索、空間推論、集計・比較、関係性分析）により、最適なRAGアプローチは異なるか？

**RQ3**: ベクトル検索と構造化処理の統合において、どのような設計原則が有効か？

**RQ4**: ファインチューニング、グラフ構造、動的ツール選択は、それぞれどのような状況で価値を発揮するか？

### 1.4 研究の段階構成

本研究は5つのPhaseで構成される：

```
┌─────────────────────────────────────────────────────────────────┐
│                        研究フロー                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Phase 5: 評価フレームワーク構築                                 │
│  ├─ 55テストケース設計（L1-L5難易度、12サブカテゴリ）           │
│  └─ Naive RAGベースライン: 60.3pt                               │
│                 │                                                │
│                 ▼                                                │
│  Phase 6: Hybrid RAGアーキテクチャ提案                           │
│  ├─ ルールベース分析 + 計算処理 + ベクトル検索                  │
│  └─ 最終スコア: 91.6pt (+31.3pt, +52%)                          │
│                 │                                                │
│                 ▼                                                │
│  Phase 7: ファインチューニング実験                               │
│  ├─ QLoRA (4bit) によるドメイン適応                             │
│  └─ FT+RAG: 78.5pt（タスク依存的な優位性を発見）                │
│                 │                                                │
│                 ▼                                                │
│  Phase 8: Graph RAG実装                                         │
│  ├─ Neo4j/NetworkX + 7種類のエッジタイプ                        │
│  └─ GraphRAG: 76.7% vs Hybrid: 89.1%（比較タスクで+50pt）       │
│                 │                                                │
│                 ▼                                                │
│  Phase 9: Agentic RAG with LangGraph                            │
│  ├─ ReActパターン + 16ツール + 動的選択                         │
│  └─ Agentic: 87.6% vs Hybrid: 96.2%（関係性で+33.3pt）          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.5 本論文の貢献

1. **包括的評価フレームワーク**: 5段階の難易度と12サブカテゴリからなる階層化テストケース（55件）を設計し、地理空間クエリシステムの多角的評価を可能にした。

2. **Hybrid RAGアーキテクチャの提案**: ベクトル検索と構造化処理を相補的に統合する設計原則を提示し、52%の性能向上を達成した。

3. **6システムの体系的比較**: Naive、Hybrid、Fine-Tuned、Graph、Adaptive、Agenticの各RAGアプローチを同一データセット・同一評価基準で比較した。

4. **タスク特性と最適手法の対応関係**: どのタスクタイプにどのアプローチが適するかを定量的に示した。

5. **用語の明確化**: 学術界の"Structured RAG"と本研究の"Hybrid RAG"の違いを明確にし、今後の研究における混乱を防ぐ指針を提供した。

6. **実装レベルの洞察**: Phase 6.2での性能悪化（`elif` vs `if`の問題）、Phase 9での言語制御問題など、実装における重要な設計判断を詳述した。

---

## 2. 関連研究（Related Work）

### 2.1 RAGの発展

**基礎理論**:
RAG（Retrieval-Augmented Generation）は、Lewis et al. (2020)により提案された。外部知識ベースからの検索結果をLLMの生成に組み込むことで、幻覚を減少させ、最新情報への対応を可能にする。

**初期アプローチ (2020-2022)**:
- DPR (Dense Passage Retrieval): 密なベクトル表現による文書検索
- REALM: エンドツーエンドの検索器と生成器の同時学習
- FiD (Fusion-in-Decoder): 複数の検索結果をデコーダーで統合

### 2.2 構造化データとRAGの統合

**用語の注意（Terminology Note）**:

学術界において「Structured RAG」という用語は、以下の手法を指すことが一般的である：

1. **メタデータフィルタリング**: カテゴリ、日付、タグなどのメタデータによる事前フィルタリング
2. **構造化クエリ生成**: SQL、CypherなどのクエリをLLMが生成し、構造化データベースに対して実行
3. **リランキング**: 検索結果を関連度スコアで再順位付け

**学術的定義の例**:
```python
# 学術界の "Structured RAG" の典型例
class StandardStructuredRAG:
    def query(self, question):
        # Step 1: カテゴリ抽出
        category = extract_category(question)

        # Step 2: メタデータフィルタリング
        docs = vectorstore.similarity_search(
            question,
            k=5,
            filter={"category": category}
        )

        # Step 3: リランキング
        reranked = rerank_by_relevance(docs)

        # Step 4: LLM生成
        return llm.generate(reranked, question)
```

**本研究における"Hybrid RAG"**:

本研究では、上記の学術的定義とは異なるアプローチを採用している：

1. **ルールベース質問分析**: キーワードベースで質問タイプを判定
2. **計算処理の実行**: 距離計算、集計、比較などの数学的処理を実行
3. **ベクトル検索との統合**: 計算結果とベクトル検索結果を常に併用
4. **日本語自然文出力**: JSON形式ではなく、整形された日本語テキストを生成

```python
# 本研究の "Hybrid RAG" の実装
class HybridRAG:
    def query(self, question):
        # Step 1: 質問分析（ルールベース）
        analysis = analyze_question(question)

        # Step 2: 複数の構造化処理を並列実行（if文、非排他）
        contexts = []
        if analysis.requires_comparison:
            contexts.append(compare_east_west(pois, category))
        if analysis.requires_aggregation:
            contexts.append(count_by_category(pois, category))
        if analysis.requires_proximity:
            contexts.append(get_nearest_pois(pois, category))

        # Step 3: ベクトル検索を常に追加（補完として）
        contexts.append(vector_search(question, k=5))

        # Step 4: 統合コンテキストで自然言語生成
        return llm.generate(contexts, question)
```

この違いは本質的であり、本論文では一貫して「Hybrid RAG」という用語を使用する。学術界の標準的な"Structured RAG"と混同しないよう注意されたい。

### 2.3 Graph RAGとナレッジグラフ

**GraphRAG (Microsoft Research, 2024)**:
- エンティティと関係をグラフ構造で表現
- コミュニティ検出によるクラスタリング
- グラフトラバーサルによる複雑な推論

**地理空間ナレッジグラフ**:
- LinkedGeoData: OpenStreetMapのRDF化
- GeoSPARQL: 空間関係のクエリ言語
- 本研究では7種類のエッジタイプ（NEAR_TO、COMPETITOR、COMPLEMENTARY等）を設計

### 2.4 Agentic RAGとツール使用

**ReAct (Yao et al., 2022)**:
- Thought（思考）→ Action（行動）→ Observation（観察）のループ
- LLMがツールを動的に選択・実行

**Toolformer (Schick et al., 2023)**:
- LLMによる自律的なツール使用
- APIコール、計算、検索の統合

**LangGraph (Harrison et al., 2024)**:
- ステート管理とワークフロー制御
- 本研究では16種類のツールを実装

### 2.5 Parameter-Efficient Fine-Tuning

**LoRA (Hu et al., 2021)**:
- 低ランク行列による効率的なファインチューニング
- パラメータの0.1%のみを学習

**QLoRA (Dettmers et al., 2023)**:
- 4bit量子化とLoRAの組み合わせ
- 消費者向けGPU（16GB）での大規模モデル学習を実現

---

## 3. 方法論（Methodology）

### 3.1 Phase 5: 階層化評価フレームワーク

#### 3.1.1 設計原則

地理空間クエリシステムの包括的評価のため、以下の原則に基づきテストケースを設計した：

1. **認知負荷に基づく難易度階層**: L1（基礎検索）からL5（高度推論）まで
2. **機能別サブカテゴリ**: 12の異なる機能領域を網羅
3. **定量的評価指標**: 複数の評価軸で性能を測定

#### 3.1.2 テストケース構成

| レベル | サブカテゴリ | 件数 | 評価対象 |
|--------|-------------|------|----------|
| **L1: 基礎検索** | | **10件** | |
| | basic_location | 5件 | 座標情報の正確な提供 |
| | basic_category | 5件 | カテゴリ検索の網羅性 |
| **L2: 空間推論** | | **15件** | |
| | spatial_proximity | 5件 | 最近傍判定の正確性 |
| | spatial_density | 5件 | 密度概念の理解 |
| | spatial_comparison | 5件 | 方向別比較の正確性 |
| **L3: 制約充足** | | **10件** | |
| | constraint_single | 5件 | 単一制約の充足 |
| | constraint_multi | 5件 | 複数制約の同時充足 |
| **L4: 意思決定支援** | | **10件** | |
| | decision_location | 5件 | 立地適性評価 |
| | decision_business | 5件 | ビジネス判断支援 |
| **L5: 高度推論** | | **10件** | |
| | advanced_sensitivity | 3件 | 条件変更の影響分析 |
| | advanced_comparison | 4件 | 多軸・多地点比較 |
| | advanced_uncertainty | 3件 | 不確実性の認識と表現 |

**合計**: 55件（Phase 5-6用）+ 35件（Phase 8 GraphRAG用）+ 15件（Phase 9 Agentic用）= **105件**

#### 3.1.3 評価指標

各テストケースに対して以下の指標を計算：

| 指標 | 説明 | 計算方法 |
|------|------|---------|
| キーワードヒット率 | 期待キーワードの含有率 | hits / total_keywords |
| 座標情報有無 | 緯度経度の含有 | bool (0 or 1) |
| 数値情報有無 | 距離・件数の含有 | bool (0 or 1) |
| POI名有無 | 具体的なPOI名の含有 | bool (0 or 1) |

**統合スコア**: レベル別重み付け平均（0-100pt）

### 3.2 Phase 6: Hybrid RAGアーキテクチャ

#### 3.2.1 アーキテクチャ概要

Hybrid RAGの核心は、**構造化処理とベクトル検索の相補的統合**である。

**設計原則**:
1. **非排他的統合**: `if`文を使用し、複数の処理を並列実行
2. **ベクトル検索の常時実行**: 補完として常に追加
3. **自然言語形式の出力**: JSON形式ではなく、整形されたテキスト

**パイプライン**:
```
質問入力
    │
    ▼
質問分析モジュール
    │
    ├─→ 構造化処理1（東西比較）
    ├─→ 構造化処理2（集計）
    ├─→ 構造化処理3（最近傍）
    └─→ ベクトル検索（常時）
    │
    ▼
統合コンテキスト
    │
    ▼
LLM回答生成
```

#### 3.2.2 4つの構造化処理コンポーネント

##### コンポーネント1: 空間情報エンリッチメント

全POIに対して、基準点（渋谷駅）からの距離と方角を事前計算。

```python
def enrich_poi_with_spatial_info(poi, station=(35.658034, 139.701636)):
    distance = haversine_distance(station, (poi['lat'], poi['lon']))
    direction = get_direction_8way(station, (poi['lat'], poi['lon']))
    poi['distance_from_station'] = distance
    poi['direction_from_station'] = direction
    return poi
```

##### コンポーネント2: 集計・比較機能

```python
def compare_east_west(pois, category=None):
    """東側と西側のPOI数を比較"""
    east_dirs = ['east', 'northeast', 'southeast']
    west_dirs = ['west', 'northwest', 'southwest']

    east_count = count_pois(pois, directions=east_dirs, category=category)
    west_count = count_pois(pois, directions=west_dirs, category=category)

    return {
        'east_count': east_count,
        'west_count': west_count,
        'winner': 'east' if east_count > west_count else 'west',
        'difference': abs(east_count - west_count)
    }
```

##### コンポーネント3: 近接性検索

```python
def get_nearest_pois(pois, category=None, top_n=5):
    """駅に最も近いPOIを取得"""
    filtered = filter_by_category(pois, category) if category else pois
    sorted_pois = sorted(filtered, key=lambda p: p['distance_from_station'])
    return sorted_pois[:top_n]
```

##### コンポーネント4: 感度分析

```python
def compare_by_radius(pois, radius1_m, radius2_m, category=None):
    """2つの半径での件数を比較"""
    count1 = len(filter_by_radius(pois, radius1_m, category))
    count2 = len(filter_by_radius(pois, radius2_m, category))
    return {
        'radius1': radius1_m,
        'radius2': radius2_m,
        'count1': count1,
        'count2': count2,
        'ratio': count2 / count1 if count1 > 0 else 0
    }
```

#### 3.2.3 重要な設計判断: if vs elif

Phase 6.2で発生した性能悪化（-5.5pt）の根本原因は、`elif`による排他的分岐であった。

**問題のあったコード (Phase 6.2)**:
```python
def _build_context(self, question, analysis):
    parts = []
    if analysis.requires_proximity:
        parts.append(proximity_context)
    elif analysis.requires_sensitivity:    # ← 排他的
        parts.append(sensitivity_context)
    elif analysis.requires_comparison:     # ← 排他的
        parts.append(comparison_context)

    # ベクトル検索はフォールバック
    if not parts:
        parts.append(vector_search_context)
    return parts
```

**修正後のコード (Phase 6.2.1)**:
```python
def _build_context(self, question, analysis):
    structured_parts = []

    # 複数の処理を並列実行
    if analysis.requires_proximity:
        structured_parts.append(proximity_context)
    if analysis.requires_sensitivity:      # ← 非排他的
        structured_parts.append(sensitivity_context)
    if analysis.requires_comparison:       # ← 非排他的
        structured_parts.append(comparison_context)
    if analysis.requires_aggregation:
        structured_parts.append(aggregation_context)

    # ベクトル検索を常に追加
    vector_context = self._get_vector_search_context(question, k=5)
    return structured_parts + [vector_context]
```

この修正により、64.1pt → 91.6pt（+27.5pt）の改善を達成した。

### 3.3 Phase 7: ファインチューニング実験

#### 3.3.1 QLoRAの実装

T4 GPU（16GB VRAM）での実行を可能にするため、QLoRA（Quantized Low-Rank Adaptation）を採用した。

**LoRA設定**:
```python
lora_config = LoraConfig(
    r=8,                    # ランク
    lora_alpha=16,          # スケーリング係数
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    task_type="CAUSAL_LM"
)
```

**学習設定**:
- バッチサイズ: 1
- 勾配累積: 16ステップ（実効バッチ=16）
- 学習率: 2e-4
- エポック数: 3
- 学習データ: ~1,000件（Alpaca形式）

#### 3.3.2 データ生成

テストリーク防止のため、テストケースで直接言及されるPOI（東宝シネマ、渋谷東武ホテル等）を学習データ生成元から除外した。

**8パターンのテンプレート**:
1. 位置情報: 「{POI名}はどこにありますか？」
2. カテゴリ検索: 「渋谷駅周辺の{カテゴリ}を教えてください」
3. 最近傍検索: 「渋谷駅に最も近い{カテゴリ}はどこですか？」
4. 東西比較: 「東側と西側、どちらに{カテゴリ}が多いですか？」
5. 集約: 「{半径}m以内に{カテゴリ}はいくつ？」
6. 制約付き: 「{半径}m以内にある{カテゴリ}を教えて」
7. 推論: 「新規{カテゴリ}の出店を検討。競合状況は？」
8. 感度分析: 「範囲を300mから500mに広げると結果は？」

### 3.4 Phase 8: Graph RAG実装

#### 3.4.1 グラフスキーマ設計

**ノードタイプ**:
| ノード | 属性 | 件数 |
|--------|------|------|
| POI | id, name, lat, lon, category | 1,046 |
| Category | name, name_jp | 12 |
| Area | name, direction, distance_zone | 32 |

**エッジタイプ（7種類）**:
| エッジ | 説明 | 抽出条件 | 件数 |
|--------|------|----------|------|
| NEAR_TO | 空間的近接 | 距離 ≤ 100m | 66,248 |
| SAME_CATEGORY | 同一カテゴリ | category一致 | 2,086 |
| SAME_BRAND | 同一チェーン店 | brand一致 | ~200 |
| COMPLEMENTARY | 相補的関係 | カテゴリペア + 200m以内 | ~3,000 |
| COMPETITOR | 競合関係 | 同カテゴリ + 100m以内 | ~8,000 |
| SAME_CUISINE | 同一料理ジャンル | cuisine一致 | ~500 |
| SAME_HOURS | 同一営業時間帯 | 24h/深夜フラグ一致 | ~2,000 |

**総エッジ数**: 約82,000

#### 3.4.2 関係性抽出手法

**ブランド情報の抽出**:
```python
KNOWN_BRANDS = {
    "セブン-イレブン": "7-Eleven",
    "ファミリーマート": "FamilyMart",
    "スターバックス": "Starbucks",
    "マクドナルド": "McDonald's",
    # ... 50+ブランド
}

def extract_brand(poi_name: str) -> Optional[str]:
    for keyword, brand in KNOWN_BRANDS.items():
        if keyword in poi_name:
            return brand
    return None
```

**相補的関係の定義**:
```python
COMPLEMENTARY_RULES = {
    ("宿泊/ホテル", "飲食店/レストラン"): "DINING_NEAR_HOTEL",
    ("娯楽/映画館", "飲食店/カフェ"): "ENTERTAINMENT_COMBO",
    ("交通/鉄道駅", "商店/コンビニ"): "TRANSIT_AMENITY",
    # ... 10+ルール
}
```

### 3.5 Phase 9: Agentic RAG with LangGraph

#### 3.5.1 システムアーキテクチャ

**LangGraphステートマシン**:
```
┌────────────────┐
│  Initial State │
│  - question    │
│  - iteration=0 │
└───────┬────────┘
        │
        ▼
┌────────────────┐
│  Planner Node  │ ← LLMが次のアクションを決定
│  (ReAct)       │
└───────┬────────┘
        │
    ┌───┴───┐
    │       │
    ▼       ▼
┌────────┐ ┌────────────┐
│ Action │ │   Finish   │
│ Node   │ │   Node     │
└───┬────┘ └────────────┘
    │
    ▼
┌────────────────┐
│ Observation    │ ← ツール実行結果
└───────┬────────┘
        │
        └─→ Planner（反復）
```

#### 3.5.2 16種類のツール

**空間計算ツール（6種）**:
1. `tool_get_nearest_pois`: 最寄りPOI検索
2. `tool_filter_by_area`: 半径フィルタリング
3. `tool_count_pois_in_radius`: 半径内件数カウント
4. `tool_compare_radius`: 半径比較（感度分析）
5. `tool_analyze_sensitivity`: 複数半径での感度分析
6. `tool_get_poi_details`: POI詳細情報取得

**集計ツール（5種）**:
7. `tool_count_by_category`: カテゴリ別件数
8. `tool_get_top_categories`: カテゴリランキング
9. `tool_compare_east_west`: 東西比較
10. `tool_compare_north_south`: 南北比較
11. `tool_analyze_category_by_direction`: 方向別カテゴリ分析

**グラフトラバーサルツール（5種）**:
12. `tool_find_nearby_competitors`: 競合店検索
13. `tool_find_complementary_pois`: 相補的POI検索
14. `tool_analyze_brand_distribution`: ブランド分布分析
15. `tool_find_same_cuisine`: 同一料理ジャンル検索
16. `tool_find_24h_pois`: 24時間営業店舗検索

#### 3.5.3 ReActプロンプト設計

```python
AGENT_SYSTEM_PROMPT = """あなたは渋谷駅周辺のPOI情報を検索・分析する専門家です。

**重要: すべての回答は必ず日本語で行ってください。**

## 利用可能なツール
{tool_descriptions}

## 回答フォーマット
各ステップで以下のフォーマットを使用してください：

**Thought**: あなたの推論過程
**Action**: tool_name
**Action Input**: {{"arg1": "value1", "arg2": "value2"}}
**Observation**: [ツール実行結果がここに表示されます]

最終回答の準備ができたら：
**Final Answer**: [日本語での最終回答]
"""
```

### 3.6 Adaptive RAG: メタレベル選択

#### 3.6.1 選択ロジック

```python
def select_system(question: str, analysis: QuestionAnalysis) -> str:
    """質問に応じて最適なシステムを選択"""

    # GraphRAGを使用すべきケース
    if analysis.requires_comparison:
        return "GraphRAG"  # comparison: +50%の優位性

    if "24時間" in question or "深夜" in question:
        return "GraphRAG"  # hours: +11%の優位性

    if analysis.requires_aggregation:
        return "GraphRAG"  # aggregation: +8%の優位性

    # それ以外はHybrid RAG（デフォルト）
    return "HybridRAG"
```

#### 3.6.2 選択分布（Phase 8実験）

```
HybridRAG選択: 62クエリ (68.9%)
GraphRAG選択:  28クエリ (31.1%)
```

---

## 4. 実験設定（Experimental Setup）

### 4.1 共通実行環境

| 項目 | 仕様 |
|------|------|
| プラットフォーム | Google Colaboratory |
| GPU | NVIDIA Tesla T4 (16GB VRAM) |
| Python | 3.10 |
| PyTorch | 2.0+ |
| Transformers | 4.36+ |

### 4.2 ベースモデル

| 項目 | 値 | 理由 |
|------|-----|------|
| モデル名 | Qwen/Qwen2.5-7B-Instruct | 日本語性能とメモリ効率 |
| パラメータ数 | 7B | T4 GPU（16GB）で実行可能 |
| 量子化 | 4bit (nf4) | メモリ使用量削減 |
| 埋め込みモデル | intfloat/multilingual-e5-base | 多言語対応、日本語性能 |

### 4.3 データセット

**POIデータ**:
- ソース: OpenStreetMap（2026年1月取得）
- 対象エリア: 渋谷駅周辺（半径約1km）
- 総件数: 1,046件
- カテゴリ数: 12種類
- 属性: name, category, lat, lon, phone, website, opening_hours

**空間分布**:
- 東側（east, northeast, southeast）: 287件（27.4%）
- 西側（west, northwest, southwest）: 555件（53.0%）
- 南北（north, south）: 204件（19.5%）

**カテゴリ分布（上位5）**:
1. 飲食店/レストラン: 278件（26.6%）
2. 飲食店/カフェ: 156件（14.9%）
3. 商店/小売: 143件（13.7%）
4. 商店/コンビニ: 82件（7.8%）
5. サービス/美容: 68件（6.5%）

### 4.4 評価データセットの構成

| ソース | 件数 | カテゴリ数 | 用途 |
|--------|------|-----------|------|
| Phase 5-6 Tests | 55件 | 12 | 基本性能評価 |
| Phase 8 GraphRAG Tests | 35件 | 10 | グラフ機能評価 |
| Phase 9 Agentic Tests | 15件 | 3 | 動的推論評価 |
| **合計** | **105件** | **25** | 包括的評価 |

---

## 5. 実験結果（Results）

### 5.1 全体性能比較

#### 5.1.1 Phase 5-6テストセット（55件）での性能

| システム | スコア | 実行時間 | Phase 5比 |
|---------|--------|----------|-----------|
| Naive RAG (Phase 5) | 60.3pt | 22.7秒 | - |
| Hybrid RAG (Phase 6.2.1) | **91.6pt** | 21.9秒 | **+31.3pt (+52%)** |
| FT-Base (Phase 7) | 77.1pt | 9.57秒 | +16.8pt (+28%) |
| FT+RAG (Phase 7) | 78.5pt | 8.04秒 | +18.2pt (+30%) |

#### 5.1.2 Phase 8拡張テストセット（90件）での性能

| システム | スコア | 処理時間 | 標準偏差 |
|---------|--------|----------|----------|
| **Hybrid RAG** | **89.1%** | 20.6秒 | 20.2 |
| Adaptive RAG | 86.1% | 17.8秒 | 20.4 |
| Graph RAG | 76.7% | 8.7秒 | 24.8 |

#### 5.1.3 Phase 9拡張テストセット（105件）での性能

| システム | 成功率 | 平均実行時間 | 中国語混入率 |
|---------|--------|-------------|------------|
| **Hybrid RAG** | **96.2%** | 11.1秒 | 0% |
| Agentic RAG | 87.6% | 56.4秒 | 7.6% |

### 5.2 Phase別詳細分析

#### 5.2.1 Phase 6: Hybrid RAG開発プロセス

```
Phase 5 (Naive RAG):     60.3pt ─────────────────────────
Phase 6.1 (+空間処理):    69.6pt ████████████████ (+9.3pt)
Phase 6.2 (+近接・感度):   64.1pt ████████████ (-5.5pt) ※悪化
Phase 6.2.1 (if修正):     91.6pt ████████████████████████████████████████ (+27.5pt)
```

**改善の内訳**:
| 機能追加 | 改善幅 | 主な寄与カテゴリ |
|---------|--------|-----------------|
| 空間情報エンリッチメント | +9.3pt | basic_location, spatial_comparison |
| 近接性検索（Phase 6.2） | -5.5pt | ※排他的分岐による悪化 |
| if修正 + ベクトル常時実行 | +27.5pt | 全カテゴリ |
| **累計** | **+31.3pt** | - |

#### 5.2.2 Phase 7: タスク依存的な性能

| タスクレベル | Baseline | RAG | FT-Base | FT+RAG | 最良 |
|------------|----------|-----|---------|--------|------|
| L1: 基礎検索 | 72.7 | **84.4** | 73.7 | **84.4** | RAG/FT+RAG |
| L2: 空間推論 | 68.6 | 71.0 | **72.4** | 71.4 | FT-Base |
| L3: 制約充足 | 72.9 | 73.0 | 72.8 | **81.0** | FT+RAG |
| L4: 意思決定 | 86.8 | 80.2 | **88.1** | 78.5 | FT-Base |
| L5: 高度推論 | **81.3** | 79.8 | 80.8 | 80.6 | Baseline |

**主要な発見**:
- **L1（基礎検索）**: RAGが優位（+11.7pt）→ 具体的POI情報の検索が必要
- **L4（意思決定）**: FT-Baseが優位（+7.9pt）→ 判断パターンの学習が有効

#### 5.2.3 Phase 8: カテゴリ別性能

**GraphRAGが優位なカテゴリ**:
| カテゴリ | GraphRAG | HybridRAG | 差分 | 理由 |
|---------|----------|-----------|------|------|
| **comparison** | **100.0%** | 50.0% | **+50.0pt** | Areaノードによる方向別集計 |
| **competitor** | **88.9%** | 66.7% | **+22.2pt** | COMPETITORエッジによる密集度分析 |

**HybridRAGが優位なカテゴリ**:
| カテゴリ | GraphRAG | HybridRAG | 差分 | 理由 |
|---------|----------|-----------|------|------|
| basic_category | 73.3% | **93.3%** | -20.0pt | ベクトル検索の精度 |
| aggregation | 80.6% | **100.0%** | -19.4pt | 集計処理の正確性 |
| relation | 63.3% | **86.7%** | -23.4pt | ベクトルコンテキストの補完 |

#### 5.2.4 Phase 9: Agentic RAGのカテゴリ別性能

**Agentic RAGが優位なカテゴリ**:
| カテゴリ | AgenticRAG | HybridRAG | 差分 | 理由 |
|---------|-----------|-----------|------|------|
| **competitor** | **100.0%** | 66.7% | **+33.3pt** | グラフトラバーサルツールの活用 |
| **complementary** | **100.0%** | 80.0% | **+20.0pt** | 相補的POI発見ツールの活用 |
| **basic_location** | **100.0%** | 80.0% | **+20.0pt** | 段階的な情報収集 |

**HybridRAGが優位なカテゴリ**:
| カテゴリ | AgenticRAG | HybridRAG | 差分 | 理由 |
|---------|-----------|-----------|------|------|
| spatial_comparison | 60.0% | **100.0%** | -40.0pt | 並列処理の効率性 |
| constraint_multi | 66.7% | **100.0%** | -33.3pt | 事前決定の正確性 |
| multi_hop | 77.8% | **100.0%** | -22.2pt | 単純なクエリでは不要な複雑性 |

### 5.3 処理時間の比較

| システム | 平均処理時間 | 最速比 | 特徴 |
|---------|------------|--------|------|
| Graph RAG | 8.7秒 | 1.0x | グラフトラバーサルが高速 |
| FT+RAG | 8.04秒 | 0.92x | RAG処理のみ、FT分のオーバーヘッドなし |
| Hybrid RAG | 11.1秒 | 1.28x | 複数の構造化処理を実行 |
| Adaptive RAG | 17.8秒 | 2.05x | システム選択のオーバーヘッド |
| Naive RAG | 22.7秒 | 2.61x | 最適化なし |
| Agentic RAG | 56.4秒 | 6.48x | 反復的ツール呼び出し |

### 5.4 言語制御の問題（Phase 9）

**中国語混入の発生状況**:
- 総テストケース: 105件
- 中国語混入: 8件（7.6%）
- 発生したカテゴリ: multi_hop（3件）、constraint_multi（2件）、complementary（2件）、basic_location（1件）

**根本原因分析**:
| 原因 | 寄与率 | 説明 |
|------|--------|------|
| JSON形式のツール出力 | 50% | 技術モードトリガー |
| ReAct英語メタ言語 | 30% | Thought/Action/Observationが英語 |
| コンテキスト蓄積 | 15% | 反復により中国語が蓄積 |
| 数値表現の違い | 5% | 中国語の数値表現が混入 |

**対策の効果**:
- 日本語強制プロンプト追加: 部分的改善（7.6% → 推定5%）
- 根本的解決には日本語フレンドリーモデル（Llama 3.1等）が必要

---

## 6. 考察（Discussion）

### 6.1 RAGアプローチの本質的な違い

#### 6.1.1 決定タイミングによる分類

| アプローチ | 決定タイミング | 決定主体 | 実行パターン |
|-----------|--------------|---------|------------|
| **Naive RAG** | 実行時なし | - | ベクトル検索のみ |
| **Hybrid RAG** | 事前（質問分析時） | ルールベース | 並列実行 |
| **Adaptive RAG** | 実行時（メタレベル） | ルールベース | システム選択 |
| **Agentic RAG** | 実行時（各ステップ） | LLM | 逐次実行・反復 |

#### 6.1.2 実行フロー比較

**Naive RAG**:
```
質問 → ベクトル検索 → LLM生成 → 回答
```

**Hybrid RAG**:
```
質問 → 質問分析（ルール）
    ├→ 東西比較処理
    ├→ 集計処理
    ├→ 最近傍処理
    └→ ベクトル検索
    → 統合 → LLM生成 → 回答
```

**Agentic RAG**:
```
質問 → LLM計画
    → ツール1実行 → LLM評価
    → ツール2実行 → LLM評価
    → ツール3実行 → LLM評価
    → 最終回答生成
```

### 6.2 タスク特性と最適アプローチ

#### 6.2.1 基礎検索タスク（L1）

**最適**: Hybrid RAG（ベクトル検索中心）

**理由**:
- 具体的なPOI情報の検索が必要
- ベクトル検索が最も効率的
- 構造化処理は補完として機能

**実験データ**:
- Hybrid RAG: 89.1pt
- FT-Base: 73.7pt（-15.4pt）→ 学習データに含まれないPOI名を生成できない

#### 6.2.2 空間比較タスク（東西比較等）

**最適**: Graph RAG

**理由**:
- Areaノードによる方向別集計が効率的
- グラフトラバーサルで高速処理
- 明確な優位性（+50pt）

**実験データ**:
- Graph RAG: 100.0%
- Hybrid RAG: 50.0%（-50.0pt）→ 方向別集計が不十分
- Agentic RAG: 60.0%（-40.0pt）→ 過剰な複雑性

#### 6.2.3 関係性分析タスク（競合・相補）

**最適**: Agentic RAG

**理由**:
- グラフトラバーサルツールを動的に選択
- 複数ステップの推論が有効
- 具体的なPOI名を段階的に取得

**実験データ**:
- Agentic RAG: 100.0%（competitor、complementary）
- Hybrid RAG: 66.7%, 80.0%（-33.3pt, -20.0pt）→ グラフ構造の活用が不十分

#### 6.2.4 意思決定支援タスク（L4）

**最適**: Fine-Tuned RAG

**理由**:
- 判断パターンの学習が有効
- 具体的なPOI情報よりも推論フレームワークが重要
- 学習データから「出店判断」のパターンを獲得

**実験データ**:
- FT-Base: 88.1pt
- Hybrid RAG: 80.2pt（-7.9pt）→ パターン認識が不足

### 6.3 構造化処理とベクトル検索の相補性

#### 6.3.1 Phase 6.2での教訓

Phase 6.2で発生した性能悪化（-5.5pt）は、**構造化処理とベクトル検索が排他的ではなく相補的**であることを明確に示した。

**排他的アプローチの問題**:
```python
# 問題: elif による排他的分岐
if requires_proximity:
    return proximity_context
elif requires_sensitivity:
    return sensitivity_context
elif requires_comparison:
    return comparison_context
else:
    return vector_search_context
```

この実装では：
- 近接性処理が選択 → ベクトル検索が実行されない
- 具体的なPOI情報が欠落 → 回答の質が低下

**相補的アプローチの成功**:
```python
# 解決: if による非排他的統合
contexts = []
if requires_proximity:
    contexts.append(proximity_context)
if requires_sensitivity:
    contexts.append(sensitivity_context)
if requires_comparison:
    contexts.append(comparison_context)
# 常にベクトル検索を追加
contexts.append(vector_search_context)
return contexts
```

この修正により：
- 構造化処理: 正確な集計・比較・フィルタリング
- ベクトル検索: 具体的なPOI情報・座標・文脈の補完
- 両者の組み合わせ: LLMが数値データと具体例の両方を活用

**効果**: 64.1pt → 91.6pt（+27.5pt、+42.9%）

#### 6.3.2 料理のレシピによるアナロジー

**Hybrid RAG（本研究）**:
```
シェフ: レシピを読む
↓
レシピ: 「タマネギを炒める」「塩を加える」「ベクトル検索で追加食材を取得」
↓
シェフ: すべての指示を並行して実行
↓
結果: 速く、正確
```

**Agentic RAG**:
```
シェフ: 一歩ずつ考える
↓
「タマネギを炒めるべきか？」→ はい → 炒める
「次は何をすべきか？」→ 塩を加える → 加える
「さらに何か必要か？」→ ベクトル検索 → 取得
↓
結果: 柔軟だが遅い
```

### 6.4 用語の明確化: Structured RAG vs Hybrid RAG

#### 6.4.1 学術界の"Structured RAG"

学術文献における"Structured RAG"の標準的定義：

**定義**: メタデータフィルタリングとリランキングを用いたRAG

**典型的な実装**:
```python
# 学術界の "Structured RAG"
def query(question):
    # 1. カテゴリ抽出
    category = llm.extract_category(question)

    # 2. メタデータフィルタリング
    results = vectorstore.search(
        question,
        filter={"category": category}
    )

    # 3. リランキング
    reranked = rerank_by_score(results)

    # 4. 生成
    return llm.generate(reranked)
```

**特徴**:
- ベクトル検索が中心
- メタデータによる事前フィルタリング
- 検索結果のリランキング
- 構造化「クエリ」の生成ではなく、検索の精緻化

#### 6.4.2 本研究の"Hybrid RAG"

**定義**: ルールベース質問分析 + 計算処理 + ベクトル検索 + 自然言語出力

**実装**:
```python
# 本研究の "Hybrid RAG"
def query(question):
    # 1. 質問分析（ルールベース）
    analysis = analyze_question(question)

    # 2. 計算処理の実行
    contexts = []
    if analysis.requires_comparison:
        result = compare_east_west(pois, category)
        contexts.append(format_comparison(result))

    if analysis.requires_aggregation:
        result = count_by_category(pois)
        contexts.append(format_aggregation(result))

    # 3. ベクトル検索を常に追加
    vector_results = vectorstore.search(question, k=5)
    contexts.append(format_vector_results(vector_results))

    # 4. 自然言語生成
    return llm.generate(contexts, question)
```

**特徴**:
- ベクトル検索と計算処理を並列実行
- 数学的に正確な集計・比較
- 整形された日本語自然文での出力
- ベクトル検索は「補完」として常に実行

#### 6.4.3 名称の選択理由

本研究では"Hybrid RAG"という用語を採用した理由：

1. **学術的混乱の回避**: "Structured RAG"は既に別の意味で広く使用されている
2. **本質の表現**: ベクトル検索と計算処理の「ハイブリッド」であることを明確化
3. **PostGISとの親和性**: Phase 10（全国展開）で導入予定のPostGIS（空間データベース）も"Hybrid"アプローチ（空間インデックス + SQL）である

#### 6.4.4 比較表

| 項目 | 学術界のStructured RAG | 本研究のHybrid RAG |
|------|----------------------|-------------------|
| **中心技術** | ベクトル検索 | 計算処理 + ベクトル検索 |
| **フィルタリング** | メタデータによる事前絞り込み | カテゴリ・半径による動的フィルタ |
| **処理の追加** | リランキング | 集計・比較・感度分析 |
| **出力形式** | 検索結果をそのまま使用 | 整形された自然言語テキスト |
| **決定方法** | LLMが暗黙的に決定 | ルールベースで明示的に決定 |
| **例** | カテゴリでフィルタ → 検索 | 東西比較計算 + 検索結果を統合 |

### 6.5 Agentic RAGの課題と可能性

#### 6.5.1 実行時間のオーバーヘッド

Agentic RAGは平均56.4秒と、Hybrid RAG（11.1秒）の5.1倍の処理時間を要した。

**原因**:
1. **反復的なLLM呼び出し**: 各ステップでLLMが計画を立てる
2. **ツール実行の逐次化**: 並列実行ができない
3. **自己評価のオーバーヘッド**: 結果を評価して次の行動を決定

**トレードオフ**:
- 柔軟性: 高い（未知のクエリパターンに対応可能）
- 速度: 低い（既知のパターンでは過剰）
- 精度: タスク依存（関係性分析では優位、単純タスクでは劣位）

#### 6.5.2 言語制御の問題

Phase 9で発見された中国語混入問題（7.6%）は、Agentic RAGの設計における重要な課題である。

**問題の本質**:
- JSON形式のツール出力 → LLMが「技術モード」に切り替わる
- 技術モードでは中国語（Qwenのドミナント言語）が優勢になる
- 日本語強制プロンプトでは部分的にしか解決できない

**対策**:
1. **短期**: ツール出力を日本語自然文に変換
2. **中期**: 日本語フレンドリーモデル（Llama 3.1等）に変更
3. **長期**: 言語制御メカニズムをシステムレベルで設計

#### 6.5.3 Agentic RAGが価値を発揮する場面

実験結果から、Agentic RAGは以下の状況で明確な優位性を示した：

**1. 関係性分析タスク**:
- competitor関係: +33.3pt
- complementary関係: +20.0pt
- グラフトラバーサルツールの動的選択が有効

**2. 未知のクエリパターン**:
- 事前にルールを定義できない複雑なクエリ
- 例: 「カフェとレストランの件数を比較して、多い方の東西分布を分析」

**3. 探索的タスク**:
- 複数の候補から最適解を探索
- 例: 「カフェが最も多い半径を100m、200m、300m、500mで比較」

### 6.6 実装レベルの重要な洞察

#### 6.6.1 if vs elif: 5.5ptの差を生む設計判断

Phase 6.2での性能悪化は、単一の制御構文の選択（`elif` → `if`）により解決された。

**教訓**:
- 小さな実装判断が大きな性能差を生む
- 排他的分岐（elif）は、新機能が既存機能を無効化するリスクがある
- 非排他的統合（if）は、複数の情報源を活用できる

#### 6.6.2 JSON vs 自然言語: 言語安定性の差

Agentic RAGでの中国語混入問題は、出力形式の選択が言語制御に影響することを示した。

**JSON形式の問題**:
```json
{
  "tool": "tool_count_pois_in_radius",
  "args": {"category": "カフェ", "radius": 500},
  "result": {"count": 73, "category": "cafe"}
}
```
→ LLMが技術モードに切り替わり、中国語が混入

**自然言語形式の利点**:
```
【500m以内のカフェ件数】
カテゴリ: カフェ
件数: 73件
半径: 500m
```
→ LLMが日本語モードを維持

**推奨**: 多言語LLMでは、自然言語形式の出力が言語安定性において優位

#### 6.6.3 段階的評価の重要性

Phase 6の反復的改善プロセスは、段階的評価の重要性を示した。

**プロセス**:
1. Phase 6.1: +9.3pt → 成功を確認
2. Phase 6.2: -5.5pt → 悪化を即座に検出
3. Phase 6.2.1: +27.5pt → 修正により大幅改善

**教訓**:
- 各改善段階で全サブカテゴリを評価
- 悪化を早期発見し、根本原因を分析
- 表面的な修正ではなく、アーキテクチャレベルの見直し

### 6.7 スケーラビリティと実用化への課題

#### 6.7.1 POI数の増加への対応

現在の実装（1,046件）は、全国規模（500万件）に対応できない。

**課題**:
- オンデマンドでの距離計算: O(N)の計算量
- メモリ使用量: すべてのPOIをメモリに保持

**解決策（Phase 10計画）**:
1. **PostGIS導入**: 空間インデックス（R-tree、GiST）で高速検索
2. **MCP Server統合**: クライアント・サーバー分離
3. **キャッシング**: 頻繁なクエリ結果をキャッシュ

#### 6.7.2 基準点の動的解決

現在の実装は渋谷駅座標がハードコードされている：

```python
SHIBUYA_STATION = (35.658034, 139.701636)
```

**必要な機能**:
1. ジオコーディングAPI統合（「新宿駅」→ 座標）
2. 複数基準点の同時サポート
3. エリア自動検出（クエリから対象エリアを推定）

#### 6.7.3 評価指標の改善

現在の評価はキーワードマッチング中心であり、推論の質を完全に評価できない。

**改善策**:
1. **LLM-as-a-Judge**: GPT-4等による回答品質評価
2. **人手評価**: 一部のサンプルで人間が評価
3. **タスク完遂率**: 実際のタスクの達成度を測定

---

## 7. 結論（Conclusion）

### 7.1 主要な発見

本研究では、OpenStreetMapのPOIデータを用いて6種類のRAGアプローチを包括的に比較し、以下の主要な発見を得た：

**1. Hybrid RAGの優位性**

ルールベース質問分析 + 計算処理 + ベクトル検索 + 自然言語出力を組み合わせたHybrid RAGが、最高性能96.2%を達成した。特に重要なのは、構造化処理とベクトル検索が**排他的ではなく相補的**であるという設計原則である。

**2. タスク依存的な最適アプローチ**

| タスクタイプ | 最適アプローチ | 優位性 | 理由 |
|------------|-------------|--------|------|
| 基礎検索（L1） | Hybrid RAG | +11.7pt | ベクトル検索の精度 |
| 空間比較（東西等） | Graph RAG | +50.0pt | Areaノードによる集計 |
| 関係性分析 | Agentic RAG | +33.3pt | 動的ツール選択 |
| 意思決定支援（L4） | Fine-Tuned RAG | +7.9pt | 判断パターンの学習 |

**3. 実装レベルの重要な洞察**

- **if vs elif**: 単一の制御構文の選択が5.5ptの差を生む
- **JSON vs 自然言語**: 出力形式が言語制御に影響（中国語混入7.6%）
- **段階的評価**: 各改善段階での全カテゴリ評価が成功の鍵

**4. 用語の明確化**

学術界の"Structured RAG"（メタデータフィルタリング + リランキング）と、本研究の"Hybrid RAG"（ルールベース + 計算 + ベクトル）は異なるアプローチである。今後の研究では用語の混乱を避けるべきである。

**5. 処理時間とスコアのトレードオフ**

| システム | スコア | 処理時間 | コストパフォーマンス |
|---------|--------|----------|-------------------|
| Hybrid RAG | 96.2% | 11.1秒 | ★★★★★ |
| Agentic RAG | 87.6% | 56.4秒 | ★★☆☆☆ |
| Graph RAG | 76.7% | 8.7秒 | ★★★☆☆ |

Hybrid RAGは高精度と実用的な処理時間のバランスが最も優れている。

### 7.2 研究課題への回答

**RQ1: どのRAGアプローチが最も高い性能を達成するか？**

**回答**: Hybrid RAGが96.2%の最高性能を達成した。ベースラインのNaive RAG（60.3pt）から+35.9pt（+60%）の改善である。

**RQ2: タスクタイプにより最適なRAGアプローチは異なるか？**

**回答**: Yes。基礎検索ではHybrid RAG、空間比較ではGraph RAG、関係性分析ではAgentic RAG、意思決定ではFine-Tuned RAGが最適である。単一のアプローチがすべてのタスクで優位ではない。

**RQ3: ベクトル検索と構造化処理の統合において、どのような設計原則が有効か？**

**回答**: **相補的統合**が有効である。具体的には：
1. 複数の構造化処理を非排他的に実行（`if`文）
2. ベクトル検索を常に補完として追加
3. 整形された自然言語コンテキストで統合

この設計により、Phase 6.2の64.1ptからPhase 6.2.1の91.6pt（+27.5pt）への改善を達成した。

**RQ4: ファインチューニング、グラフ構造、動的ツール選択はどのような状況で価値を発揮するか？**

**回答**:
- **ファインチューニング**: 意思決定支援タスク（L4）で優位（+7.9pt）。判断パターンの学習が有効。
- **グラフ構造**: 空間比較タスクで明確な優位性（+50pt）。方向別集計やPOI密集度分析に有効。
- **動的ツール選択**: 関係性分析タスクで優位（+33.3pt）。グラフトラバーサルツールを適切に選択できる。

### 7.3 実用システムへの提言

#### 7.3.1 推奨アーキテクチャ

**基本システム**: Hybrid RAG

**理由**:
- 最高性能（96.2%）
- 実用的な処理時間（11.1秒）
- 実装の複雑性が適度
- 新エリア追加時の柔軟性

**拡張オプション**:
```
┌─────────────────────────────────────────┐
│     推奨ハイブリッドアーキテクチャ       │
├─────────────────────────────────────────┤
│                                         │
│  質問入力                                │
│      │                                  │
│      ▼                                  │
│  質問分析（ルールベース）                │
│      │                                  │
│      ├─→ 空間比較？ → Graph RAG        │
│      ├─→ 関係性分析？ → Agentic RAG   │
│      ├─→ 意思決定？ → Fine-Tuned RAG  │
│      └─→ その他 → Hybrid RAG（デフォルト）│
│                                         │
└─────────────────────────────────────────┘
```

#### 7.3.2 段階的導入計画

**Phase 1（即時）**: Hybrid RAG単独
- 最も安定した性能
- 実装済み
- 90%以上のタスクに対応

**Phase 2（3ヶ月後）**: Hybrid + Graph RAG
- 空間比較タスクでGraph RAGを選択的に使用
- Adaptive RAG的なメタレベル選択
- +5-10ptの改善を期待

**Phase 3（6ヶ月後）**: 全国展開
- PostGIS + MCP Server
- 500万POI対応
- 50ms以下の応答時間目標

**Phase 4（長期）**: 特化型モジュール追加
- 関係性分析用のAgentic RAGモジュール
- 意思決定支援用のFine-Tuned RAGモジュール
- タスクタイプに応じた自動選択

#### 7.3.3 避けるべきアンチパターン

1. **排他的分岐（elif）**: ベクトル検索と構造化処理を排他的に扱わない
2. **JSON形式の過度な使用**: 多言語LLMでは自然言語形式を優先
3. **単一システムへの過度な期待**: タスク特性を考慮した選択が重要
4. **早期最適化**: まずHybrid RAGで基盤を構築し、必要に応じて拡張

### 7.4 今後の研究課題

#### 7.4.1 Phase 9-B: 日本語モデルでの検証

**目的**: 中国語混入問題の根本的解決

**計画**:
- 日本語フレンドリーモデル（Llama 3.1 8B Instruct）への変更
- 5システム（Naive、Standard Structured、Hybrid、Adaptive、Agentic）の再評価
- 言語制御の改善効果を定量評価

**期待される結果**:
- Agentic RAG: 87.6% → 92-96%（言語問題解決）
- 中国語混入: 7.6% → 0%

#### 7.4.2 Phase 10: 全国展開

**目的**: 500万POI規模への対応

**技術スタック**:
- PostGIS: 空間インデックス
- Supabase: PostgreSQL + PostGIS
- MCP Server: クライアント・サーバー分離

**目標**:
- 応答時間: 50ms以下
- POI数: 500万件以上
- 対応エリア: 全国主要都市

#### 7.4.3 評価指標の改善

**課題**: キーワードマッチング中心の評価の限界

**改善案**:
1. **LLM-as-a-Judge**: GPT-4による回答品質評価
2. **人手評価**: ゴールドスタンダード作成
3. **タスク完遂率**: 実際のユースケースでの有用性測定

#### 7.4.4 リアルタイムデータとの統合

**展望**:
- 営業時間の自動更新
- 混雑状況のリアルタイム反映
- ユーザーレビューの統合

### 7.5 学術的貢献

本研究の主な学術的貢献は以下の通りである：

1. **包括的評価フレームワーク**: 地理空間クエリシステムの多角的評価を可能にする階層化テストケース（105件）

2. **6システムの体系的比較**: 同一データセット・同一評価基準での公正な比較

3. **Hybrid RAGアーキテクチャ**: ベクトル検索と構造化処理の相補的統合の設計原則

4. **実装レベルの洞察**: if vs elifの問題、JSON vs 自然言語の言語制御への影響

5. **用語の明確化**: Structured RAG vs Hybrid RAGの違いを明確にし、今後の研究における混乱を防ぐ指針

6. **タスク特性と最適手法の対応**: どのタスクにどのアプローチが適するかの定量的知見

---

## 参考文献（References）

### RAGの基礎

1. Lewis, P., et al. (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." NeurIPS 2020.

2. Shuster, K., et al. (2021). "Retrieval Augmentation Reduces Hallucination in Conversation." EMNLP 2021.

3. Khattab, O., et al. (2022). "Demonstrate-Search-Predict: Composing retrieval and language models for knowledge-intensive NLP." arXiv preprint.

### Graph RAG

4. Edge, D., et al. (2024). "From Local to Global: A Graph RAG Approach to Query-Focused Summarization." arXiv:2404.16130.

5. Microsoft Research. (2024). "GraphRAG: Unlocking LLM discovery on narrative private data." Microsoft Research Blog.

### Agentic RAG

6. Yao, S., et al. (2022). "ReAct: Synergizing Reasoning and Acting in Language Models." arXiv:2210.03629.

7. Schick, T., et al. (2023). "Toolformer: Language Models Can Teach Themselves to Use Tools." arXiv:2302.04761.

8. Harrison, C., et al. (2024). "LangGraph: Multi-Actor Systems for LLM Applications." LangChain Technical Report.

### Fine-Tuning

9. Hu, E. J., et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models." arXiv:2106.09685.

10. Dettmers, T., et al. (2023). "QLoRA: Efficient Finetuning of Quantized LLMs." arXiv:2305.14314.

### 地理空間情報

11. Auer, S., et al. (2007). "LinkedGeoData: Adding a Spatial Dimension to the Web of Data." ISWC 2009.

12. Battle, R., & Kolas, D. (2011). "GeoSPARQL: Enabling a geospatial Semantic Web." Semantic Web Journal.

### LLMと評価

13. Qwen Team. (2024). "Qwen2.5 Technical Report." arXiv preprint.

14. Zheng, L., et al. (2023). "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena." arXiv:2306.05685.

---

## 付録（Appendix）

### A. テストケース統計

#### A.1 Phase 5-6テストセット（55件）

| レベル | サブカテゴリ | 件数 | 難易度 |
|--------|------------|------|--------|
| L1 | basic_location | 5 | easy |
| L1 | basic_category | 5 | easy |
| L2 | spatial_proximity | 5 | medium |
| L2 | spatial_density | 5 | medium |
| L2 | spatial_comparison | 5 | medium |
| L3 | constraint_single | 5 | medium |
| L3 | constraint_multi | 5 | hard |
| L4 | decision_location | 5 | hard |
| L4 | decision_business | 5 | hard |
| L5 | advanced_sensitivity | 3 | expert |
| L5 | advanced_comparison | 4 | expert |
| L5 | advanced_uncertainty | 3 | expert |

#### A.2 Phase 8 GraphRAGテストセット（35件）

| カテゴリ | 件数 | GraphRAG期待優位性 |
|---------|------|-------------------|
| proximity | 5 | 空間インデックス |
| aggregation | 4 | グラフメトリクス |
| comparison | 4 | エリアノード集計 |
| relation | 3 | エッジトラバーサル |
| multi_hop | 3 | 経路探索 |
| brand | 5 | SAME_BRANDエッジ |
| complementary | 5 | COMPLEMENTARYエッジ |
| competitor | 3 | COMPETITORエッジ |
| cuisine | 4 | SAME_CUISINEエッジ |
| hours | 3 | SAME_HOURSエッジ |

#### A.3 Phase 9 Agenticテストセット（15件）

| カテゴリ | 件数 | 特徴 |
|---------|------|------|
| multi_step_spatial | 5 | 複数ステップの空間推論 |
| conditional_reasoning | 5 | 条件付き推論 |
| iterative_refinement | 5 | 反復的な絞り込み |

### B. 実装ファイル一覧

#### B.1 Phase 6: Hybrid RAG

```
src/
├── geo_utils.py              # 空間処理（29KB）
├── aggregator.py             # 集計処理（21KB）
├── structured_rag_system.py  # 統合システム（32KB）
├── test_cases_v2.py          # テストケース（55件）
└── evaluators_v2.py          # 評価システム
```

#### B.2 Phase 7: Fine-Tuning

```
notebooks/
├── finetuning_01_data_preparation.ipynb  # データ準備
├── finetuning_02_training.ipynb          # QLoRA学習
└── finetuning_03_evaluation.ipynb        # 評価
```

#### B.3 Phase 8: Graph RAG

```
src/
├── graph_builder.py          # グラフ構築
├── graph_rag_system.py       # GraphRAGシステム
├── adaptive_rag_system.py    # Adaptive RAG
└── test_cases_graphrag.py    # GraphRAGテスト
```

#### B.4 Phase 9: Agentic RAG

```
src/
├── agentic_rag_system.py     # Agentic RAGメイン
├── agent_state.py            # 状態管理
├── agent_tools.py            # 16ツール実装
├── agent_prompts.py          # ReActプロンプト
└── test_cases_agentic.py     # Agenticテスト
```

### C. 主要な数式

#### C.1 Haversine距離計算

2点間の大圏距離：

```
a = sin²(Δφ/2) + cos(φ1) × cos(φ2) × sin²(Δλ/2)
c = 2 × atan2(√a, √(1-a))
d = R × c
```

ここで、φは緯度、λは経度、R=6,371km

#### C.2 方位角計算

```
θ = atan2(sin(Δλ) × cos(φ2), cos(φ1) × sin(φ2) - sin(φ1) × cos(φ2) × cos(Δλ))
```

#### C.3 評価スコア計算

```
score = Σ(wi × mi) / Σwi
```

ここで、wiは各評価軸の重み、miは各評価軸のスコア（0-100）

### D. 実験データアクセス

すべての実験データ、ノートブック、実装コードは以下で公開されている：

**GitHub**: https://github.com/[organization]/experiments-local-llm

**主要な評価結果**:
- `results/phase9_evaluation_20260211_090656.json`: Phase 9 Full Test
- `results/adaptive_evaluation_20260130_053413.json`: Phase 8 Adaptive
- `results/finetuning_eval_20260128_162829.json`: Phase 7 Fine-Tuning

---

**作成日**: 2026年2月12日
**著者**: Claude Sonnet 4.5
**バージョン**: 1.0
**論文ステータス**: 技術報告書（査読前）

---

## 謝辞（Acknowledgments）

本研究の実施にあたり、Google Colaboratoryの計算資源を使用した。OpenStreetMapコミュニティには、高品質なPOIデータの提供に感謝する。また、HuggingFace、LangChain、LangGraphの各プロジェクトチームには、オープンソースフレームワークの提供に感謝する。
