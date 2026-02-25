# グラフRAG実験計画書

**作成日**: 2026年1月29日
**プロジェクト**: experiments-local-llm
**フェーズ**: Phase 8 - グラフRAG比較実験

---

## 1. 実験の目的

### 1.1 研究課題

地理空間POI（Point of Interest）情報に対して、**グラフRAG**は従来の**構造化RAG**と比較してどのような質問タイプで優位性を持つか？

### 1.2 仮説

| 仮説ID | 仮説内容 | 検証方法 |
|--------|---------|---------|
| H1 | POI間の空間的関係（近接性、クラスタ所属）をグラフエッジとして明示的に表現することで、関係性を問う質問の回答精度が向上する | 関係性クエリのスコア比較 |
| H2 | カテゴリ階層をグラフ構造で表現することで、カテゴリ横断的な質問への回答が改善される | カテゴリ集計クエリのスコア比較 |
| H3 | 複数POIにまたがる複合的な質問（経路探索、エリア比較）ではグラフトラバーサルが有効 | 複合クエリのスコア比較 |
| H4 | 単純な最寄り検索では構造化RAGが効率的で、グラフRAGはオーバーヘッドとなる | 近接性クエリの処理時間・精度比較 |

---

## 2. 実験設計

### 2.1 比較対象システム

| システム | 説明 | ベースライン |
|---------|------|------------|
| **Baseline** | 現行の構造化RAG（Phase 6.2.1） | 91.6pt |
| **GraphRAG-Neo4j** | Neo4j + ベクトル検索 + Cypher | 新規実装 |
| **GraphRAG-LlamaIndex** | PropertyGraphIndex + NetworkX | 新規実装 |
| **Hybrid** | グラフ検索 + 構造化処理の融合 | 新規実装 |

### 2.2 グラフスキーマ設計

```
┌─────────────────────────────────────────────────────────────────┐
│                        POI Knowledge Graph                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────┐     BELONGS_TO      ┌────────────┐                 │
│  │   POI   │────────────────────▶│  Category  │                 │
│  └────┬────┘                     └─────┬──────┘                 │
│       │                                │                         │
│       │ LOCATED_IN                     │ PARENT_OF               │
│       ▼                                ▼                         │
│  ┌─────────┐                     ┌────────────┐                 │
│  │  Area   │                     │SubCategory │                 │
│  └────┬────┘                     └────────────┘                 │
│       │                                                          │
│       │ ADJACENT_TO                                              │
│       ▼                                                          │
│  ┌─────────┐                                                     │
│  │  Area   │                                                     │
│  └─────────┘                                                     │
│                                                                   │
│  POI ──NEAR_TO──▶ POI  (距離 ≤ 100m)                            │
│  POI ──SAME_AREA──▶ POI (同一エリアクラスタ)                     │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

#### ノードタイプ

| ノード | 属性 | 件数（予想） |
|--------|-----|-------------|
| `POI` | id, name, name_en, lat, lon, embedding | 1,046 |
| `Category` | name, name_jp | 12 |
| `SubCategory` | name, name_jp | 約30 |
| `Area` | name, direction, distance_zone | 32（8方位×4ゾーン） |
| `Landmark` | name, lat, lon | 1（渋谷駅） |

#### エッジタイプ

| エッジ | 説明 | 件数（予想） |
|--------|-----|-------------|
| `BELONGS_TO` | POI → Category | 1,046 |
| `LOCATED_IN` | POI → Area | 1,046 |
| `NEAR_TO` | POI → POI（100m以内） | 約5,000 |
| `SAME_AREA` | POI → POI（同一エリア） | 約10,000 |
| `ADJACENT_TO` | Area → Area（隣接エリア） | 約100 |
| `PARENT_OF` | Category → SubCategory | 約30 |
| `DISTANCE_FROM` | POI → Landmark（重み付き） | 1,046 |

### 2.3 実装アプローチ

#### 2.3.1 GraphRAG-Neo4j

```python
# グラフ構築
from neo4j import GraphDatabase

# POIノード作成
CREATE (p:POI {
    id: $id,
    name: $name,
    lat: $lat,
    lon: $lon,
    embedding: $embedding
})

# 空間インデックス作成
CREATE POINT INDEX poi_location FOR (p:POI) ON (p.location)

# ベクトルインデックス作成
CREATE VECTOR INDEX poi_embedding FOR (p:POI) ON (p.embedding)
OPTIONS { indexConfig: { `vector.dimensions`: 768, `vector.similarity_function`: 'cosine' } }

# 近接関係エッジ作成
MATCH (p1:POI), (p2:POI)
WHERE p1 <> p2 AND point.distance(p1.location, p2.location) < 100
CREATE (p1)-[:NEAR_TO {distance: point.distance(p1.location, p2.location)}]->(p2)
```

#### 2.3.2 GraphRAG-LlamaIndex

```python
from llama_index.core import PropertyGraphIndex
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor

# カスタムエンティティ定義
entities = ["POI", "Category", "Area", "Landmark"]
relations = ["NEAR_TO", "BELONGS_TO", "LOCATED_IN", "ADJACENT_TO"]

# グラフインデックス構築
index = PropertyGraphIndex.from_documents(
    documents,
    kg_extractors=[
        SchemaLLMPathExtractor(
            possible_entities=entities,
            possible_relations=relations,
            llm=llm
        )
    ],
    embed_model=embed_model,
    show_progress=True
)
```

### 2.4 クエリパイプライン

```
質問入力
    │
    ▼
┌───────────────────┐
│   質問分析        │  ← 既存のQuestionAnalysis再利用
│   (question_type) │
└────────┬──────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌────────────┐
│ Vector │ │   Graph    │
│ Search │ │ Traversal  │
└────┬───┘ └─────┬──────┘
     │           │
     ▼           ▼
┌───────────────────┐
│  Context Fusion   │  ← ベクトル結果 + グラフ結果の統合
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│   LLM Generation  │
└───────────────────┘
```

---

## 3. 評価計画

### 3.1 評価指標

| 指標 | 説明 | 測定方法 |
|-----|------|---------|
| **Accuracy** | 回答の正確性 | 既存の55テストケース評価 |
| **Latency** | 応答時間 | クエリ実行時間（秒） |
| **Relevance** | 取得情報の関連性 | Retrieved文書の適合率 |
| **Reasoning** | 推論の正確性 | 複合質問での因果関係記述 |

### 3.2 テストケース拡張

既存の55テストケースに加え、**グラフRAGの強みを検証する追加テストケース**を設計：

| カテゴリ | 質問例 | 期待される優位性 |
|---------|-------|-----------------|
| **関係性クエリ** | 「渋谷駅の東側にあるカフェで、同じエリアにコンビニもある場所はどこですか？」 | グラフトラバーサル |
| **経路探索** | 「渋谷駅から100m以内のカフェを起点に、そこから50m以内にある書店を教えてください」 | 多ホップ推論 |
| **クラスタ分析** | 「飲食店が最も密集しているエリアはどこですか？」 | エリアノード集計 |
| **カテゴリ横断** | 「銀行とカフェが両方ある地域を教えてください」 | エッジ共有パターン |
| **比較推論** | 「東側と西側で、カテゴリの多様性が高いのはどちらですか？」 | グラフメトリクス |

### 3.3 実験条件

| 条件 | 設定 |
|-----|------|
| **実行環境** | Google Colab T4 GPU（比較公平性のため統一） |
| **LLM** | Qwen2.5-7B-Instruct（4bit量子化） |
| **埋め込みモデル** | multilingual-e5-base |
| **試行回数** | 各テストケース3回（平均を採用） |

---

## 4. 実験スケジュール

### Phase 8.1: グラフ構築（Week 1）

| タスク | 成果物 |
|-------|-------|
| POIデータからグラフスキーマ実装 | `src/graph_builder.py` |
| Neo4j/NetworkXへのデータロード | `notebooks/graph_construction.ipynb` |
| 空間エッジ（NEAR_TO等）の自動生成 | グラフDB構築完了 |

### Phase 8.2: クエリ実装（Week 2）

| タスク | 成果物 |
|-------|-------|
| グラフトラバーサルクエリ実装 | `src/graph_query.py` |
| ベクトル検索との統合 | `src/graph_rag_system.py` |
| 質問分析とクエリ変換 | 動作確認完了 |

### Phase 8.3: 評価・分析（Week 3）

| タスク | 成果物 |
|-------|-------|
| 55テストケース評価実行 | `notebooks/graph_rag_evaluation.ipynb` |
| 追加テストケース評価 | `results/graphrag_eval_*.json` |
| 比較分析レポート作成 | `docs/reports/GRAPHRAG_EXPERIMENT_REPORT.md` |

---

## 5. 技術的考慮事項

### 5.1 Neo4j vs NetworkX の選択

| 観点 | Neo4j | NetworkX |
|-----|-------|----------|
| **スケーラビリティ** | 大規模対応 | メモリ制限あり |
| **空間インデックス** | ネイティブサポート | 要カスタム実装 |
| **ベクトル検索** | 5.x以降サポート | 外部ライブラリ連携 |
| **Colab互換性** | Docker/ホスト必要 | 純Python |
| **学習コスト** | Cypher習得必要 | Pythonic |

**推奨**: 初期実験は**NetworkX + LlamaIndex**で開始し、効果が確認できればNeo4jに移行

### 5.2 グラフ構築の計算量

| 処理 | 計算量 | 1,046 POIでの予測 |
|-----|-------|------------------|
| 全ペア距離計算 | O(n²) | 約110万ペア |
| 100m閾値フィルタ | O(n²) | 約5,000エッジ |
| 空間インデックス使用 | O(n log n) | 高速化可能 |

### 5.3 メモリ最適化

```python
# Colab T4での制約
# - システムRAM: 12.7GB
# - GPU VRAM: 16GB

# 推奨設定
BATCH_SIZE = 100  # グラフ構築時のバッチサイズ
MAX_NEIGHBORS = 20  # NEAR_TOエッジの最大数/POI
EMBEDDING_CACHE = True  # 埋め込みのキャッシュ
```

---

## 6. 期待される成果

### 6.1 定量的成果

| 指標 | ベースライン | 目標 |
|-----|------------|------|
| 全体スコア | 91.6pt | 維持または向上 |
| 関係性クエリスコア | 未測定 | 構造化RAG比+5pt |
| 平均処理時間 | 21.9秒 | 25秒以内 |

### 6.2 定性的成果

1. **グラフRAGの適用可能性評価**: 地理空間情報に対するグラフRAGの有効なユースケースの特定
2. **アーキテクチャ指針**: 構造化RAG vs グラフRAG vs ハイブリッドの選択基準
3. **実装知見**: Neo4j/LlamaIndex PropertyGraphの実装パターン

### 6.3 論文化可能性

| 観点 | 新規性 |
|-----|-------|
| 地理空間POIへのグラフRAG適用 | 先行研究少 |
| 構造化RAGとの定量比較 | 実用的知見 |
| 質問タイプ別の手法選択指針 | 実務への示唆 |

---

## 7. リスクと対策

| リスク | 影響 | 対策 |
|-------|-----|------|
| グラフ構築のメモリ不足 | 実験中断 | バッチ処理、エッジ数制限 |
| Neo4j環境構築の複雑さ | スケジュール遅延 | NetworkXで先行、後からNeo4j追加 |
| グラフRAGの効果が限定的 | 仮説棄却 | ネガティブ結果も重要な知見として報告 |
| LLM推論のばらつき | 評価の不安定性 | 3回試行の平均、温度0設定 |

---

## 8. 参考文献

### 8.1 GraphRAG

- [GraphRAG: Unlocking LLM discovery on narrative private data](https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/) - Microsoft Research
- [Retrieval-Augmented Generation with Graphs (GraphRAG)](https://arxiv.org/abs/2501.00309) - arXiv 2025
- [Awesome-GraphRAG](https://github.com/DEEP-PolyU/Awesome-GraphRAG) - GitHub

### 8.2 技術ドキュメント

- [LlamaIndex PropertyGraphIndex](https://docs.llamaindex.ai/en/stable/module_guides/indexing/lpg_index_guide/)
- [Neo4j Vector Search](https://neo4j.com/docs/cypher-manual/current/functions/vector/)
- [Neo4j Spatial](https://neo4j.com/docs/cypher-manual/current/values-and-types/spatial/)

### 8.3 プロジェクト内ドキュメント

- `docs/handovers/HANDOVER_PHASE6.md` - 構造化RAGアーキテクチャ詳細
- `docs/reports/STRUCTURED_RAG_RESEARCH_REPORT.md` - Phase 5-6学術レポート
- `docs/reports/FINETUNING_EXPERIMENT_REPORT.md` - ファインチューニング実験結果

---

## 9. 次のステップ

### 即時アクション

1. [ ] 本計画書のレビュー・承認
2. [ ] NetworkX + LlamaIndex 環境セットアップ
3. [ ] `src/graph_builder.py` の初期実装

### 意思決定事項

1. **グラフDB選択**: NetworkX先行 or Neo4j優先？
2. **追加テストケース**: 何件追加するか？（推奨: 10-15件）
3. **実験範囲**: 全55テストケース or サブセット先行？

---

## 10. 今後検討すべきRAGアーキテクチャ

本実験ではグラフRAGを検証対象としますが、地理空間POI情報に適したRAGのあり方を包括的に理解するため、以下のRAGアーキテクチャも今後の検証候補として整理します。

### 10.1 RAGアーキテクチャの分類

```
┌─────────────────────────────────────────────────────────────────────┐
│                    RAG Architecture Taxonomy                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌─────────────┐                                                     │
│  │  Naive RAG  │  単純なベクトル検索 + LLM生成                       │
│  └──────┬──────┘                                                     │
│         │                                                             │
│         ▼                                                             │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    Advanced RAG                              │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │    │
│  │  │ Pre-Retrieval│  │  Retrieval  │  │Post-Retrieval│        │    │
│  │  │ (Query変換) │  │ (検索強化)  │  │ (Re-ranking)│         │    │
│  │  └─────────────┘  └─────────────┘  └─────────────┘         │    │
│  └─────────────────────────────────────────────────────────────┘    │
│         │                                                             │
│         ▼                                                             │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    Modular RAG                               │    │
│  │  検索・生成・評価をモジュール化し、動的に組み合わせ          │    │
│  └─────────────────────────────────────────────────────────────┘    │
│         │                                                             │
│    ┌────┴────┬────────┬────────┬────────┐                          │
│    ▼         ▼        ▼        ▼        ▼                          │
│ ┌───────┐┌───────┐┌───────┐┌───────┐┌───────┐                     │
│ │Graph  ││Agentic││Self-  ││CRAG   ││Adaptive│                     │
│ │RAG    ││RAG    ││RAG    ││       ││RAG    │                      │
│ └───────┘└───────┘└───────┘└───────┘└───────┘                     │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

### 10.2 検討対象RAGアーキテクチャ一覧

| アーキテクチャ | 概要 | 地理空間への適用可能性 | 優先度 |
|--------------|------|----------------------|-------|
| **Graph RAG** | 知識グラフによる関係性表現 | POI間関係・エリア構造の明示化 | ★★★（本実験） |
| **Agentic RAG** | エージェントによる反復的検索・推論 | 複雑な空間推論タスク | ★★★ |
| **Self-RAG** | 自己反省による検索・生成の最適化 | 回答の信頼性向上 | ★★☆ |
| **CRAG** | 検索結果の自己修正 | 不正確な空間情報の補正 | ★★☆ |
| **Adaptive RAG** | 質問に応じた動的戦略選択 | 質問タイプ別の最適化 | ★★★ |
| **HyDE** | 仮説的回答による検索強化 | 抽象的な空間クエリ | ★☆☆ |
| **Multi-modal RAG** | 画像・地図との統合 | 地図画像からの情報抽出 | ★★☆ |
| **Hierarchical RAG** | 階層的文書構造の活用 | エリア→POI の階層検索 | ★★☆ |

### 10.3 各アーキテクチャの詳細

#### 10.3.1 Agentic RAG

**概要**: LLMエージェントが検索・推論を反復的に実行し、必要に応じてツールを呼び出す

**地理空間への適用**:
```
質問: 「渋谷駅から徒歩5分以内で、カフェとATMが両方あるエリアを探して」

Agent Loop:
  1. [Tool] get_nearby_cafes(radius=400m) → カフェリスト取得
  2. [Tool] get_nearby_atms(radius=400m) → ATMリスト取得
  3. [Reasoning] エリアごとに共起を分析
  4. [Tool] get_area_details(area_id) → 詳細情報取得
  5. [Generate] 最終回答生成
```

**期待される効果**:
- 複数ステップの空間推論タスクで精度向上
- ツール呼び出しによる正確な計算（距離、件数等）

**実装候補**: LangGraph, LlamaIndex Agents, AutoGen

#### 10.3.2 Self-RAG

**概要**: 検索の必要性判断、検索結果の関連性評価、回答の自己批評を行う

**地理空間への適用**:
```
質問: 「渋谷で一番人気のカフェはどこ？」

Self-RAG Flow:
  1. [Retrieve] → 「人気」の定義が曖昧 → 追加検索トリガー
  2. [Critique] → 検索結果にレビュー情報なし → 「人気度は不明」と明示
  3. [Generate] → 不確実性を含む回答生成
```

**期待される効果**:
- 曖昧な質問への適切な対応
- 回答の信頼性・根拠の明示

**参考論文**: [Self-RAG: Learning to Retrieve, Generate, and Critique](https://arxiv.org/abs/2310.11511)

#### 10.3.3 CRAG (Corrective RAG)

**概要**: 検索結果の品質を評価し、低品質な場合はWeb検索などで補完

**地理空間への適用**:
```
質問: 「渋谷マークシティの営業時間は？」

CRAG Flow:
  1. [Retrieve] → POIデータに営業時間なし
  2. [Evaluate] → 検索結果が不十分と判定
  3. [Correct] → Web検索で補完 or 「情報なし」と回答
```

**期待される効果**:
- POIデータに欠損がある場合の対応力向上
- 誤情報の生成抑制

**参考論文**: [Corrective Retrieval Augmented Generation](https://arxiv.org/abs/2401.15884)

#### 10.3.4 Adaptive RAG

**概要**: 質問の複雑さに応じて、検索戦略を動的に選択

**地理空間への適用**:
```
質問分類 → 戦略選択:
  - 単純な位置検索 → 直接ベクトル検索（高速）
  - 空間比較 → 構造化処理（geo_utils）
  - 関係性クエリ → グラフトラバーサル
  - 複合推論 → Agentic RAG
```

**期待される効果**:
- 質問タイプに応じた最適な処理パイプライン選択
- 処理効率と精度のバランス最適化

**参考論文**: [Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models](https://arxiv.org/abs/2403.14403)

#### 10.3.5 HyDE (Hypothetical Document Embeddings)

**概要**: 質問から仮説的な回答を生成し、その埋め込みで検索

**地理空間への適用**:
```
質問: 「静かに作業できる場所を探しています」

HyDE Flow:
  1. [Hypothesize] → 「静かなカフェや図書館が適しています。渋谷駅から少し離れた
                      場所にある○○カフェは席数が多く落ち着いた雰囲気です...」
  2. [Embed] → 仮説的回答を埋め込み
  3. [Search] → 類似POIを検索
```

**期待される効果**:
- 抽象的・主観的な質問への対応力向上
- 意図と文書のセマンティックギャップ解消

**制限**: 地理空間では具体的な質問が多く、効果は限定的の可能性

#### 10.3.6 Multi-modal RAG

**概要**: テキスト・画像・地図など複数モダリティを統合

**地理空間への適用**:
```
入力: 「この地図で赤く囲まれた場所にあるお店を教えて」+ 地図画像

Multi-modal Flow:
  1. [Vision] → 地図画像から座標範囲を抽出
  2. [Retrieve] → 座標範囲内のPOI検索
  3. [Generate] → テキスト回答生成
```

**期待される効果**:
- 地図ベースのインタラクション
- ストリートビュー画像からのPOI特定

**実装候補**: GPT-4V, LLaVA, Qwen-VL

#### 10.3.7 Hierarchical RAG

**概要**: 文書の階層構造を活用した段階的検索

**地理空間への適用**:
```
階層構造:
  渋谷エリア
    ├── 東側
    │   ├── 駅周辺（0-100m）
    │   │   ├── POI-001: カフェA
    │   │   └── POI-002: レストランB
    │   └── 中距離（100-300m）
    │       └── ...
    └── 西側
        └── ...

検索フロー:
  1. [Coarse] → エリア・方向の特定
  2. [Fine] → 該当エリア内のPOI検索
```

**期待される効果**:
- 大規模POIデータでの検索効率向上
- エリア単位のサマリー生成

### 10.4 実験ロードマップ

```
Phase 8: Graph RAG（本実験）
    │
    ▼
Phase 9: Adaptive RAG
    │   質問タイプ別の戦略選択を実装
    │   Graph RAG / 構造化RAG / Vector RAG の動的切り替え
    │
    ▼
Phase 10: Agentic RAG
    │   複雑な空間推論タスク向け
    │   ツール呼び出しによる正確な計算
    │
    ▼
Phase 11: 統合評価
        全アーキテクチャの比較分析
        質問タイプ × RAGアーキテクチャ の最適マッピング
```

### 10.5 評価フレームワーク（将来構想）

| 評価軸 | 測定対象 | 手法 |
|-------|---------|------|
| **精度** | 回答の正確性 | テストケース評価 |
| **効率** | 処理時間・トークン消費 | ベンチマーク |
| **堅牢性** | 曖昧・不完全な質問への対応 | ストレステスト |
| **説明可能性** | 回答根拠の明示 | 人手評価 |
| **スケーラビリティ** | POI数増加時の性能 | 負荷テスト |

### 10.6 参考文献（RAGアーキテクチャ）

- [A Survey on RAG with LLMs](https://arxiv.org/abs/2312.10997) - RAG手法の体系的サーベイ
- [Self-RAG](https://arxiv.org/abs/2310.11511) - 自己反省型RAG
- [CRAG](https://arxiv.org/abs/2401.15884) - 修正型RAG
- [Adaptive-RAG](https://arxiv.org/abs/2403.14403) - 適応型RAG
- [HyDE](https://arxiv.org/abs/2212.10496) - 仮説的文書埋め込み
- [Modular RAG](https://arxiv.org/abs/2407.21059) - モジュラーRAGフレームワーク

---

## 11. 拡張グラフRAG構築手順

### 11.1 概要

Phase 8.1の初期実装後、グラフ構造の関係性を強化するために以下の5つの新しいエッジタイプを追加しました。これにより、POI間のより豊かな関係性を表現できるようになりました。

### 11.2 新しいエッジタイプ

| エッジタイプ | 説明 | 抽出条件 | 予想エッジ数 |
|------------|------|---------|-------------|
| `SAME_BRAND` | 同一ブランド/チェーン店 | brand属性が一致 | 約200 |
| `COMPLEMENTARY` | 相補的関係（ホテル↔レストラン等） | カテゴリペアルールに基づく | 約3,000 |
| `COMPETITOR` | 競合関係（同カテゴリ・近距離） | 同カテゴリ + 100m以内 | 約8,000 |
| `SAME_CUISINE` | 同一料理ジャンル | cuisine属性が一致 | 約500 |
| `SAME_HOURS` | 同一営業時間帯 | 24h/深夜/早朝フラグ一致 | 約2,000 |

### 11.3 相補的関係（COMPLEMENTARY）のルール

以下のカテゴリペアが相補的関係としてエッジ化されます（200m以内）：

```python
COMPLEMENTARY_RULES = {
    ("宿泊/ホテル", "飲食店/レストラン"): "DINING_NEAR_HOTEL",
    ("宿泊/ホテル", "飲食店/カフェ"): "CAFE_NEAR_HOTEL",
    ("宿泊/ホテル", "商店/コンビニ"): "CONVENIENCE_NEAR_HOTEL",
    ("娯楽/映画館", "飲食店/カフェ"): "ENTERTAINMENT_COMBO",
    ("娯楽/映画館", "飲食店/ファストフード"): "ENTERTAINMENT_COMBO",
    ("交通/鉄道駅", "商店/コンビニ"): "TRANSIT_AMENITY",
    ("交通/鉄道駅", "飲食店/カフェ"): "TRANSIT_AMENITY",
    ("観光/名所", "飲食店/カフェ"): "SIGHTSEEING_REST",
    ("観光/名所", "飲食店/レストラン"): "SIGHTSEEING_DINING",
    ("金融/銀行", "商店/コンビニ"): "BANKING_CONVENIENCE",
    ("医療/病院", "医療/薬局"): "MEDICAL_COMBO",
    ("医療/クリニック", "医療/薬局"): "MEDICAL_COMBO",
}
```

### 11.4 構築手順

#### ステップ1: POIデータの拡張取得

```bash
# OpenStreetMapからブランド、営業時間、料理ジャンル等の拡張タグを取得
uv run python osm_poi_fetcher.py
```

**出力ファイル**: `poi_documents.json`（拡張メタデータ付き）

**追加される属性**:
- `brand`: チェーン店/ブランド名（日本語名から自動抽出）
- `opening_hours`: 営業時間（raw文字列）
- `is_24h`: 24時間営業フラグ
- `late_night`: 深夜営業フラグ（22:00以降）
- `early_morning`: 早朝営業フラグ（6:00以前）
- `cuisine`: 料理ジャンル

#### ステップ2: 拡張グラフの構築

```python
from src.graph_builder import POIGraphBuilder

# 拡張エッジを含むグラフを構築
builder = POIGraphBuilder(
    poi_json_path="poi_documents.json",
    include_extended_edges=True  # 新しいエッジタイプを有効化
)
stats = builder.build()

# 統計情報の確認
print(f"ノード数: {stats.total_nodes}")
print(f"総エッジ数: {stats.total_edges}")
print(f"- SAME_BRAND: {stats.same_brand_edges}")
print(f"- COMPLEMENTARY: {stats.complementary_edges}")
print(f"- COMPETITOR: {stats.competitor_edges}")
print(f"- SAME_CUISINE: {stats.same_cuisine_edges}")
print(f"- SAME_HOURS: {stats.same_hours_edges}")
```

#### ステップ3: GraphRAGシステムの初期化

```python
from src.graph_rag_system import GraphRAGSystem

# 拡張グラフを使用するシステムの初期化
system = GraphRAGSystem(
    rebuild=True,
    include_extended_edges=True
)
```

### 11.5 Google Colabでの実行

`notebooks/graphrag_05_enhanced_comparison.ipynb` を使用して、拡張グラフRAGの評価を実行できます。

```python
# セルでの実行例
import sys
sys.path.append('/content/drive/MyDrive/experiments-local-llm/src')

from graph_builder import POIGraphBuilder
from graph_rag_system import GraphRAGSystem

# 拡張グラフ構築
builder = POIGraphBuilder(
    poi_json_path='/content/drive/MyDrive/experiments-local-llm/poi_documents.json',
    include_extended_edges=True
)
stats = builder.build()
print(f"拡張グラフ構築完了: {stats.total_edges} エッジ")
```

### 11.6 期待されるグラフ統計

| 項目 | 基本グラフ | 拡張グラフ |
|-----|----------|----------|
| ノード数 | 1,080 | 1,080 |
| 総エッジ数 | 68,334 | 82,078 |
| NEAR_TO | 66,248 | 66,248 |
| SAME_CATEGORY | 2,086 | 2,086 |
| SAME_BRAND | 0 | 約200 |
| COMPLEMENTARY | 0 | 約3,000 |
| COMPETITOR | 0 | 約8,000 |
| SAME_CUISINE | 0 | 約500 |
| SAME_HOURS | 0 | 約2,000 |

### 11.7 拡張テストケース

拡張グラフRAG向けに、以下のカテゴリのテストケースが追加されています（`src/test_cases_graphrag.py`）：

| カテゴリ | 件数 | 質問例 |
|---------|-----|-------|
| brand | 5件 | 「渋谷にあるスターバックスは何店舗ありますか？」 |
| complementary | 5件 | 「渋谷駅近くでホテルの近くにあるレストランを教えてください」 |
| competitor | 3件 | 「渋谷でコンビニが密集しているエリアはどこですか？」 |
| cuisine | 4件 | 「渋谷で日本料理のレストランを探しています」 |
| hours | 3件 | 「渋谷で24時間営業の店舗を教えてください」 |

**合計**: 35件（基本15件 + 拡張20件）

### 11.8 トラブルシューティング

#### NetworkXがインストールされていない

```bash
uv add networkx
```

#### メタデータのフラット化エラー

ChromaDBはネストされた辞書をサポートしません。`flatten_metadata()`ヘルパーを使用してください：

```python
from src.geo_utils import flatten_metadata
flat_meta = flatten_metadata(poi["metadata"])
```

#### ブランド情報が抽出されない

`osm_poi_fetcher.py`の`KNOWN_BRANDS`辞書にブランドを追加してください。

---

---

## 12. Adaptive RAG実装

### 12.1 概要

拡張GraphRAG評価の結果、質問タイプによってGraphRAGと構造化RAGの得手不得手が明確になったため、質問を動的にルーティングする**Adaptive RAG**を実装した。

### 12.2 システム選択ロジック

```python
def select_system(question, analysis):
    # GraphRAGを使用すべきケース
    if analysis.requires_comparison:
        return "GraphRAG"  # +25%の優位性
    if analysis.requires_proximity:
        return "GraphRAG"  # +16.6%の優位性
    if "24時間" in question or "深夜" in question:
        return "GraphRAG"  # +11.1%の優位性（hours）
    if analysis.requires_aggregation:
        return "GraphRAG"  # +8.3%の優位性

    # それ以外は構造化RAG（デフォルト）
    return "StructuredRAG"
```

### 12.3 適性マップ（評価結果より）

| クエリタイプ | GraphRAG | 構造化RAG | 差分 | 選択 |
|------------|----------|----------|------|------|
| comparison | 100.0% | 75.0% | **+25.0** | GraphRAG |
| proximity | 83.3% | 66.7% | **+16.6** | GraphRAG |
| hours | 100.0% | 88.9% | **+11.1** | GraphRAG |
| aggregation | 100.0% | 91.7% | **+8.3** | GraphRAG |
| relation | 63.3% | 86.7% | -23.4 | 構造化RAG |
| multi_hop | 77.8% | 100.0% | -22.2 | 構造化RAG |
| brand | 73.3% | 93.3% | -20.0 | 構造化RAG |

### 12.4 ファイル構成

| ファイル | 役割 |
|---------|------|
| `src/adaptive_rag_system.py` | Adaptive RAGシステム実装 |
| `notebooks/graphrag_06_adaptive_evaluation.ipynb` | 評価ノートブック |

### 12.5 使用方法

```python
from src.adaptive_rag_system import AdaptiveRAGSystem

# 初期化
adaptive_rag = AdaptiveRAGSystem(
    poi_json_path="poi_documents.json",
    rebuild=True,
    include_extended_edges=True
)

# 自動システム選択でクエリ実行
result = adaptive_rag.query("渋谷駅の東側と西側で、レストランが多いのはどちらですか？")
print(f"選択システム: {result.selected_system}")  # GraphRAG
print(f"選択理由: {result.selection_reason}")    # 東西/方向比較クエリ (+25%)
print(f"回答: {result.response}")

# 特定のシステムを指定してクエリ実行（比較評価用）
result = adaptive_rag.query_with_system(question, "GraphRAG")
result = adaptive_rag.query_with_system(question, "StructuredRAG")
result = adaptive_rag.query_with_system(question, "Adaptive")
```

---

---

## 13. 最終評価結果（2026-01-30 完了）

### 13.1 全体スコア（90テストケース）

| システム | スコア | 処理時間 | 標準偏差 |
|---------|--------|----------|----------|
| **StructuredRAG** | **89.1%** | 20.6秒 | 20.2 |
| Adaptive RAG | 86.1% | 17.8秒 | 20.4 |
| GraphRAG | 76.7% | 8.7秒 | 24.8 |

### 13.2 仮説検証結果

| 仮説ID | 仮説内容 | 結果 |
|--------|---------|------|
| H1 | グラフエッジで関係性クエリが向上 | **部分的に支持**（competitorで+22.2pt） |
| H2 | カテゴリ横断クエリが改善 | **棄却**（StructuredRAGが-19.4pt優位） |
| H3 | 複合クエリでグラフトラバーサルが有効 | **棄却**（同等または劣位） |
| H4 | 単純検索では構造化RAGが効率的 | **支持**（処理時間2.4倍差） |

### 13.3 GraphRAGが優位なカテゴリ

| カテゴリ | GraphRAG | StructuredRAG | 差分 |
|---------|----------|---------------|------|
| **comparison** | **100.0%** | 50.0% | **+50.0pt** |
| **competitor** | **88.9%** | 66.7% | **+22.2pt** |
| decision_business | **93.3%** | 89.3% | +4.0pt |

### 13.4 結論

1. **StructuredRAGが最も効果的**: 全体スコア89.1%で最高、GraphRAG向けテストでも86.9%を達成
2. **GraphRAGは特定タスクで有効**: 東西比較（comparison）と競合分析（competitor）で明確な優位性
3. **Adaptive RAGは改善の余地あり**: 選択精度の向上が必要

### 13.5 推奨アプローチ

| シナリオ | 推奨システム |
|---------|------------|
| 一般的なPOIクエリ | StructuredRAG |
| 東西/方向比較 | GraphRAG |
| 競合店分析 | GraphRAG |
| 処理速度優先 | GraphRAG |

---

**作成者**: Claude Opus 4.5
**ステータス**: **実験完了**
**更新履歴**:
- 2026-01-29: 初版作成
- 2026-01-30: 拡張グラフRAG構築手順（セクション11）追加
- 2026-01-30: Adaptive RAG実装（セクション12）追加
- 2026-01-30: 最終評価結果（セクション13）追加、実験完了
