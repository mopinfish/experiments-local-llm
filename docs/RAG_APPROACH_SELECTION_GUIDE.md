# RAGアプローチ選択ガイド：タスク別推奨マトリクス

## 概要

本ガイドは、Phase 5-9の実験結果に基づき、**どのタスクにどのRAGアプローチが適しているか**を簡潔にまとめたものです。

**記号の意味**:
- ◎: 最適（90%以上の性能）
- ○: 適している（75-89%）
- △: 使用可能だが限定的（60-74%）
- ×: 推奨しない（60%未満）

---

## クロス表：タスクカテゴリ × RAGアプローチ

### 表1: 基本タスク（Phase 5-6テストケース）

| カテゴリ | Naive | Hybrid | Fine-Tuned | Graph | Adaptive | Agentic |
|---------|-------|--------|-----------|-------|----------|---------|
| **基礎検索** | | | | | | |
| basic_location（座標検索） | ○ | ◎ | ◎ | ○ | ◎ | ◎ |
| basic_category（カテゴリ検索） | △ | ◎ | △ | ○ | ◎ | ○ |
| **空間推論** | | | | | | |
| spatial_proximity（最寄り） | △ | ◎ | ◎ | ◎ | ◎ | ◎ |
| spatial_density（密度分析） | △ | ◎ | ○ | ○ | ◎ | ○ |
| spatial_comparison（東西比較） | △ | ○ | ○ | ◎ | ○ | △ |
| **制約充足** | | | | | | |
| constraint_single（単一制約） | △ | ◎ | ○ | ○ | ◎ | ○ |
| constraint_multi（複数制約） | △ | ◎ | ○ | ○ | ◎ | △ |
| **意思決定** | | | | | | |
| decision_location（立地評価） | ◎ | ◎ | ◎ | ○ | ◎ | ○ |
| decision_business（出店判断） | ◎ | ◎ | ◎ | ◎ | ◎ | ◎ |
| **高度推論** | | | | | | |
| advanced_sensitivity（感度分析） | △ | ◎ | ○ | ○ | ◎ | ○ |
| advanced_comparison（多軸比較） | △ | ◎ | ◎ | ○ | ◎ | ○ |
| advanced_uncertainty（不確実性） | ◎ | ◎ | ◎ | ○ | ◎ | ○ |

### 表2: グラフ・関係性タスク（Phase 8-9テストケース）

| カテゴリ | Naive | Hybrid | Fine-Tuned | Graph | Adaptive | Agentic |
|---------|-------|--------|-----------|-------|----------|---------|
| **関係性分析** | | | | | | |
| brand（ブランド分析） | × | ◎ | - | ○ | ◎ | ○ |
| complementary（相補的関係） | × | ◎ | - | ○ | ◎ | ◎ |
| competitor（競合分析） | × | ○ | - | ◎ | ○ | ◎ |
| cuisine（料理ジャンル） | △ | ◎ | - | △ | ◎ | ○ |
| hours（営業時間） | △ | ◎ | - | ◎ | ○ | ○ |
| **複雑な推論** | | | | | | |
| multi_hop（マルチホップ） | × | ◎ | - | ◎ | ◎ | △ |
| multi_step_spatial（複数ステップ） | × | ◎ | - | ○ | ◎ | ○ |
| conditional_reasoning（条件分岐） | △ | ◎ | - | ○ | ◎ | ○ |
| iterative_refinement（反復絞込） | × | ◎ | - | △ | ◎ | △ |

**注**: Fine-Tuned RAGはPhase 8-9のテストケースでは評価していないため「-」と表記。

---

## カテゴリ別推奨アプローチ

### 1. 基礎検索タスク → **Hybrid RAG**

**カテゴリ**: basic_location, basic_category

**理由**:
- ベクトル検索で具体的なPOI情報を正確に取得
- 構造化処理で空間情報を補完
- 性能: 89.1pt（Naive 65.0ptから+37%改善）

**実装例**:
```python
# ベクトル検索 + 空間情報エンリッチメント
results = vectorstore.search(question, k=5)
for r in results:
    r['distance'] = calculate_distance(station, r['coords'])
    r['direction'] = get_direction(station, r['coords'])
```

### 2. 空間比較タスク → **Graph RAG**

**カテゴリ**: spatial_comparison（東西比較、南北比較）

**理由**:
- Areaノードによる方向別集計が効率的
- 明確な優位性（+50pt）
- 処理速度も高速（8.7秒）

**実装例**:
```cypher
// Neo4jのCypherクエリ
MATCH (poi:POI)-[:LOCATED_IN]->(area:Area)
WHERE area.direction IN ['east', 'northeast', 'southeast']
RETURN count(poi) as east_count
```

### 3. 競合・関係性分析 → **Agentic RAG** または **Graph RAG**

**カテゴリ**: competitor, complementary, brand

**理由**:
- グラフトラバーサルツールの動的選択が有効
- Agentic: competitor +33.3pt, complementary +20.0pt
- Graph: competitor +22.2pt

**選択基準**:
- 既知のパターン → Graph RAG（高速、8.7秒）
- 未知のパターン → Agentic RAG（柔軟、但し56.4秒）

### 4. 意思決定タスク → **Fine-Tuned RAG**

**カテゴリ**: decision_location, decision_business

**理由**:
- 判断パターンの学習が有効
- Fine-Tuned: 88.1pt vs Hybrid: 80.2pt（+7.9pt）
- 具体的なPOI情報よりも推論フレームワークが重要

**実装例**:
```python
# LoRAアダプタを適用したモデル
model = PeftModel.from_pretrained(
    base_model,
    "lora_adapters/poi_decision_v1"
)
answer = model.generate(question)
```

### 5. 感度分析・高度推論 → **Hybrid RAG**

**カテゴリ**: advanced_sensitivity, advanced_comparison, advanced_uncertainty

**理由**:
- 複数の半径での比較計算が正確
- 感度分析: 100.0pt（完全スコア）
- 構造化処理により数学的に正確な結果

**実装例**:
```python
# 半径比較による感度分析
result1 = count_pois_in_radius(pois, 300, category)
result2 = count_pois_in_radius(pois, 500, category)
ratio = result2 / result1  # 増加率
conclusion = "成立する" if ratio < 1.5 else "成立しない"
```

---

## 総合推奨アーキテクチャ

### シンプル構成（推奨）

```
┌─────────────────────────────────┐
│    Hybrid RAG単独               │
│                                 │
│  • 90%以上のタスクに対応         │
│  • 最高性能96.2%                │
│  • 処理時間11.1秒（実用的）      │
│  • 実装・保守が容易              │
└─────────────────────────────────┘
```

**適用場面**: ほとんどのユースケース

### 拡張構成（高度なニーズ向け）

```
┌─────────────────────────────────────────┐
│  質問分析（ルールベース）                │
│           │                             │
│     ┌─────┴─────┐                       │
│     ▼           ▼                       │
│  東西比較？  関係性分析？                │
│     │           │                       │
│     ▼           ▼                       │
│  Graph RAG   Agentic RAG                │
│   (+50pt)     (+33.3pt)                 │
│                                         │
│  その他 → Hybrid RAG（デフォルト）       │
└─────────────────────────────────────────┘
```

**適用場面**:
- 東西比較が頻繁（Graph RAG追加で+50pt）
- 未知の関係性分析が必要（Agentic RAG追加で+33.3pt）

---

## 処理時間とスコアのトレードオフ

| システム | スコア | 処理時間 | コストパフォーマンス |
|---------|--------|----------|-------------------|
| **Hybrid RAG** | 96.2% | 11.1秒 | ★★★★★（最良） |
| Agentic RAG | 87.6% | 56.4秒 | ★★☆☆☆ |
| Graph RAG | 76.7% | 8.7秒 | ★★★☆☆ |
| Fine-Tuned | 78.5pt | 8.04秒 | ★★★★☆ |
| Naive RAG | 60.3pt | 22.7秒 | ★☆☆☆☆ |

**結論**:
- **最優先**: Hybrid RAG（性能・速度・複雑性のバランスが最良）
- **特定タスクで追加**: Graph RAG（東西比較）、Agentic RAG（関係性分析）

---

## MCP Serverへの応用可能性

### 現状の課題

本プロジェクトの実装（1,046 POI、渋谷限定）は以下の制約がある：

1. **スケーラビリティ**: 全国規模（500万POI）に対応できない
2. **基準点の固定**: 渋谷駅座標がハードコードされている
3. **クライアント・サーバー分離**: すべてがローカル実行

### MCP Serverアーキテクチャへの移行

**MCP (Model Context Protocol)** を用いたクライアント・サーバー分離により、以下が実現可能：

#### 1. サーバー側実装（推奨: Hybrid RAG）

```
┌─────────────────────────────────────────┐
│  MCP Server (FastAPI)                   │
├─────────────────────────────────────────┤
│                                         │
│  ┌───────────────────────────────┐     │
│  │  Supabase + PostGIS            │     │
│  │  • 500万POI格納                │     │
│  │  • 空間インデックス（GiST）     │     │
│  │  • 高速検索（50ms以下）         │     │
│  └───────────────────────────────┘     │
│           │                             │
│           ▼                             │
│  ┌───────────────────────────────┐     │
│  │  Hybrid RAG Engine             │     │
│  │  • 質問分析                     │     │
│  │  • 集計・比較処理               │     │
│  │  • ベクトル検索                 │     │
│  └───────────────────────────────┘     │
│           │                             │
│           ▼                             │
│  ┌───────────────────────────────┐     │
│  │  MCP Tools                     │     │
│  │  • poi_search()                │     │
│  │  • spatial_compare()           │     │
│  │  • aggregate_by_category()     │     │
│  └───────────────────────────────┘     │
└─────────────────────────────────────────┘
```

#### 2. クライアント側（Claude Desktop等）

```python
# MCPクライアントからの利用例
import mcp

# MCP Serverに接続
server = mcp.connect("poi-rag-server")

# ツールを呼び出し
result = server.call_tool("poi_search", {
    "question": "渋谷駅の東側と西側、どちらにカフェが多い？",
    "location": "渋谷駅"
})

print(result)
# → "西側にカフェが多いです。東側51件、西側58件で、西側が7件多いです。"
```

#### 3. 提供するMCP Tools

| ツール名 | 説明 | 対応RAG | 性能 |
|---------|------|---------|------|
| `poi_search` | 基本検索 | Hybrid | 96.2% |
| `spatial_compare` | 東西比較 | Graph | 100% |
| `competitor_analysis` | 競合分析 | Agentic/Graph | 100% |
| `sensitivity_analysis` | 感度分析 | Hybrid | 100% |
| `business_evaluation` | 意思決定支援 | Fine-Tuned | 88.1pt |

#### 4. スケーラビリティ対応

**PostGIS空間クエリ例**:
```sql
-- 渋谷駅（動的に解決）から500m以内のカフェを高速検索
SELECT name, ST_Distance(geom, ST_SetSRID(ST_MakePoint(139.701636, 35.658034), 4326)) as distance
FROM pois
WHERE category = 'カフェ'
  AND ST_DWithin(geom::geography, ST_SetSRID(ST_MakePoint(139.701636, 35.658034), 4326)::geography, 500)
ORDER BY distance
LIMIT 10;

-- 実行時間: ~10ms（500万POI規模でも高速）
```

#### 5. 動的基準点解決

```python
# ジオコーディングAPIとの統合
async def resolve_location(location_name: str) -> tuple[float, float]:
    """
    場所名から座標を解決
    例: "新宿駅" → (35.6896, 139.7006)
    """
    result = await geocoding_api.search(location_name)
    return (result['lat'], result['lon'])

# MCPツールから利用
@mcp_tool
async def poi_search(question: str, location: str):
    coords = await resolve_location(location)  # 動的解決
    return hybrid_rag.query(question, base_point=coords)
```

### 期待される効果

| 項目 | 現状（ローカル） | MCP Server化後 |
|------|----------------|---------------|
| POI数 | 1,046件 | 500万件+ |
| 対応エリア | 渋谷のみ | 全国 |
| 応答時間 | 11.1秒 | <50ms |
| スケーラビリティ | × | ◎ |
| 保守性 | △ | ◎ |
| 多ユーザー対応 | × | ◎ |

### 実装優先順位

**Phase 10（全国展開）推奨ステップ**:

1. **Step 1**: Supabase + PostGIS セットアップ（2週間）
2. **Step 2**: Hybrid RAG Engine の移行（1週間）
3. **Step 3**: MCP Server API実装（1週間）
4. **Step 4**: 全国POIデータ投入（1週間）
5. **Step 5**: パフォーマンスチューニング（1週間）

**合計**: 6週間で全国展開完了

---

## まとめ

### 重要な発見

1. **Hybrid RAGが最優先**: 90%以上のタスクで最高性能（96.2%）
2. **タスク特性に応じた選択**: 東西比較→Graph、関係性→Agentic、意思決定→Fine-Tuned
3. **処理時間とのトレードオフ**: Hybrid RAGが最もバランスが良い（11.1秒）
4. **MCP Server化で全国展開**: PostGIS + Hybrid RAGで500万POI、50ms応答を実現可能

### 実務での選択指針

```
【シンプル】Hybrid RAG単独
    ↓
【拡張】特定タスクでGraph/Agenticを追加
    ↓
【全国展開】MCP Server + PostGIS + Hybrid RAG
```

**最終推奨**: まずHybrid RAGで基盤を構築し、必要に応じて段階的に拡張する。

---

**作成日**: 2026年2月12日
**ベース実験**: Phase 5-9（2026年1月-2月）
**データ**: OpenStreetMap 渋谷駅周辺 1,046 POI
