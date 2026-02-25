# POI問合せにおけるベクトル検索の役割分析

**作成日**: 2026-02-19
**プロジェクト**: experiments-local-llm
**関連フェーズ**: Phase 9-B（4エリアRAG比較評価）
**結論**: POI問合せにおいてベクトル検索の寄与は限定的であり、MCPツール＋構造化処理で代替可能

---

## 1. 分析の背景

Phase 9-Bでは、渋谷・新宿・池袋・東京の4エリア（約3,600 POI）を対象に4つのRAGシステムを130テストケース（520クエリ）で比較評価した。その過程で、**ベクトルDB（ChromaDB + multilingual-e5-base）の精度貢献度が想定より低い**ことが明らかになった。

本ドキュメントでは、実験データと実装の両面からベクトル検索の役割を分析し、Phase 10（全国展開）に向けたアーキテクチャ設計指針を示す。

---

## 2. 各RAGシステムにおけるベクトル検索の利用状況

### 2.1 実装上の役割

| システム | ベクトル検索の実装 | 構造化処理との関係 |
|---------|-----------------|-----------------|
| **Hybrid RAG** | `vectorstore.similarity_search(question, k=5)` を毎回実行 | 独立して並行実行。近接性検索時はベクトル結果を除外 |
| **Graph RAG** | **一切使用しない** | NetworkXグラフトラバーサルのみ |
| **Adaptive RAG** | Hybrid RAGにルーティングされた場合のみ間接使用 | GraphRAG選択時はベクトル検索なし |
| **Agentic RAG** | `tool_vector_search()` が定義されているが簡易キーワードマッチ。ChromaDB未接続 | ツール選択はLLM判断 |

### 2.2 Hybrid RAGの処理フロー詳細

```
query_with_structured_rag():
  1. エリア特定 → target_pois（エリア内POIに限定）
  2. 質問分析（ルールベース）→ requires_proximity 等のフラグ
  3. ベクトル検索 → similarity_search(question, k=5)     ← ChromaDB
  4. 構造化処理（質問タイプに応じて並列実行）            ← all_pois全件走査
     ├─ 集計: count_by_category(all_pois)
     ├─ 比較: compare_east_west(all_pois)
     ├─ 近接性: get_nearest_pois(all_pois)
     ├─ 空間フィルタ: filter_by_distance(all_pois)
     └─ 感度分析: compare_by_radius(all_pois)
  5. コンテキスト構築 → 両結果を結合（ただし近接性検索時はベクトル結果を除外）
  6. LLM回答生成
```

重要な点:
- 構造化処理はベクトル検索結果を**一切参照しない**。`all_pois`（全POIリスト）の直接走査で動作する
- ベクトル検索と構造化処理は**2つの独立したデータソース**として並行動作する
- 精度が最も求められる近接性クエリで、ベクトル検索結果は**意図的に除外**される（`_build_context`: `if search_results and not proximity`）

---

## 3. 実験データによる検証

### 3.1 ベクトル検索有無と成功率の関係

| システム | ベクトル検索 | 成功率 | 差分（vs Hybrid） |
|---------|-----------|-------|-----------------|
| Hybrid RAG | あり（毎回） | **96.2%** | — |
| Graph RAG | **なし** | 92.3% | -3.9% |
| Adaptive RAG | 部分的 | 92.3% | -3.9% |
| Agentic RAG | 簡易のみ | 90.8% | -5.4% |

Graph RAGはベクトル検索を全く使用せずに92.3%を達成している。Hybrid RAGとの3.9%の差は、ベクトル検索の有無ではなく以下の設計差で説明可能:
- ルールベース質問分析の精度差（Hybrid RAGの`analyze_question()`はGraph RAGの`analyze_graph_query()`より詳細）
- コンテキスト構築パイプラインの差（Hybrid RAGは構造化計算結果をより体系的に統合）

### 3.2 サブカテゴリ別の分析

ベクトル検索が最も価値を発揮するはずの「自由文検索」系サブカテゴリの結果:

| サブカテゴリ | Hybrid (ベクトルあり) | Graph (ベクトルなし) | ベクトル検索の寄与 |
|------------|---------------------|--------------------|-|
| basic_location | 62.5% | 62.5% | **差なし** |
| brand | 100.0% | 100.0% | **差なし** |
| landmark_origin | 96.3% | 100.0% | Graph RAGが上回る |
| competitor | 75.0% | 100.0% | Graph RAGが上回る |

basic_locationやbrandのようなPOI名検索でもベクトル検索の有無による差は見られない。むしろlandmark_originやcompetitorではGraph RAG（ベクトルなし）の方が高い成功率を記録しており、グラフ構造による関係性検索の方が有効であることを示している。

### 3.3 多次元品質スコアの比較

| システム | Avg Composite | Avg Evidence | Avg Reasoning |
|---------|-------------|-------------|--------------|
| Hybrid RAG | **52.2** | **2.34** | 1.43 |
| Graph RAG | 46.1 | 1.64 | 1.42 |
| Adaptive RAG | 48.1 | 1.70 | 1.24 |
| Agentic RAG | 47.8 | 2.33 | **1.68** |

evidence_score（根拠の明示度）ではHybrid RAGが最高だが、これはベクトル検索結果にPOI名・座標が含まれるためコンテキストの情報密度が上がることに起因する。ただしAgentic RAG（ベクトル未接続）も2.33とほぼ同等であり、ツール出力の構造化情報で同等の根拠明示が達成できている。

---

## 4. POIデータの本質とベクトル検索の不適合性

### 4.1 POI問合せの本質

POIに対する典型的な問合せとその本質的な操作:

| 質問タイプ | 例 | 本質的操作 | 適切な処理方法 |
|-----------|---|----------|-------------|
| 最寄り検索 | 「最寄りのコンビニは？」 | 距離計算＋ソート | `ORDER BY ST_Distance() LIMIT n` |
| 件数集計 | 「カフェは何件ある？」 | カテゴリ別COUNT | `COUNT(*) WHERE category = ?` |
| 方角比較 | 「東と西でどちらが多い？」 | 空間分割＋集計 | `GROUP BY (lon > center_lon)` |
| 距離フィルタ | 「500m以内の店は？」 | 空間バッファ＋フィルタ | `ST_DWithin(geom, center, 500)` |
| 複合条件 | 「500m以内の24時間営業は？」 | 空間＋属性フィルタ | `WHERE ST_DWithin(...) AND hours = '24h'` |
| 感度分析 | 「半径を変えると結論は変わる？」 | 複数半径での繰り返し計算 | 複数の`COUNT(*) WHERE ST_DWithin(...)` |

これらはすべて**構造化データに対する構造化クエリ**であり、GIS/空間DBの基本操作である。ベクトル空間での類似度計算とは本質的に異なる。

### 4.2 ベクトル埋め込みによる情報損失

POIデータの主要属性と、ベクトル埋め込みにおける扱い:

| 属性 | データ型 | ベクトル化時の問題 |
|------|---------|----------------|
| **座標** (lat, lon) | 数値ペア | 空間的近接性はベクトル空間で保存されない。近い座標が近いベクトルになる保証がない |
| **カテゴリ** | 階層的列挙型 | 「飲食店/カフェ」と「飲食店/ラーメン」の関係は埋め込みで正確に表現困難 |
| **営業時間** | 構造化テキスト | 「24時間」と「深夜3時まで」の包含関係はベクトル類似度では判定できない |
| **名称** | 固有名詞 | ブランド名の完全一致は文字列比較の方が確実 |

POIの埋め込みテキスト（例: 「スターバックス 渋谷スクランブル交差点店 飲食店/カフェ 渋谷区」）は意味的類似性の検索には使えるが、「渋谷駅から最も近いカフェ」という質問に対して、ベクトル類似度の上位5件に空間的最近傍が含まれる保証はない。

### 4.3 ベクトル検索 vs 構造化クエリの特性比較

| 特性 | ベクトル検索 | 構造化クエリ（PostGIS等） |
|------|-----------|----------------------|
| **結果の確定性** | 近似的（類似度上位k件） | 確定的（条件合致の全件） |
| **空間演算** | 不可 | ネイティブ対応 |
| **集計・カウント** | 不可 | ネイティブ対応 |
| **複合条件** | 困難 | WHERE句で自在に組合せ |
| **説明可能性** | 低（なぜこの5件が選ばれたか不透明） | 高（SQL条件がそのまま根拠） |
| **スケーラビリティ** | k=5固定で500万件から選ぶ精度問題 | 空間インデックスでO(log n) |
| **得意領域** | 非構造化テキストからの情報検索 | 構造化データの条件検索・集計 |

---

## 5. MCPツール＋構造化処理アーキテクチャの優位性

### 5.1 アーキテクチャ比較

**現行（ベクトル検索RAG）**:
```
質問 → 埋め込み → ChromaDB類似度検索(k=5) → 上位5件のPOIテキスト
                                                ↓
    構造化処理（全POI走査） ─────────────────→ コンテキスト結合 → LLM回答
```

問題点:
- ベクトル検索と構造化処理が独立しており、両方のコストが発生
- ベクトル検索のk=5は500万POIに対して精度が劣化する
- 構造化処理の全件走査はO(n)でスケールしない

**提案（MCPツール＋PostGIS）**:
```
質問 → LLM（意図理解・ツール選択）
         ↓
       MCPツール呼び出し
         ├─ get_nearest_pois(category, n)  → PostGIS: ORDER BY ST_Distance LIMIT n
         ├─ count_in_radius(center, r)     → PostGIS: COUNT WHERE ST_DWithin
         ├─ compare_directions(category)   → PostGIS: GROUP BY direction
         └─ filter_by_constraints(...)     → PostGIS: WHERE条件
         ↓
       確定的な結果 → LLM（自然文回答生成）
```

利点:
- ベクトルDB/埋め込みモデルが不要（VRAM/メモリ節約）
- 結果が確定的で説明可能
- PostGIS空間インデックスで500万POI以上にスケール
- LLMは意図理解とツール選択に専念（得意領域）
- 空間計算はGISエンジンに委譲（得意領域）

### 5.2 MCPツール設計案

Phase 9-Bの構造化処理関数をMCPツールに対応付ける:

| 現行関数 (`geo_utils.py` / `aggregator.py`) | MCPツール | PostGIS実装 |
|---|---|---|
| `get_nearest_pois(pois, category, top_n)` | `get_nearest_pois` | `ORDER BY ST_Distance(geom, ?) WHERE category = ? LIMIT ?` |
| `filter_by_distance(pois, radius)` | `filter_by_radius` | `SELECT * WHERE ST_DWithin(geom, ?, ?)` |
| `compare_east_west(pois, category)` | `compare_directions` | `COUNT(*) GROUP BY (ST_X(geom) > ST_X(?))` |
| `count_by_category(pois)` | `count_by_category` | `COUNT(*) GROUP BY category WHERE ST_DWithin(...)` |
| `compare_by_radius(pois, r1, r2)` | `compare_radii` | 2回の`COUNT WHERE ST_DWithin` |
| `analyze_radius_sensitivity(pois, cat, radii)` | `analyze_sensitivity` | 複数半径の`COUNT`をバッチ実行 |
| `detect_target_area(question, areas)` | `detect_area` | ジオコーディングAPI + `ST_Within` |

### 5.3 ベクトル検索が価値を持ち得るニッチ

公平を期すため、ベクトル検索が代替困難な場面も挙げる:

| ユースケース | 例 | ベクトル検索が必要な理由 |
|------------|---|---------------------|
| 曖昧な自然言語検索 | 「駅前のあのおしゃれなイタリアン」 | POI説明文・口コミとの意味的マッチング |
| カテゴリ横断検索 | 「子連れで行きやすい場所」 | 構造化カテゴリに載らない属性 |
| 類似POI推薦 | 「この店に似た雰囲気の店」 | 非構造化属性の類似度計算 |

ただし、これらも以下の方法で代替可能:
- **LLMの言語理解能力**: 「おしゃれなイタリアン」→ `category="イタリア料理"` への変換はLLM自身が可能
- **属性タグの拡充**: 「子連れ」「おしゃれ」等をPOIの構造化属性として追加
- **口コミ・レビュー検索**: POI本体とは別にレビューテキストに対するベクトル検索を行う（POI自体の検索は構造化クエリ）

---

## 6. 結論と設計指針

### 6.1 結論

1. **POI問合せはGISの基本操作（空間クエリ＋属性フィルタ＋集計）であり、ベクトル検索の適用領域ではない**。Phase 9-Bの実験データもこれを裏付ける。

2. **MCPツール＋PostGIS構造化処理は、ベクトル検索層を省いた分だけシンプルであり、結果の確定性・説明可能性・スケーラビリティの全てで優れる**。

3. **LLMの役割は意図理解・ツール選択・自然文回答生成に限定すべき**であり、POI検索自体をLLMやベクトル空間に委ねるのは非効率。

### 6.2 Phase 10設計指針

| 設計方針 | 内容 |
|---------|------|
| **ベクトルDB不要** | ChromaDB/埋め込みモデルを廃止。VRAM/メモリを節約 |
| **PostGIS中心** | 空間クエリ・属性フィルタ・集計をPostGIS空間SQLに集約 |
| **MCPツール化** | `geo_utils.py`/`aggregator.py`の関数をMCPツールとして公開 |
| **LLMは意図理解に専念** | ツール選択とパラメータ抽出、回答の自然文生成 |
| **口コミ等の非構造化データは別系統** | 必要に応じてレビューテキストのベクトル検索を独立して構築（POI検索とは分離） |

---

## 付録: 参照データ

- **実験結果**: `results/phase9b_evaluation_20260219_005810.json`
- **Hybrid RAG実装**: `src/structured_rag_system.py` L837-930 (`query_with_structured_rag`)
- **Graph RAG実装**: `src/graph_rag_system.py` L426-484 (`query`)
- **構造化処理**: `src/geo_utils.py`, `src/aggregator.py`
- **実験レポート**: `docs/reports/PHASE9B_MULTI_AREA_EXPERIMENT_REPORT.md`
- **引き継ぎ資料**: `docs/handovers/HANDOVER_PHASE9B.md` セクション4.5
