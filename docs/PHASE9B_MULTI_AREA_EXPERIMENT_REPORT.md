# Phase 9-B: 4エリアRAG比較評価 実験レポート

**研究期間**: 2026-02-18 〜 2026-02-XX
**プロジェクト**: experiments-local-llm
**LLMモデル**: Qwen2.5-7B-Instruct (4-bit NF4量子化)
**埋め込みモデル**: intfloat/multilingual-e5-base
**評価規模**: 130テストケース × 4 RAGシステム = 520クエリ
**実行環境**: Google Colab T4 GPU (15GB VRAM)

---

## 概要 (Abstract)

{PLACEHOLDER: 実験結果に基づく概要を記述。主要な発見（最も成績の良いシステム、エリア汎化性の程度、クロスエリア対応力の差異）を3-4文で要約する。}

---

## 1. 序論

### 1.1 研究背景

Phase 6〜9では渋谷駅周辺（1,047 POI）に限定してRAG比較実験を行い、以下の成果を得た：

| Phase | アプローチ | 渋谷単一スコア | テスト件数 | 特徴 |
|-------|----------|--------------|----------|------|
| Phase 6 | Hybrid RAG（構造化RAG） | 96.2% | 105 | ルールベース質問分析 + 計算処理 + ベクトル検索 |
| Phase 8 | Graph RAG | 76.7% | 90 | NetworkXナレッジグラフ + グラフトラバーサル |
| Phase 8 | Adaptive RAG | 86.1% | 90 | クエリ複雑度に基づくシステム動的選択 |
| Phase 9 | Agentic RAG | 87.6% | 105 | LangGraph + ReActパターン + 16ツール |

しかし、これらの結果は渋谷駅周辺という**単一エリア**に最適化された環境での評価であり、以下の疑問が残る：

1. **汎化性能**: 渋谷以外のエリアでも同等の精度が出るのか？
2. **エリア横断クエリ**: エリア間比較クエリに対応できるのか？
3. **スケーラビリティ**: POI数増加（1,047→~3,600）時の性能劣化は？
4. **エリア特定能力**: 質問文からターゲットエリアを正しく特定できるか？

### 1.2 研究目的

本実験は、渋谷・新宿・池袋・東京の4エリアに対象範囲を拡大し、4つのRAGシステムを公平に比較評価することで、以下を明らかにする：

1. 各RAGアプローチのエリア汎化性能
2. クロスエリアクエリにおけるシステム間の優劣
3. エリア特定精度が全体性能に与える影響
4. Phase 10（全国展開）で採用すべきRAGアーキテクチャの選定根拠

### 1.3 検証する仮説

| ID | 仮説 | 判定基準 |
|----|------|---------|
| **H1** | Hybrid RAGの構造化パイプラインは、エリア特定が正しければ渋谷以外でも同等精度を維持する | 新エリア成功率 ≥ 渋谷の90%で支持 |
| **H2** | Agentic RAGは複数ツール呼び分けによりクロスエリアで他システムを上回る | クロスエリア成功率が他の最高値+10%以上で支持 |
| **H3** | エリア特定精度が全体の成功率を律速する | エリア特定誤りが失敗ケースの50%以上で支持 |
| **H4** | POI総数3.5倍増加でベクトル検索精度が低下する | 統合collection vs エリア別で成功率差10%以上で支持 |
| **H5** | 中国語混入率はエリアによって変動しない（モデル固有の問題） | 4エリア間の混入率差が±3%以内で支持 |

---

## 2. 実験設計

### 2.1 対象エリア

| エリア | 基準駅座標 | POI数 | エリア特性 |
|--------|----------|-------|-----------|
| 渋谷 (shibuya) | (35.658034, 139.701636) | ~1,000 | 商業・娯楽中心、若年層向け |
| 新宿 (shinjuku) | (35.689592, 139.700413) | ~1,200 | ビジネス＋歓楽街＋商業 |
| 池袋 (ikebukuro) | (35.728926, 139.711086) | ~800 | 商業施設集中、サンシャイン |
| 東京 (tokyo) | (35.681236, 139.767125) | ~600 | オフィス・観光・交通ハブ |

### 2.2 テストケース構成

合計130件を以下の4カテゴリに分類：

| カテゴリ | 件数 | 説明 |
|---------|------|------|
| **A: エリア内クエリ** | 80 | 4エリア × 20件（L1-L5各4件） |
| **B: クロスエリアクエリ** | 20 | エリア間比較・横断クエリ |
| **C: ランドマーク起点クエリ** | 15 | ランドマーク名からのエリア暗黙推定 |
| **D: エリア特定テスト** | 15 | エリア特定の精度を直接評価 |

レベル別分布:

| レベル | 説明 | 件数 |
|--------|------|------|
| L1 | 基本検索 | ~30 |
| L2 | 空間推論 | ~30 |
| L3 | 制約付き | ~25 |
| L4 | 意思決定 | ~25 |
| L5 | 高度推論 | ~20 |

### 2.3 評価対象システム

| # | システム | 実装 | 特徴 | 追加VRAM |
|---|---------|------|------|---------|
| 1 | **Hybrid RAG** | StructuredRAGSystem | ルールベース分析 + 構造化計算 + ベクトル検索 | なし |
| 2 | **Graph RAG** | GraphRAGSystem | NetworkXグラフ + LLM回答生成 | なし (CPU) |
| 3 | **Adaptive RAG** | AdaptiveRAGSystem | Hybrid/Graphの動的切り替え | なし (再利用) |
| 4 | **Agentic RAG** | AgenticRAGSystem | LangGraph + ReAct + 16ツール | なし |

### 2.4 評価指標

| 指標 | 説明 | 計算方法 |
|------|------|---------|
| **success_rate** | キーワードヒットによる成功率 | hit_rate ≥ 閾値 のケース割合 |
| **avg_keyword_hit_rate** | 平均キーワードヒット率 | 全ケースのhit_rateの平均 |
| **area_detection_accuracy** | エリア特定精度 | detected_area == expected_area の割合 |
| **area_consistency** | エリア間スコア分散 | エリア別success_rateの分散（低いほど良い） |
| **avg_time_sec** | 平均実行時間 | 全ケースの実行時間平均 |
| **language_issue_count** | 中国語混入件数 | 回答に中国語文字が含まれるケース数 |

### 2.5 実行環境

- **GPU**: NVIDIA Tesla T4 (15GB VRAM)
- **LLM**: Qwen2.5-7B-Instruct (4-bit NF4量子化, ~4.5GB VRAM)
- **埋め込みモデル**: intfloat/multilingual-e5-base
- **ベクトルDB**: ChromaDB (in-memory, 5 collections)
- **グラフDB**: NetworkX (in-memory, 4 area graphs)
- **チェックポイント**: システム別JSONファイルで中断再開可能

---

## 3. 実験結果

### 3.1 全体比較

| システム | Success Rate | Avg Hit Rate | Avg Time (s) | Errors | Language Issues |
|---------|-------------|-------------|--------------|--------|----------------|
| hybrid_rag | {PLACEHOLDER}% | {PLACEHOLDER} | {PLACEHOLDER} | {PLACEHOLDER} | {PLACEHOLDER} |
| graph_rag | {PLACEHOLDER}% | {PLACEHOLDER} | {PLACEHOLDER} | {PLACEHOLDER} | {PLACEHOLDER} |
| adaptive_rag | {PLACEHOLDER}% | {PLACEHOLDER} | {PLACEHOLDER} | {PLACEHOLDER} | {PLACEHOLDER} |
| agentic_rag | {PLACEHOLDER}% | {PLACEHOLDER} | {PLACEHOLDER} | {PLACEHOLDER} | {PLACEHOLDER} |

**ランキング (by success rate)**:
1. {PLACEHOLDER}
2. {PLACEHOLDER}
3. {PLACEHOLDER}
4. {PLACEHOLDER}

### 3.2 エリア別結果

#### Hybrid RAG

| エリア | Success Rate | Hit Rate | n |
|--------|-------------|----------|---|
| shibuya | {PLACEHOLDER}% | {PLACEHOLDER} | {PLACEHOLDER} |
| shinjuku | {PLACEHOLDER}% | {PLACEHOLDER} | {PLACEHOLDER} |
| ikebukuro | {PLACEHOLDER}% | {PLACEHOLDER} | {PLACEHOLDER} |
| tokyo | {PLACEHOLDER}% | {PLACEHOLDER} | {PLACEHOLDER} |

#### Graph RAG

| エリア | Success Rate | Hit Rate | n |
|--------|-------------|----------|---|
| shibuya | {PLACEHOLDER}% | {PLACEHOLDER} | {PLACEHOLDER} |
| shinjuku | {PLACEHOLDER}% | {PLACEHOLDER} | {PLACEHOLDER} |
| ikebukuro | {PLACEHOLDER}% | {PLACEHOLDER} | {PLACEHOLDER} |
| tokyo | {PLACEHOLDER}% | {PLACEHOLDER} | {PLACEHOLDER} |

#### Adaptive RAG

| エリア | Success Rate | Hit Rate | n |
|--------|-------------|----------|---|
| shibuya | {PLACEHOLDER}% | {PLACEHOLDER} | {PLACEHOLDER} |
| shinjuku | {PLACEHOLDER}% | {PLACEHOLDER} | {PLACEHOLDER} |
| ikebukuro | {PLACEHOLDER}% | {PLACEHOLDER} | {PLACEHOLDER} |
| tokyo | {PLACEHOLDER}% | {PLACEHOLDER} | {PLACEHOLDER} |

#### Agentic RAG

| エリア | Success Rate | Hit Rate | n |
|--------|-------------|----------|---|
| shibuya | {PLACEHOLDER}% | {PLACEHOLDER} | {PLACEHOLDER} |
| shinjuku | {PLACEHOLDER}% | {PLACEHOLDER} | {PLACEHOLDER} |
| ikebukuro | {PLACEHOLDER}% | {PLACEHOLDER} | {PLACEHOLDER} |
| tokyo | {PLACEHOLDER}% | {PLACEHOLDER} | {PLACEHOLDER} |

#### 渋谷ベースライン比較

| システム | Phase 9 渋谷単一 | Phase 9-B 渋谷 | 差分 |
|---------|-----------------|---------------|------|
| Hybrid RAG | 96.2% | {PLACEHOLDER}% | {PLACEHOLDER} |
| Agentic RAG | 87.6% | {PLACEHOLDER}% | {PLACEHOLDER} |

### 3.3 レベル別結果

| レベル | Hybrid RAG | Graph RAG | Adaptive RAG | Agentic RAG |
|--------|-----------|-----------|-------------|-------------|
| L1 | {PLACEHOLDER}% | {PLACEHOLDER}% | {PLACEHOLDER}% | {PLACEHOLDER}% |
| L2 | {PLACEHOLDER}% | {PLACEHOLDER}% | {PLACEHOLDER}% | {PLACEHOLDER}% |
| L3 | {PLACEHOLDER}% | {PLACEHOLDER}% | {PLACEHOLDER}% | {PLACEHOLDER}% |
| L4 | {PLACEHOLDER}% | {PLACEHOLDER}% | {PLACEHOLDER}% | {PLACEHOLDER}% |
| L5 | {PLACEHOLDER}% | {PLACEHOLDER}% | {PLACEHOLDER}% | {PLACEHOLDER}% |

### 3.4 Subcategory別結果

{PLACEHOLDER: subcategory別の4システム比較テーブル}

### 3.5 クロスエリアクエリ結果

| システム | Success Rate | n | Avg Time (s) |
|---------|-------------|---|-------------|
| hybrid_rag | {PLACEHOLDER}% | {PLACEHOLDER} | {PLACEHOLDER} |
| graph_rag | {PLACEHOLDER}% | {PLACEHOLDER} | {PLACEHOLDER} |
| adaptive_rag | {PLACEHOLDER}% | {PLACEHOLDER} | {PLACEHOLDER} |
| agentic_rag | {PLACEHOLDER}% | {PLACEHOLDER} | {PLACEHOLDER} |

### 3.6 エリア特定精度

| システム | Accuracy | Correct/Total |
|---------|---------|--------------|
| hybrid_rag | {PLACEHOLDER}% | {PLACEHOLDER}/{PLACEHOLDER} |
| graph_rag | {PLACEHOLDER}% | {PLACEHOLDER}/{PLACEHOLDER} |
| adaptive_rag | {PLACEHOLDER}% | {PLACEHOLDER}/{PLACEHOLDER} |
| agentic_rag | {PLACEHOLDER}% | {PLACEHOLDER}/{PLACEHOLDER} |

エリア特定タイプ別:
- 明示的（駅名指定）: {PLACEHOLDER}%
- 暗黙的（ランドマーク推定）: {PLACEHOLDER}%
- 不明（フォールバック）: {PLACEHOLDER}%

### 3.7 エリア一貫性（スコア分散）

| システム | Variance | 解釈 |
|---------|---------|------|
| hybrid_rag | {PLACEHOLDER} | {PLACEHOLDER} |
| graph_rag | {PLACEHOLDER} | {PLACEHOLDER} |
| adaptive_rag | {PLACEHOLDER} | {PLACEHOLDER} |
| agentic_rag | {PLACEHOLDER} | {PLACEHOLDER} |

### 3.8 中国語混入分析

| システム | 全体混入率 | 渋谷 | 新宿 | 池袋 | 東京 |
|---------|----------|------|------|------|------|
| hybrid_rag | {PLACEHOLDER}% | {PLACEHOLDER}% | {PLACEHOLDER}% | {PLACEHOLDER}% | {PLACEHOLDER}% |
| graph_rag | {PLACEHOLDER}% | {PLACEHOLDER}% | {PLACEHOLDER}% | {PLACEHOLDER}% | {PLACEHOLDER}% |
| adaptive_rag | {PLACEHOLDER}% | {PLACEHOLDER}% | {PLACEHOLDER}% | {PLACEHOLDER}% | {PLACEHOLDER}% |
| agentic_rag | {PLACEHOLDER}% | {PLACEHOLDER}% | {PLACEHOLDER}% | {PLACEHOLDER}% | {PLACEHOLDER}% |

### 3.9 実行時間分析

| システム | 全体平均 | エリア内平均 | クロスエリア平均 | 合計時間 |
|---------|---------|------------|----------------|---------|
| hybrid_rag | {PLACEHOLDER}s | {PLACEHOLDER}s | {PLACEHOLDER}s | {PLACEHOLDER}min |
| graph_rag | {PLACEHOLDER}s | {PLACEHOLDER}s | {PLACEHOLDER}s | {PLACEHOLDER}min |
| adaptive_rag | {PLACEHOLDER}s | {PLACEHOLDER}s | {PLACEHOLDER}s | {PLACEHOLDER}min |
| agentic_rag | {PLACEHOLDER}s | {PLACEHOLDER}s | {PLACEHOLDER}s | {PLACEHOLDER}min |

---

## 4. 仮説検証

### 4.1 H1: 構造化パイプラインのエリア汎化性

> **仮説**: Hybrid RAGの構造化パイプラインは、エリア特定が正しければ渋谷以外でも同等精度を維持する。

**判定**: {PLACEHOLDER: 支持 / 部分的支持 / 棄却}

| エリア | Hybrid RAG Success Rate | 渋谷比 |
|--------|------------------------|--------|
| 渋谷 | {PLACEHOLDER}% | 100% |
| 新宿 | {PLACEHOLDER}% | {PLACEHOLDER}% |
| 池袋 | {PLACEHOLDER}% | {PLACEHOLDER}% |
| 東京 | {PLACEHOLDER}% | {PLACEHOLDER}% |

{PLACEHOLDER: 判定根拠と考察を記述}

### 4.2 H2: エージェント型のクロスエリア優位性

> **仮説**: Agentic RAGは複数ツール呼び分けによりクロスエリアで他システムを上回る。

**判定**: {PLACEHOLDER: 支持 / 部分的支持 / 棄却}

| システム | クロスエリア Success Rate |
|---------|------------------------|
| hybrid_rag | {PLACEHOLDER}% |
| graph_rag | {PLACEHOLDER}% |
| adaptive_rag | {PLACEHOLDER}% |
| agentic_rag | {PLACEHOLDER}% |

{PLACEHOLDER: 判定根拠と考察を記述}

### 4.3 H3: エリア特定がボトルネックになる

> **仮説**: エリア特定精度が全体の成功率を律速する。

**判定**: {PLACEHOLDER: 支持 / 部分的支持 / 棄却}

{PLACEHOLDER: 失敗ケースのうちエリア特定誤りが占める割合を分析}

### 4.4 H4: POI数の増加が検索精度に影響する

> **仮説**: POI総数3.5倍増加でベクトル検索精度が低下する。

**判定**: {PLACEHOLDER: 支持 / 部分的支持 / 棄却}

{PLACEHOLDER: 統合collection vs エリア別collectionの比較データが必要。本実験では統合collectionのみを使用しているため、間接的な評価として渋谷エリアのPhase 9（1,047 POI）との比較で判定する。}

### 4.5 H5: 中国語混入がエリアによって変動する

> **仮説**: 中国語混入率はエリアによって変動しない（モデル固有の問題）。

**判定**: {PLACEHOLDER: 支持 / 棄却}

{PLACEHOLDER: 3.8節のデータに基づいて判定}

---

## 5. 考察

### 5.1 Phase 9（渋谷単一）との比較

{PLACEHOLDER: Phase 9ベースラインとの比較考察}

**Phase 9ベースライン** (渋谷単一105件):
- Structured RAG: 96.2% success, 11.08s avg
- Agentic RAG: 87.6% success, 56.36s avg

### 5.2 各RAGアプローチの適性まとめ

{PLACEHOLDER: 実験結果に基づく適性マトリクス}

|                | エリア内 | クロスエリア | エリア特定 | 実行速度 | 総合評価 |
|----------------|---------|------------|----------|---------|---------|
| Hybrid RAG     | {PLACEHOLDER} | {PLACEHOLDER} | {PLACEHOLDER} | {PLACEHOLDER} | {PLACEHOLDER} |
| Graph RAG      | {PLACEHOLDER} | {PLACEHOLDER} | {PLACEHOLDER} | {PLACEHOLDER} | {PLACEHOLDER} |
| Adaptive RAG   | {PLACEHOLDER} | {PLACEHOLDER} | {PLACEHOLDER} | {PLACEHOLDER} | {PLACEHOLDER} |
| Agentic RAG    | {PLACEHOLDER} | {PLACEHOLDER} | {PLACEHOLDER} | {PLACEHOLDER} | {PLACEHOLDER} |

---

## 6. Phase 10への提言

### 6.1 推奨RAGアーキテクチャ

以下の判断基準（計画書セクション12.2）に基づき、Phase 10のアーキテクチャを推奨する：

| 結果パターン | Phase 10推奨 |
|-------------|-------------|
| Hybrid RAGが全エリアで安定（90%+） | Hybrid RAGベースで全国展開 |
| Agentic RAGがクロスエリアで優位 | Hybrid + Agentic RAGのハイブリッド |
| 全システムで新エリアスコアが大幅低下 | エリア特定ロジック強化が先 |
| Graph RAGがエリア間関係で優位 | Graph RAG要素をHybridに統合 |

**本実験の結果パターン**: {PLACEHOLDER}

**推奨**: {PLACEHOLDER}

### 6.2 残課題

{PLACEHOLDER: 実験結果から明らかになった残課題を列挙}

1. {PLACEHOLDER}
2. {PLACEHOLDER}
3. {PLACEHOLDER}

---

## 7. 結論

{PLACEHOLDER: 主要な結論を3-5点で記述}

---

## 付録

### A. Phase 9ベースライン（渋谷単一105件の結果）

**評価日**: 2026-02-11
**テストケース**: 105件

| カテゴリ | Structured RAG | Agentic RAG | 差分 |
|---------|---------------|-------------|------|
| competitor | 66.7% | 100.0% | +33.3% |
| complementary | 80.0% | 100.0% | +20.0% |
| basic_location | 80.0% | 100.0% | +20.0% |
| basic_category | 80.0% | 80.0% | 0.0% |
| spatial_proximity | 100.0% | 100.0% | 0.0% |
| decision_business | 100.0% | 100.0% | 0.0% |
| decision_location | 100.0% | 100.0% | 0.0% |
| spatial_density | 100.0% | 100.0% | 0.0% |
| constraint_single | 100.0% | 100.0% | 0.0% |
| conditional_reasoning | 100.0% | 100.0% | 0.0% |
| multi_step_spatial | 100.0% | 100.0% | 0.0% |
| advanced_sensitivity | 100.0% | 100.0% | 0.0% |
| relation | 100.0% | 100.0% | 0.0% |
| proximity | 100.0% | 100.0% | 0.0% |
| cuisine | 100.0% | 100.0% | 0.0% |
| comparison | 100.0% | 100.0% | 0.0% |
| aggregation | 100.0% | 100.0% | 0.0% |
| iterative_refinement | 100.0% | 100.0% | 0.0% |
| hours | 100.0% | 100.0% | 0.0% |
| spatial_comparison | 100.0% | 80.0% | -20.0% |
| multi_hop | 100.0% | 66.7% | -33.3% |
| brand | 100.0% | 60.0% | -40.0% |
| advanced_comparison | 100.0% | 50.0% | -50.0% |
| constraint_multi | 100.0% | 40.0% | -60.0% |
| advanced_uncertainty | 100.0% | 0.0% | -100.0% |

**全体スコア**:
- Structured RAG: 96.2% success, 11.08s avg
- Agentic RAG: 87.6% success, 56.36s avg

### B. 結果データファイル一覧

| ファイル | 説明 |
|---------|------|
| `results/phase9b_evaluation_{timestamp}.json` | Full Test結果（JSON） |
| `results/phase9b_summary_{timestamp}.txt` | テキストサマリー |
| `results/phase9b_overall_comparison.png` | 全体比較グラフ |
| `results/phase9b_area_comparison.png` | エリア別比較グラフ |
| `results/phase9b_level_comparison.png` | レベル別比較グラフ |
| `results/checkpoint_{system_name}.json` | 各システムのチェックポイント |
