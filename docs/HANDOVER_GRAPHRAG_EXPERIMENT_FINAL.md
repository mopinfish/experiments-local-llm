# GraphRAG実験 最終引き継ぎドキュメント

**作成日**: 2026年1月30日
**プロジェクト**: experiments-local-llm
**ブランチ**: feature/graphrag-experiment
**関連Issue**: #4 (Phase 8: グラフRAG実験の実装)

---

## 1. 実験概要

### 1.1 目的

地理空間POI情報に対して、**GraphRAG**が従来の**構造化RAG**と比較してどのような質問タイプで優位性を持つかを検証する。

### 1.2 比較対象システム

| システム | 説明 | 実装状況 |
|---------|------|----------|
| **StructuredRAG** | Phase 6で実装した構造化RAG | 完了 |
| **GraphRAG** | NetworkXベースのグラフRAG | 完了 |
| **Adaptive RAG** | 質問タイプに応じた動的システム選択 | 完了 |

---

## 2. 最終評価結果

### 2.1 全体スコア（90テストケース）

| システム | スコア | 処理時間 | 標準偏差 |
|---------|--------|----------|----------|
| **StructuredRAG** | **89.1%** | 20.6秒 | 20.2 |
| Adaptive RAG | 86.1% | 17.8秒 | 20.4 |
| GraphRAG | 76.7% | 8.7秒 | 24.8 |

### 2.2 テストソース別スコア

| テストソース | GraphRAG | StructuredRAG | Adaptive |
|-------------|----------|---------------|----------|
| Structured Tests (55件) | 74.5% | **90.5%** | 86.6% |
| GraphRAG Tests (35件) | 80.2% | **86.9%** | 85.2% |

### 2.3 Adaptive RAGの選択分布

```
StructuredRAG選択: 62クエリ (68.9%)
GraphRAG選択:      28クエリ (31.1%)
```

### 2.4 カテゴリ別詳細結果

#### GraphRAGが優位なカテゴリ（3カテゴリ）

| カテゴリ | GraphRAG | StructuredRAG | 差分 |
|---------|----------|---------------|------|
| **comparison** | **100.0%** | 50.0% | +50.0pt |
| **competitor** | **88.9%** | 66.7% | +22.2pt |
| decision_business | **93.3%** | 89.3% | +4.0pt |

#### StructuredRAGが優位なカテゴリ（多数）

| カテゴリ | GraphRAG | StructuredRAG | 差分 |
|---------|----------|---------------|------|
| cuisine | 66.7% | **100.0%** | -33.3pt |
| hours | 83.3% | **100.0%** | -16.7pt |
| basic_location | 78.0% | **100.0%** | -22.0pt |
| constraint_single | 75.0% | **100.0%** | -25.0pt |
| aggregation | 80.6% | **100.0%** | -19.4pt |
| multi_hop | 100.0% | **100.0%** | 0pt |
| proximity | 100.0% | **100.0%** | 0pt |

---

## 3. 仮説検証結果

### 3.1 当初の仮説と結果

| 仮説ID | 仮説内容 | 結果 | 考察 |
|--------|---------|------|------|
| H1 | POI間の空間的関係をグラフエッジで表現することで関係性クエリが向上 | **部分的に支持** | competitorカテゴリで+22.2pt |
| H2 | カテゴリ階層のグラフ表現でカテゴリ横断クエリが改善 | **棄却** | StructuredRAGが-19.4pt優位 |
| H3 | 複数POIにまたがる複合クエリでグラフトラバーサルが有効 | **棄却** | multi_hopでは同等、relationでStructuredRAG優位 |
| H4 | 単純な最寄り検索では構造化RAGが効率的 | **支持** | 処理時間: GraphRAG 8.7秒 vs StructuredRAG 20.6秒 |

### 3.2 新たな発見

1. **StructuredRAGの汎用性が予想以上に高い**
   - GraphRAG向けテストでも86.9%を達成
   - 専用設計のGraphRAG（80.2%）を上回る

2. **GraphRAGの優位性は限定的**
   - 明確な優位性は「comparison」「competitor」の2カテゴリのみ
   - 全体スコアへの寄与は小さい

3. **Adaptive RAGの選択精度が不十分**
   - 68.9%でStructuredRAGを選択するが、それでも-3.1pt悪化
   - 一部カテゴリで誤選択による大幅な性能低下

---

## 4. 実装成果物

### 4.1 新規ファイル

| ファイル | 役割 |
|---------|------|
| `src/graph_builder.py` | POIナレッジグラフ構築（NetworkX） |
| `src/graph_rag_system.py` | GraphRAGシステム本体 |
| `src/adaptive_rag_system.py` | Adaptive RAGシステム |
| `src/test_cases_graphrag.py` | GraphRAG向けテストケース（35件） |

### 4.2 更新ファイル

| ファイル | 変更内容 |
|---------|---------|
| `osm_poi_fetcher.py` | ブランド/営業時間/料理ジャンル抽出追加 |
| `src/structured_rag_system.py` | `get_context()`メソッド追加（統合評価用） |

### 4.3 Notebooks

| ファイル | 内容 |
|---------|------|
| `notebooks/graphrag_01_graph_construction.ipynb` | グラフ構築 |
| `notebooks/graphrag_02_query_implementation.ipynb` | クエリ実装 |
| `notebooks/graphrag_03_initial_evaluation.ipynb` | 初期評価 |
| `notebooks/graphrag_04_unified_comparison.ipynb` | 統一比較（70件） |
| `notebooks/graphrag_05_enhanced_comparison.ipynb` | 拡張比較（90件） |
| `notebooks/graphrag_06_adaptive_evaluation.ipynb` | Adaptive RAG評価 |

### 4.4 評価結果

| ファイル | 内容 |
|---------|------|
| `results/adaptive_comparison_overall.png` | 全体スコア比較グラフ |
| `results/adaptive_comparison_by_category.png` | カテゴリ別比較グラフ |
| `results/adaptive_evaluation_20260130_053413.json` | 詳細評価データ |

---

## 5. グラフ構造

### 5.1 ノードタイプ

| ノード | 属性 | 件数 |
|--------|-----|------|
| `POI` | id, name, lat, lon, category, embedding | 1,046 |
| `Category` | name | 12 |
| `Area` | name, direction, distance_zone | 32 |

### 5.2 エッジタイプ

| エッジ | 説明 | 件数 |
|--------|-----|------|
| `NEAR_TO` | POI間の近接関係（100m以内） | 66,248 |
| `SAME_CATEGORY` | 同一カテゴリ | 2,086 |
| `SAME_BRAND` | 同一チェーン店 | 約200 |
| `COMPLEMENTARY` | 相補的関係（ホテル↔レストラン等） | 約3,000 |
| `COMPETITOR` | 競合関係（同カテゴリ・100m以内） | 約8,000 |
| `SAME_CUISINE` | 同一料理ジャンル | 約500 |
| `SAME_HOURS` | 同一営業時間帯 | 約2,000 |

**総エッジ数**: 約82,000

---

## 6. 結論と推奨事項

### 6.1 主要な結論

1. **StructuredRAGが最も効果的**
   - 全体スコア89.1%で最高
   - GraphRAG向けテストでも高いスコアを達成
   - 処理時間は長いが、精度優先なら推奨

2. **GraphRAGは特定タスクで有効**
   - 「comparison」（東西比較等）で+50pt優位
   - 「competitor」（競合分析）で+22pt優位
   - 処理速度は最速（8.7秒）

3. **Adaptive RAGは改善の余地あり**
   - 選択アルゴリズムの精度向上が必要
   - 現状ではStructuredRAG単体より劣る

### 6.2 推奨アプローチ

| シナリオ | 推奨システム | 理由 |
|---------|------------|------|
| 一般的なPOIクエリ | StructuredRAG | 最高精度（89.1%） |
| 東西/方向比較 | GraphRAG | +50pt優位 |
| 競合店分析 | GraphRAG | +22pt優位 |
| 処理速度優先 | GraphRAG | 2.4倍高速 |
| 自動選択が必要 | Adaptive RAG（要改善） | 選択精度向上後 |

### 6.3 今後の改善方向

| アプローチ | 期待効果 |
|-----------|----------|
| **StructuredRAGにGraph機能を統合** | 単一システムで全カテゴリ対応 |
| **Adaptive RAGの選択ロジック改善** | カテゴリ別の最適選択 |
| **ハイブリッド出力** | 両システムの結果を統合して回答生成 |

---

## 7. 次のフェーズへの提案

### 7.1 Phase 9: 他のRAGアーキテクチャ比較

GraphRAG実験で得られた知見を踏まえ、以下のRAGアーキテクチャを検討：

| アーキテクチャ | 優先度 | 期待効果 |
|--------------|-------|---------|
| **Agentic RAG** | ★★★ | 複雑な空間推論タスク |
| **Self-RAG** | ★★☆ | 回答の信頼性向上 |
| **CRAG** | ★★☆ | 不正確な空間情報の補正 |
| **HyDE** | ★☆☆ | 抽象的な空間クエリ |

### 7.2 Phase 10: 全国展開

| 課題 | 対策 |
|-----|------|
| 渋谷固有のハードコーディング | 動的基準点解決 |
| スケーラビリティ | PostGIS/Supabase導入 |
| 処理時間 | 空間インデックス最適化 |

---

## 8. 参考リソース

### 8.1 Issue・PR

- **Issue #4**: GraphRAG実験の進捗追跡（評価結果コメント済み）
- **PR #3**: ファインチューニング実験（マージ待ち）

### 8.2 評価データ

- `results/adaptive_evaluation_20260130_053413.json`: 90テストケースの詳細結果
- `results/adaptive_comparison_*.png`: 比較グラフ

### 8.3 関連ドキュメント

- `docs/GRAPHRAG_EXPERIMENT_PLAN.md`: 実験計画書
- `docs/STRUCTURED_RAG_RESEARCH_REPORT.md`: Phase 5-6研究レポート
- `docs/FINETUNING_EXPERIMENT_REPORT.md`: ファインチューニング実験レポート

---

## 9. 次回セッションへの引き継ぎ

### 9.1 完了済みタスク

- [x] GraphRAG実装（基本 + 拡張エッジ）
- [x] Adaptive RAG実装
- [x] 90テストケースでの評価
- [x] 評価結果のIssue #4への報告
- [x] 最終引き継ぎドキュメント作成

### 9.2 保留タスク

- [ ] PR #3（ファインチューニング実験）のマージ判断
- [ ] Phase 9（他RAGアーキテクチャ）の計画策定
- [ ] Phase 10（全国展開）の計画策定

### 9.3 意思決定事項

1. **GraphRAG実験の継続**: これ以上の改善を追求するか、現状で終了するか
2. **次フェーズの優先度**: Agentic RAG vs 全国展開のどちらを先に進めるか
3. **Adaptive RAGの改善**: 選択アルゴリズムの改善に投資するか

---

**作成者**: Claude Opus 4.5
**実験期間**: 2026年1月29日〜30日
**最終スコア**: StructuredRAG 89.1%（最高）、Adaptive RAG 86.1%、GraphRAG 76.7%
