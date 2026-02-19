# Phase 9-B: 複数エリアRAG比較評価 - 引き継ぎドキュメント

**作成日**: 2026-02-19
**プロジェクト**: experiments-local-llm
**担当**: Claude Code + User
**ステータス**: Phase 9-B完了、レポートPRレビュー中

---

## 1. Phase 9-B概要

### 1.1 目的

Phase 9までの渋谷単一エリア評価を4エリア（渋谷・新宿・池袋・東京）に拡張し、4つのRAGシステムのエリア汎化性能を比較評価する。

### 1.2 達成した成果

| 指標 | 値 |
|------|-----|
| 評価規模 | 130テストケース × 4システム = **520クエリ** |
| 対象エリア | 渋谷・新宿・池袋・東京（~3,600 POI） |
| 最高キーワード成功率 | Hybrid RAG **96.2%** |
| 最高多次元composite | Hybrid RAG **52.2/100** |
| エリア特定精度 | 全システム **98.1%** |
| 実験レポート | 完成（643行、PLACEHOLDERゼロ） |

### 1.3 主要な結論

1. **Hybrid RAGが最も高い汎化性能** — 全エリア92〜100%で安定
2. **Agentic RAGのクロスエリア優位性は未確認** — 95% vs 他3システム100%
3. **キーワード評価と多次元評価の乖離が大きい** — 成功率90〜96% vs composite成功率29〜34%
4. **Phase 10にはHybrid RAGベースを推奨**

---

## 2. 実施内容と進捗

### 2.1 実装ステップ（全7段階、すべて完了）

| # | コミット | 内容 | PR |
|---|---------|------|-----|
| 1 | `be30923` | POIデータ取得の4エリア対応 | #14 (merged) |
| 2 | `b1df4b9` | コアモジュール（geo_utils, aggregator）の広域対応 | #15 (merged) |
| 3 | `c63297f` | RAGシステム4種の広域対応 + Agentic中国語対策 | #16 (merged) |
| 4 | `32a4e7f` | テストケース130件・評価モジュール・評価ノートブック | #17 (merged) |
| 5 | `476153c` | 4システム対応ノートブック + 実験レポートテンプレート | current PR |
| 6 | `b66d043` | 多次元評価スコアリングを追加 | current PR |
| 7 | `f6fa595` | 実験レポート本文を記述（PLACEHOLDERをすべて実データで置換） | current PR |

### 2.2 ブランチ構成

```
main
├── feature/phase9b-poi-data         (#14, merged)
├── feature/phase9b-core-modules     (#15, merged)
├── feature/phase9b-rag-systems      (#16, merged)
├── feature/phase9b-test-framework   (#17, merged)
└── feature/phase9b-evaluation-report (current, PRレビュー中)
    ├── #5 ノートブック + テンプレート
    ├── #6 多次元評価
    └── #7 レポート本文
```

### 2.3 現在のPR状態

- **ブランチ**: `feature/phase9b-evaluation-report`
- **ステータス**: PRレビュー中
- **含まれるコミット**: 5件（#5, CUDA OOM fix, #6, #7, results追加）
- **merge後に不要になるブランチ**: `feature/phase9b-evaluation-report`

---

## 3. ファイル構成

### 3.1 Phase 9-Bで新規作成・変更したファイル

| ファイル | 説明 | Phase 9-Bでの変更 |
|---------|------|------------------|
| `src/geo_utils.py` | 空間計算 | 4エリア座標・ランドマーク辞書を追加 |
| `src/aggregator.py` | データ集約 | エリア別集計に対応 |
| `src/structured_rag_system.py` | Hybrid RAG | 汎用プロンプト + `detect_area()` 追加 |
| `src/graph_rag_system.py` | Graph RAG | 4エリアグラフ対応 |
| `src/adaptive_rag_system.py` | Adaptive RAG | エリア対応ルーティング |
| `src/agentic_rag_system.py` | Agentic RAG | 中国語混入対策を強化 |
| `src/test_cases_multi_area.py` | テストケース130件 | **新規作成** |
| `src/evaluators_multi_area.py` | 評価モジュール | **新規作成**（多次元評価含む） |
| `notebooks/phase9b_multi_area_evaluation.ipynb` | 評価ノートブック | **新規作成** |
| `docs/PHASE9B_MULTI_AREA_EXPERIMENT_PLAN.md` | 実験計画書 | **新規作成** |
| `docs/PHASE9B_MULTI_AREA_EXPERIMENT_REPORT.md` | 実験レポート | **新規作成**→完成 |

### 3.2 結果データファイル（未追跡）

| ファイル | 説明 | 備考 |
|---------|------|------|
| `results/phase9b_evaluation_20260219_005810.json` | **最終結果**（520クエリ全データ） | 590KB |
| `results/phase9b_summary_20260219_005810.txt` | 最終サマリー | レポートのデータソース |
| `results/phase9b_overall_comparison.png` | 全体比較グラフ | |
| `results/phase9b_level_comparison.png` | レベル別比較グラフ | |
| `results/phase9b_evaluation_20260218_*.json` | 中間結果（デバッグ用） | 旧バージョン |

> **注意**: `results/phase9b_evaluation_20260219_005810.json` はレポートの全数値の根拠データ。削除しないこと。

### 3.3 未追跡ファイル（整理候補）

以下はgit未追跡で、merge後に整理可能：

| ファイル | 判断 |
|---------|------|
| `agentic_rag_results.json` | 削除可（Phase 9の一時ファイル） |
| `debug_cell_v3.py` | 削除可（デバッグ用） |
| `fixed_evaluation_cell.py` | 削除可（デバッグ用） |
| `quick_test_output.log` | 削除可（ログ） |
| `docs/RAG_APPROACH_SELECTION_GUIDE.md` | 要確認（有用なら追跡対象に） |
| `results/phase9b_evaluation_20260218_054933.json` | 削除可（Quick Test中間結果） |
| `results/phase9b_evaluation_20260218_065531.json` | 削除可（Quick Test中間結果） |
| `results/phase9b_evaluation_20260218_234531.json` | 削除可（Full Test中間結果、多次元スコアなし） |

---

## 4. 技術的知見

### 4.1 CUDA OOMへの対処

Google Colab T4 (15GB VRAM) で4システムを順次実行する際にCUDA OOMが頻発した。対処法：

```python
# システム切り替え時のVRAM解放パターン
import gc, torch
del current_system
gc.collect()
torch.cuda.empty_cache()
```

- LLMインスタンスは1つだけロードし、4システムで共有
- チェックポイント機能（10件ごとにJSON保存）でOOM後の再開を可能に

### 4.2 エリア特定ロジック

`detect_area()` は全4システムで共通使用：

1. **明示的検出**: 質問文に「渋谷駅」「新宿」等のエリア名を含む場合
2. **ランドマーク推定**: 「東京タワー」→東京、「サンシャイン」→池袋
3. **フォールバック**: 該当なしの場合はNone（クロスエリア扱い）

失敗2件はランドマーク辞書にない施設名が原因。辞書拡充で対応可能。

### 4.3 多次元評価の事後計算

520クエリの再実行なしで多次元スコアを追加するため、保存済み回答テキストに対してルールベースの事後スコアリングを実装：

- `reasoning_score`: 推論指標語（「なぜなら」「したがって」等）の出現数ベース
- `evidence_score`: POI名引用数 + 座標有無 + 引用表現
- `constraint_score`: 制約キーワードの充足割合
- `uncertainty_score`: 留保表現（「かもしれ」「推測」等）の出現
- `composite_score`: レベル別重み付き（L1はキーワード重視、L5は推論重視）

### 4.4 中国語混入パターン

Agentic RAGで6件検出。原因はReActパターンのJSON形式ツール出力がQwen2.5の中国語技術文書モードを誘発すること。対策として `agent_prompts.py` に日本語強制指示を追加済みだが完全には解消されていない。

---

## 5. 主要な実験結果サマリー

### 5.1 全体比較

| システム | 成功率 | Composite | 実行時間 | 中国語 |
|---------|-------|-----------|---------|--------|
| **Hybrid RAG** | **96.2%** | **52.2** | 8.8s | 1件 |
| Graph RAG | 92.3% | 46.1 | 8.2s | 0件 |
| Adaptive RAG | 92.3% | 48.1 | 8.1s | 0件 |
| Agentic RAG | 90.8% | 47.8 | 54.4s | 6件 |

### 5.2 仮説検証結果

| 仮説 | 判定 | 要約 |
|------|------|------|
| H1: Hybrid汎化性 | **支持** | 新エリア92〜100%（基準83.1%以上） |
| H2: Agenticクロスエリア優位 | **棄却** | 95% < 他100% |
| H3: エリア特定がボトルネック | **棄却** | 98.1%の高精度、失敗主因は別 |
| H4: POI増加で精度低下 | **部分的支持** | -3.9%だが基準10%未満 |
| H5: 中国語混入エリア非依存 | **棄却** | システム依存（Agentic=6件） |

---

## 6. 今後の課題

### 6.1 短期（PR merge後）

| 優先度 | タスク | 詳細 |
|--------|-------|------|
| 高 | PRのmerge | `feature/phase9b-evaluation-report` をmainにmerge |
| 高 | 未追跡ファイルの整理 | セクション3.3の削除候補を処理 |
| 中 | 結果データの保全 | `results/phase9b_evaluation_20260219_005810.json` をgit追跡に含めるか判断 |

### 6.2 中期（Phase 10準備）

| 優先度 | タスク | 詳細 |
|--------|-------|------|
| 高 | プロンプト改善 | 推論説明・根拠引用の指示をsystem_promptに追加（composite_score改善） |
| 高 | MCPサーバー設計 | `geo_utils.py`/`aggregator.py`のMCPツール化設計 |
| 中 | モデル検討 | 14B以上またはAPI経由の大規模モデル評価 |
| 中 | PostGIS移行設計 | ChromaDB → PostGIS空間インデックスの移行計画 |

### 6.3 長期（Phase 10全国展開）

| タスク | 詳細 |
|-------|------|
| 動的基準点解決 | ジオコーディングAPI（Nominatim等）連携 |
| 500万POI対応 | PostGIS + R-tree空間インデックス |
| エリア特定強化 | ランドマーク辞書拡充 + 外部APIフォールバック |
| 評価フレームワーク | 多次元評価をCI/CDパイプラインに統合 |

---

## 7. Phase間の関係性

```
Phase 5 (ベースライン 60.3pt)
  ↓
Phase 6 (構造化RAG 91.6pt / 渋谷55件)
  ↓
Phase 8 (Graph RAG / Adaptive RAG追加)
  ↓
Phase 9 (Agentic RAG追加 / 渋谷105件)
  ↓
Phase 9-B (4エリア拡張 / 130件×4sys=520クエリ) ← 現在地
  ↓
Phase 10 (全国展開 / MCP / PostGIS) ← 次のフェーズ
```

---

## 8. 関連ドキュメント

| ドキュメント | 内容 |
|-------------|------|
| `docs/PHASE9B_MULTI_AREA_EXPERIMENT_PLAN.md` | 実験計画書（テスト設計・評価基準） |
| `docs/PHASE9B_MULTI_AREA_EXPERIMENT_REPORT.md` | **実験レポート（最終成果物）** |
| `docs/HANDOVER_PHASE6.md` | Phase 6の引き継ぎ（構造化RAGの基盤設計） |
| `docs/HANDOVER_PHASE9_AGENTIC_RAG.md` | Phase 9の引き継ぎ（Agentic RAG実装） |
| `docs/RAG_APPROACH_SELECTION_GUIDE.md` | RAGアプローチ選定ガイド（未追跡） |
| `CLAUDE.md` | プロジェクト全体のガイダンス |
