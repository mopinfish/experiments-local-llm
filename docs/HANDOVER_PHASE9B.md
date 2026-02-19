# Phase 9-B: 複数エリアRAG比較評価 - 引き継ぎドキュメント

**作成日**: 2026-02-19
**プロジェクト**: experiments-local-llm
**担当**: Claude Code + User
**ステータス**: Phase 9-B完了、Phase 9-C準備中

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
| 5 | `476153c` | 4システム対応ノートブック + 実験レポートテンプレート | #18 (merged) |
| 6 | `b66d043` | 多次元評価スコアリングを追加 | #18 (merged) |
| 7 | `f6fa595` | 実験レポート本文を記述（PLACEHOLDERをすべて実データで置換） | #18 (merged) |

### 2.2 ブランチ構成

Phase 9-Bの全ブランチはmainにmerge済み・削除済み。

```
main  ← Phase 9-B全PRマージ完了
  ├── #14 feature/phase9b-poi-data         (merged, deleted)
  ├── #15 feature/phase9b-core-modules     (merged, deleted)
  ├── #16 feature/phase9b-rag-systems      (merged, deleted)
  ├── #17 feature/phase9b-test-framework   (merged, deleted)
  └── #18 feature/phase9b-evaluation-report (merged, deleted)
```

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

### 4.5 POI問合せにおけるベクトル検索の限定的寄与

Phase 9-Bの比較実験から、**ベクトル検索はPOI問合せの精度向上にほとんど寄与していない**ことが判明した。

**実験的根拠**:
- Graph RAG（ベクトル検索を一切使用しない）が92.3%を達成し、ベクトル検索を使用するHybrid RAG（96.2%）との差はわずか3.9%
- Hybrid RAG自身も、精度が求められる近接性クエリでベクトル検索結果をコンテキストから意図的に除外している
- 構造化処理（距離計算・集計・比較）はベクトル検索結果を参照せず、全POIリストの直接走査で動作している

**本質的理由**: POI問合せの核心は「意味的類似性」ではなく「空間的・属性的な条件合致」（最寄り検索、件数集計、方角比較、距離フィルタ）であり、これらは構造化クエリ（PostGIS空間SQL等）の領域である。座標をベクトル埋め込みに変換しても空間的近接性は保存されず、k=5の類似度検索で最寄りPOIが含まれる保証もない。

**Phase 10への設計指針**: MCPツール＋PostGIS構造化処理の組み合わせでベクトル検索層を省略し、アーキテクチャのシンプル化・結果の確定性・説明可能性・スケーラビリティを同時に達成する。LLMは意図理解とツール選択に専念し、空間計算はGISエンジンに委譲する。

> 詳細分析は `docs/VECTOR_SEARCH_ANALYSIS_FOR_POI.md` を参照。

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

## 6. Phase 9-C: プロンプト改善・モデル変更・ファインチューニング比較実験

### 6.1 背景と目的

Phase 9-Bで明らかになった最大の課題は**キーワード成功率（96.2%）と多次元composite（52.2/100）の乖離**である。「答えは含まれているが、推論説明・根拠引用が不十分」な状態は実用品質として不足している。

Phase 9-Cでは、Phase 9-Bと**同一のテストケース（130件）・質問解析モジュール（`evaluators_multi_area.py`）・評価方法（キーワード＋多次元二層評価）**を固定し、以下の3つの変数を段階的に変更して比較実験を行う：

1. プロンプト改善
2. LLMモデルの変更
3. ファインチューニングモデルとの比較

### 6.2 実験ステップ

#### Step 1: プロンプト改善（低コスト・即着手可能）

**変更対象**: `src/structured_rag_system.py` の `system_prompt`

現在のプロンプト:
> 「あなたは東京都内の主要駅周辺エリア（...）の地理情報に詳しいアシスタントです。提供された情報に基づいて、正確かつ簡潔に回答してください。」

改善案（推論・根拠・不確実性の明示指示を追加）:
> 「...正確に回答してください。**推論過程を段階的に説明し**、**根拠となるPOI名・座標・データを引用してください**。**情報が不確実または不足している場合はその旨を明記してください**。」

**評価**: 改善プロンプトで4システム×130件=520クエリを再実行し、Phase 9-Bと同一の評価パイプラインで比較

**期待効果**: reasoning_score（現1.2〜1.7/5.0）とuncertainty_scoreの改善、composite_scoreの向上

#### Step 2: LLMモデル変更

Phase 9-Bの結果から、7B 4bit量子化モデルの推論能力が多次元品質のボトルネックであることが判明している。以下のモデルを候補として評価する：

| モデル | パラメータ | 量子化 | VRAM目安 | 期待される改善 |
|-------|----------|--------|---------|--------------|
| Qwen2.5-7B-Instruct | 7B | 4bit (現行) | ~4.5GB | ベースライン |
| Qwen2.5-14B-Instruct | 14B | 4bit | ~8GB | 推論能力向上 |
| Qwen2.5-14B-Instruct | 14B | 8bit | ~14GB | 推論＋生成品質向上 |
| Qwen2.5-7B-Instruct | 7B | 8bit | ~8GB | 量子化劣化の測定 |

**評価**: Step 1の改善プロンプトを適用した状態で、モデルを差し替えて同一評価パイプラインで比較。モデル変更の寄与とプロンプト改善の寄与を分離して測定する。

**T4 15GB VRAM制約**: 14B 8bitはT4ではギリギリの容量。RAGシステムのメモリ消費も考慮し、実行可能性を事前検証する。

#### Step 3: ファインチューニングモデルとの比較

Phase 9-Bの130テストケースおよびPOIデータを活用し、地理的質問応答タスクに特化したファインチューニングモデルを作成・評価する。

**ファインチューニング方針**:

| 項目 | 内容 |
|------|------|
| ベースモデル | Step 2で最良だったモデル |
| 手法 | QLoRA（4bit量子化 + LoRAアダプター） |
| 学習データ | Phase 9-Bの高品質回答（composite_score ≥ 60）+ 手動作成の模範回答 |
| 学習タスク | 地理的質問→推論過程付き回答の生成 |
| 評価 | 130テストケースの再実行（学習データとの重複に注意し、Hold-outまたはCross-validationを適用） |

**比較マトリクス**:

| 条件 | プロンプト | モデル | 備考 |
|------|----------|-------|------|
| Phase 9-B (ベースライン) | 現行 | Qwen2.5-7B 4bit | 既存データ再利用 |
| 9-C Step 1 | 改善 | Qwen2.5-7B 4bit | プロンプト効果の測定 |
| 9-C Step 2 | 改善 | 上位モデル | モデル効果の測定 |
| 9-C Step 3 | 改善 | ファインチューニング済み | FT効果の測定 |

各条件で4 RAGシステム（Hybrid/Graph/Adaptive/Agentic）を横断的に評価し、**プロンプト改善・モデルスケールアップ・ファインチューニングそれぞれの寄与**を分離して分析する。

### 6.3 再利用する資産（Phase 9-Bから固定）

| 資産 | ファイル | 用途 |
|------|---------|------|
| テストケース130件 | `src/test_cases_multi_area.py` | 全Step共通の評価入力 |
| 質問解析モジュール | `src/evaluators_multi_area.py` | キーワード＋多次元二層評価 |
| 評価ノートブック | `notebooks/phase9b_multi_area_evaluation.ipynb` | 実行・可視化基盤 |
| Phase 9-Bベースライン | `results/phase9b_evaluation_20260219_005810.json` | 比較基準データ |
| 4 RAGシステム | `src/{structured,graph,adaptive,agentic}_rag_system.py` | 評価対象 |

### 6.4 成功基準

| 指標 | Phase 9-B現状 | Phase 9-C目標 |
|------|-------------|-------------|
| composite_score (Hybrid) | 52.2/100 | 70以上 |
| composite_success_rate | 33.9% | 60%以上 |
| reasoning_score 平均 | 1.43/5.0 | 3.0以上 |
| キーワード成功率 | 96.2% | 95%以上維持 |
| 中国語混入 (Agentic) | 6件 | 0件 |

---

## 7. Phase 10以降の課題

### 7.1 MCPサーバー化（ベクトル検索レスアーキテクチャ）

Phase 9-Bの知見（セクション4.5）を踏まえ、Phase 10ではベクトル検索層を省略したMCPツール＋PostGIS構造化処理のアーキテクチャを採用する：

- `geo_utils.py`/`aggregator.py` の空間計算ロジックをMCPツールとして移植
- PostGIS/Supabase MCP経由での空間クエリ実行（ChromaDBベクトル検索は不要）
- エリア特定の `detect_area()` をMCPツールとして独立化 + ジオコーディングAPI連携
- LLMは意図理解・ツール選択・自然文回答生成に専念し、空間計算はMCPツール経由でGISエンジンに委譲

### 7.2 全国展開（PostGIS移行）

| タスク | 詳細 |
|-------|------|
| 動的基準点解決 | ジオコーディングAPI（Nominatim等）連携 |
| 500万POI対応 | PostGIS + R-tree空間インデックス |
| エリア特定強化 | ランドマーク辞書拡充 + 外部APIフォールバック |
| 評価フレームワーク | 多次元評価をCI/CDパイプラインに統合 |

---

## 8. Phase間の関係性

```
Phase 5 (ベースライン 60.3pt)
  ↓
Phase 6 (構造化RAG 91.6pt / 渋谷55件)
  ↓
Phase 8 (Graph RAG / Adaptive RAG追加)
  ↓
Phase 9 (Agentic RAG追加 / 渋谷105件)
  ↓
Phase 9-B (4エリア拡張 / 130件×4sys=520クエリ) ← 完了
  ↓
Phase 9-C (プロンプト改善・モデル変更・FT比較) ← 次のフェーズ
  ├── Step 1: プロンプト改善
  ├── Step 2: LLMモデル変更 (14B等)
  └── Step 3: ファインチューニングモデル比較
  ↓
Phase 10 (全国展開 / MCP / PostGIS)
```

---

## 9. 関連ドキュメント

| ドキュメント | 内容 |
|-------------|------|
| `docs/PHASE9B_MULTI_AREA_EXPERIMENT_PLAN.md` | 実験計画書（テスト設計・評価基準） |
| `docs/PHASE9B_MULTI_AREA_EXPERIMENT_REPORT.md` | **実験レポート（最終成果物）** |
| `docs/HANDOVER_PHASE6.md` | Phase 6の引き継ぎ（構造化RAGの基盤設計） |
| `docs/HANDOVER_PHASE9_AGENTIC_RAG.md` | Phase 9の引き継ぎ（Agentic RAG実装） |
| `docs/VECTOR_SEARCH_ANALYSIS_FOR_POI.md` | **POI問合せにおけるベクトル検索の役割分析** |
| `docs/RAG_APPROACH_SELECTION_GUIDE.md` | RAGアプローチ選定ガイド（未追跡） |
| `CLAUDE.md` | プロジェクト全体のガイダンス |
