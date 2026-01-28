# CLAUDE.md

このファイルは、Claude Code (claude.ai/code) がこのリポジトリのコードを扱う際のガイダンスを提供します。

## プロジェクト概要

地理的POI（Point of Interest）クエリ向けの構造化RAG（Retrieval-Augmented Generation）システムです。渋谷駅エリアに特化して最適化されており、ベクトル検索と構造化空間処理を組み合わせて位置情報に関する質問に回答します。

**現在の状態**: Phase 6.2.1 完了 - スコア 91.6pt（ベースライン 60.3pt から向上）

## ビルド・実行コマンド

```bash
# 依存関係のインストール
uv sync

# OpenStreetMapからPOIデータを取得
uv run python osm_poi_fetcher.py

# ベクトルストアの初期化（初回のみ）
uv run python -c "from rag_system import POI_RAG_System; POI_RAG_System(rebuild=True)"

# インタラクティブモードで実行
uv run python rag_system.py

# クイックテスト実行（5ケース）
uv run python test_runner.py --quick

# フルテスト実行（30ケース）
uv run python test_runner.py
```

### Google Colab評価

主要な評価は `notebooks/phase6_full_evaluation.ipynb` でGoogle Colab（T4 GPUランタイム）上で実行されます。このノートブックは `src/` モジュールを使用し、55テストケース（L1-L5難易度）に対して評価を行います。

## アーキテクチャ

### コア設計パターン: 相補的統合

重要なアーキテクチャ上の洞察は、構造化処理とベクトル検索が**相互排他ではなく相補的**であるという点です。システムは `if` 文（`elif` ではなく）を使用して、複数のコンテキストタイプが共存できるようにしています：

```python
def _build_context(self, question, analysis):
    structured_parts = []

    if analysis.requires_proximity and category:
        structured_parts.append(proximity_context)
    if analysis.requires_sensitivity and category:
        structured_parts.append(sensitivity_context)
    if analysis.requires_comparison:
        structured_parts.append(comparison_context)
    if analysis.requires_aggregation:
        structured_parts.append(aggregation_context)

    # ベクトル検索は常にフォールバックではなく補完として実行
    vector_context = self._get_vector_search_context(question, k=5)

    return structured_parts + [vector_context]
```

### モジュール構成 (src/)

- **geo_utils.py**: 空間計算 - 渋谷駅からの距離/方角、最寄りPOIランキング、半径フィルタリング、感度分析
- **aggregator.py**: データ集約 - カテゴリ件数、東西比較、トップカテゴリランキング
- **structured_rag_system.py**: 質問分析とコンテキスト構築のオーケストレーション
- **test_cases_v2.py**: 12サブカテゴリにわたる55テストケース（L1-L5難易度）
- **evaluators_v2.py**: 評価用スコアリングシステム

### 主要定数

```python
SHIBUYA_STATION = (35.658034, 139.701636)  # すべての空間計算の基準点
```

### 質問タイプ検出キーワード

- **近接性** ("最も近い", "一番近い", "最寄り"): 距離ソート結果をトリガー
- **感度** ("変えても", "範囲を", "成立"): 半径比較分析をトリガー
- **比較** ("東", "西", "比較"): 方角分析をトリガー
- **集約** ("いくつ", "何件", "カテゴリ"): 件数サマリーをトリガー

## データ

- **poi_documents.json**: 渋谷エリアの1,047 POI
- **chroma_db/**: 永続化ベクトルストア（ChromaDB）
- 埋め込みモデル: multilingual-e5-base
- LLM: Qwen2.5-7B-Instruct（Colab用4ビット量子化）

## 既知の制限事項

### 渋谷固有のハードコーディング
すべての空間計算がハードコードされた渋谷駅座標を使用しています。全国展開には以下が必要です：
1. 動的な基準点解決（ジオコーディングAPI）
2. 大規模パフォーマンスのためのPostGISまたは同等の空間インデックス
3. 場所に依存しない質問分析

### パフォーマンス
現在: クエリあたり約22秒（評価には許容範囲、本番運用には遅すぎる）

### ChromaDBの制約
メタデータでネストされた辞書をサポートしていません - `flatten_metadata()` ヘルパーを使用してください。

## 将来のフェーズ

- **Phase 7**: LLMファインチューニング（LoRA/QLoRA）で95pt以上を目標
- **Phase 8**: PostGIS/Supabaseによる全国展開で500万POI以上の規模に対応

## 主要ドキュメント

- `docs/HANDOVER_PHASE6.md`: アーキテクチャ詳細を含む完全な技術引き継ぎ
- `docs/PHASE6_IMPROVEMENT_REPORT.md`: 機能別スコア改善の詳細分析
- `docs/PHASE6_PROGRESS_REPORT.md`: フェーズごとの進捗追跡
