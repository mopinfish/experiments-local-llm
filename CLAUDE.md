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
- LLM: Qwen3-32B（Colab A100用4ビット量子化、VRAM約19GB）

## Google Colab でのローカル LLM 運用ノウハウ

### GPU ランタイムの選択

- **A100 40GB** が必要（Qwen3-32B 4bit で VRAM 約19GB 使用）
- T4 16GB では Qwen3-32B は動作しない（Qwen2.5-7B なら可）

### 大規模モデルロード時の OOM 対策

4bit 量子化モデルでも、ロード過程では重みを一時的に fp16/bf16/fp32 で GPU に展開してから量子化するため、**最終モデルサイズの2倍近いピークメモリ**が必要になる場合がある。以下の3パラメータで回避する：

```python
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.float16,                        # 中間テンソルをfp16に制限
    low_cpu_mem_usage=True,                            # レイヤー単位でCPU→GPU逐次転送
    max_memory={0: "36GiB", "cpu": "24GiB"},           # GPU上限制限＋CPUオフロード許可
)
```

| パラメータ | 効果 |
|-----------|------|
| `torch_dtype=torch.float16` | 重みの中間展開を fp16 に制限（bf16/fp32 で展開されるのを防ぐ） |
| `low_cpu_mem_usage=True` | 全レイヤーを一括 GPU 展開せずレイヤー単位で逐次転送 |
| `max_memory={0: "36GiB", "cpu": "24GiB"}` | GPU に載りきらない分を CPU RAM にオフロード |

**注意**: `transformers` / `bitsandbytes` のバージョン更新でロード時のメモリ挙動が変わることがある。OOM が発生したらまずこの3パラメータを確認すること。

### Qwen3 の Non-thinking モード

Qwen3 はデフォルトで `<think>` タグ付きの推論モードが有効。RAG 用途では不要なので無効化する：

```python
original_apply = tokenizer.apply_chat_template
def patched_apply(*args, **kwargs):
    kwargs['enable_thinking'] = False
    return original_apply(*args, **kwargs)
tokenizer.apply_chat_template = patched_apply
```

### ランタイムリセットの注意点

- `Runtime > Restart runtime` では GPU メモリが完全に解放されない場合がある
- OOM が発生したら `Runtime > Disconnect and delete runtime` で**ファクトリーリセット**してから再接続する

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

## ドキュメント命名規則

`docs/` 配下のドキュメントには、関連する実験IDをプレフィックスとして付与する：

```
docs/<カテゴリ>/<実験ID>_<内容>.md
```

- **実験ID形式**: `PHASE<番号>` または `PHASE<番号>_<サブID>`（例: `PHASE6`, `PHASE9_B`）
- **カテゴリ**: `handovers/`, `plans/`, `reports/`, `experiments/`

例:
- `docs/reports/PHASE9_B_MULTI_AREA_EXPERIMENT_REPORT.md`
- `docs/plans/PHASE7_FINETUNING_EXPERIMENT_PLAN.md`
- `docs/handovers/PHASE6_HANDOVER.md`

## 主要ドキュメント

- `docs/handovers/HANDOVER_PHASE6.md`: アーキテクチャ詳細を含む完全な技術引き継ぎ
- `docs/reports/PHASE6_IMPROVEMENT_REPORT.md`: 機能別スコア改善の詳細分析
- `docs/reports/PHASE6_PROGRESS_REPORT.md`: フェーズごとの進捗追跡
