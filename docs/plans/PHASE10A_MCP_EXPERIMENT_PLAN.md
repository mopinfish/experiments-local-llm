# Phase 10-A: MCP サーバー構造化ツール拡張実験計画

## 概要

Phase 9-C で構築した質問分析 + 構造化処理パイプライン（composite 70.4pt）の知見を MCP サーバーのツールとして実装し、「拡張 MCP サーバー vs 既存シンプル MCP サーバー」の性能比較を行う。パイプライン方式と LLM エージェント方式の両方で評価し、オーケストレーション方式の影響も検証する。

## 実験設計

### 4 システム比較（2×2 マトリクス）

|  | パイプライン方式 | LLM エージェント方式 |
|--|----------------|-------------------|
| **拡張 MCP** (構造化ツールあり) | **A: Enhanced+Pipeline** (主) | **C: Enhanced+Agent** (補) |
| **既存 MCP** (基本ツールのみ) | **B: Simple+Pipeline** (主) | **D: Simple+Agent** (補) |

### 寄与分離

- **A vs B** (Pipeline): 構造化ツールの寄与（オーケストレーション統一）
- **C vs D** (Agent): 構造化ツールの寄与（LLM 自律選択）
- **A vs C**: パイプライン vs エージェント（構造化ツールあり）
- **B vs D**: パイプライン vs エージェント（基本ツールのみ）

### ネットワーク構成

```
[Google Colab A100]
  │ Qwen3-32B 4bit
  │ 評価ノートブック
  │
  │ HTTPS (ngrok tunnel)
  ▼
[ローカル PC]
  MCP Server (streamable-http transport)
  GeoTechAgent-mapfanmcp
  │
  │ HTTPS
  ▼
[MapFan REST API]
```

### テストケース

- **Variant A**: 既存 130 テストケース（expected_keywords そのまま）
- **Variant B**: データソース非依存版（OSM 固有 POI 名を汎用キーワードに置換）

### 評価指標

既存 `evaluators_multi_area.py` を使用。Composite Score を主指標、Reasoning / Evidence / Constraint / Success Rate を副指標。

## 実装済みファイル

### GeoTechAgent-mapfanmcp（MCP サーバー拡張）

| ファイル | 目的 |
|---------|------|
| `mapfan/geo_utils.py` | 空間計算（haversine, direction, 駅座標） |
| `mapfan/category_mapper.py` | キーワード→ジャンルコード |
| `mapfan/structured_tools.py` | 構造化処理ツール 6 個 |
| `main.py` | ツール登録追加 |

### 新規 MCP ツール

| ツール名 | 説明 |
|---------|------|
| `geo_analyze_question` | 質問分析（タイプ・カテゴリ・駅・方角検出） |
| `geo_nearest_pois` | 最寄り POI 検索（距離ソート + 統計） |
| `geo_count_by_category` | カテゴリ別件数集計 |
| `geo_compare_directions` | 東西南北 POI 件数比較 |
| `geo_sensitivity_analysis` | 半径感度分析 |
| `geo_search_with_context` | all-in-one 構造化検索 |

### experiments-local-llm（評価側）

| ファイル | 目的 |
|---------|------|
| `src/mcp_client.py` | MCP サーバー接続ラッパー |
| `src/mcp_enhanced_pipeline.py` | System A: Enhanced+Pipeline |
| `src/mcp_simple_pipeline.py` | System B: Simple+Pipeline |
| `src/mcp_agent_system.py` | System C/D: LLM Agent 方式 |
| `src/test_cases_multi_area_v2.py` | Variant B テストケース |
| `notebooks/phase10a_mcp_evaluation.ipynb` | 評価ノートブック |

## 実行手順

### 1. MCP サーバー起動（ローカル PC）

```bash
cd GeoTechAgent-mapfanmcp
MCP_TRANSPORT=streamable-http MCP_HOST=0.0.0.0 MCP_PORT=8000 uv run python main.py
```

### 2. ngrok トンネル設定

```bash
ngrok http 8000
```

### 3. Colab で評価実行

1. `notebooks/phase10a_mcp_evaluation.ipynb` を開く
2. ngrok URL を設定
3. 接続テスト → クイックテスト → フル評価の順に実行

## 実行時間見積もり

- Pipeline: 130 ケース × ~1.5 分 × 2 システム × 2 Variant = ~13 時間
- Agent: 130 ケース × ~3 分 × 2 システム = ~13 時間
- 合計: ~26 時間（複数 Colab セッションに分割、チェックポイント対応）

## リスクと対策

| リスク | 対策 |
|-------|------|
| ngrok トンネルの安定性 | チェックポイントで中断再開対応 |
| MapFan API レート制限 | API コール間ディレイ、セッションキャッシュ |
| Colab での MCP SDK 互換性 | `mcp[cli]>=1.9.0` を pip install、`nest_asyncio` |
| Qwen3-32B の function calling 品質 | エージェント方式は補足扱い |

## ベースライン

Phase 9-C C2 (Qwen3-32B + C1 prompt): **composite 70.4pt**
