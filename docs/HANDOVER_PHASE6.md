# Phase 6 構造化RAG 引き継ぎドキュメント

**作成日**: 2026年1月23日  
**プロジェクト**: experiments-local-llm  
**ステータス**: Phase 6.2.1 完了

---

## 1. 現在の状況

### 達成した成果

| 指標 | Phase 5（ベースライン） | Phase 6.2.1（現在） | 改善 |
|------|------------------------|---------------------|------|
| **全体スコア** | 60.3pt | **91.6pt** | **+31.3pt（52%向上）** |
| 処理時間 | - | 21.9秒 | - |
| テストケース | 55件（L1-L5） | 55件（L1-L5） | - |

### サブカテゴリ別最終スコア

```
advanced_sensitivity:  100.0pt 🏆
decision_location:      98.4pt
constraint_multi:       98.0pt
decision_business:      97.3pt
basic_location:         96.8pt
spatial_proximity:      95.7pt
spatial_comparison:     92.0pt
advanced_comparison:    90.8pt
constraint_single:      85.3pt
advanced_uncertainty:   82.0pt
basic_category:         81.3pt
spatial_density:        80.8pt
```

### 改善の要約

単純なベクトル検索RAGでは空間的・数値的な質問に対応できなかった。Phase 6で全POIに距離・方角を付与し、東西比較・カテゴリ集計・最近傍検索・感度分析機能を実装。これらの構造化処理とベクトル検索結果を常に組み合わせてコンテキストを構築することで、60.3pt→91.6ptを達成した。

---

## 2. 実装済み機能

### 2.1 geo_utils.py

| 関数 | 機能 | 用途 |
|-----|------|------|
| `enrich_all_pois()` | 全POIに距離・方角を付与 | 空間クエリの基盤 |
| `get_nearest_pois()` | 距離ソートで最寄りPOI取得 | 近接性検索 |
| `filter_by_radius()` | 半径内POIをフィルタ | 距離制約 |
| `compare_by_radius()` | 2つの半径での件数比較 | 感度分析 |
| `generate_proximity_context()` | 近接性コンテキスト生成 | LLMプロンプト用 |
| `generate_sensitivity_context()` | 感度分析コンテキスト生成 | LLMプロンプト用 |

### 2.2 aggregator.py

| 関数 | 機能 | 用途 |
|-----|------|------|
| `compare_east_west()` | 東西のPOI数を比較 | 方向別比較 |
| `get_top_categories()` | カテゴリ別ランキング | 集計クエリ |
| `filter_by_category()` | カテゴリでフィルタ | カテゴリ検索 |
| `analyze_category_by_direction()` | 方向×カテゴリ分析 | 詳細分析 |

### 2.3 structured_rag_system.py

| クラス/関数 | 機能 | 用途 |
|------------|------|------|
| `QuestionAnalysis` | 質問分析結果のデータクラス | 質問タイプ判定 |
| `analyze_question()` | 質問文を解析 | 処理パス選択 |
| キーワード定義 | 近接性・感度分析キーワード | 質問タイプ検出 |

### 2.4 Notebook (phase6_full_evaluation.ipynb)

| セクション | 内容 |
|-----------|------|
| Section 1-3 | 環境セットアップ、モデルロード |
| Section 4 | StructuredRAGEvaluatorクラス（コア実装） |
| Section 5 | 評価実行 |
| Section 6-7 | 結果分析、保存 |

---

## 3. ファイル構成

```
experiments-local-llm/
├── src/
│   ├── geo_utils.py              # 空間処理（29KB）
│   ├── aggregator.py             # 集計処理（21KB）
│   ├── structured_rag_system.py  # 質問分析（32KB）
│   ├── test_cases_v2.py          # テストケース定義
│   └── __init__.py               # v0.6.2
├── notebooks/
│   └── phase6_full_evaluation.ipynb  # 評価Notebook
├── data/
│   └── poi_documents.json        # POIデータ（1,047件）
├── results/
│   ├── phase621_eval_*.json      # 評価結果
│   └── phase621_report_*.md      # 評価レポート
└── docs/
    ├── PHASE6_IMPROVEMENT_REPORT.md  # 改善詳細レポート
    └── HANDOVER_PHASE6.md            # 本ドキュメント
```

---

## 4. 技術的詳細

### 4.1 コンテキスト構築の設計（重要）

Phase 6.2.1の成功の鍵は、**構造化処理とベクトル検索の補完的統合**：

```python
def _build_context(self, question, analysis):
    structured_parts = []
    
    # 構造化コンテキスト（複数が同時に追加可能）
    if analysis.requires_proximity and cat:
        structured_parts.append(proximity_context)
    if analysis.requires_sensitivity and cat:
        structured_parts.append(sensitivity_context)
    if analysis.requires_comparison:
        structured_parts.append(comparison_context)
    if analysis.requires_aggregation:
        structured_parts.append(aggregation_context)
    
    # ベクトル検索は常に追加（補完として）
    vector_context = self._get_vector_search_context(question, k=5)
    
    return structured_parts + [vector_context]
```

**注意**: `elif`ではなく`if`を使用し、ベクトル検索を常に実行すること。

### 4.2 質問タイプの判定

```python
PROXIMITY_KEYWORDS = ["最も近い", "一番近い", "最寄り", "近い順", "最短"]
SENSITIVITY_KEYWORDS = ["変えても", "変更しても", "広げても", "狭めても", "範囲を", "半径を", "成立"]
```

### 4.3 モデル構成

- **LLM**: Qwen/Qwen2.5-7B-Instruct（4bit量子化）
- **Embedding**: intfloat/multilingual-e5-base
- **ベクトルストア**: ChromaDB（ローカル、非永続化）
- **実行環境**: Google Colab（T4 GPU）

---

## 5. 今後の取り組み候補

### 5.1 Phase 7: ファインチューニング

**目的**: 91.6ptをベースラインとして、モデル調整で95pt+を目指す

**アプローチ**:
- LoRA/QLoRAによる効率的ファインチューニング
- 地理クエリ特化のデータセット作成
- プロンプトテンプレートの最適化

### 5.2 処理時間最適化

**現状**: 21.9秒/質問

**改善案**:
- ベクトル検索結果のキャッシュ
- 質問タイプに応じたベクトル検索のスキップ（集計のみの場合など）
- バッチ処理の導入

### 5.3 Phase 8: 全国展開

**目的**: 渋谷以外のエリアへの適用

**課題**:
- POIデータの収集（MapFan API活用）
- 地域特性に応じたカテゴリマッピング
- スケーラビリティの確保

### 5.4 MapFan MCP統合

**目的**: 本改善をMapFan MCPサーバーに適用

**タスク**:
- geo_utils.py、aggregator.pyのMCPサーバーへの移植
- ツール定義の追加（nearest_poi、compare_radius等）
- 評価フレームワークの統合

### 5.5 グラフRAG導入（Phase 6.3）

**目的**: POI間の関係性を活用

**アプローチ**:
- 近接関係グラフの構築
- 同一建物・同一エリアの関係
- 経路探索との統合

---

## 6. 実行手順

### 6.1 評価の再実行

```fish
# 1. Google Colabでノートブックを開く
# experiments-local-llm/notebooks/phase6_full_evaluation.ipynb

# 2. ランタイム → GPUを選択（T4推奨）

# 3. セルを順次実行
# - Section 1-3: 環境セットアップ（約5分）
# - Section 4: RAGシステム初期化
# - Section 5: 評価実行（約30-40分）
# - Section 6-7: 結果分析・保存
```

### 6.2 ローカルでの開発

```fish
cd /path/to/experiments-local-llm

# 依存パッケージ
pip install langchain langchain-chroma chromadb sentence-transformers

# モジュールのインポートテスト
python -c "from src.geo_utils import enrich_all_pois; print('OK')"
```

---

## 7. 既知の課題・制限

### 7.1 処理時間

- 現在21.9秒/質問は本番利用には長い
- ベクトル検索の常時実行が主因

### 7.2 メタデータの制約

- ChromaDBはネストした辞書を受け付けない
- `flatten_metadata()`で平坦化が必要

### 7.3 テストケースの属性名

- `level` vs `difficulty`、`subcategory` vs `sub_category`の不統一
- Notebookで両方に対応するコードを実装済み

### 7.4 モデル依存

- Qwen2.5-7B-Instructに最適化
- 他モデルでは調整が必要な可能性

---

## 8. 参考資料

### リポジトリ内ドキュメント

- `docs/PHASE6_IMPROVEMENT_REPORT.md` - 改善プロセスの詳細
- `results/phase621_eval_*.json` - 評価結果（生データ）
- `results/phase621_report_*.md` - 評価レポート

### 関連プロジェクト

- GeoTechAgent-mapfanmcp - MapFan MCPサーバー
- mcp-benchmark-tool - MCPベンチマークツール

---

## 9. 連絡事項

### 成果サマリー

```
Phase 5:    60.3pt ──────────────────
Phase 6.2.1: 91.6pt ████████████████████████████████████ (+31.3pt, 52%↑)
```

### 重要な学び

1. **排他的分岐（elif）は危険** - 新機能追加時に既存機能を壊す
2. **補完的アプローチが有効** - 構造化処理とベクトル検索は排他ではなく補完
3. **段階的評価が重要** - 各フェーズで全サブカテゴリを評価し、悪化を早期発見

### 次のアクション推奨

1. **短期**: 処理時間最適化（21.9秒→15秒以下）
2. **中期**: ファインチューニングで95pt+を目指す
3. **長期**: MapFan MCPサーバーへの統合

---

**以上で Phase 6 の引き継ぎを完了します。**
