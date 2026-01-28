# セッション引き継ぎドキュメント

**作成日**: 2026年1月29日
**プロジェクト**: experiments-local-llm
**ブランチ**: feature/finetuning-experiment
**PR**: #3

---

## 1. 本セッションで完了した作業

### 1.1 ファインチューニング実験（Phase 7準備）

| 成果物 | 内容 |
|--------|------|
| `docs/FINETUNING_EXPERIMENT_PLAN.md` | 実験計画書 |
| `notebooks/finetuning_01_data_preparation.ipynb` | データ準備（8パターン生成） |
| `notebooks/finetuning_02_training.ipynb` | QLoRA学習（T4最適化済み） |
| `notebooks/finetuning_03_evaluation.ipynb` | 4モデル比較評価 |
| `docs/FINETUNING_EXPERIMENT_REPORT.md` | 実験結果の学術レポート |

**実験結果**:
- Baseline: 75.7pt
- RAG: 77.1pt
- FT-Base: 77.1pt
- **FT+RAG: 78.5pt**（最高スコア）

**結論**: ファインチューニング単体では+1.4ptの効果、RAGとの併用でさらに+1.4pt向上。Phase 7ではより大規模なデータセットでの検証が必要。

### 1.2 Phase 5-6 アカデミックレポート

| 成果物 | 内容 |
|--------|------|
| `docs/STRUCTURED_RAG_RESEARCH_REPORT.md` | Phase 5-6の学術論文形式レポート |

**内容**:
- 階層化テストフレームワーク（55件、L1-L5）の設計
- 構造化RAGアーキテクチャの提案
- 60.3pt → 91.6pt（+52%）の改善プロセス
- if vs elif の重要な設計判断

### 1.3 ドキュメント整備

| 成果物 | 内容 |
|--------|------|
| `README.md` 更新 | Phase 5-6の説明、質問分析モジュールの解説追加 |

---

## 2. プロジェクト全体の状況

### 2.1 フェーズ進行状況

```
Phase 1-3: 環境構築・基本RAG [完了]
    │
Phase 4: テスト・評価基盤 [完了]
    │
Phase 5: 階層化テストフレームワーク [完了]
    │       55件テストケース（L1-L5）
    │       ベースライン: 60.3pt
    │
Phase 6: 構造化RAG [完了] ★現在地
    │       Phase 6.1: 69.6pt (+9.3pt)
    │       Phase 6.2: 64.1pt (-5.5pt) ※一時悪化
    │       Phase 6.2.1: 91.6pt (+27.5pt) ★最終スコア
    │
Phase 7: ファインチューニング [実験完了・本格実装待ち]
    │       予備実験: FT+RAG 78.5pt
    │       目標: 95pt以上
    │
Phase 8: 全国展開 [未着手]
    │       PostGIS/Supabase導入
    │       目標: 500万POIで50ms以下
    │
Phase 9: MCP統合 [未着手]
        MapFan MCPサーバーへの統合
```

### 2.2 現在のスコア

| 指標 | 値 |
|------|-----|
| 全体スコア | **91.6pt** |
| 最高サブカテゴリ | advanced_sensitivity: 100.0pt |
| 最低サブカテゴリ | spatial_density: 80.8pt |
| 平均処理時間 | 21.9秒/質問 |

### 2.3 Git状況

```
ブランチ: feature/finetuning-experiment
最新コミット: docs: README.mdにPhase 5-6構造化RAGと質問分析モジュールの解説を追加
PR #3: オープン中（マージ待ち）
```

---

## 3. 主要ファイル一覧

### 3.1 ソースコード（src/）

| ファイル | 役割 | 重要度 |
|---------|------|--------|
| `geo_utils.py` | 空間処理（距離、方角、近接性、感度分析） | ★★★ |
| `aggregator.py` | 集計処理（東西比較、カテゴリ集計） | ★★★ |
| `structured_rag_system.py` | 質問分析、コンテキスト構築 | ★★★ |
| `test_cases_v2.py` | 55件テストケース定義 | ★★ |
| `evaluators_v2.py` | 評価スコアリング | ★★ |

### 3.2 Notebooks

| ファイル | 用途 | 実行環境 |
|---------|------|----------|
| `phase6_full_evaluation.ipynb` | Phase 6評価 | Colab T4 |
| `finetuning_01_data_preparation.ipynb` | FTデータ生成 | Colab T4 |
| `finetuning_02_training.ipynb` | QLoRA学習 | Colab T4 |
| `finetuning_03_evaluation.ipynb` | 4モデル比較 | Colab T4 |

### 3.3 ドキュメント（docs/）

| ファイル | 内容 |
|---------|------|
| `STRUCTURED_RAG_RESEARCH_REPORT.md` | Phase 5-6 学術レポート |
| `PHASE6_IMPROVEMENT_REPORT.md` | Phase 6 改善詳細 |
| `HANDOVER_PHASE6.md` | Phase 6 技術引き継ぎ |
| `FINETUNING_EXPERIMENT_REPORT.md` | FT実験レポート |
| `FINETUNING_EXPERIMENT_PLAN.md` | FT実験計画書 |
| `STRUCTURED_DATA_DESIGN_GUIDE.md` | 構造化データ設計ガイド |

---

## 4. 今後の取り組み

### 4.1 短期（Phase 7 本格実装）

**目標**: 91.6pt → 95pt以上

**タスク**:
1. [ ] 学習データの拡充（現在2,640件 → 10,000件以上）
2. [ ] データ品質の向上（実際のテストケースとの整合性）
3. [ ] ハイパーパラメータチューニング
4. [ ] A100 GPU環境での学習（より大きなモデル/バッチサイズ）
5. [ ] 評価の精緻化（LLM-as-Judge導入検討）

**参考**: 予備実験ではFT+RAGで78.5pt（ベースライン比+3.8%）を達成。より大規模な学習で効果増大が期待される。

### 4.2 中期（Phase 8 全国展開）

**目標**: 500万POIで50ms以下の応答時間

**タスク**:
1. [ ] PostGIS/Supabase環境構築
2. [ ] 空間インデックスの設計
3. [ ] 動的基準点解決機能の実装
4. [ ] 渋谷以外のエリア（新宿、横浜等）でのテスト
5. [ ] パフォーマンスベンチマーク

**注意点**:
- 現在の実装は渋谷駅固有のハードコーディングあり
- `SHIBUYA_STATION = (35.658034, 139.701636)` が固定値
- 全国展開には動的な基準点解決が必須

### 4.3 長期（Phase 9 MCP統合）

**目標**: MapFan MCPサーバーへの統合

**タスク**:
1. [ ] geo_utils.py, aggregator.py のTypeScript移植
2. [ ] MCPツール定義（nearest_poi, compare_areas等）
3. [ ] MapFan APIとの連携
4. [ ] 本番デプロイ

---

## 5. 技術的な注意事項

### 5.1 重要な設計判断

**if文 vs elif文の選択**:

Phase 6.2で`elif`を使用したことで性能が悪化（-5.5pt）。Phase 6.2.1で`if`に修正し+27.5ptの改善。

```python
# NG: elif による排他的分岐
if analysis.requires_proximity:
    ...
elif analysis.requires_sensitivity:  # ← 近接性が有効だと実行されない
    ...

# OK: if による並列実行
if analysis.requires_proximity:
    ...
if analysis.requires_sensitivity:   # ← 両方実行される
    ...
```

**教訓**: 構造化処理とベクトル検索は排他ではなく相補的に実行すべき。

### 5.2 Colab実行時のメモリ最適化

T4 GPU（16GB VRAM）での学習には以下の設定が必要:

```python
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 16
MAX_SEQ_LENGTH = 512
LORA_R = 8
target_modules = ["q_proj", "v_proj"]
```

### 5.3 テストケース属性の不統一

`test_cases_v2.py`では`level`と`subcategory`を使用するが、一部のコードでは`difficulty`や`sub_category`を参照する場合がある。評価Notebookでは両方に対応するコードを実装済み。

---

## 6. 参考リンク

### 6.1 プロジェクト内

- PR #3: feature/finetuning-experiment
- `CLAUDE.md`: Claude Code用ガイダンス

### 6.2 外部リソース

- [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- [unsloth](https://github.com/unslothai/unsloth): 高速ファインチューニング
- [ChromaDB](https://docs.trychroma.com/): ベクトルストア

---

## 7. 次回セッションへの引き継ぎ事項

### 7.1 即時対応

1. **PR #3のマージ判断**: ファインチューニング実験の成果をmainにマージするか
2. **Colab環境の確認**: ノートブックが最新状態で動作するか検証

### 7.2 検討事項

1. **Phase 7の方針**: 学習データ拡充 or ハイパーパラメータ最適化のどちらを優先するか
2. **全国展開のタイムライン**: Phase 8をいつ開始するか
3. **評価指標の見直し**: LLM-as-Judgeを導入するか

### 7.3 質問・確認事項

- 特になし（本セッションで主要な作業は完了）

---

**作成者**: Claude Opus 4.5
**セッション終了時点**: 2026年1月29日
