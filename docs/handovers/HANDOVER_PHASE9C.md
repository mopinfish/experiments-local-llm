# Phase 9-C: 回答品質改善実験 - 引き継ぎドキュメント

**作成日**: 2026-02-27（初版）/ 2026-03-01（Step 3 完了更新）
**プロジェクト**: experiments-local-llm
**ステータス**: Step 1/2/3 全完了、Phase 9-C 終結
**PR**: #23 (Step 1+2, merged), **#24 (Step 3, open)**

---

## 1. Phase 9-C 総括

### 1.1 目的

Phase 9-B で判明したキーワード成功率 96.2% vs composite 成功率 33.9% の乖離を解消するため、3 つの変数（プロンプト/モデル/FT）を段階的に変更し、各変数の寄与を分離測定する。

### 1.2 最終結果

| ステップ | 変更内容 | Composite | Reasoning | Evidence | Success% | 判定 |
|----------|----------|-----------|-----------|----------|----------|------|
| C0 (ベースライン) | C0 プロンプト + Qwen2.5-7B | 52.2 | 1.43 | 2.34 | 33.9% | — |
| **C1 (Step 1)** | C1 改善プロンプト + Qwen2.5-7B | **67.1** (+14.9) | **2.65** (+1.22) | **3.71** (+1.37) | **76.2%** (+42.3%) | 最大効果 |
| **C2 (Step 2)** | C1 改善プロンプト + Qwen3-32B | **70.4** (+3.3) | **3.07** (+0.42) | **3.85** (+0.13) | **83.1%** (+6.9%) | 有効 |
| **C3 (Step 3)** | QLoRA FT + Qwen3-32B + RAG | 69.5 (-0.9) | 3.08 (+0.01) | **4.32** (+0.47) | 83.1% (±0.0%) | **無効（採用せず）** |
| **累積改善** | | **+18.2** | **+1.64** | **+1.51** | **+49.2%** | |

**Phase 9-C の最終到達点は C2（Step 2）の composite 70.4pt**。Step 3 の QLoRA FT は効果がなく採用しない。

### 1.3 主要な知見

1. **プロンプト改善が全改善量の 81.9% を占める** — composite +14.9pt。モデル変更 +3.3pt (18.1%)、FT -1.0pt (-4.9%)
2. **QLoRA FT の純粋効果はゼロ** — 59 件の学習データでは汎化した地理知識の獲得に不足。学習は回答形式を暗記しただけ
3. **FT+RAG の相乗効果は負** — FT がパターンを固定化し RAG の動的コンテキスト活用を阻害
4. **RAG は最も安定的な品質基盤** — FT なしの RAG + 改善プロンプトが最高スコア
5. **過学習分析で選定バイアスを検出** — FT なしの System A の FT-data/Non-FT gap が +15.6 で最大。学習データ選定の非ランダム性がスコア比較を歪める

### 1.4 Step 3 で追加された 4 システム比較実験設計

Step 3 では従来の 3 システム比較に **System D（FT + C0 オリジナルプロンプト）** を追加し、FT・プロンプト・RAG の寄与を完全分離する実験設計を採用した。

| System | モデル | プロンプト | RAG | Composite | 目的 |
|--------|--------|-----------|-----|-----------|------|
| A: RAG (C2) | Qwen3-32B | C1 改善版 | あり | **70.4** | ベースライン |
| B: FT+prompt | QLoRA FT | C1 改善版 | なし | 67.9 | FT + 改善プロンプト |
| C: FT+RAG | QLoRA FT | C1 改善版 | あり | 69.5 | FT + RAG 相補 |
| **D: FT-bare** | QLoRA FT | **C0 オリジナル** | なし | 51.2 | **FT 単体測定用** |

D の追加により `B - D = +16.7pt`（プロンプト効果）、`D - C0 = -1.0pt`（FT 純粋効果）の分離が可能になった。

---

## 2. ブランチ・PR 構成

```
main
  ├── feature/phase9c-step2-model   ← PR #23 (Step 1 + Step 2, merged)
  │     ├── Step 1: プロンプト改善 (src/ 変更 + 評価結果)
  │     └── Step 2: モデル変更 (ノートブック + 評価結果)
  │
  └── feature/phase9c-step3-qlora   ← PR #24 (Step 3, open)
        ├── QLoRA 学習環境構築 + 学習データ作成
        ├── QLoRA 学習実行 (Qwen3-32B, 59 件)
        ├── 4 システム比較評価 (130 件 × 4)
        └── 実験レポート完成
```

PR #24 は 40 ファイル変更、69,280 行追加。Issue #21 を自動クローズ。

---

## 3. ファイル構成

### 3.1 src/ の変更（Step 1 プロンプト改善、PR #23 で merged）

| ファイル | 変更内容 |
|---------|---------|
| `src/structured_rag_system.py` | system_prompt を 3 部構成（結論→根拠→補足）に改善 |
| `src/adaptive_rag_system.py` | 同上（structured_rag と同期） |
| `src/agent_prompts.py` | AGENT_SYSTEM_PROMPT、ANSWER_GENERATION_PROMPT_TEMPLATE を強化 |

### 3.2 ノートブック

| ファイル | 説明 | 実行環境 | 所要時間 |
|---------|------|----------|---------|
| `notebooks/phase9c_step1_prompt_evaluation.ipynb` | C0 vs C1: 4 システム × 130 件 = 520 クエリ | Colab T4 | ~6 時間 |
| `notebooks/phase9c_step2_model_evaluation.ipynb` | C1 vs C2: Hybrid RAG × 130 件 | Colab Pro A100 | ~1.8 時間 |
| `notebooks/phase9c_step3_qlora_training.ipynb` | QLoRA 学習 (Qwen3-32B, 59 件, 5 epochs) | Colab Pro A100 | ~30 分 |
| `notebooks/phase9c_step3_evaluation.ipynb` | **4 システム比較評価** (130 件 × 4 = 520 クエリ) | Colab Pro A100 | ~7-8 時間 |

### 3.3 結果ファイル

| ファイル | 内容 |
|---------|------|
| `results/phase9c_step1_20260226_022240.json` | Step 1 評価結果 (520 クエリ、4 システム) |
| `results/phase9c_step2_20260227_071313.json` | Step 2 評価結果 (130 件、Qwen3-32B) |
| `results/phase9c_step3_20260301_103931.json` | **Step 3 評価結果 (130 件 × 4 システム、最終版)** |
| `results/phase9c_step3_training_metadata.json` | QLoRA 学習メタデータ |
| `results/checkpoint_c3_ft_only.json` | System B チェックポイント |
| `results/checkpoint_c3_ft_rag.json` | System C チェックポイント |
| `results/checkpoint_c3_ft_bare.json` | System D チェックポイント |
| `results/phase9c_step3_*.png` | 全体比較/レベル別/過学習分析グラフ (3 枚) |
| `data/phase9c_training_data.json` | QLoRA 学習データ (59 件、Alpaca 形式) |
| `scripts/prepare_training_data.py` | 学習データ作成スクリプト |

### 3.4 ドキュメント

| ファイル | 内容 |
|---------|------|
| `docs/plans/PHASE9C_QUALITY_IMPROVEMENT_PLAN.md` | 実験計画書（実績値更新済み） |
| `docs/reports/PHASE9C_STEP1_PROMPT_IMPROVEMENT_REPORT.md` | Step 1 詳細レポート |
| `docs/reports/PHASE9C_STEP2_MODEL_UPGRADE_REPORT.md` | Step 2 詳細レポート |
| `docs/reports/PHASE9C_STEP3_QLORA_FINETUNING_REPORT.md` | **Step 3 詳細レポート（学術論文形式、536 行）** |
| `docs/handovers/HANDOVER_PHASE9C.md` | 本ドキュメント |

---

## 4. 技術的知見

### 4.1 Qwen3 モデルの扱い方

Qwen3 は thinking/non-thinking dual mode を持つ。

**非思考モードの適用方法**:

```python
original_apply = tokenizer.apply_chat_template
def patched_apply(*args, **kwargs):
    kwargs['enable_thinking'] = False
    return original_apply(*args, **kwargs)
tokenizer.apply_chat_template = patched_apply
```

**出力からの `<think>` タグ除去** (学習・推論両方で必要):

```python
import re
answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL).strip()
answer = re.sub(r'</think>', '', answer).strip()  # 閉じタグのみ残る場合
```

### 4.2 bitsandbytes NF4 量子化

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,  # Step 3 で float16 → bfloat16 に修正
    bnb_4bit_use_double_quant=True
)
```

- **重要**: 学習ノートブックと評価ノートブックで `compute_dtype` を統一すること。不一致があると学習時と推論時で数値精度が異なり、結果が不安定になる
- Qwen3-32B NF4: VRAM ~18-20GB（A100 80GB で十分な余裕）
- AWQ 事前量子化版は互換性問題が多い。**bitsandbytes NF4 を推奨**

### 4.3 QLoRA 学習の注意点（Step 3 で得た知見）

**学習データ量の閾値**:
- 59 件では効果ゼロ。パターン空間（320 パターン）の 18.4% しかカバーしていない
- 推奨: 最低 500 件以上、全レベル・全サブカテゴリを均等にカバー

**学習データ選定バイアス**:
- `composite_score >= 70` で自動選定すると、RAG で「解きやすい」問題に偏る
- 評価時に FT-data vs Non-FT-data の gap が発生するが、これは FT 過学習ではなく選定バイアス
- System A（FT なし）の gap が +15.6 で最大であることがその証拠

**Train/Eval loss の解釈**:
- Train loss 0.086, Eval loss 0.355 (gap 4.1 倍) — 表面上は過学習だが、eval loss は大きく悪化していない
- Train loss の収束は「回答形式の暗記」であり「知識の獲得」ではない

### 4.4 Colab Pro A100 必須

- Qwen3-32B 4bit は L4 (22GB VRAM) では動作不可
- ランタイム設定で **A100 を明示的に選択** すること
- ノートブックに VRAM < 30GB の場合に `RuntimeError` で停止するガードを実装済み

### 4.5 チェックポイント機構

`evaluators_multi_area.py` の `evaluate_all()` は `checkpoint_file` パラメータで JSON 保存可能。Colab セッション切断後も再開できる。

- Step 3 では 4 システム × 130 件 = 520 クエリで計 7-8 時間。チェックポイント必須
- モデル変更時は古いチェックポイントを削除すること

### 4.6 Colab ノートブックの git push 問題

Colab 上で `!git add results/phase9c_step3_*` とするとシェル glob が CWD で展開される。`git -C {REPO_PATH}` と組み合わせて解決:

```python
!git -C {REPO_PATH} add results/phase9c_step3_*
```

---

## 5. Step 3 詳細分析

### 5.1 寄与分離分析（Step 3 の核心）

```
C0 ベースライン:    52.2
                     │
                     ├── FT 純粋効果: -1.0 ──→ D (FT-bare): 51.2
                     │                          │
                     │                          └── プロンプト改善: +16.7 ──→ B (FT+prompt): 67.9
                     │                                                         │
                     │                                                         └── RAG 追加: +1.6 ──→ C (FT+RAG): 69.5
                     │
                     └── RAG + プロンプト: +18.2 ──→ A (RAG): 70.4
```

| 要素 | 寄与量 | 寄与率 | 費用対効果 |
|------|--------|--------|-----------|
| プロンプト改善 (B - D) | +16.7pt | **81.9%** | 極めて高い（コストゼロ） |
| モデル変更 32B (C2 - C1) | +3.3pt | 18.1% | 中程度（計算コスト増） |
| QLoRA FT (D - C0) | -1.0pt | -4.9% | **負**（学習コスト + 品質低下） |

### 5.2 レベル別パターン

| Level | A: RAG | B: FT+p | C: FT+RAG | D: bare | Best | 解釈 |
|-------|--------|---------|-----------|---------|------|------|
| L1 Basic | 64.4 | 64.5 | 59.7 | **82.2** | D | FT 暗記が有効（単純パターン） |
| L2 Spatial | 64.2 | **65.7** | 62.4 | 43.6 | B | プロンプトが必要 |
| L3 Constraint | 84.8 | 80.5 | **85.5** | 59.1 | C | FT+RAG の相補性あり |
| L4 Decision | 69.1 | 62.3 | **72.9** | 31.2 | C | FT+RAG の相補性あり |
| L5 Advanced | **68.1** | 64.7 | 63.6 | 36.9 | A | 高度推論は RAG 単体が最善 |

### 5.3 過学習分析

| System | FT data (59 件) | Non-FT (71 件) | Gap | 判定 |
|--------|-----------------|----------------|-----|------|
| A: RAG (参考) | 79.0 | 63.3 | +15.6 | **選定バイアス** |
| B: FT+prompt | 68.1 | 67.7 | +0.5 | 過学習なし |
| C: FT+RAG | 75.3 | 64.7 | +10.6 | バイアス + 軽度 FT 過学習 |
| D: FT-bare | 50.2 | 52.0 | -1.8 | 過学習なし |

---

## 6. 次フェーズへの方向性

### 6.1 Phase 9-C の結論に基づく推奨

Step 3 の結果を踏まえ、以下の優先順位を推奨する:

**優先度 1: プロンプトの更なる改善（C1 → C2 プロンプト）**
- Step 1 で +14.9pt の実績がある最も費用対効果の高い施策
- L4 Decision / L5 Advanced の弱点に特化した改良で +3-5pt を見込む
- temperature 最適化（0.7 → 0.3-0.5）、enable_thinking=True の検討

**優先度 2: RAG 検索精度の向上**
- 埋め込みモデルの更新（multilingual-e5-base → e5-large）
- reranker の導入
- RAG は +18.2pt の安定寄与があり、検索精度改善は直接的にスコア向上に寄与

**優先度 3: Phase 10（全国展開）への移行**
- Phase 9-C の composite 70.4pt を基盤に、PostGIS/Supabase による 500 万 POI 規模への展開
- RAG + 改善プロンプトの組み合わせが最も堅牢な構成

**優先度 低: QLoRA FT の再挑戦**
- 500 件以上の多様な学習データ、人手キュレーション、独立評価セットの確保が前提
- 現状では投資対効果が低い

### 6.2 Phase 10 への申し送り

- **最適構成**: RAG + C1 改善プロンプト + Qwen3-32B（FT なし）
- **A100 必須**: Qwen3-32B 4bit はローカル推論サーバー (vLLM 等) の検討が必要
- **プロンプト設計の知見**: 3 部構成（結論→根拠→補足）は他の LLM でも効果が期待できる
- **評価フレームワーク**: `evaluators_multi_area.py` の多次元評価は Phase 10 でも再利用可能
- **4 システム比較実験設計**: FT 実験を行う際は System D 相当のコントロール群を必ず設けること

---

## 7. Issue / PR 一覧

| # | タイトル | 状態 | 備考 |
|---|---------|------|------|
| #19 | Phase 9-C Step 1: プロンプト改善 | Closed | PR #23 で完了 |
| #20 | Phase 9-C Step 2: LLM モデル変更 | Closed | PR #23 で完了 |
| #21 | Phase 9-C Step 3: QLoRA FineTuning | Open → **PR #24 で Close 予定** | 4 システム比較結果をコメント済み |
| #23 | Phase 9-C: プロンプト改善 + モデルスケールアップ | **Merged** | Step 1 + 2 |
| #24 | Phase 9-C Step 3: QLoRA 4 システム比較実験 | **Open** | Step 3、レビュー待ち |

---

## 8. Phase 間の関係性

```
Phase 5 (ベースライン 60.3pt / 渋谷55件)
  ↓
Phase 6 (構造化RAG 91.6pt / 渋谷55件)
  ↓
Phase 8 (Graph RAG / Adaptive RAG追加)
  ↓
Phase 9 (Agentic RAG追加 / 渋谷105件)
  ↓
Phase 9-B (4エリア拡張 / 130件×4sys=520クエリ / composite評価導入)
  ↓
Phase 9-C (回答品質改善)  ← 完了
  ├── Step 1: プロンプト改善     ✅ composite 52.2 → 67.1 (+14.9)
  ├── Step 2: Qwen3-32B         ✅ composite 67.1 → 70.4 (+3.3)
  └── Step 3: QLoRA FT          ✅ 効果なし (-0.9)、採用せず
  ↓
Phase 10 (全国展開 / MCP / PostGIS)  ← 次フェーズ
```

---

## 9. 関連ドキュメント

| ドキュメント | 内容 | 重要度 |
|-------------|------|--------|
| `docs/reports/PHASE9C_STEP3_QLORA_FINETUNING_REPORT.md` | **Step 3 詳細レポート（学術論文形式）** | 必須 |
| `docs/reports/PHASE9C_STEP2_MODEL_UPGRADE_REPORT.md` | Step 2 詳細レポート | 必須 |
| `docs/reports/PHASE9C_STEP1_PROMPT_IMPROVEMENT_REPORT.md` | Step 1 詳細レポート | 参考 |
| `docs/plans/PHASE9C_QUALITY_IMPROVEMENT_PLAN.md` | 実験計画書（実績値更新済み） | 参考 |
| `docs/handovers/HANDOVER_PHASE9B.md` | Phase 9-B 引き継ぎ（前フェーズ） | 参考 |
| `CLAUDE.md` | プロジェクト全体のガイダンス | 必須 |
