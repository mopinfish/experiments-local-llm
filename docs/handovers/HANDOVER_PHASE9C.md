# Phase 9-C: 回答品質改善実験 - 引き継ぎドキュメント

**作成日**: 2026-02-27
**プロジェクト**: experiments-local-llm
**ステータス**: Step 1/2 完了、Step 3 未着手
**PR**: #23 (`feature/phase9c-step2-model` → `main`)

---

## 1. Phase 9-C概要

### 1.1 目的

Phase 9-Bで判明したキーワード成功率96.2% vs composite成功率33.9%の乖離を解消するため、3つの変数（プロンプト/モデル/FT）を段階的に変更し、各変数の寄与を分離測定する。

### 1.2 達成した成果

| ステップ | 変更内容 | Composite | Reasoning | Evidence | Success% |
|----------|----------|-----------|-----------|----------|----------|
| C0 (ベースライン) | — | 52.2 | 1.43 | 2.34 | 33.9% |
| **C1 (Step 1)** | プロンプト改善 | **67.1** (+14.9) | **2.65** (+1.22) | **3.71** (+1.37) | **76.2%** (+42.3%) |
| **C2 (Step 2)** | Qwen3-32B | **70.4** (+3.3) | **3.07** (+0.42) | **3.85** (+0.13) | **83.1%** (+6.9%) |
| **累積改善** | | **+18.2** | **+1.64** | **+1.51** | **+49.2%** |

**Phase 9-C全体の成功基準は Step 2 時点で全達成済み。**

### 1.3 主要な知見

1. **プロンプト改善の効果が圧倒的** — composite +14.9pt (全改善の82%)。モデル変更+3.3pt (18%)
2. **モデルスケールアップは根拠抽出力で効く** — 具体的POI名・件数の引用能力が大幅向上
3. **32Bモデルの弱点: 過度な慎重さ** — sensitivity/decision_supportで「データ不足」と誤判断する傾向
4. **Colab Pro A100は必須** — Qwen3-32B 4bitはL4 (22GB)では動作不可

---

## 2. ブランチ・PR構成

```
main
  └── feature/phase9c-step2-model  ← PR #23 (Step 1 + Step 2 の全変更を含む)
        ├── Step 1: プロンプト改善 (src/ 変更 + 評価結果)
        └── Step 2: モデル変更 (ノートブック + 評価結果)
```

PR #23 には Step 1 と Step 2 の両方が含まれる（同一ブランチで作業）。

---

## 3. ファイル構成

### 3.1 src/ の変更（Step 1 プロンプト改善）

| ファイル | 変更内容 |
|---------|---------|
| `src/structured_rag_system.py` | system_prompt を3部構成（結論→根拠→補足）に改善、ユーザープロンプトから固定回答開始文を削除 |
| `src/adaptive_rag_system.py` | 同上（structured_ragと同期） |
| `src/agent_prompts.py` | AGENT_SYSTEM_PROMPT、ANSWER_GENERATION_PROMPT_TEMPLATE を強化 |

### 3.2 ノートブック

| ファイル | 説明 | 実行環境 |
|---------|------|----------|
| `notebooks/phase9c_step1_prompt_evaluation.ipynb` | C0 vs C1: 4システム×130件=520クエリ評価 | Colab T4 |
| `notebooks/phase9c_step2_model_evaluation.ipynb` | C1 vs C2: Hybrid RAG×130件評価 | **Colab Pro A100** |

### 3.3 結果ファイル

| ファイル | 内容 |
|---------|------|
| `results/phase9c_step1_20260226_022240.json` | C1評価結果 (520クエリ、4システム) |
| `results/phase9c_step2_20260227_071313.json` | **C2評価結果 (130件、Qwen3-32B)** |
| `results/phase9c_step2_20260227_035416.json` | C2中間結果 (Qwen3-14Bフォールバック版、参考) |
| `results/phase9c_step2_*.png` | レベル別/エリア別/メトリクス比較グラフ |
| `results/checkpoint_c1_*.json` | Step 1 チェックポイント (4システム分) |
| `results/checkpoint_c2_hybrid_rag.json` | Step 2 チェックポイント |

### 3.4 ドキュメント

| ファイル | 内容 |
|---------|------|
| `docs/plans/PHASE9C_QUALITY_IMPROVEMENT_PLAN.md` | 実験計画書（実績値更新済み） |
| `docs/reports/PHASE9C_STEP1_PROMPT_IMPROVEMENT_REPORT.md` | Step 1 詳細レポート |
| `docs/reports/PHASE9C_STEP2_MODEL_UPGRADE_REPORT.md` | Step 2 詳細レポート |
| `docs/handovers/HANDOVER_PHASE9C.md` | 本ドキュメント |

---

## 4. 技術的知見

### 4.1 Qwen3モデルの扱い方

Qwen3はQwen2.5の後継（2025年4月リリース）で、thinking/non-thinking dual modeを持つ。

**非思考モードの適用方法**:

```python
# tokenizer.apply_chat_template に enable_thinking=False を注入
original_apply = tokenizer.apply_chat_template
def patched_apply(*args, **kwargs):
    kwargs['enable_thinking'] = False
    result = original_apply(*args, **kwargs)
    # テンプレートに <think> が残る場合はテキストレベルで除去
    if isinstance(result, str) and '<think>' in result:
        result = result.replace('<think>\n', '').replace('<think>', '')
    return result
tokenizer.apply_chat_template = patched_apply
```

**注意**: `enable_thinking=False` を渡してもテンプレート出力に `<think>` が残ることがある。テキストレベルでの除去が必要。さらにモデル出力にも `<think>...</think>` ブロックが生成される場合があり、`re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)` で除去する。

### 4.2 bitsandbytes NF4 on A100

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True  # ダブル量子化でメモリ節約
)
```

- Qwen3-32B NF4: VRAM ~18-20GB（A100 40GBで十分な余裕）
- bitsandbytesはロード時にfp16で読み込んでからレイヤーごとに量子化するため、ピーク時VRAMはモデルサイズより大きい
- AWQ事前量子化版 (`Qwen/Qwen3-32B-AWQ`) は transformers >= 4.51.0 で `gptqmodel` が必要だが、互換性問題が多い。**bitsandbytes NF4を推奨**

### 4.3 Colab Pro A100 GPU選択

Colab Proにアップグレードしても、ランタイム設定で**A100を明示的に選択**しないとL4が割り当てられることがある。

```
Runtime > Change runtime type > GPU > A100 を明示選択 > Save
```

ノートブックではVRAM < 30GBの場合に `RuntimeError` で停止するガードを実装済み。

### 4.4 Colabノートブックのgit push問題

Colab上で `!git add results/phase9c_step2_*` とすると、シェルglobが `/content` (CWD) で展開され、PROJECT_PATH配下のファイルが見つからない。Python `glob.glob()` で展開してからgit addに渡す方式で解決:

```python
import glob as g
files = [os.path.relpath(f, PROJECT_PATH) for f in g.glob(f"{PROJECT_PATH}/results/phase9c_step2_*")]
for f in files:
    !git -C {PROJECT_PATH} add "{f}"
```

### 4.5 チェックポイント機構

`evaluators_multi_area.py` の `evaluate_all()` は `checkpoint_file` パラメータでJSON保存可能。Colabセッション切断後もチェックポイントから再開できる。

**注意**: モデル変更時は古いチェックポイントを削除すること。Step 2ノートブックでは `model_id == "Qwen/Qwen3-32B"` の場合に自動削除するロジックを実装。

---

## 5. C2結果の詳細分析

### 5.1 レベル別

| Level | C1 | C2 | Delta | 傾向 |
|-------|------|------|-------|------|
| L1 Basic | 64.0 | 64.4 | +0.4 | 変化なし |
| L2 Spatial | 59.8 | 64.2 | **+4.4** | 改善 |
| L3 Constraint | 77.0 | 84.8 | **+7.9** | 大幅改善 |
| L4 Decision | 70.1 | 69.1 | -1.0 | 微減 |
| L5 Advanced | 62.8 | 68.1 | **+5.4** | 改善 |

### 5.2 サブカテゴリ別（特筆すべきもの）

- **大幅改善**: competitor (+20.2), comparison (+11.9), basic_location (+11.3), constraint_single (+11.0), multi_hop (+10.2)
- **劣化**: sensitivity (-5.8), decision_support (-5.0), relation (-3.8)

### 5.3 32Bモデルの特性

- **強み**: 根拠抽出力（POI名・件数の正確な引用）、制約適用、安定性（エラー率0%）
- **弱み**: 過度な慎重さ（hedging）、推論コスト（22秒→50秒）

---

## 6. Step 3 以降の方向性

### 6.1 Step 3 (QLoRA) の実施判断

Step 2で全目標達成済みのため、Step 3の必要性は再検討が必要。以下の選択肢がある:

**Option A: FTなしの改善施策を先に試す**
1. **sensitivity/decision_support劣化対策**: 「データに基づき断定的に回答せよ」等のプロンプト追加
2. **temperature最適化**: 0.7 → 0.3~0.5 でhedging軽減
3. **enable_thinking=True**: thinking modeで推論品質向上（時間増のトレードオフ）

**Option B: Step 3 QLoRAを実施**
- ベースモデル: Qwen3-32B
- 学習データ: C2の130件出力からcomposite 70+を選定
- 目標: composite 75+, sensitivity/decision_support改善

**Option C: Phase 10（MCP/PostGIS全国展開）に進む**
- Phase 9-Cの目標は達成済み。これ以上の品質改善より、アーキテクチャ刷新に投資する判断もあり得る

### 6.2 Phase 10 への申し送り

Phase 9-Bの引き継ぎ（`HANDOVER_PHASE9B.md` セクション7）で示したMCPサーバー化の方針に加え:

- **Qwen3-32Bの利用**: A100必須。ローカル推論サーバー(vLLM等)の検討が必要
- **プロンプト設計の知見**: 3部構成（結論→根拠→補足）プロンプトは他のLLMでも効果が期待できる
- **評価フレームワーク**: `evaluators_multi_area.py` の多次元評価はPhase 10でも再利用可能

---

## 7. Issue/PR一覧

| # | タイトル | 状態 |
|---|---------|------|
| #20 | Phase 9-C Step 2: LLMモデル変更実験 | Open (PR #23で Close予定) |
| #21 | Phase 9-C Step 3: QLoRAファインチューニング実験 | Open |
| #23 | Phase 9-C: プロンプト改善 + モデルスケールアップで composite 70.4 達成 | Open |

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
Phase 9-B (4エリア拡張 / 130件×4sys=520クエリ)
  ↓
Phase 9-C (回答品質改善)  ← 現在地
  ├── Step 1: プロンプト改善     ✅ composite 52.2 → 67.1
  ├── Step 2: Qwen3-32B         ✅ composite 67.1 → 70.4
  └── Step 3: QLoRA             ⬜ 未着手（実施判断保留）
  ↓
Phase 10 (全国展開 / MCP / PostGIS)
```

---

## 9. 関連ドキュメント

| ドキュメント | 内容 |
|-------------|------|
| `docs/plans/PHASE9C_QUALITY_IMPROVEMENT_PLAN.md` | **実験計画書（実績値更新済み）** |
| `docs/reports/PHASE9C_STEP1_PROMPT_IMPROVEMENT_REPORT.md` | Step 1 詳細レポート |
| `docs/reports/PHASE9C_STEP2_MODEL_UPGRADE_REPORT.md` | Step 2 詳細レポート |
| `docs/handovers/HANDOVER_PHASE9B.md` | Phase 9-B引き継ぎ（前フェーズ） |
| `docs/plans/PHASE9B_MULTI_AREA_EXPERIMENT_PLAN.md` | Phase 9-B実験計画 |
| `CLAUDE.md` | プロジェクト全体のガイダンス |
