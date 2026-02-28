# Phase 9-C Step 3: QLoRA FineTuning 実験レポート

**実験日**: 2026-03-XX (TBD)
**実験ID**: PHASE9C_STEP3
**ブランチ**: `feature/phase9c-step3-qlora`

---

## 1. 実験概要

### 目的

QLoRA FineTuning の効果を測定し、RAG との相補性を検証する。3 システム比較により、FT 単体・RAG 単体・FT+RAG の寄与を分離測定する。

### 実験構成

| System | 構成 | 目的 |
|--------|------|------|
| **A: RAG (C2)** | Qwen3-32B + RAG | ベースライン（Step 2 既存結果） |
| **B: FT-only** | QLoRA Qwen3-32B + システムプロンプト（RAGなし） | FT 単体の効果測定 |
| **C: FT+RAG** | QLoRA Qwen3-32B + RAG | FT と RAG の相補効果測定 |

### 実行環境

- **GPU**: NVIDIA A100 (Google Colab Pro)
- **ベースモデル**: Qwen/Qwen3-32B (NF4 4-bit quantization)
- **QLoRA**: r=16, alpha=32, target=全attention+FFN
- **学習データ**: C2 高品質回答 59 件 (composite>=70, reasoning>=3)
- **評価対象**: 130 テストケース（5 エリア、L1-L5）

### C3 目標値

| 指標 | C2 実績 | C3 目標 |
|------|---------|---------|
| composite_score | 70.4 | **75+** |
| reasoning_score | 3.07 | **3.5+** |
| evidence_score | 3.85 | **4.0+** |
| composite_success_rate | 83.1% | **88%+** |

---

## 2. 学習データ統計

### 選定条件

- C2 結果 130 件から `composite_score >= 70` AND `reasoning_score >= 3` で選定
- 59 件（130 件中 45.4%）

### 分割

| 分割 | 件数 |
|------|------|
| Train | 47 |
| Validation | 12 |
| **合計** | **59** |

### レベル分布

| Level | 件数 |
|-------|------|
| L1 (Basic) | 9 |
| L2 (Spatial) | 9 |
| L3 (Constraint) | 18 |
| L4 (Decision) | 16 |
| L5 (Advanced) | 7 |

### エリア分布

| エリア | 件数 |
|--------|------|
| shibuya | 11 |
| shinjuku | 10 |
| ikebukuro | 10 |
| tokyo | 15 |
| cross_area | 13 |

### データ形式

Alpaca 形式 → Qwen3 chat template に変換:
- system: C1 改善版システムプロンプト
- user: テストケースの質問文
- assistant: C2 モデルの高品質回答（`<think>` タグ除去済み）

---

## 3. 学習過程

### QLoRA パラメータ

| パラメータ | 値 |
|-----------|-----|
| LoRA rank (r) | 16 |
| LoRA alpha | 32 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Dropout | 0.05 |
| Epochs | 5 |
| Effective batch size | 4 (batch=1, accumulation=4) |
| Learning rate | 2e-4 |
| Scheduler | cosine |
| Warmup ratio | 0.1 |
| Max grad norm | 0.3 |
| Optimizer | paged_adamw_8bit |
| fp16 | True |
| Gradient checkpointing | True |

### Loss 推移

| Epoch | Train Loss | Eval Loss |
|-------|-----------|-----------|
| 1 | TBD | TBD |
| 2 | TBD | TBD |
| 3 | TBD | TBD |
| 4 | TBD | TBD |
| 5 | TBD | TBD |

### 過学習判定

- Best eval loss: **TBD**
- Final eval loss: **TBD**
- 判定: **TBD**

---

## 4. 3 システム比較結果

### 4.1 全体結果

| 指標 | A: RAG (C2) | B: FT-only | C: FT+RAG | B-A | C-A | C3目標 | 判定 |
|------|-------------|------------|------------|-----|-----|--------|------|
| Composite | 70.4 | TBD | TBD | TBD | TBD | 75+ | TBD |
| Reasoning | 3.07 | TBD | TBD | TBD | TBD | 3.5+ | TBD |
| Evidence | 3.85 | TBD | TBD | TBD | TBD | 4.0+ | TBD |
| Success% | 83.1% | TBD | TBD | TBD | TBD | 88%+ | TBD |
| Keyword Hit Rate | 96.8% | TBD | TBD | TBD | TBD | — | — |
| Avg Time (s) | 50.1 | TBD | TBD | TBD | TBD | — | — |

### 4.2 レベル別結果

| Level | A: RAG (C2) | B: FT-only | C: FT+RAG | C-A | Best |
|-------|-------------|------------|------------|-----|------|
| L1 Basic | 64.4 | TBD | TBD | TBD | TBD |
| L2 Spatial | 64.2 | TBD | TBD | TBD | TBD |
| L3 Constraint | 84.8 | TBD | TBD | TBD | TBD |
| L4 Decision | 69.1 | TBD | TBD | TBD | TBD |
| L5 Advanced | 68.1 | TBD | TBD | TBD | TBD |

### 4.3 サブカテゴリ別結果

| Subcategory | N | A: RAG | B: FT | C: FT+RAG | C-A | Best |
|-------------|---|--------|-------|-----------|-----|------|
| basic_location | 8 | TBD | TBD | TBD | TBD | TBD |
| brand | 8 | TBD | TBD | TBD | TBD | TBD |
| area_detection | 19 | TBD | TBD | TBD | TBD | TBD |
| proximity | 4 | TBD | TBD | TBD | TBD | TBD |
| aggregation | 4 | TBD | TBD | TBD | TBD | TBD |
| comparison | 4 | TBD | TBD | TBD | TBD | TBD |
| landmark_origin | 27 | TBD | TBD | TBD | TBD | TBD |
| constraint_single | 4 | TBD | TBD | TBD | TBD | TBD |
| constraint_multi | 4 | TBD | TBD | TBD | TBD | TBD |
| decision_support | 4 | TBD | TBD | TBD | TBD | TBD |
| relation | 4 | TBD | TBD | TBD | TBD | TBD |
| sensitivity | 8 | TBD | TBD | TBD | TBD | TBD |
| multi_hop | 4 | TBD | TBD | TBD | TBD | TBD |
| competitor | 4 | TBD | TBD | TBD | TBD | TBD |
| complementary | 4 | TBD | TBD | TBD | TBD | TBD |
| cross_area_comparison | 20 | TBD | TBD | TBD | TBD | TBD |

---

## 5. 寄与分離分析

### 5.1 寄与分解

| 指標 | RAG効果 (A-B) | FT効果 (B-A) | 相乗効果 (C-max(A,B)) | 合計 (C-min(A,B)) |
|------|---------------|-------------|----------------------|-------------------|
| Composite | TBD | TBD | TBD | TBD |
| Reasoning | TBD | TBD | TBD | TBD |
| Evidence | TBD | TBD | TBD | TBD |
| Success% | TBD | TBD | TBD | TBD |

### 5.2 解釈

- **RAG効果 > 0**: RAG は FT 単体を超える価値を提供
- **FT効果 > 0**: FT は RAG 単体を超える価値を提供
- **相乗効果 > 0**: FT+RAG は個々を超える相補的効果あり
- **相乗効果 < 0**: FT と RAG の間に一部冗長性あり

TBD: 実験結果に基づく解釈を記載

---

## 6. C0→C3 累積改善分析

### 6.1 ステップ別改善量

| Step | 手法 | Composite | Reasoning | Evidence | Success% |
|------|------|-----------|-----------|----------|----------|
| C0 (Baseline) | 現行プロンプト + Qwen2.5-7B | 52.2 | 1.43 | 2.34 | 33.9% |
| C1 (Step 1) | 改善プロンプト + Qwen2.5-7B | 67.1 (+14.9) | 2.65 (+1.22) | 3.71 (+1.37) | 76.2% (+42.3%) |
| C2 (Step 2) | 改善プロンプト + Qwen3-32B | 70.4 (+3.3) | 3.07 (+0.42) | 3.85 (+0.13) | 83.1% (+6.9%) |
| C3 (Step 3) | QLoRA FT + Qwen3-32B + RAG | TBD | TBD | TBD | TBD |

### 6.2 総改善量 (C0→C3)

| 指標 | C0 | C3 | Delta | 改善率 |
|------|----|----|-------|--------|
| Composite | 52.2 | TBD | TBD | TBD |
| Reasoning | 1.43 | TBD | TBD | TBD |
| Evidence | 2.34 | TBD | TBD | TBD |
| Success% | 33.9% | TBD | TBD | TBD |

### 6.3 ステップ別寄与率

| Step | Composite寄与 | 寄与率 |
|------|---------------|--------|
| Step 1 (プロンプト改善) | +14.9 | TBD% |
| Step 2 (モデル変更) | +3.3 | TBD% |
| Step 3 (QLoRA FT) | TBD | TBD% |
| **合計** | **TBD** | **100%** |

---

## 7. 過学習分析

### 7.1 学習データ vs 非学習データのスコア比較

| System | FT-data (59件) | Non-FT (71件) | Gap |
|--------|---------------|---------------|-----|
| A: RAG (C2) | TBD | TBD | TBD |
| B: FT-only | TBD | TBD | TBD |
| C: FT+RAG | TBD | TBD | TBD |

### 7.2 判定基準

| Gap | 判定 |
|-----|------|
| <= 5pt | 過学習なし |
| 5-10pt | 軽度の過学習 |
| > 10pt | 重度の過学習 |

### 7.3 結論

TBD: 実験結果に基づく過学習判定を記載

---

## 8. C2 弱点改善分析

### C2 で 50-70 点だったケースの改善推移

| System | 改善件数 | 改善率 | 平均Delta |
|--------|---------|--------|-----------|
| B: FT-only | TBD | TBD | TBD |
| C: FT+RAG | TBD | TBD | TBD |

### C2 弱点サブカテゴリの改善

C2 で劣化した 3 サブカテゴリ（sensitivity, decision_support, relation）の改善度:

| Subcategory | C2 | B: FT | C: FT+RAG | 改善 |
|-------------|-----|-------|-----------|------|
| sensitivity | 62.5 | TBD | TBD | TBD |
| decision_support | 72.0 | TBD | TBD | TBD |
| relation | 59.8 | TBD | TBD | TBD |

---

## 9. 結論と次ステップ

### 9.1 結論

TBD: 実験結果に基づく結論を記載

- C3 目標達成: TBD
- FT と RAG の相補性: TBD
- 過学習の有無: TBD
- 最も効果的なシステム構成: TBD

### 9.2 Phase 9-C 総括

| Step | 手法 | Composite改善 | 累積 |
|------|------|---------------|------|
| Step 1 | プロンプト改善 | +14.9 | 67.1 |
| Step 2 | モデル変更 (32B) | +3.3 | 70.4 |
| Step 3 | QLoRA FT | TBD | TBD |

### 9.3 次ステップへの提言

TBD: 実験結果に基づく次ステップの提言

候補:
1. **Phase 10 への移行**: PostGIS/Supabase による全国展開
2. **追加の FT データ**: C3 結果から再帰的にデータ拡充
3. **パラメータ最適化**: temperature, top_p, top_k の最適化
4. **別モデルの検討**: Qwen3 の他のサイズや別モデルファミリー

---

## 付録

### A. ファイル一覧

| ファイル | 内容 |
|---------|------|
| `data/phase9c_training_data.json` | 学習データ（59 件、Alpaca 形式） |
| `notebooks/phase9c_step3_qlora_training.ipynb` | QLoRA 学習ノートブック |
| `notebooks/phase9c_step3_evaluation.ipynb` | 3 システム比較評価ノートブック |
| `results/phase9c_step3_*.json` | 評価結果 |
| `results/phase9c_step3_training_metadata.json` | 学習メタデータ |
| `results/phase9c_step3_*.png` | 可視化グラフ |

### B. 実験の再現手順

1. `notebooks/phase9c_step3_qlora_training.ipynb` を Colab Pro A100 で実行
2. アダプターが Google Drive に保存されることを確認
3. `notebooks/phase9c_step3_evaluation.ipynb` を Colab Pro A100 で実行
4. 結果 JSON が `results/` に保存されることを確認
5. 本レポートの TBD 箇所を実績値で更新
