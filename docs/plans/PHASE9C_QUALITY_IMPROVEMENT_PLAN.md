# Phase 9-C: 回答品質改善実験計画（プロンプト/モデル/FT）

**作成日**: 2026-02-25
**プロジェクト**: experiments-local-llm
**ステータス**: 計画策定中

---

## 目次

- [1. 実験概要](#1-実験概要)
- [2. 実験設計](#2-実験設計)
- [3. 実装詳細](#3-実装詳細)
- [4. 成功基準](#4-成功基準)
- [5. リスク管理](#5-リスク管理)
- [6. 成果物・スケジュール](#6-成果物スケジュール)

---

## 1. 実験概要

### 1.1 背景と目的

Phase 9-Bで4エリア（渋谷/新宿/池袋/東京）×4システムの520クエリ評価を完了した。結果、**キーワード成功率96.2%に対してcomposite成功率33.9%**という大きな乖離が判明した。

| 指標 | Phase 9-B結果 (Hybrid RAG) |
|------|---------------------------|
| success_rate (keyword) | 96.15% |
| avg_keyword_hit_rate | 89.86% |
| **avg_composite_score** | **52.2** |
| **avg_reasoning_score** | **1.43** |
| avg_evidence_score | 2.34 |
| **composite_success_rate** | **33.85%** |

#### 原因分析

1. **プロンプトの不足**: 現行のsystem_promptは「正確かつ簡潔に回答」とだけ指示しており、推論過程の明示・根拠引用・不確実性の明記を誘導していない
2. **Qwen2.5-7B 4bitの推論能力限界**: 4ビット量子化による精度劣化と7Bパラメータの推論能力上限

#### 目的

3つの変数（プロンプト/モデル/ファインチューニング）を段階的に変更し、**各変数の寄与を分離測定**する。最終的にcomposite_score 70+、composite成功率 60%+を目指す。

### 1.2 比較マトリクス

| ID | 構成 | プロンプト | モデル | FT | 期待効果 |
|----|------|-----------|--------|-----|---------|
| **C0** | ベースライン | 現行 | Qwen2.5-7B 4bit | なし | — (composite 52.2) |
| **C1** | Step 1 | **改善版** | Qwen2.5-7B 4bit | なし | reasoning向上 |
| **C2** | Step 2 | 改善版 | **上位モデル** | なし | 推論能力向上 |
| **C3** | Step 3 | 改善版 | 上位モデル | **QLoRA** | 最終到達点 |

各StepでC0からの差分のみを変更し、改善効果を分離測定する。

### 1.3 使用環境

- **LLM**: Qwen2.5-7B-Instruct 4bit（C0/C1）→ 上位モデル（C2/C3）
- **埋め込みモデル**: multilingual-e5-base（全Step共通、変更なし）
- **実行環境**: Google Colab T4 GPU（15GB VRAM）
- **ベクトルDB**: ChromaDB（全Step共通、変更なし）

---

## 2. 実験設計

### 2.1 テストケース

Phase 9-Bで使用した**130テストケース**をそのまま使用（4エリア×L1-L5難易度）。テストケースの変更は行わず、評価条件を統一する。

| エリア | テストケース数 |
|--------|-------------|
| shibuya | 26 |
| shinjuku | 27 |
| ikebukuro | 26 |
| tokyo | 26 |
| cross_area | 25 |
| **合計** | **130** |

### 2.2 評価指標

Phase 9-Bと同一の評価パイプライン（`src/evaluators_multi_area.py`）を使用。

| 指標 | 説明 | 重み |
|------|------|-----|
| keyword_hit_rate | キーワード一致率 | composite算出に使用 |
| reasoning_score | 推論の論理性・段階性（1-5） | composite算出に使用 |
| evidence_score | 根拠引用の具体性（1-5） | composite算出に使用 |
| composite_score | 上記3指標の加重平均（0-100） | **主要KPI** |
| composite_success_rate | composite_score ≥ 60の割合 | **主要KPI** |

### 2.3 実験ステップ

```
C0 (ベースライン: composite 52.2)
  │
  ▼ Step 1: プロンプト改善
C1 (目標: composite 60+, reasoning 2.0+)
  │
  ▼ Step 2: モデル変更
C2 (目標: composite 68+, reasoning 2.5+)
  │
  ▼ Step 3: QLoRAファインチューニング
C3 (目標: composite 70+, reasoning 3.0+, 成功率 60%+)
```

各Stepの評価は**Hybrid RAG×130件**で統一する（Phase 9-Bで最もバランスが良かったシステム）。

ただしStep 1のみ、プロンプト改善の汎用性を確認するため**4システム×130件=520クエリ**の全量評価も実施する。

---

## 3. 実装詳細

### 3.1 Step 1: プロンプト改善

#### 変更対象ファイル

| ファイル | 変更箇所 | 内容 |
|---------|---------|------|
| `src/structured_rag_system.py` | L335-349 system_prompt | 推論指示追加 |
| `src/structured_rag_system.py` | L905-913 ユーザープロンプト | 回答フォーマット指定 |
| `src/adaptive_rag_system.py` | L220-234 system_prompt | 同上（structured_ragと同期） |
| `src/agent_prompts.py` | L17-90 AGENT_SYSTEM_PROMPT | 回答形式の強化 |
| `src/agent_prompts.py` | L155-172 ANSWER_GENERATION_PROMPT_TEMPLATE | 推論・根拠テンプレート |

#### 改善方針

現行プロンプトの問題点と改善方向：

**1. system_prompt（structured_rag / adaptive_rag共通）**

```
# 現行（5行、抽象的な指示のみ）
あなたは...地理情報に詳しいアシスタントです。
提供された情報に基づいて、正確かつ簡潔に回答してください。
座標情報がある場合は必ず含めてください。
数値データがある場合は具体的な数字を使って回答してください。
情報がない場合は「情報がありません」と正直に回答してください。
```

改善方向:
- **推論過程の明示**: 「回答に至った推論の過程を示してください」
- **根拠引用の義務化**: 「提供情報のどの部分を根拠としたか明記してください」
- **不確実性の表現**: 「確信度が低い場合はその旨を明記してください」
- **回答構造の指定**: 結論→根拠→補足の構造を指示

**2. ユーザープロンプト（structured_rag）**

```
# 現行（回答開始文が固定で推論を抑制）
【回答】
上記の情報を基に、具体的な数値や場所名を含めて回答します。
```

改善方向:
- 回答開始文を削除（モデルの自由な推論を許可）
- 「まず根拠を示し、次に結論を述べてください」等の構造指示を追加

**3. ANSWER_GENERATION_PROMPT_TEMPLATE（agent系）**

改善方向:
- ツール結果からの情報抽出→推論→結論の段階的プロセスを指示
- 「どのツール結果をどう解釈したか」を明記させる

### 3.2 Step 2: LLMモデル変更

#### モデル候補とVRAM見積もり

| モデル | パラメータ | 量子化 | 推定VRAM | T4適合 | 期待効果 |
|--------|----------|--------|---------|--------|---------|
| Qwen2.5-7B | 7B | 4bit | ~5GB | ○ | ベースライン |
| Qwen2.5-7B | 7B | 8bit | ~8GB | ○ | 量子化劣化軽減 |
| Qwen2.5-14B | 14B | 4bit | ~9GB | ○ | パラメータ増加 |
| Qwen2.5-14B | 14B | 8bit | ~16GB | △（要検証） | 最大品質 |

#### 実行手順

1. **VRAM検証フェーズ**: 各モデルをロードし、実際のVRAM使用量を計測。T4 15GBに収まるか確認
2. **クイック評価**: VRAM収まるモデルで10件クイックテスト（応答品質・速度の概算）
3. **本評価**: 最良候補でHybrid RAG×130件の全量評価

#### モデルロードコード（参考）

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# 8bit量子化
quantization_config_8bit = BitsAndBytesConfig(load_in_8bit=True)

# 4bit量子化
quantization_config_4bit = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto",
)
```

### 3.3 Step 3: QLoRAファインチューニング

#### 学習データ作成

Step 2の最良モデルの出力から、高品質な回答を人手で選定・修正して学習データを作成する。

| 項目 | 値 |
|------|-----|
| 学習データ件数 | 50-80件 |
| データソース | Step 2の130件出力から選定 |
| フォーマット | instruction-input-output形式 |
| 品質基準 | composite_score 70+かつreasoning_score 3+の回答を選定・修正 |

#### QLoRAパラメータ（候補）

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,                      # LoRAランク
    lora_alpha=32,             # スケーリング係数
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
```

| パラメータ | 値 | 備考 |
|-----------|-----|------|
| エポック数 | 3-5 | 過学習監視で早期停止 |
| バッチサイズ | 4 | gradient_accumulation=4で実効16 |
| 学習率 | 2e-4 | cosineスケジューラ |
| max_seq_length | 2048 | コンテキスト+回答の最大長 |
| 推定学習時間 | 30-60分 | T4 GPU |

---

## 4. 成功基準

### 4.1 Step別目標

| 指標 | C0 (現状) | C1目標 (Step1) | C2目標 (Step2) | C3目標 (Step3) |
|------|----------|---------------|---------------|---------------|
| composite_score | 52.2 | 60+ | 68+ | **70+** |
| reasoning_score | 1.43 | 2.0+ | 2.5+ | **3.0+** |
| evidence_score | 2.34 | 2.8+ | 3.0+ | 3.5+ |
| composite_success_rate | 33.85% | 45%+ | 55%+ | **60%+** |
| keyword_success_rate | 96.15% | 95%+（維持） | 95%+（維持） | 95%+（維持） |

### 4.2 最終成功基準（Phase 9-C全体）

- **composite_score 70以上**（C0: 52.2 → +17.8pt改善）
- **composite_success_rate 60%以上**（C0: 33.85% → +26.15pt改善）
- **reasoning_score 3.0以上**（C0: 1.43 → +1.57pt改善）
- keyword_success_rateの維持（95%以上）

---

## 5. リスク管理

| リスク | 影響度 | 対策 |
|--------|-------|------|
| T4 VRAM不足で14B 8bitが動かない | 中 | 14B 4bitにフォールバック。Step 2のVRAM検証フェーズで早期確認 |
| プロンプト改善がreasoning向上に寄与しない | 中 | Few-shot例の追加を検討。プロンプトパターンを複数試行 |
| QLoRA学習データの品質確保が困難 | 中 | Step 2で高品質出力が少ない場合、人手修正で補完 |
| QLoRAで汎化性能が低下（過学習） | 中 | validationセット（20%）で監視、早期停止を適用 |
| Colabセッション切断による実験中断 | 低 | チェックポイント保存を頻繁に実施、結果はJSON逐次保存 |
| 評価パイプライン自体のバイアス | 低 | evaluator_multi_areaは変更せず、C0と同一条件で比較 |

---

## 6. 成果物・スケジュール

### 6.1 成果物一覧

| # | 成果物 | 形式 | 備考 |
|---|--------|------|------|
| 1 | 本計画書 | `docs/plans/PHASE9C_QUALITY_IMPROVEMENT_PLAN.md` | |
| 2 | Step 1 改善プロンプト | `src/` 各ファイルの修正 | |
| 3 | Step 1 評価ノートブック | `notebooks/phase9c_step1_prompt_evaluation.ipynb` | |
| 4 | Step 1 評価結果 | `results/phase9c_step1_*.json` | |
| 5 | Step 2 モデル比較ノートブック | `notebooks/phase9c_step2_model_evaluation.ipynb` | |
| 6 | Step 2 評価結果 | `results/phase9c_step2_*.json` | |
| 7 | Step 3 FTノートブック | `notebooks/phase9c_step3_qlora_training.ipynb` | |
| 8 | Step 3 学習データ | `data/phase9c_training_data.json` | |
| 9 | Step 3 評価結果 | `results/phase9c_step3_*.json` | |
| 10 | 最終レポート | `docs/reports/PHASE9C_QUALITY_IMPROVEMENT_REPORT.md` | |
| 11 | ハンドオーバー | `docs/handovers/PHASE9C_HANDOVER.md` | |

### 6.2 実行順序

```
Step 1: プロンプト改善実験
  ├── プロンプト修正・レビュー
  ├── 4システム×130件評価（520クエリ）
  └── 結果分析・C0との比較
      │
      ▼
Step 2: モデル変更実験（Step 1完了後）
  ├── VRAM検証（3モデル候補）
  ├── クイック評価（10件×候補数）
  ├── 本評価（Hybrid RAG×130件）
  └── 結果分析・C1との比較
      │
      ▼
Step 3: QLoRAファインチューニング（Step 2完了後）
  ├── 学習データ作成（50-80件）
  ├── QLoRA学習（30-60分）
  ├── 本評価（Hybrid RAG×130件）
  └── 最終レポート作成
```
