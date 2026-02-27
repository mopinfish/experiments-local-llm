# Phase 9-C: 回答品質改善実験計画（プロンプト/モデル/FT）

**作成日**: 2026-02-25
**プロジェクト**: experiments-local-llm
**ステータス**: Step 1/2 完了、Step 3 未着手
**最終更新**: 2026-02-27

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
| **C1** | Step 1 | **改善版** | Qwen2.5-7B 4bit | なし | **実績: composite 67.1 (+14.9)** |
| **C2** | Step 2 | 改善版 | **Qwen3-32B 4bit** | なし | **実績: composite 70.4 (+3.3)** |
| **C3** | Step 3 | 改善版 | Qwen3-32B 4bit | **QLoRA** | 未実施 |

各StepでC0からの差分のみを変更し、改善効果を分離測定する。

### 1.3 使用環境

- **LLM**: Qwen2.5-7B-Instruct 4bit（C0/C1）→ **Qwen3-32B** 4bit NF4（C2/C3）
- **埋め込みモデル**: multilingual-e5-base（全Step共通、変更なし）
- **実行環境**: Google Colab T4 GPU（C0/C1）→ **Google Colab Pro A100 GPU**（C2以降）
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
C1 (目標: composite 60+ → ★実績: 67.1 (+14.9))  ✅ 完了
  │
  ▼ Step 2: モデル変更 (Qwen3-32B)
C2 (目標: composite 70+ → ★実績: 70.4 (+3.3))   ✅ 完了
  │
  ▼ Step 3: QLoRAファインチューニング
C3 (目標: composite 75+?, reasoning 3.5+?)        ⬜ 未着手
```

各Stepの評価は**Hybrid RAG×130件**で統一する（Phase 9-Bで最もバランスが良かったシステム）。

ただしStep 1のみ、プロンプト改善の汎用性を確認するため**4システム×130件=520クエリ**の全量評価も実施した。

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

### 3.2 Step 2: LLMモデル変更 ✅ 完了

#### 最終選定モデル

| モデル | パラメータ | 量子化 | 実測VRAM | GPU | 結果 |
|--------|----------|--------|---------|------|------|
| **Qwen3-32B** | 32B | NF4 4bit | ~18-20GB | **A100 40GB** | **composite 70.4** |

当初計画ではQwen2.5-14Bを候補としていたが、Qwen3ファミリー(2025年4月リリース)の登場により、より大きなモデルでの評価に変更。

#### 実行経緯

1. Qwen3-32B-AWQ → transformers互換性問題でbitsandbytes NF4に変更
2. Qwen3-32B NF4 on L4 (22GB) → OOM
3. Qwen3-14B NF4 on L4 → OOM
4. Colab Proアップグレード → A100 (40GB) で Qwen3-32B NF4 動作確認
5. 初回実行: L4割り当て → 14Bフォールバック (composite 65.7)
6. GPU検証強化 (VRAM < 30GBでエラー停止) → A100明示選択で32B動作 → **composite 70.4**

#### Qwen3固有の対応

- **enable_thinking=False**: tokenizer.apply_chat_templateにmonkey-patchで非思考モード強制
- **`<think>`タグ除去**: テンプレート出力とモデル出力の両方でテキストレベル除去
- **生成パラメータ**: Qwen3推奨値 (temperature=0.7, top_p=0.8, top_k=20)

### 3.3 Step 3: QLoRAファインチューニング ⬜ 未着手

> **注**: Step 2でC2目標を全達成したため、Step 3の必要性・方針は再検討が必要。
> C2結果の分析から、sensitivity/decision_support劣化やtemperatureチューニング等、
> FTなしで改善可能な施策が複数ある。Step 3実施前にこれらを優先検討すべき。

#### 学習データ作成

Step 2の最良モデルの出力から、高品質な回答を人手で選定・修正して学習データを作成する。

| 項目 | 値 |
|------|-----|
| ベースモデル | **Qwen3-32B** (Step 2で確定) |
| 学習データ件数 | 50-80件 |
| データソース | Step 2の130件出力から選定 |
| フォーマット | instruction-input-output形式 |
| 品質基準 | composite_score 70+かつreasoning_score 3+の回答を選定・修正 |
| 実行環境 | **Colab Pro A100** (32Bモデルのため) |

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
| 推定学習時間 | 60-120分 | A100 GPU (32Bモデル) |

---

## 4. 成功基準

### 4.1 Step別目標と実績

| 指標 | C0 (現状) | C1目標 | **C1実績** | C2目標 | **C2実績** | C3目標 |
|------|----------|--------|-----------|--------|-----------|--------|
| composite_score | 52.2 | 60+ | **67.1** ✅ | 70+ | **70.4** ✅ | 75+? |
| reasoning_score | 1.43 | 2.0+ | **2.65** ✅ | 3.0+ | **3.07** ✅ | 3.5+? |
| evidence_score | 2.34 | 2.8+ | **3.71** ✅ | 3.5+ | **3.85** ✅ | 4.0+? |
| composite_success% | 33.85% | 45%+ | **76.2%** ✅ | 80%+ | **83.1%** ✅ | 85%+? |
| success_rate | 96.15% | 95%+ | **96.9%** ✅ | 95%+ | **100%** ✅ | 95%+ |

> **C3目標値は暫定**。Step 3の実施判断と合わせて再設定が必要。

### 4.2 最終成功基準（Phase 9-C全体）— **Step 2時点で達成済み**

- ✅ **composite_score 70以上** — C2実績: 70.4（C0: 52.2 → **+18.2pt改善**）
- ✅ **composite_success_rate 60%以上** — C2実績: 83.1%（C0: 33.85% → **+49.2pt改善**）
- ✅ **reasoning_score 3.0以上** — C2実績: 3.07（C0: 1.43 → **+1.64pt改善**）
- ✅ **keyword_success_rateの維持** — C2実績: 100%（維持以上）

---

## 5. リスク管理

| リスク | 影響度 | 対策 | 結果 |
|--------|-------|------|------|
| T4 VRAM不足で14B 8bitが動かない | 中 | 14B 4bitにフォールバック | **発生**: 32BはT4/L4不可。Colab Pro A100で解決 |
| プロンプト改善がreasoning向上に寄与しない | 中 | Few-shot例の追加を検討 | **未発生**: C1で+1.22pt大幅改善 |
| QLoRA学習データの品質確保が困難 | 中 | Step 2で高品質出力が少ない場合、人手修正で補完 | Step 3未着手 |
| QLoRAで汎化性能が低下（過学習） | 中 | validationセットで監視、早期停止 | Step 3未着手 |
| Colabセッション切断による実験中断 | 低 | チェックポイント保存を頻繁に実施 | **軽微**: チェックポイント機構で対応済み |
| 評価パイプライン自体のバイアス | 低 | evaluator_multi_areaは変更せず | **未発生**: 全Step同一条件で比較 |
| **Colab GPUガチャ** (新規) | 高 | VRAM < 30GBでエラー停止 | **発生**: L4割当→14Bフォールバック。A100明示選択で解決 |
| **Qwen3 thinkingモード混入** (新規) | 中 | monkey-patch + テキスト除去 | **発生・解決**: `<think>`タグ除去で対応 |

---

## 6. 成果物・スケジュール

### 6.1 成果物一覧

| # | 成果物 | 形式 | 状態 |
|---|--------|------|------|
| 1 | 本計画書 | `docs/plans/PHASE9C_QUALITY_IMPROVEMENT_PLAN.md` | ✅ 完了 |
| 2 | Step 1 改善プロンプト | `src/` 各ファイルの修正 | ✅ 完了 |
| 3 | Step 1 評価ノートブック | `notebooks/phase9c_step1_prompt_evaluation.ipynb` | ✅ 完了 |
| 4 | Step 1 評価結果 | `results/phase9c_step1_20260226_022240.json` | ✅ 完了 |
| 5 | Step 1 レポート | `docs/reports/PHASE9C_STEP1_PROMPT_IMPROVEMENT_REPORT.md` | ✅ 完了 |
| 6 | Step 2 モデル評価ノートブック | `notebooks/phase9c_step2_model_evaluation.ipynb` | ✅ 完了 |
| 7 | Step 2 評価結果 | `results/phase9c_step2_20260227_071313.json` | ✅ 完了 |
| 8 | Step 2 レポート | `docs/reports/PHASE9C_STEP2_MODEL_UPGRADE_REPORT.md` | ✅ 完了 |
| 9 | 引き継ぎ資料 | `docs/handovers/HANDOVER_PHASE9C.md` | ✅ 完了 |
| 10 | Step 3 FTノートブック | `notebooks/phase9c_step3_qlora_training.ipynb` | ⬜ 未着手 |
| 11 | Step 3 学習データ | `data/phase9c_training_data.json` | ⬜ 未着手 |
| 12 | Step 3 評価結果 | `results/phase9c_step3_*.json` | ⬜ 未着手 |

### 6.2 実行順序

```
Step 1: プロンプト改善実験                          ✅ 完了 (2026-02-25〜26)
  ├── プロンプト修正・レビュー
  ├── 4システム×130件評価（520クエリ）
  └── 結果分析・C0との比較
      │  結果: composite 52.2 → 67.1 (+14.9)
      ▼
Step 2: モデル変更実験（Step 1完了後）              ✅ 完了 (2026-02-26〜27)
  ├── VRAM検証 → A100必須が判明
  ├── Colab Proアップグレード
  ├── 本評価（Hybrid RAG×130件、Qwen3-32B）
  └── 結果分析・C1との比較
      │  結果: composite 67.1 → 70.4 (+3.3)
      ▼
Step 3: QLoRAファインチューニング（Step 2完了後）   ⬜ 未着手
  ├── 学習データ作成（50-80件）
  ├── QLoRA学習
  ├── 本評価（Hybrid RAG×130件）
  └── 最終レポート作成
```
