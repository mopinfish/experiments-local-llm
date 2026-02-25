# Phase 9: Agentic RAG実験 引き継ぎドキュメント

**作成日**: 2026年2月3日
**プロジェクト**: experiments-local-llm
**ブランチ**: main
**関連Issue**: #6 (Phase 9: Agentic RAG実験の実装)

---

## 1. 実験概要

### 1.1 目的

**Agentic RAG**が、静的なルールベース選択（Adaptive RAG）や単一システム（StructuredRAG/GraphRAG）と比較して、複雑な地理空間クエリにおいてどのような優位性を持つかを検証する。

### 1.2 研究課題

Agentic RAGは、複数ステップの空間推論タスク（L4-L5）や条件付き推論において、従来のシステムを上回るか？

### 1.3 仮説

| 仮説ID | 仮説内容 |
|--------|---------|
| H1 | Agentic RAGは複数ステップの空間推論タスク（L4-L5）でStructuredRAGを上回る |
| H2 | エージェントによるツール選択は、ルールベースのAdaptive RAGより精度が高い |
| H3 | Self-correctionメカニズムにより、曖昧な質問への対応力が向上する |
| H4 | 処理時間はStructuredRAGより増加するが、精度向上で相殺される |

---

## 2. 実装成果物

### 2.1 新規ファイル

| ファイル | 役割 | 行数 |
|---------|------|------|
| `src/agent_tools.py` | 16個のツール定義 | ~850 |
| `src/agent_state.py` | LangGraph状態管理 | ~250 |
| `src/agent_prompts.py` | プロンプトテンプレート | ~350 |
| `src/agentic_rag_system.py` | エージェントシステム本体 | ~450 |
| `src/test_cases_agentic.py` | Agentic RAG向けテストケース（15件） | ~550 |
| `notebooks/phase9_agentic_rag_evaluation.ipynb` | Google Colab評価Notebook | - |
| `evaluate_agentic_rag.py` | ローカル評価スクリプト | ~350 |
| `test_tools_functionality.py` | ツール単体テスト | ~200 |

### 2.2 更新ファイル

| ファイル | 変更内容 |
|---------|---------|
| `pyproject.toml` | langgraph依存関係追加 |

---

## 3. システムアーキテクチャ

### 3.1 Agentic RAGシステム構成

```
┌─────────────────────────────────────────────────────────┐
│             Agentic RAG System (LangGraph)              │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌────────────────────────────────────────────────┐   │
│  │          Agent Node (LLM with Tools)           │   │
│  │  • 質問分析・プランニング                      │   │
│  │  • ツール選択・実行                            │   │
│  │  • 結果統合・回答生成                          │   │
│  └──────────────────┬─────────────────────────────┘   │
│         ┌───────────┴───────────┐                      │
│         ▼                       ▼                      │
│  ┌─────────────┐         ┌─────────────┐              │
│  │  Tool Node  │         │Finalize Node│              │
│  │  (実行)     │         │ (回答生成)  │              │
│  └─────────────┘         └─────────────┘              │
│         │                                               │
│         ▼                                               │
│  ┌─────────────────────────────────────────────┐      │
│  │          16 Tools (agent_tools.py)          │      │
│  ├─────────────────────────────────────────────┤      │
│  │ 空間計算 (6):                                │      │
│  │  • tool_get_nearest_pois                    │      │
│  │  • tool_count_pois_in_radius                │      │
│  │  • tool_compare_radius                      │      │
│  │  • tool_analyze_sensitivity                 │      │
│  │  • tool_filter_by_area                      │      │
│  │  • tool_calculate_distance                  │      │
│  │                                               │      │
│  │ 比較・集計 (5):                              │      │
│  │  • tool_compare_east_west                   │      │
│  │  • tool_compare_north_south                 │      │
│  │  • tool_count_by_category                   │      │
│  │  • tool_get_top_categories                  │      │
│  │  • tool_analyze_category_by_direction       │      │
│  │                                               │      │
│  │ 検索 (2):                                    │      │
│  │  • tool_vector_search                       │      │
│  │  • tool_find_pois_by_keyword                │      │
│  │                                               │      │
│  │ グラフトラバーサル (3):                      │      │
│  │  • tool_find_nearby_similar_pois            │      │
│  │  • tool_find_complementary_pois             │      │
│  │  • tool_find_pois_in_same_area              │      │
│  └─────────────────────────────────────────────┘      │
│                                                          │
└─────────────────────────────────────────────────────────┘
              ▼
    POI Data: 1,047件（渋谷エリア）
```

### 3.2 状態管理（LangGraph）

```python
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    question: str
    intermediate_steps: List[Dict[str, Any]]
    tool_results: List[Dict[str, Any]]
    answer: Optional[str]
    iteration: int
    next_action: str  # "continue" | "finish"
    error: Optional[str]
```

### 3.3 エージェントループ

```
User Question
    ↓
Agent Node (LLM)
    ├─ ツール呼び出しあり → Tool Node → Agent Node (繰り返し)
    └─ ツール呼び出しなし/最大イテレーション → Finalize Node
                                                   ↓
                                              Final Answer
```

---

## 4. テストケース

### 4.1 テストケース構成

| テストソース | 件数 | 説明 |
|-------------|------|------|
| Structured Tests (Phase 5-6) | 55件 | L1-L5の階層化テストケース |
| GraphRAG Tests (Phase 8) | 35件 | brand, complementary, competitor等 |
| **Agentic Tests (Phase 9)** ★ | **15件** | **複数ステップ推論向け** |
| **合計** | **105件** | |

### 4.2 Agentic RAG向けテストケース詳細

#### Category 1: multi_step_spatial（複数ステップの空間推論）- 5件

| ID | 質問例 | 期待ツール |
|----|--------|-----------|
| A-01 | 「最も近いカフェから300m以内の他のカフェは何件？」 | get_nearest_pois, count_pois_in_radius |
| A-02 | 「東側と西側で最も近いコンビニ、どちらが駅に近い？」 | get_nearest_pois, compare_east_west |
| A-03 | 「500m→700mに広げると何件増える？」 | count_pois_in_radius, compare_radius |
| A-04 | 「カフェ上位3カテゴリのうち東側に多いのは？」 | get_top_categories, compare_east_west |
| A-05 | 「300m以内のレストランで最も東にあるのは？」 | filter_by_area, get_nearest_pois |

#### Category 2: conditional_reasoning（条件付き推論）- 5件

| ID | 質問例 | 期待動作 |
|----|--------|---------|
| A-06 | 「50件未満なら『少ない』、50件以上なら『多い』」 | 条件分岐判定 |
| A-07 | 「多い方が50件以上なら『人気エリア』」 | 比較→条件判定 |
| A-08 | 「1.5倍以上増えるなら『集中度が低い』」 | 増加率→条件判定 |
| A-09 | 「100m以内なら『駅直結』、100-300mなら『駅近』」 | 距離→多段階分類 |
| A-10 | 「多い方の東西分布を教えて」 | 動的な分析対象選択 |

#### Category 3: iterative_refinement（反復的絞り込み）- 5件

| ID | 質問例 | 期待動作 |
|----|--------|---------|
| A-11 | 「カフェが最も多い半径を見つけて」 | 複数半径の試行 |
| A-12 | 「東>西なら終了、西>東なら西側分析」 | 条件付き追加分析 |
| A-13 | 「20件未満なら500mで再検索」 | 動的半径調整 |
| A-14 | 「レストランが多い方向で最寄り3件」 | 方向分析→最寄り検索 |
| A-15 | 「東西が同程度なら南北も比較」 | 段階的多軸分析 |

---

## 5. 動作確認結果

### 5.1 ツール単体テスト結果

```
============================================================
Agentic RAG Tools Functionality Test
============================================================

空間計算ツール:
  ✓ 最寄りカフェ: Urth Caffé (56.72m)
  ✓ 500m圏内: 86件のカフェ
  ✓ 半径比較: 300m→500mで49件増加 (2.32倍)

集計ツール:
  ✓ 東西比較: 西側が7件多い (東51件, 西58件)
  ✓ トップカテゴリ: レストラン367件, カフェ149件

グラフトラバーサルツール:
  ✓ 競合店検索: スターバックス周辺に13店舗のカフェ
  ✓ 相補的POI: 東宝シネマ周辺に42店舗のカフェ
  ✓ 同エリア検索: east_nearエリアに102店舗の飲食店

✓ All tool tests completed successfully
```

### 5.2 システム制約

**⚠ 重要**: ローカル環境ではOllamaがインストールされていないため、LLMを使用した完全な評価は未実行。
- ツール単体: ✓ 動作確認済み
- エージェントループ: Google Colab環境での評価が必要

---

## 6. Google Colab評価手順

### 6.1 事前準備

1. **Notebookのアップロード**
   - `notebooks/phase9_agentic_rag_evaluation.ipynb`をGoogle Colabにアップロード

2. **GitHubリポジトリ**
   - リポジトリをpublicに設定するか、認証情報を準備

3. **ランタイム設定**
   - GPU: T4（Phase 8と同様）
   - ランタイムタイプ: Python 3

### 6.2 実行手順

#### Step 1: 環境セットアップ（セル1-3）
```python
# Ollamaインストール・起動
# モデルダウンロード（qwen2.5:7b-instruct）
# 依存関係インストール（langgraph等）
```

#### Step 2: データ読み込み（セル4-5）
```python
# POIデータ読み込み（1,047件）
# テストケース読み込み（105件）
```

#### Step 3: システム初期化（セル6）
```python
# Agentic RAGシステム初期化
# Structured RAGシステム初期化（比較用）
```

#### Step 4: 評価実行（セル7-8）
```python
# クイックテスト: quick_test = True（10ケース、約5-10分）
# フル評価: quick_test = False（105ケース、約30-60分）
```

#### Step 5: 結果分析（セル9-12）
```python
# 全体スコア計算
# カテゴリ別分析
# 可視化
# ツール使用統計
```

#### Step 6: 結果保存（セル13）
```python
# JSONファイル保存
# サマリーレポート保存
# グラフ画像保存
```

### 6.3 期待される評価時間

| モード | テストケース数 | 予想時間 |
|--------|--------------|----------|
| クイックテスト | 10件 | 5-10分 |
| フル評価 | 105件 | 30-60分 |

### 6.4 出力ファイル

| ファイル | 内容 |
|---------|------|
| `agentic_rag_evaluation_YYYYMMDD_HHMMSS.json` | 詳細評価データ |
| `agentic_rag_summary_YYYYMMDD_HHMMSS.txt` | サマリーレポート |
| `agentic_rag_overall_comparison.png` | 全体スコア比較グラフ |
| `agentic_rag_category_comparison.png` | カテゴリ別比較グラフ |
| `agentic_rag_tool_usage.png` | ツール使用統計グラフ |

---

## 7. ベースラインとの比較

### 7.1 Phase 8までの結果（90テストケース）

| システム | スコア | 処理時間 |
|---------|--------|----------|
| **StructuredRAG** | **89.1%** | 20.6秒 |
| Adaptive RAG | 86.1% | 17.8秒 |
| GraphRAG | 76.7% | 8.7秒 |

### 7.2 Phase 9の目標

| 指標 | ベースライン | 目標 |
|-----|------------|------|
| 全体スコア | 89.1% (StructuredRAG) | **92%以上** |
| L4-L5スコア | 約90% | **95%以上** |
| 平均処理時間 | 20.6秒 | 30秒以内 |

### 7.3 評価軸

1. **精度**: キーワードヒット率、成功率
2. **ツール選択**: 適切なツールを選択できているか
3. **処理時間**: StructuredRAGとの比較
4. **カテゴリ別性能**: 特に`multi_step_spatial`での優位性

---

## 8. 技術的詳細

### 8.1 主要な設計判断

#### 1. LangGraphの採用理由
- ✓ 状態遷移の可視化が容易
- ✓ サイクル（反復処理）への対応
- ✓ ツール実行の統合サポート
- × LangChainよりも複雑な学習曲線

#### 2. ツール設計原則
- **粒度**: 1ツール = 1つの明確な機能
- **冗長性**: 類似ツールも意図的に提供（エージェントの選択肢を増やす）
- **エラー処理**: JSONフォーマットで一貫したエラー返却

#### 3. プロンプト設計
```python
AGENT_SYSTEM_PROMPT = """
あなたは渋谷駅周辺の地理空間POI情報に精通したエージェントです。

# 実行ガイドライン
1. 段階的思考: 一度に全てを実行せず、段階的にツールを実行
2. 適切なツール選択: 質問のタイプに応じて最適なツールを選択
3. パラメータ検証: ツールを呼び出す前にパラメータが適切か確認
4. 結果検証: ツール実行結果が期待通りか確認
5. エラーハンドリング: エラーが発生した場合は代替手段を試みる
"""
```

### 8.2 制限事項

#### 実装済み機能
- ✓ 16個のツール
- ✓ LangGraphベースのエージェントループ
- ✓ ツール実行と結果統合
- ✓ 最大10イテレーションの制御

#### 未実装機能
- ☐ Self-correction機構（検証→再実行）
- ☐ ベクトル検索のChromaDB連携
- ☐ グラフRAGとの完全統合

#### 技術的制約
- 渋谷駅固定（ハードコーディング）
- メモリベースのPOI管理（スケーラビリティ制限）
- LLMの推論能力に依存（GPT-4クラス推奨）

---

## 9. 今後の改善方向

### 9.1 Phase 9.3候補タスク

| タスク | 優先度 | 期待効果 |
|--------|--------|----------|
| Self-correction実装 | ★★★ | 精度向上（誤検索の修正） |
| ベクトル検索統合 | ★★☆ | 曖昧な質問への対応力向上 |
| プロンプト最適化 | ★★☆ | ツール選択精度の向上 |
| ツール追加（時系列分析等） | ★☆☆ | 新タスクへの対応 |

### 9.2 Phase 10以降の展望

| フェーズ | 内容 |
|---------|------|
| Phase 10 | 全国展開（PostGIS/Supabase、動的基準点解決） |
| Phase 11 | MCP統合（MapFan MCPサーバー） |
| Phase 12 | プロダクション化（API化、スケーラビリティ） |

---

## 10. 参考リソース

### 10.1 Issue・PR

- **Issue #6**: Phase 9 Agentic RAG実験の実装

### 10.2 関連ドキュメント

- `docs/handovers/HANDOVER_SESSION_20260130.md`: Phase 8までの進捗
- `docs/handovers/HANDOVER_GRAPHRAG_EXPERIMENT_FINAL.md`: GraphRAG実験結果
- `CLAUDE.md`: プロジェクト全体のガイダンス

### 10.3 主要論文・リソース

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)
- [Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection](https://arxiv.org/abs/2310.11511)

---

## 11. 次回セッションへの引き継ぎ

### 11.1 完了済みタスク

- [x] Phase 9.1: 基盤実装（ツール、状態管理、エージェントシステム）
- [x] Phase 9.2: 機能強化（POIローディング修正、テストケース作成、グラフツール追加）
- [x] Google Colab評価Notebook作成
- [x] 引き継ぎドキュメント作成

### 11.2 保留タスク

- [ ] Google Colabでの完全評価実行
- [ ] 評価結果の分析とレポート作成
- [ ] Self-correction機構の実装
- [ ] ベクトル検索のChromaDB統合

### 11.3 意思決定事項

1. **評価実行**: Google Colab環境での評価を実施するか
2. **Phase 9.3継続**: さらなる機能強化を行うか
3. **Phase 10移行**: 全国展開に着手するか

---

**作成者**: Claude Sonnet 4.5
**実装期間**: 2026年2月3日
**実装状況**: Phase 9.1-9.2完了、Phase 9.3（評価）準備完了
**次のステップ**: Google Colabでの評価実行
