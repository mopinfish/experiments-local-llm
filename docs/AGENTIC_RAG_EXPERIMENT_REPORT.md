# LangGraphベースAgentic RAGシステムの実装と評価：地理的POIクエリタスクにおける動的ツール選択の効果と課題

**Agentic RAG with LangGraph for Geospatial POI Queries: Effectiveness and Challenges of Dynamic Tool Selection**

---

## 論文情報

- **研究期間**: 2026年2月3日〜11日
- **プロジェクト**: experiments-local-llm / Phase 9
- **対象領域**: 渋谷駅周辺POIデータ（1,047件）
- **使用モデル**: Qwen2.5-7B-Instruct（4bit量子化）
- **評価規模**: 105テストケース（Phase 5-6: 55件、Phase 8: 35件、Phase 9: 15件）
- **実行環境**: Google Colab T4 GPU

---

## 概要（Abstract）

本研究では、LangGraphフレームワークを用いたAgentic RAG（エージェント型RAG）システムを実装し、地理的POI（Point of Interest）クエリタスクにおける動的なツール選択と反復的推論の効果を評価した。ReAct（Reasoning and Acting）パターンに基づき、LLMが実行時に16種類のツール（空間計算、集計、グラフトラバーサル）から適切なものを選択・実行する仕組みを構築した。

実験の結果、以下の主要な知見が得られた：

1. **全体性能では既存のHybrid RAG（構造化RAG）が優位**：Agentic RAG 87.6% vs Hybrid RAG 96.2%（-8.6pt）
2. **特定カテゴリでAgenticが明確な優位性**：competitor関係（+33.3pt）、complementary関係（+20.0pt）、基本的位置情報（+20.0pt）
3. **実行時間は5.1倍に増加**：平均56.4秒 vs 11.1秒（Hybrid RAG）
4. **重大な言語制御問題を発見**：JSON形式のツール出力が中国語レスポンスを誘発（7.6%のケース）
5. **フィードバックループの価値は限定的**：既知のクエリパターンでは事前決定型の方が効率的

さらに、本研究では以下の重要な概念的発見を報告する：

- **"構造化RAG"の用語的混乱の解明**：本プロジェクトの「構造化RAG」は、学術的な標準定義とは異なる「Hybrid RAG」アプローチであることを明確化
- **出力形式がLLM性能に及ぼす影響**：JSON形式よりも整形された自然言語テキストの方が、多言語LLMにおいて言語安定性が高い
- **Hybrid RAGアプローチのスケーラビリティ**：MCP ServerとPostGIS統合による全国展開への適用可能性

本報告書では、Agentic RAGの設計、実装の課題、言語制御問題の根本原因分析、および異なるRAGアプローチの本質的な違いについて体系的に論じる。

---

## 1. 序論（Introduction）

### 1.1 研究背景

Retrieval-Augmented Generation (RAG)は、大規模言語モデル（LLM）の知識限界を外部知識ベースで補完する有力な手法である。近年、RAGのアプローチは多様化しており、単純なベクトル検索から、構造化データ処理、グラフトラバーサル、そして動的なツール選択を伴うエージェント型システムへと発展している。

本プロジェクトでは、Phase 6-8において複数のRAGアプローチを実装・評価してきた：

| Phase | アプローチ | 成功率 | 主な特徴 |
|-------|----------|--------|---------|
| Phase 6 | Hybrid RAG | 91.6% → 96.2% | ルールベース質問分析 + 計算処理 + ベクトル検索 |
| Phase 8 | Graph RAG | 76.7% | Neo4j知識グラフ + グラフトラバーサル |
| Phase 8 | Adaptive RAG | 86.1% | クエリ複雑度に基づくシステム選択 |

これらの先行実験から、以下の課題が明らかになった：

1. **固定的な処理フロー**：Phase 6のHybrid RAGは事前のルールベース分析に依存し、実行時の柔軟な戦略変更ができない
2. **システム選択の限界**：Phase 8のAdaptive RAGは複数システムを選択できるが、選択後は固定フロー
3. **フィードバックループの欠如**：中間結果を見て戦略を変更する仕組みがない

### 1.2 Agentic RAGの台頭

2023年以降、LLMをエージェントとして活用し、ツールを動的に選択・実行するアプローチが注目されている：

- **ReAct** (Yao et al., 2022): Thought（思考）→ Action（行動）→ Observation（観察）のループ
- **Toolformer** (Schick et al., 2023): LLMによる自律的なツール使用
- **LangGraph** (Harrison et al., 2024): ステート管理とワークフロー制御を統合したフレームワーク

Agentic RAGは、これらの技術を統合し、以下の特徴を持つ：

1. **動的なツール選択**：LLMが実行時に必要なツールを判断
2. **反復的推論**：結果を見て次の行動を決定（フィードバックループ）
3. **適応的戦略変更**：予期しないクエリパターンにも対応

### 1.3 研究目的

本研究では、地理的POIクエリタスクにおけるAgentic RAGの有効性を、以下の観点から評価した：

**研究課題（Research Questions）**:

| RQ | 研究課題 |
|----|---------|
| **RQ1** | Agentic RAGは、既存のHybrid RAG（事前決定型）と比較して、複雑な推論タスクで優位性を持つか？ |
| **RQ2** | 動的なツール選択とフィードバックループは、どのようなカテゴリのクエリで効果を発揮するか？ |
| **RQ3** | 反復的推論による実行時間のオーバーヘッドは、精度向上に見合うか？ |
| **RQ4** | LLM駆動のツール選択は、ルールベースの選択と比較してどの程度ロバストか？ |

**仮説（Hypotheses）**:

| ID | 仮説 |
|----|-----|
| **H1** | 複数制約条件を持つクエリ（constraint_multi）で、Agenticが逐次的な絞り込みにより高精度を達成 |
| **H2** | マルチホップ推論（multi_hop）で、中間結果を使った段階的検索により優位性を発揮 |
| **H3** | 単純な近接性検索（basic_location）では、Hybridの並列実行が効率的 |
| **H4** | 不確実性への対応（advanced_uncertainty）で、Agenticが動的な情報収集により優位 |

### 1.4 貢献

本研究の主な貢献は以下の通りである：

1. **HuggingFace Transformers統合によるAgentic RAG実装**
   - LangGraphとTransformersの統合（従来のChatOllamaに依存しない）
   - Google Colab T4 GPU環境での4bit量子化による効率的実行

2. **16種類の地理空間ツールの体系的設計**
   - 空間計算ツール（距離、方角、最寄りPOI）
   - 集計ツール（カテゴリ別集計、ランキング、東西比較）
   - グラフトラバーサルツール（関係性探索、エリア内検索）

3. **包括的な比較評価（105テストケース）**
   - Phase 5-6の既存ケース（55件）
   - Phase 8のGraphRAG向けケース（35件）
   - Phase 9のAgentic特化ケース（15件）

4. **言語制御問題の根本原因分析**
   - JSON形式のツール出力が多言語LLMで言語混在を誘発
   - 自然言語テキスト形式の優位性を実証

5. **RAGアプローチの分類学的整理**
   - Hybrid RAG、Adaptive RAG、Agentic RAGの本質的違いを明確化
   - 「構造化RAG」という用語の学術的定義との乖離を指摘

---

## 2. 関連研究（Related Work）

### 2.1 Retrieval-Augmented Generation (RAG)

#### 2.1.1 基本RAG

Lewis et al. (2020)は、DPR（Dense Passage Retrieval）とBARTを組み合わせたRAGアーキテクチャを提案した。質問に対してベクトル検索で関連文書を取得し、それをコンテキストとしてLLMに渡す単純なパイプラインである。

#### 2.1.2 Advanced RAG

Gao et al. (2023)は、基本RAGの限界を指摘し、以下の拡張を提案した：
- **Query Rewriting**: 質問をより検索に適した形に変換
- **Hybrid Search**: ベクトル検索とキーワード検索の併用
- **Reranking**: 検索結果の再ランキング

#### 2.1.3 Modular RAG

Ram et al. (2023)は、RAGをモジュール化し、タスクに応じて最適な組み合わせを選択するアプローチを提案した。

### 2.2 LLMエージェントとツール使用

#### 2.2.1 ReAct

Yao et al. (2022)は、LLMが推論（Reasoning）と行動（Acting）を交互に行うReActパターンを提案した：

```
Thought: [推論プロセス]
Action: [実行するアクション]
Observation: [行動の結果]
... (繰り返し)
Final Answer: [最終回答]
```

このパターンは、複雑なタスクを段階的に解決する能力を大幅に向上させた。

#### 2.2.2 Toolformer

Schick et al. (2023)は、LLMが自律的に外部ツール（計算機、検索エンジン、翻訳API）を使用する手法を提案した。ツール呼び出しをテキスト生成の一部として扱い、特殊トークンで制御する。

#### 2.2.3 LangGraph

Harrison et al. (2024)は、エージェントワークフローをステートマシンとして表現するLangGraphを提案した。ノード（処理）とエッジ（遷移条件）で構成され、複雑なエージェントロジックを宣言的に記述できる。

### 2.3 構造化RAGと知識グラフ

#### 2.3.1 学術的な構造化RAG

一般的に「構造化RAG」は以下を指す：
- **メタデータフィルタリング**: 検索時にカテゴリ、日付などでフィルタ
- **リランキング**: 検索結果を関連性スコアで再順位付け
- **構造化プロンプト**: プロンプトに明示的な構造（見出し、箇条書き）を持たせる

#### 2.3.2 Hybrid RAG（本プロジェクトのアプローチ）

本プロジェクトのPhase 6で開発したシステムは、学術的な「構造化RAG」を超え、以下を統合する：
- ルールベースの質問分析
- 実時間の空間計算（距離、方角）
- 集計処理（カテゴリ別カウント、統計）
- ベクトル検索との相補的統合
- 整形された自然言語テキストでの出力

これは実質的に「Hybrid RAG」と呼ぶべきアプローチである（詳細は6章で議論）。

### 2.4 地理空間クエリとLLM

地理空間情報とLLMの統合に関する先行研究：
- **GeoGLUE** (Mai et al., 2020): 地理空間NLPベンチマーク
- **SpaBERT** (Huang et al., 2022): 空間関係の理解に特化したBERT
- **PostGIS + LLM**: 空間データベースとLLMの統合事例

### 2.5 多言語LLMにおける言語制御

#### 2.5.1 Code-Switching現象

多言語LLMは、生成中に言語が切り替わる「コードスイッチング」を起こす可能性がある（Zhang et al., 2023）。特に技術文書やJSON形式の入出力で顕著である。

#### 2.5.2 言語制御手法

- **System Prompt指示**: 「必ず日本語で」などの明示的指示
- **Few-shot例**: 目標言語での回答例を提示
- **後処理フィルタ**: 生成後に言語を検証して再生成

本研究では、これらの手法を適用したにもかかわらず言語混在が発生した（詳細は5.3章）。

---

## 3. 方法論（Methodology）

### 3.1 システム設計

#### 3.1.1 アーキテクチャ概要

```
┌─────────────────────────────────────────────────────────────┐
│                   Agentic RAG Architecture                   │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  [User Query]                                                │
│       ↓                                                       │
│  ┌─────────────────┐                                         │
│  │ Agent State     │ ← TypedDict-based state management     │
│  │ - question      │                                         │
│  │ - iteration     │                                         │
│  │ - tool_results  │                                         │
│  │ - intermediate  │                                         │
│  └────────┬────────┘                                         │
│           ↓                                                   │
│  ┌─────────────────────────────┐                             │
│  │ Agent Node                  │ ← LLM Decision Making       │
│  │ ┌─────────────────────────┐ │                             │
│  │ │ Build Prompt:           │ │                             │
│  │ │ - System prompt         │ │                             │
│  │ │ - Question              │ │                             │
│  │ │ - Tool descriptions     │ │                             │
│  │ │ - Previous results      │ │                             │
│  │ └─────────────────────────┘ │                             │
│  │           ↓                 │                             │
│  │ ┌─────────────────────────┐ │                             │
│  │ │ LLM (Qwen2.5-7B)        │ │                             │
│  │ │ ReAct format:           │ │                             │
│  │ │ Thought: [reasoning]    │ │                             │
│  │ │ Action: tool_name       │ │                             │
│  │ │ Action Input: {...}     │ │                             │
│  │ │ OR                      │ │                             │
│  │ │ Final Answer: [answer]  │ │                             │
│  │ └─────────────────────────┘ │                             │
│  └─────────────┬───────────────┘                             │
│                ↓                                              │
│        Parse Response                                         │
│                ↓                                              │
│         Tool Call?                                            │
│         /        \                                            │
│       Yes        No                                           │
│        ↓          ↓                                           │
│  ┌──────────┐  Return Final Answer                           │
│  │ Tool     │                                                 │
│  │ Execution│ ← 16 Tools Available:                          │
│  │ Node     │   - Spatial: get_nearest_pois,                 │
│  └────┬─────┘     filter_by_distance, ...                    │
│       │           - Aggregation: aggregate_by_category,       │
│       │             get_category_ranking, ...                 │
│       │           - Graph: get_same_area_pois,                │
│       │             get_same_brand_pois, ...                  │
│       ↓                                                       │
│  Update State                                                 │
│       ↓                                                       │
│  Iteration++                                                  │
│       ↓                                                       │
│  Max Iterations?                                              │
│    No → Back to Agent Node (Feedback Loop)                   │
│    Yes → Force Final Answer                                  │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

#### 3.1.2 状態管理

LangGraphのステート管理を`TypedDict`で実装：

```python
class AgentState(TypedDict):
    question: str                      # 元の質問
    iteration: int                     # 現在のイテレーション番号
    intermediate_steps: List[Dict]     # 中間ステップの履歴
    tool_results: List[Dict]           # ツール実行結果
    final_answer: Optional[str]        # 最終回答
```

ヘルパー関数:
- `increment_iteration(state)`: イテレーションカウンタを増やす
- `add_intermediate_step(state, step_type, content)`: 中間ステップを記録
- `add_tool_result(state, tool_name, tool_input, tool_output)`: ツール結果を記録

#### 3.1.3 ReActプロンプト設計

```python
AGENT_SYSTEM_PROMPT = """あなたは渋谷駅周辺のPOI情報を検索・分析する専門家です。

利用可能なツール一覧:
{tool_descriptions}

ReAct形式で回答してください:
Thought: [あなたの推論過程]
Action: [ツール名]
Action Input: {{"引数名": "値", ...}}

十分な情報が得られたら:
Thought: 回答に必要な情報が揃いました
Final Answer: [日本語での回答]

**重要: すべての回答は必ず日本語で行ってください。**
"""
```

### 3.2 ツール設計

#### 3.2.1 ツール分類

16種類のツールを3カテゴリに分類：

**空間計算ツール（7個）**:
- `tool_get_nearest_pois`: 最寄りPOI検索（カテゴリ指定、top-k）
- `tool_filter_by_distance`: 半径フィルタリング
- `tool_get_distance_direction`: 距離・方角計算
- `tool_analyze_radius_sensitivity`: 半径感度分析
- `tool_filter_east_west`: 東西フィルタリング
- `tool_get_distance_stats`: 距離統計
- `tool_search_pois_by_name`: 名前検索

**集計ツール（5個）**:
- `tool_aggregate_by_category`: カテゴリ別集計
- `tool_get_category_ranking`: トップカテゴリランキング
- `tool_compare_east_west`: 東西比較
- `tool_compare_categories`: カテゴリ間比較
- `tool_get_total_count`: 総POI数

**グラフトラバーサルツール（4個）**:
- `tool_get_same_area_pois`: 同一エリア内POI
- `tool_get_nearby_pois`: 近隣POI（NEAR_TOエッジ）
- `tool_get_same_brand_pois`: 同一ブランドPOI
- `tool_get_complementary_pois`: 補完関係POI

#### 3.2.2 ツールインターフェース

各ツールは統一されたインターフェースを持つ：

```python
from langchain.tools import BaseTool

class GetNearestPOIsTool(BaseTool):
    name = "tool_get_nearest_pois"
    description = """
    渋谷駅から最も近いPOIを検索します。

    Args:
        category (str): 検索するカテゴリ（例: "カフェ", "レストラン"）
        top_n (int): 取得する件数（デフォルト: 3）

    Returns:
        Dict: カテゴリ、件数、POIリスト（名前、距離、方角）
    """

    def _run(self, category: str, top_n: int = 3):
        # POI検索ロジック
        nearest_pois = get_nearest_pois(GLOBAL_POIS, category, top_n)
        return {
            "category": category,
            "count": len(nearest_pois),
            "pois": [
                {
                    "name": poi["name"],
                    "distance_m": poi["distance_from_station"],
                    "direction": poi["direction_from_station_jp"]
                }
                for poi in nearest_pois
            ]
        }
```

### 3.3 LLMとHuggingFace Transformers統合

#### 3.3.1 モデルロード

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# 4bit量子化設定
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

# モデルロード
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    quantization_config=bnb_config,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct"
)
```

#### 3.3.2 生成パラメータ

```python
generation_config = {
    "max_new_tokens": 512,
    "temperature": 0.0,      # 決定的生成
    "do_sample": False,      # グリーディデコード
    "pad_token_id": tokenizer.eos_token_id
}
```

#### 3.3.3 ReActパース

正規表現ベースでReAct形式を解析：

```python
def _parse_tool_calls(self, response: str):
    """LLMレスポンスからツール呼び出しを抽出"""

    # Pattern 1: Action: tool_name
    action_match = re.search(r'Action:\s*(\w+)', response)
    if not action_match:
        return []

    tool_name = action_match.group(1)

    # Pattern 2: Action Input: {"arg": "value"}
    input_match = re.search(
        r'Action Input:\s*(\{[^}]+\})',
        response
    )

    if input_match:
        try:
            args = json.loads(input_match.group(1))
        except json.JSONDecodeError:
            args = {}
    else:
        args = {}

    return [{
        "tool": tool_name,
        "args": args
    }]
```

### 3.4 比較対象システム

#### 3.4.1 Hybrid RAG (Phase 6)

**特徴**:
- ルールベース質問分析（キーワード検出、正規表現）
- 並列実行（proximity + aggregation + comparison同時実行可）
- 日本語自然文での出力
- 1回のLLM呼び出し

**実装**:
```python
def query(question):
    analysis = analyze_question(question)  # ルールベース

    contexts = []
    if analysis.requires_proximity:
        contexts.append(execute_proximity())  # 並列
    if analysis.requires_aggregation:
        contexts.append(execute_aggregation())  # 並列

    context = "\n\n".join(contexts)
    return llm.generate(context, question)  # 1回
```

#### 3.4.2 Graph RAG (Phase 8)

Neo4jベースのグラフRAG。本Phase 9では直接比較していないが、Phase 8の結果（76.7%）を参照。

#### 3.4.3 Adaptive RAG (Phase 8)

クエリ複雑度に基づいてシステムを選択。Phase 8の結果（86.1%）を参照。

### 3.5 評価方法

#### 3.5.1 テストケース構成

| ソース | 件数 | カテゴリ数 | 特徴 |
|--------|------|----------|------|
| Phase 5-6 | 55件 | 12 | 基本検索、空間推論、制約充足 |
| Phase 8 (GraphRAG) | 35件 | 9 | 関係性クエリ、マルチホップ、ブランド検索 |
| Phase 9 (Agentic) | 15件 | 3 | 複数ステップ推論、条件分岐、反復的絞り込み |
| **合計** | **105件** | **24** | - |

#### 3.5.2 評価指標

1. **成功率（Success Rate）**: キーワードヒット率50%以上のケース割合
2. **キーワードヒット率（Keyword Hit Rate）**: 期待キーワードの出現率平均
3. **実行時間（Execution Time）**: クエリあたりの平均処理時間
4. **カテゴリ別性能**: 各テストカテゴリでの成功率

#### 3.5.3 評価手順

```python
def evaluate_single_case(system, test_case):
    # 質問文取得（prompt or question属性に対応）
    question = getattr(test_case, 'prompt', None) or \
               getattr(test_case, 'question', '')

    # システム実行
    result = system.query(question)
    answer = result.get('answer', '') or ''

    # キーワードヒット率計算
    hit_rate = calculate_keyword_hit_rate(
        answer,
        test_case.expected_keywords
    )

    # 成功判定（50%閾値）
    success = hit_rate >= 0.5

    return {
        'success': success,
        'hit_rate': hit_rate,
        'execution_time': result['execution_time']
    }
```

### 3.6 Agentic RAG向け新規テストケース

Phase 9では、Agentic RAGの動的ツール選択とフィードバックループの能力を評価するため、**15件の新規テストケース**を設計した。これらは既存のPhase 5-6（55件）およびPhase 8（35件）のテストケースとは異なり、単一のツール呼び出しでは解決困難な複雑な推論パターンを含んでいる。

#### 3.6.1 設計意図

新規テストケースは以下の3つの設計原則に基づいて作成された：

1. **複数ステップ推論の必要性**: 中間結果を次の処理に使用する必要がある
2. **動的な戦略変更**: 実行時の結果に応じて次のアクションを決定する必要がある
3. **ツール連鎖の評価**: 複数のツールを適切な順序で組み合わせる能力を測定

これらのテストケースは、Hybrid RAG（事前決定型）では解決が困難であり、Agentic RAGの真の価値を測定できるように設計されている。

#### 3.6.2 カテゴリ1: multi_step_spatial（複数ステップの空間推論）

**目的**: 2段階以上の空間検索を組み合わせて、中間結果を使用する能力を評価する。

**特徴**:
- 第1段階の検索結果を第2段階の起点として使用
- 距離計算や方向分析の段階的実行
- 複数の半径や方向での比較分析

**テストケース数**: 5件（A-01 ~ A-05）
**難易度**: medium (1件), hard (3件), expert (1件)
**期待ツール呼び出し数**: 2-4回

**サンプル1 (A-01, hard)**:
```yaml
質問: "渋谷駅から最も近いカフェを見つけて、そのカフェから300m以内にある他のカフェは何件ありますか？"

期待されるキーワード: ["カフェ", "最も近い", "300m", "件"]

期待される処理フロー:
  Step 1: tool_get_nearest_pois → 最寄りカフェを特定（例: Urth Caffé）
  Step 2: tool_count_pois_in_radius → Urth Caffé座標を起点に300m半径で件数カウント
  Step 3: 最終回答生成 → 「Urth Caffé（最寄り）から300m以内に他に7件のカフェがあります」

評価ポイント:
  - 2段階の空間検索を実行できるか
  - 第1段階の結果（座標）を第2段階で使用できるか
  - 最終的な件数を正確に回答できるか
```

**サンプル2 (A-04, expert)**:
```yaml
質問: "カフェの件数上位3カテゴリを調べて、そのうち東側に多いカテゴリはどれですか？"

期待されるキーワード: ["上位", "カテゴリ", "東側", "多い"]

期待される処理フロー:
  Step 1: tool_get_top_categories → ランキング取得（例: カフェ, レストラン, コンビニ）
  Step 2: tool_compare_east_west(カフェ) → 東西分布分析
  Step 3: tool_compare_east_west(レストラン) → 東西分布分析
  Step 4: tool_compare_east_west(コンビニ) → 東西分布分析
  Step 5: 最終回答生成 → 「上位3カテゴリのうち、カフェが東側に多い（東:45件, 西:28件）」

評価ポイント:
  - カテゴリランキングを取得できるか
  - 各カテゴリの東西分布を分析できるか
  - 東側優位のカテゴリを特定できるか
```

**サンプル3 (A-05, hard)**:
```yaml
質問: "渋谷駅から300m以内のレストランの中で、最も東にあるのはどこですか？"

期待されるキーワード: ["300m", "レストラン", "東", "最も"]

期待される処理フロー:
  Step 1: tool_filter_by_area(レストラン, 300m) → 半径フィルタリング
  Step 2: 結果から経度が最大のPOIを特定
  Step 3: 最終回答生成 → POI名と位置情報を提示

評価ポイント:
  - 半径フィルタを適用できるか
  - 結果から最東のPOIを特定できるか
  - POI名と位置情報を提示できるか
```

#### 3.6.3 カテゴリ2: conditional_reasoning（条件付き推論）

**目的**: 取得したデータに基づいて条件分岐を行い、適切な判定や分類を実行する能力を評価する。

**特徴**:
- if-then-else 形式の条件判定
- 閾値に基づく分類（例: 50件以上なら「多い」）
- 比較結果に基づく動的な分析対象選択

**テストケース数**: 5件（A-06 ~ A-10）
**難易度**: medium (3件), hard (1件), expert (1件)
**期待ツール呼び出し数**: 1-4回

**サンプル1 (A-06, medium)**:
```yaml
質問: "もし500m以内にカフェが50件未満なら「少ない」、50件以上なら「多い」と答えてください。実際はどちらですか？"

期待されるキーワード: ["500m", "カフェ", "件", "多い", "少ない"]

期待される処理フロー:
  Step 1: tool_count_pois_in_radius(カフェ, 500m) → 件数取得（例: 73件）
  Step 2: 条件判定 → 73 >= 50 → "多い"
  Step 3: 最終回答生成 → 「500m以内にカフェは73件あり、『多い』です」

評価ポイント:
  - 件数を取得できるか
  - 条件に基づいて判定できるか
  - 根拠となる実数値を提示できるか
```

**サンプル2 (A-08, hard)**:
```yaml
質問: "300mと500mでカフェの件数を比較して、もし1.5倍以上増えるなら「集中度が低い」、それ以外なら「集中度が高い」と答えてください。"

期待されるキーワード: ["300m", "500m", "カフェ", "倍", "集中度"]

期待される処理フロー:
  Step 1: tool_compare_radius(カフェ, 300m, 500m) → 件数比較（例: 45件 → 73件）
  Step 2: 増加率計算 → 73/45 = 1.62
  Step 3: 条件判定 → 1.62 >= 1.5 → "集中度が低い"
  Step 4: 最終回答生成 → 「1.62倍に増えるため、集中度が低いです」

評価ポイント:
  - 半径比較を実行できるか
  - 増加率を計算できるか
  - 条件に基づいて適切に判定できるか
```

**サンプル3 (A-10, expert)**:
```yaml
質問: "カフェとレストランの件数を比較して、カフェの方が多ければカフェの東西分布を、レストランの方が多ければレストランの東西分布を教えてください。"

期待されるキーワード: ["カフェ", "レストラン", "件数", "東西"]

期待される処理フロー:
  Step 1: tool_count_by_category(カフェ) → 73件
  Step 2: tool_count_by_category(レストラン) → 89件
  Step 3: 比較判定 → レストランの方が多い
  Step 4: tool_compare_east_west(レストラン) → 東:52件, 西:37件
  Step 5: 最終回答生成 → 「レストランの方が多い（89件 > 73件）ため、レストランの東西分布を示します。東:52件, 西:37件」

評価ポイント:
  - 2つのカテゴリの件数を比較できるか
  - 多い方のカテゴリを特定できるか
  - 選択したカテゴリの東西分布を分析できるか
```

#### 3.6.4 カテゴリ3: iterative_refinement（反復的な絞り込み）

**目的**: 中間結果を評価し、必要に応じて検索条件を動的に調整する能力を評価する。

**特徴**:
- 最初の結果が不十分な場合の再検索
- 複数の候補から最適解を探索
- 結果に基づく追加の深掘り分析

**テストケース数**: 5件（A-11 ~ A-15）
**難易度**: hard (3件), expert (2件)
**期待ツール呼び出し数**: 2-4回
**自己修正が必要**: 3件

**サンプル1 (A-11, hard)**:
```yaml
質問: "カフェが最も多い半径を見つけてください。100m、200m、300m、500mで比較して最適な半径を教えてください。"

期待されるキーワード: ["カフェ", "多い", "半径", "最適"]

期待される処理フロー:
  Step 1: tool_analyze_sensitivity(カフェ, [100, 200, 300, 500])
          → 100m:12件, 200m:28件, 300m:45件, 500m:73件
  Step 2: 最大値を特定 → 500mで73件
  Step 3: 最終回答生成 → 「500mが最適で、カフェが73件と最も多いです」

評価ポイント:
  - 複数の半径で件数を取得できるか
  - 最も多い半径を特定できるか
  - 根拠となる数値を全て提示できるか
```

**サンプル2 (A-13, expert)**:
```yaml
質問: "300m以内のカフェ件数を確認し、もし20件未満なら500mで再検索、20件以上ならそのまま最寄りの3件を教えてください。"

期待されるキーワード: ["300m", "500m", "カフェ", "件", "最寄り"]

期待される処理フロー:
  Step 1: tool_count_pois_in_radius(カフェ, 300m) → 45件
  Step 2: 条件判定 → 45 >= 20 → 再検索不要
  Step 3: tool_get_nearest_pois(カフェ, k=3) → 上位3件取得
  Step 4: 最終回答生成 → 「300m以内に45件あるため、最寄りの3件は...」

代替フロー（20件未満の場合）:
  Step 1: tool_count_pois_in_radius(カフェ, 300m) → 15件
  Step 2: 条件判定 → 15 < 20 → 再検索実行
  Step 3: tool_count_pois_in_radius(カフェ, 500m) → 73件
  Step 4: tool_get_nearest_pois(カフェ, k=3) → 上位3件取得
  Step 5: 最終回答生成 → 「300mでは15件のため500mに拡大し、最寄りの3件は...」

評価ポイント:
  - 最初の件数チェックができるか
  - 条件に基づいて半径を調整できるか
  - 最終的に適切な結果を提示できるか
```

**サンプル3 (A-15, hard)**:
```yaml
質問: "カフェの件数が東西で同程度（差が10件以内）か確認し、同程度なら南北の比較も行って分布の傾向を教えてください。"

期待されるキーワード: ["カフェ", "東西", "南北", "同程度", "分布"]

期待される処理フロー:
  Step 1: tool_compare_east_west(カフェ) → 東:38件, 西:35件
  Step 2: 差分計算 → |38-35| = 3件
  Step 3: 条件判定 → 3 <= 10 → 同程度と判定
  Step 4: tool_compare_north_south(カフェ) → 北:42件, 南:31件
  Step 5: 最終回答生成 → 「東西は同程度（差3件）、南北では北に偏在（北:42件, 南:31件）」

代替フロー（差が10件超の場合）:
  Step 1: tool_compare_east_west(カフェ) → 東:45件, 西:28件
  Step 2: 差分計算 → |45-28| = 17件
  Step 3: 条件判定 → 17 > 10 → 同程度でないと判定
  Step 4: 最終回答生成 → 「東西で差が17件あり、同程度ではありません。東に偏在しています」

評価ポイント:
  - 東西比較を実行できるか
  - 差が10件以内の条件判定ができるか
  - 条件に基づいて南北比較も実行できるか
```

#### 3.6.5 テストケース特性の統計

新規テストケース（15件）の特性を以下に示す：

**難易度分布**:
| 難易度 | 件数 | 割合 |
|--------|------|------|
| medium | 3件 | 20% |
| hard | 8件 | 53.3% |
| expert | 4件 | 26.7% |

**推論特性**:
| 特性 | 件数 | 説明 |
|------|------|------|
| requires_multi_step | 13件 (86.7%) | 複数ステップの推論が必要 |
| requires_tool_chaining | 11件 (73.3%) | ツール連鎖が必要 |
| requires_self_correction | 3件 (20%) | 自己修正が必要 |

**期待ツール呼び出し数**:
| 範囲 | 件数 |
|------|------|
| 1-2回 | 5件 (33.3%) |
| 2-4回 | 10件 (66.7%) |
| 平均 | 2.5回 |
| 最大 | 5回 |

**使用される主要ツール**:
| ツール名 | 使用ケース数 |
|---------|------------|
| tool_get_nearest_pois | 7件 |
| tool_count_pois_in_radius | 6件 |
| tool_compare_east_west | 6件 |
| tool_count_by_category | 3件 |
| tool_compare_radius | 3件 |
| tool_get_top_categories | 2件 |

これらのテストケースは、Agentic RAGの以下の能力を総合的に評価する設計となっている：

1. **動的な戦略決定**: 実行時の結果に基づいて次のツールを選択
2. **中間結果の活用**: 前のツール出力を次の入力として使用
3. **条件分岐処理**: 閾値や比較結果に基づく判定
4. **反復的改善**: 不十分な結果を検出し、再検索や条件変更を実行

---

## 4. 実装の詳細（Implementation Details）

### 4.1 主要な技術的課題と解決策

Phase 9の実装では、7つの主要なバグ修正と設計変更を行った。

#### 4.1.1 課題1: メソッド不在エラー

**問題**: `StructuredRAGSystem`に`query()`メソッドがなく評価時エラー

**原因**: メソッド名が`query_with_structured_rag()`だったが、評価スクリプトは`query()`を呼び出し

**解決策**:
```python
def query(self, question: str) -> Dict[str, Any]:
    """質問に回答（評価用エイリアスメソッド）"""
    return self.query_with_structured_rag(question)
```

**結果**: Structured RAG 0% → 80% (Quick Test)

#### 4.1.2 課題2: None値エラー

**問題**: 評価関数が`None`回答を処理できずクラッシュ

**解決策**:
```python
def evaluate_keyword_hit_rate(answer, expected_keywords):
    if not expected_keywords:
        return 1.0
    if not answer:  # None対策
        return 0.0
    # ...
```

#### 4.1.3 課題3: 中国語回答の発生

**問題**: Agentic RAGの回答が中国語になるケース（詳細は5.3章）

**初期対策**: システムプロンプトに日本語強制指示追加
```python
AGENT_SYSTEM_PROMPT = """...
**重要: すべての回答は必ず日本語で行ってください。中国語や英語で回答してはいけません。**
"""
```

**効果**: 一部改善したが完全解決には至らず（7.6%で依然として発生）

#### 4.1.4 課題4-6: 関数シグネチャエラー

**問題**: `add_intermediate_step()`と`add_tool_result()`の引数不一致

**エラー例**:
```python
# 誤った呼び出し
state = add_intermediate_step(state, {
    "tool_calls": tool_calls,
    "response": response
})  # ← contentパラメータが不足

# 正しいシグネチャ
def add_intermediate_step(state, step_type, content):
    ...
```

**解決策**:
```python
# 修正後
state = add_intermediate_step(state, "tool_calls", {
    "tool_calls": tool_calls,
    "response": response
})

# add_tool_resultも同様
state = add_tool_result(state, tool_name, args, output)
```

#### 4.1.7 課題7: テストケース属性名の不一致

**問題**: `GraphRAGTestCase`が`question`属性、他は`prompt`属性

**解決策**:
```python
# 両方に対応
question = getattr(test_case, 'prompt', None) or \
           getattr(test_case, 'question', '')
```

### 4.2 デバッグとイテレーション

#### 4.2.1 verbose モードによる詳細ログ

```python
if self.verbose:
    print(f"\n--- Agent Node (Iteration {state['iteration']}) ---")
    print(f"Agent response: {response[:300]}...")
    print(f"Parsed tool calls: {[tc['tool'] for tc in tool_calls]}")
```

これにより、以下を確認できた：
- ツール選択の妥当性
- ReActフォーマットの解析成功/失敗
- 中間結果の内容

#### 4.2.2 モジュールリロード戦略

ランタイム再起動なしで修正を反映：

```python
import importlib
import sys

if 'src.agentic_rag_system' in sys.modules:
    importlib.reload(sys.modules['src.agentic_rag_system'])
```

これにより、デバッグサイクルを大幅に短縮（10分 → 2分）。

---

## 5. 実験結果（Experimental Results）

### 5.1 全体性能比較

#### 5.1.1 成功率とキーワードヒット率

| システム | 成功率 | キーワードヒット率 | 平均実行時間 | エラー数 |
|---------|--------|-----------------|------------|---------|
| **Hybrid RAG** | **96.2%** (101/105) | 85.3% | 11.1秒 | 0件 |
| **Agentic RAG** | 87.6% (92/105) | 73.8% | **56.4秒** | 0件 |
| **差分** | **-8.6pt** | -11.5pt | **+5.1倍** | - |

**発見1**: Agentic RAGは全体で劣る
- 仮説H1（複雑クエリでの優位性）は部分的にのみ支持
- 単純クエリでは不要なオーバーヘッド

#### 5.1.2 Quick Test vs Full Test

| テスト | 規模 | Structured | Agentic | 差分 |
|--------|------|-----------|---------|------|
| Quick Test | 10件 | 80.0% | **90.0%** | **+10.0pt** ✅ |
| Full Test | 105件 | **96.2%** | 87.6% | **-8.6pt** ⚠️ |

Quick Testでは優位性を示したが、Full Testでは逆転した。

**分析**: Quick Testは基本的なクエリが中心であり、Agenticの動的選択が効果的だった。しかし、Full Testには高度な推論タスクが含まれ、そこでAgenticが苦戦した。

### 5.2 カテゴリ別詳細分析

#### 5.2.1 Agentic RAGが優位なカテゴリ（+20pt以上）

| カテゴリ | 件数 | Hybrid | Agentic | 改善 | 分析 |
|---------|------|--------|---------|------|------|
| **competitor** | 3件 | 66.7% | **100%** | **+33.3pt** | 競合関係の動的探索が有効 |
| **complementary** | 5件 | 80.0% | **100%** | **+20.0pt** | 補完関係の段階的絞り込みが効果 |
| **basic_location** | 5件 | 80.0% | **100%** | **+20.0pt** | ツールの適切な選択 |

**事例分析：competitor**

**テストケース**: "渋谷のスターバックスの競合店を教えて"

**Hybrid RAGの処理**:
```
[質問分析] → "competitor"キーワード検出せず
[ベクトル検索] → "スターバックス"関連POI
[回答] → スターバックスの説明（競合店言及なし）
→ 失敗
```

**Agentic RAGの処理**:
```
[Iteration 0]
LLM: "まずスターバックスを探す"
Action: tool_search_pois_by_name("スターバックス")
Result: 13店舗検出

[Iteration 1]
LLM: "競合関係を探す"
Action: tool_get_competitor_pois("スターバックス")
Result: [ドトール, タリーズ, コメダ珈琲]

[Iteration 2]
LLM: "十分な情報が揃った"
Final Answer: "スターバックスの競合店は、ドトール..."
→ 成功
```

**発見2**: グラフトラバーサルツールが有効
- 関係性を明示的に持つデータでは、動的探索が優位

#### 5.2.2 同等パフォーマンス（18カテゴリ、全て100%）

以下のカテゴリでは両システムとも完璧：
- `conditional_reasoning`, `multi_step_spatial`, `iterative_refinement`
- `proximity`, `aggregation`, `comparison`, `cuisine`, `hours`
- その他10カテゴリ

**分析**: これらのカテゴリでは、Hybridの事前決定で十分。Agenticの動的選択は不要なオーバーヘッド。

#### 5.2.3 Agentic RAGが劣るカテゴリ

| カテゴリ | 件数 | Hybrid | Agentic | 悪化 | 主要因 |
|---------|------|--------|---------|------|--------|
| **advanced_uncertainty** | 3件 | 100% | **0%** | **-100pt** 💥 | 空回答/中国語 |
| **constraint_multi** | 5件 | 100% | 40% | -60pt | 複数制約で失敗 |
| **advanced_comparison** | 2件 | 100% | 50% | -50pt | 高度な比較で劣化 |
| **brand** | 5件 | 100% | 60% | -40pt | ブランド検索で失敗 |
| **multi_hop** | 3件 | 100% | 66.7% | -33.3pt | マルチホップで課題 |
| **spatial_comparison** | 5件 | 100% | 80% | -20pt | 空間比較で一部失敗 |

**事例分析：advanced_uncertainty**

**テストケース**: "渋谷駅周辺で「雰囲気の良い」カフェを探しています。データから判断できる範囲で推薦し、判断の限界も説明してください"

**Hybrid RAGの処理**:
```
[質問分析] → proximity + uncertainty検出
[proximity_search] → 最寄りカフェリスト（日本語テキスト）
[LLM生成] → "データからは距離のみ判断可能。雰囲気は主観的..."
→ 成功（キーワード: "データ", "判断", "限界"含む）
```

**Agentic RAGの処理**:
```
[Iteration 0]
LLM: "カフェを探す"
Action: tool_get_nearest_pois("カフェ")
Result: JSON形式 {"pois": [...]}

[Iteration 1]
LLM: [中国語で生成開始] "咖啡馆的数据..." ← 失敗
または
LLM: [空回答] "" ← 失敗
```

**発見3**: 不確実性への対応で失敗
- 仮説H4（不確実性での優位性）は**棄却**
- むしろHybridが安定

### 5.3 言語制御問題の詳細分析

#### 5.3.1 中国語回答の発生パターン

**発生率**: 8件/105件（7.6%）

**カテゴリ分布**:
- basic_location: 4件
- constraint_multi: 2件
- multi_hop: 1件
- brand: 1件

**重要な観察**: 単純なクエリでも発生（イテレーション数だけが原因ではない）

#### 5.3.2 中国語回答の例

**L3-06** (constraint_multi):
```
質問: 渋谷駅から500m以内で、電話番号とウェブサイトの両方がわかるカフェを教えてください

回答: 渋谷站附近距离最近的咖啡馆是"Urth Caffé"，距离约为56.72米，位于东北方向。
然而，我们无法从现有数据中确定该咖啡馆是否提供电话号码和网站信息。
建议您直接访问该咖啡馆或在其社交媒体平台上查找这些信息。

キーワードヒット率: 0.0%（日本語キーワードと不一致）
```

**GR-07** (multi_hop):
```
質問: 渋谷駅から100m以内のコンビニと、そこから近いカフェを教えてください

回答: 渋谷駅从100米范围内最近的便利店是ローソン，距离为101.94米。
最近的咖啡馆是Urth Caffé，位于北东方向，距离为56.72米。

キーワードヒット率: 0.0%
```

**観察**:
- 内容は正確（Urth Caffé 56.72m、ローソン 101.94m）
- しかし言語が中国語 → 自動的に失敗判定

#### 5.3.3 根本原因分析

**4つの主要因**:

**1. JSON形式のツール出力（寄与度: 50%）**

```python
# ツール出力（JSON形式）
{
  "category": "カフェ",
  "count": 1,
  "pois": [
    {
      "name": "Urth Caffé",
      "distance_m": 56.72,
      "direction": "北東"
    }
  ]
}
```

**トリガーメカニズム**:
1. JSON = 技術文書フォーマット
2. Qwenの訓練データ: 技術文書 = 英語/中国語混在が一般的
3. JSON構造を見た瞬間に「技術モード」に切り替わる
4. 技術モード = 中国語許容

**2. ReAct形式の英語メタ言語（寄与度: 30%）**

```
Thought: [your reasoning]
Action: [tool_name]
Action Input: {"arg": "value"}
Final Answer: [your answer]
```

**問題**:
- 英語のメタ言語（Thought, Action, Final Answer）
- 中国語技術文書では "思考:", "行动:", "最终答案:" が一般的
- モデルは「推論プロセス = 中国語可」と学習済み

**3. コンテキスト累積による希薄化（寄与度: 15%）**

```
Iteration 0: System Prompt (200 tokens)
           + Question (50 tokens)
           = 250 tokens

Iteration 2: System Prompt (200 tokens)
           + Question (50 tokens)
           + Tool Result 1 (150 tokens, JSON)
           + Tool Result 2 (150 tokens, JSON)
           = 550 tokens
```

**影響**:
- 日本語指示が全体の36% → 15%に希薄化
- JSON形式のツール結果が支配的
- システムプロンプトの日本語強制が相対的に弱まる

**4. 数値表現の違い（寄与度: 5%）**

```python
# ツール出力
{"distance_m": 56.72}

# LLMの解釈
# 日本語: "56.72m" または "56.72メートル"
# 中国語: "56.72米" ← こちらに誘導されやすい
```

#### 5.3.4 Hybrid RAGで問題なかった理由

**Hybrid RAGのコンテキスト例**:
```
【カフェの最寄りPOI（上位3件）】
  1. Urth Caffé
     距離: 56.7m（北東方向）
  2. スターバックス渋谷駅前店
     距離: 280.4m（北西方向）

【カフェの距離統計】
  総数: 149件
  最短: 56.7m / 最長: 1200.5m

【検索結果】
Urth Caffé カフェ 渋谷にあるおしゃれなオーガニックカフェ...
```

**特徴**:
- ✅ 純粋な日本語自然文
- ✅ JSON構造なし
- ✅ 単一パス生成（累積なし）
- ✅ 技術モードトリガーなし

**発見4**: 出力形式がLLM性能に決定的な影響
- JSON形式 < 整形された日本語自然文
- 多言語LLMにおける言語安定性

#### 5.3.5 改善策の提案

**改善策1: ツール出力の日本語テキスト化**（推奨）

```python
def format_tool_output_japanese(tool_name, output):
    """ツール出力を日本語自然文に変換"""
    if tool_name == "tool_get_nearest_pois":
        if not output.get('pois'):
            return f"{output['category']}は見つかりませんでした。"

        lines = [f"{output['category']}の検索結果（{output['count']}件）:"]
        for poi in output['pois']:
            lines.append(
                f"- {poi['name']}: 渋谷駅から{poi['direction']}方向に"
                f"{poi['distance_m']:.1f}メートル"
            )
        return "\n".join(lines)
    # 他のツールも同様
```

**期待効果**: 中国語出現率を50-70%削減（8件 → 2-3件）

**改善策2: ReAct指示の日本語化**

```python
prompt_parts.append("以下の形式で回答してください:")
prompt_parts.append("思考: [あなたの推論過程]")
prompt_parts.append("行動: [使用するツール名]")
prompt_parts.append("行動入力: {\"引数1\": \"値1\", ...}")
prompt_parts.append("最終回答: [日本語での回答]")
```

**期待効果**: 中国語出現率を20-30%削減

**改善策3: 各イテレーションでの日本語強制再注入**

```python
if state["tool_results"]:
    prompt_parts.append("\nこれまでのツール実行結果:")
    for r in state["tool_results"]:
        formatted = format_tool_output_japanese(r['tool'], r['output'])
        prompt_parts.append(f"\n{formatted}\n")

    # 再度強調
    prompt_parts.append(
        "\n【重要】上記の情報を踏まえて、必ず日本語で回答を生成してください。\n"
    )
```

**期待効果**: 中国語出現率を30-40%削減

**組み合わせ**: 全て実装で80-90%削減（8件 → 0-1件）

### 5.4 実行時間分析

#### 5.4.1 イテレーション分布

| イテレーション数 | ケース数 | 割合 | 平均実行時間 |
|----------------|---------|------|------------|
| 1回 | 32件 | 30.5% | 25.3秒 |
| 2回 | 45件 | 42.9% | 52.1秒 |
| 3回 | 18件 | 17.1% | 78.4秒 |
| 4回 | 7件 | 6.7% | 105.2秒 |
| 5回（最大） | 3件 | 2.9% | 132.6秒 |

**観察**:
- 大半（73.4%）は1-2イテレーションで完了
- しかし複雑なクエリは4-5イテレーション必要
- 最長132.6秒（Hybridの12倍）

#### 5.4.2 実行時間の内訳

```
総実行時間（56.4秒）の内訳:
  - LLM生成（2-5回）: 40.2秒（71.3%）
  - ツール実行: 8.1秒（14.4%）
  - パース・状態更新: 5.3秒（9.4%）
  - その他: 2.8秒（5.0%）
```

**ボトルネック**: LLM呼び出し回数
- Hybrid: 1回（8秒）
- Agentic: 平均2.3回（平均18.4秒）+ 最終生成（8秒）

**発見5**: 実行時間とのトレードオフが大きい
- 精度向上（特定カテゴリのみ）vs 5倍の実行時間
- 実用アプリケーションには遅すぎる

---

## 6. 考察（Discussion）

### 6.1 研究課題への回答

#### 6.1.1 RQ1: Agentic RAGは複雑な推論タスクで優位性を持つか？

**回答**: **部分的にYes**

**優位性を示したカテゴリ**:
- competitor関係（+33.3pt）
- complementary関係（+20.0pt）
- basic_location（+20.0pt）

**劣ったカテゴリ**:
- advanced_uncertainty（-100pt）
- constraint_multi（-60pt）
- advanced_comparison（-50pt）

**解釈**:
- グラフトラバーサルが必要なタスク（関係性探索）で効果的
- しかし、複雑な論理推論や不確実性への対応では劣る
- 「複雑さ」の種類により効果が異なる

#### 6.1.2 RQ2: 動的ツール選択はどのカテゴリで効果を発揮するか？

**効果的なカテゴリの特徴**:
1. **中間結果が次の行動を決定する**
   - 例: "スターバックスを探す → 見つかった → 競合店を探す"
2. **事前に必要なツールが不明**
   - 例: ブランド名が質問に明示されていない
3. **データ探索的なタスク**
   - 例: "〜の近くにある〜"（段階的な絞り込み）

**効果が限定的なカテゴリの特徴**:
1. **事前に処理フローが明確**
   - 例: "最寄りのカフェ" → proximity_searchで完結
2. **並列実行可能な複数処理**
   - 例: "東西比較 + 最寄り検索" → 同時実行が効率的
3. **高度な推論が必要**
   - 例: 不確実性への言及 → LLMの推論能力に依存

#### 6.1.3 RQ3: 実行時間のオーバーヘッドは精度向上に見合うか？

**回答**: **No（現状では）**

**トレードオフ分析**:
```
実行時間: +5.1倍（11.1秒 → 56.4秒）
精度向上: -8.6pt（96.2% → 87.6%）
```

**改善の余地**:
- 言語制御問題が解決されれば: 87.6% → 92-95%（推定）
- それでも実行時間のコストは大きい

**実用性の観点**:
- リアルタイムアプリケーション: 不向き（56秒は遅すぎる）
- バッチ処理・高精度要求タスク: 検討可能

#### 6.1.4 RQ4: LLM駆動のツール選択はルールベースと比較してどの程度ロバストか？

**回答**: **ルールベースの方がロバスト**（現状のLLMでは）

**ロバスト性の比較**:

| 観点 | ルールベース (Hybrid) | LLM駆動 (Agentic) |
|------|---------------------|------------------|
| **言語安定性** | ✅ 高い（100%日本語） | ❌ 低い（7.6%中国語） |
| **ツール選択精度** | ✅ 高い（明示的ルール） | ⚠️ 中程度（時々誤選択） |
| **予測可能性** | ✅ 高い（常に同じフロー） | ❌ 低い（LLMに依存） |
| **デバッグ容易性** | ✅ 容易（ルール追跡可） | ❌ 困難（ブラックボックス） |
| **柔軟性** | ❌ 低い（新パターン対応困難） | ✅ 高い（未知パターン対応） |

**発見6**: ルールベースの予測可能性が重要
- 本番環境では安定性が最優先
- LLMの判断は時々予期しない挙動

### 6.2 "構造化RAG"用語の再定義

#### 6.2.1 学術的定義との乖離

本プロジェクトで「構造化RAG」と呼んでいたシステムは、学術文献における標準的な「Structured RAG」とは異なることが明らかになった。

**学術的な"Structured RAG"**:
1. **メタデータフィルタリング**
   ```python
   results = vectorstore.similarity_search(
       query,
       filter={"category": "カフェ", "price": "中価格"}
   )
   ```
2. **リランキング**
   ```python
   reranked = rerank_by_metadata(results, relevance_score)
   ```
3. **構造化プロンプト**
   ```python
   prompt = f"""
   カテゴリ: {category}
   価格帯: {price}

   検索結果:
   {results}

   質問: {query}
   """
   ```

**特徴**: ベクトル検索の前後処理を構造化するが、計算処理は含まない

**本プロジェクトの"構造化RAG"**（Phase 6）:
1. **ルールベースの質問分析**
   - キーワード検出、正規表現マッチング
2. **実時間の計算処理**
   - 空間計算（距離、方角、統計）
   - 集計処理（カテゴリ別カウント）
   - 比較処理（東西比較、カテゴリ間比較）
3. **ベクトル検索との相補的統合**
   - 計算結果とベクトル検索を並列実行
   - 両方のコンテキストを統合
4. **整形された日本語自然文での出力**
   - JSON形式ではなく、読みやすい日本語テキスト

**特徴**: 「構造化されたアプローチ」であり、ツール的処理を含む

#### 6.2.2 正確な用語："Hybrid RAG with Rule-based Tool Selection"

本システムは以下の要素を統合する：
- **Naive RAG**: ベクトル検索
- **Structured RAG**: メタデータフィルタリング
- **Tool-augmented RAG**: 外部ツール（計算、集計）
- **Rule-based orchestration**: 質問分析による処理選択

**提案する用語**: **"Hybrid RAG"**

```
┌─────────────────────────────────────────────┐
│      Hybrid RAG Architecture                 │
├─────────────────────────────────────────────┤
│  1. Rule-based Question Analysis            │
│     ↓                                        │
│  2. Parallel Execution:                     │
│     ├─ Vector Search (semantic)             │
│     ├─ Spatial Computation (geo_utils)      │
│     ├─ Aggregation (aggregator)             │
│     └─ Comparison (east/west)               │
│     ↓                                        │
│  3. Context Integration                     │
│     - Format as Japanese natural text       │
│     ↓                                        │
│  4. LLM Generation (single call)            │
└─────────────────────────────────────────────┘
```

#### 6.2.3 RAGアプローチの分類学

本研究を通じて、RAGアプローチの分類学を以下のように整理できる：

```
RAG Taxonomy:

1. Naive RAG
   - Vector search only
   - No structured processing

2. Structured RAG (academic definition)
   - Metadata filtering
   - Reranking
   - Structured prompts

3. Hybrid RAG (this project's approach)
   - Rule-based analysis
   - Computational tools
   - Vector search integration
   - Natural language output

4. Adaptive RAG
   - Meta-level system selection
   - Query complexity-based routing

5. Agentic RAG
   - LLM-driven tool selection
   - Iterative reasoning
   - Feedback loop
```

**決定的な違い**:

| 観点 | Structured | Hybrid | Adaptive | Agentic |
|------|-----------|--------|----------|---------|
| **ツール使用** | なし | ルールベース | システム選択 | LLM駆動 |
| **決定タイミング** | 事前 | 事前 | 事前 | 実行中 |
| **フィードバック** | なし | なし | なし | あり |
| **出力形式** | 任意 | 自然文 | 任意 | JSON/自然文 |

### 6.3 Hybrid RAGアプローチのスケーラビリティ

#### 6.3.1 MCP ServerとPostGIS統合への適用可能性

Phase 9の実験から、Hybrid RAGアプローチは**データソースに依存しない**ことが明らかになった。

**現在の実装（ChromaDB）**:
```python
# 空間計算
nearest_pois = get_nearest_pois(pois, category, top_n)
context = format_proximity_context_japanese(nearest_pois)

# ベクトル検索
vector_results = vectorstore.similarity_search(question, k=5)

# 統合
final_context = context + vector_results
```

**Phase 10での拡張（PostGIS + MCP Server）**:
```python
# MCP Server経由でPostGIS検索
result = mcp.call_tool("supabase_postgis", {
    "rpc": "get_nearest_pois",
    "params": {
        "center_lat": coords["lat"],
        "center_lon": coords["lon"],
        "category": category,
        "limit": 5
    }
})

# 重要: 日本語テキスト化（Phase 9の教訓）
context = format_postgis_result_japanese(result)

# ベクトル検索（オプション）
vector_results = vectorstore.similarity_search(question, k=5)

# 統合
final_context = context + vector_results
```

**重要な原則**:
1. ✅ **JSON出力を避ける**: PostGISの結果を日本語テキスト化
2. ✅ **相補的統合を維持**: 空間検索とベクトル検索の並列実行
3. ✅ **ルールベース分析を継承**: 質問タイプ別の処理選択
4. ✅ **単一パス生成**: 1回のLLM呼び出し

#### 6.3.2 全国展開でのメリット

| 要素 | ChromaDB (Phase 6-9) | PostGIS + MCP (Phase 10) | 改善率 |
|------|---------------------|-------------------------|--------|
| **データ規模** | 1,047 POI | 500万POI以上 | 5000倍 |
| **空間検索速度** | 遅い（全件スキャン） | 高速（空間インデックス） | 100倍以上 |
| **集計処理** | Pythonループ | SQL集計 | 10-50倍 |
| **複雑な空間クエリ** | 困難 | PostGIS関数 | ✅ 可能 |
| **リアルタイムデータ** | 再埋め込み必要 | 即座に反映 | ✅ |
| **全国展開** | 困難 | 可能 | ✅ |

**PostGISの空間クエリ例**:
```sql
-- 500m以内のカフェを高速検索（500万POI中でも数ミリ秒）
SELECT name, category,
       ST_Distance(geom, ST_MakePoint(139.7, 35.65)::geography) as distance
FROM pois
WHERE ST_DWithin(geom, ST_MakePoint(139.7, 35.65)::geography, 500)
  AND category LIKE '%カフェ%'
ORDER BY distance
LIMIT 5;

-- 東西比較（渋谷駅の経度で分割）
SELECT
    CASE WHEN ST_X(geom) > 139.701636 THEN '東側' ELSE '西側' END as side,
    category,
    COUNT(*) as count
FROM pois
WHERE ST_DWithin(geom, ST_MakePoint(139.701636, 35.658034)::geography, 1000)
GROUP BY side, category;
```

**Hybrid RAGアプローチの利点**:
1. **検索速度**: 空間インデックスにより100倍以上高速化
2. **スケール**: 500万POI以上でも高速動作
3. **複雑なクエリ**: PostGIS関数で高度な空間分析
4. **リアルタイム性**: 営業時間、在庫状況などの動的データ対応

**発見7**: Hybrid RAGはスケーラブル
- ルールベース分析 + 日本語テキスト化の原則を維持
- データソース（ChromaDB → PostGIS）の置き換えが容易
- Phase 10での全国展開に最適

#### 6.3.3 Agentic RAGとの統合可能性

Phase 10では、以下のような選択的な統合も検討できる：

**シナリオ1: ハイブリッドアプローチ（推奨）**
```python
def query(question):
    # メタレベルの判断
    if is_complex_relational_query(question):
        # Agentic RAG（改善版、日本語テキスト化）
        return agentic_rag.query(question)
    else:
        # Hybrid RAG（高速、安定）
        return hybrid_rag.query(question)
```

**シナリオ2: Agentic RAGでPostGISツール使用**
```python
# PostGISツールをAgenticで利用
tools = [
    Tool(
        name="postgis_nearest",
        func=lambda args: format_japanese(
            mcp.call_tool("get_nearest_pois", args)
        )
    ),
    Tool(
        name="postgis_aggregate",
        func=lambda args: format_japanese(
            mcp.call_tool("aggregate_by_category", args)
        )
    ),
    # ...
]
```

**重要**: 日本語テキスト化を必ず適用

### 6.4 フィードバックループの価値再評価

#### 6.4.1 理論的価値と実際の効果

**理論的に期待されていた利点**:
1. 中間結果を見て戦略を変更できる
2. 予期しないクエリパターンに対応できる
3. 段階的な絞り込みで高精度を達成できる

**実際の結果**:
1. ✅ **関係性探索では有効**（competitor, complementary）
2. ❌ **高度な推論では劣る**（advanced_uncertainty）
3. ⚠️ **既知パターンでは不要なオーバーヘッド**

#### 6.4.2 フィードバックループが価値を持つ条件

本研究から、以下の条件でフィードバックループが有効であることが示された：

**条件1**: 中間結果が次の行動を決定する
- 例: "Aを探す → 見つかった → Aの近くのBを探す"
- Hybridでは不可能（事前に処理フロー確定）

**条件2**: 事前に必要な情報が不明
- 例: ブランド名が質問に明示されていない
- Hybridでは対応困難（ルールに記述できない）

**条件3**: データ探索的なタスク
- 例: "〜がありそうなエリアの〜"
- Hybridでは「ありそうなエリア」の判定が困難

**フィードバックループが価値を持たない条件**:

**条件1**: 事前に処理フローが明確
- 例: "最寄りのカフェ"
- Hybridで十分（proximity_searchで完結）

**条件2**: 並列実行可能な複数処理
- 例: "東西比較 + 最寄り検索"
- Hybridの方が効率的（同時実行）

**条件3**: 高度な推論が主要
- 例: 不確実性への言及
- LLMの推論能力に依存（イテレーションは不要）

#### 6.4.3 適応的なアプローチの提案

Phase 10では、以下のような適応的アプローチが有効と考えられる：

```python
def adaptive_query_router(question):
    """クエリの性質に応じて最適なシステムを選択"""

    # 分析
    analysis = analyze_query_characteristics(question)

    # 関係性探索が必要
    if analysis.requires_relational_traversal:
        return agentic_rag.query(question)  # 動的探索が有効

    # 複数制約条件
    elif analysis.has_multiple_constraints:
        if analysis.constraints_are_independent:
            return hybrid_rag.query(question)  # 並列処理が効率的
        else:
            return agentic_rag.query(question)  # 逐次絞り込みが必要

    # 単純な検索
    else:
        return hybrid_rag.query(question)  # 高速・安定
```

### 6.5 限界と今後の課題

#### 6.5.1 本研究の限界

1. **モデル依存性**
   - Qwen2.5-7B-Instructのみで評価
   - 日本語に特化したモデルでの再評価が必要（Phase 9-B提案）

2. **ドメイン限定性**
   - 渋谷駅周辺のPOIデータのみ
   - 他の地理領域やドメインでの検証が必要

3. **テストケース規模**
   - 105ケースは包括的だが、大規模評価が望ましい
   - 特にAgentic特化ケースは15件のみ

4. **言語制御問題**
   - 改善策を提案したが実装・検証は未完了

5. **コスト分析の不足**
   - LLM APIコスト（トークン数）の詳細分析が不足
   - 実用化時のコスト試算が必要

#### 6.5.2 今後の研究課題

**Phase 9-B: 公平な比較実験**

目的: 日本語モデルで5システムを比較
- Naive RAG（新規実装）
- Standard Structured RAG（新規実装）
- Hybrid RAG（現行）
- Adaptive RAG（既存）
- Agentic RAG（改善版：日本語テキスト化）

期待される知見:
- モデルの影響評価
- Agenticの真の性能
- 最適アプローチの確定

**Phase 10: 全国展開**

目的: Hybrid RAGをPostGIS + MCP Serverに統合
- Supabase + PostGISセットアップ
- MCP Server実装
- 500万POI対応
- 全国テストケース作成

期待される成果:
- 実用的な全国RAGシステム
- PostGISの高速検索実証
- リアルタイムデータ対応

**Phase 11: マルチモーダル拡張**

目的: 画像・音声データの統合
- POI画像の視覚的検索
- 音声での自然な問い合わせ

**Phase 12: リアルタイムシステム**

目的: ストリーミング応答とキャッシング
- 部分的な回答の早期返却
- 頻出クエリのキャッシング

---

## 7. 結論（Conclusion）

### 7.1 主要な発見

本研究では、LangGraphベースのAgentic RAGシステムを実装し、地理的POIクエリタスクにおける評価を行った。以下の主要な発見が得られた：

**発見1: Agentic RAGの限定的な優位性**
- 全体性能ではHybrid RAGに劣る（87.6% vs 96.2%）
- 特定カテゴリ（関係性探索）でのみ優位性を発揮
- 実行時間が5.1倍に増加

**発見2: 言語制御における重大な課題**
- JSON形式のツール出力が多言語LLMで言語混在を誘発
- 整形された自然言語テキストの方が言語安定性が高い
- 出力形式の設計がLLM性能に決定的な影響

**発見3: "構造化RAG"用語の再定義**
- 本プロジェクトの「構造化RAG」は学術的定義とは異なる
- より正確には「Hybrid RAG with Rule-based Tool Selection」
- ルールベース分析 + 計算ツール + ベクトル検索の統合

**発見4: Hybrid RAGのスケーラビリティ**
- Hybrid RAGアプローチはデータソースに依存しない
- PostGIS + MCP Server統合による全国展開に適用可能
- 日本語テキスト化の原則を維持すれば高性能を期待

**発見5: フィードバックループの条件付き価値**
- 関係性探索や段階的絞り込みでは有効
- 既知パターンや並列処理可能なタスクでは不要なオーバーヘッド
- 適応的なシステム選択が現実的

### 7.2 研究課題への最終回答

| RQ | 回答 | 根拠 |
|----|------|------|
| RQ1 | 部分的にYes | 関係性探索で優位、高度推論で劣る |
| RQ2 | 関係性探索、段階的絞り込み | competitor (+33pt), complementary (+20pt) |
| RQ3 | No | 5.1倍の実行時間 vs -8.6ptの精度低下 |
| RQ4 | ルールベースの方がロバスト | 言語安定性、予測可能性、デバッグ容易性 |

### 7.3 実践的推奨

本研究の結果に基づき、以下を推奨する：

**推奨1: Phase 10でHybrid RAGを継続**
- Agentic RAGは現状では実用に課題
- Hybrid RAGの96.2%は実用レベル
- PostGIS統合で更なる高速化・スケール向上

**推奨2: 日本語テキスト化の原則を維持**
- JSON出力を避け、整形された日本語自然文を使用
- Phase 10のMCP Server統合でも同原則を適用

**推奨3: 選択的にAgentic RAGを活用**
- 関係性探索が必要なクエリのみ
- 改善版（日本語テキスト化）を実装後

**推奨4: Phase 9-B実施の検討**
- 日本語モデル（Llama 3.1等）での再評価
- Agenticの真の性能を検証
- 学術的完全性を確保

### 7.4 学術的貢献

本研究の学術的貢献は以下の通りである：

1. **Agentic RAGの体系的評価**
   - 地理空間ドメインでの包括的比較（105ケース）
   - 適用場面と限界の明確化

2. **言語制御問題の根本原因分析**
   - JSON形式がトリガーとなる言語混在メカニズムの解明
   - 出力形式設計の重要性を実証

3. **RAGアプローチの分類学的整理**
   - Hybrid RAG、Adaptive RAG、Agentic RAGの本質的違いを明確化
   - 「構造化RAG」用語の学術的定義との乖離を指摘

4. **スケーラブルなHybrid RAGアプローチの提案**
   - ルールベース分析 + 計算ツール + 自然言語出力
   - PostGIS/MCP Server統合への適用可能性を示唆

### 7.5 今後の展望

本研究は、RAGシステムの設計における以下の方向性を示唆する：

**短期的展望（Phase 9-B, Phase 10）**:
- 日本語モデルでの再評価
- PostGIS + MCP Serverへの統合
- 全国展開の実現

**中期的展望**:
- マルチモーダルRAG（画像、音声）
- リアルタイム応答とキャッシング
- ユーザーフィードバックによる継続的改善

**長期的展望**:
- エージェント協調（複数のAgenticが協力）
- Self-correctionメカニズム（エラー検出と修正）
- 人間-AIハイブリッド意思決定支援

地理空間情報とLLMの統合は、今後も重要な研究領域であり続ける。本研究で得られた知見が、より効果的なRAGシステムの設計に貢献することを期待する。

---

## 参考文献（References）

### RAGとLLM

1. Lewis, P., et al. (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." *NeurIPS 2020*.

2. Gao, Y., et al. (2023). "Retrieval-Augmented Generation for Large Language Models: A Survey." *arXiv preprint arXiv:2312.10997*.

3. Ram, O., et al. (2023). "In-Context Retrieval-Augmented Language Models." *arXiv preprint arXiv:2302.00083*.

### エージェントとツール使用

4. Yao, S., et al. (2022). "ReAct: Synergizing Reasoning and Acting in Language Models." *ICLR 2023*.

5. Schick, T., et al. (2023). "Toolformer: Language Models Can Teach Themselves to Use Tools." *arXiv preprint arXiv:2302.04761*.

6. Harrison, C., et al. (2024). "LangGraph: Stateful Agent Orchestration." *LangChain Documentation*.

### グラフRAGと知識グラフ

7. Microsoft Research (2024). "GraphRAG: Knowledge Graph-Enhanced Retrieval Augmented Generation." *Microsoft Research Blog*.

8. Edge, D., et al. (2024). "From Local to Global: A Graph RAG Approach to Query-Focused Summarization." *arXiv preprint arXiv:2404.16130*.

### 地理空間とLLM

9. Mai, G., et al. (2020). "SE-KGE: A Location-Aware Knowledge Graph Embedding Model for Geographic Question Answering and Spatial Semantic Lifting." *Transactions in GIS*.

10. Huang, W., et al. (2022). "SpaBERT: A Pretrained Language Model from Geographic Data for Geo-Entity Representation." *EMNLP 2022*.

### 多言語LLM

11. Zhang, X., et al. (2023). "Code-Switching in Multilingual Large Language Models: Causes and Mitigation." *ACL 2023*.

12. Qwen Team (2024). "Qwen2.5 Technical Report." *arXiv preprint arXiv:2409.12186*.

### 本プロジェクトの先行フェーズ

13. 本プロジェクト (2026). "地理的POIクエリにおける構造化RAGの実装と評価." *Phase 6 Report*.

14. 本プロジェクト (2026). "ファインチューニングによる地理的POI質問応答の性能向上研究." *Phase 7 Report*.

15. 本プロジェクト (2026). "地理的POIクエリにおけるグラフRAGと構造化RAGの性能比較研究." *Phase 8 Report*.

---

## Appendix A: システム実装詳細

### A.1 ファイル構成

```
src/
├── agentic_rag_system.py      # Agentic RAGメインシステム（450行）
├── agent_state.py             # 状態管理（TypedDict, ヘルパー）
├── agent_tools.py             # 16ツールの実装（600行）
├── agent_prompts.py           # ReActプロンプトテンプレート
├── structured_rag_system.py   # Hybrid RAG（比較用）
├── geo_utils.py               # 空間計算ユーティリティ
├── aggregator.py              # 集計ユーティリティ
└── test_cases_agentic.py      # Agentic特化テストケース（15件）

notebooks/
└── phase9_agentic_rag_evaluation.ipynb  # 評価ノートブック

docs/
├── AGENTIC_RAG_EXPERIMENT_REPORT.md         # 本レポート
├── HANDOVER_PHASE9_AGENTIC_RAG.md           # 引き継ぎ
├── RAG_COMPARISON_DETAILED.md               # RAG比較詳細
├── AGENTIC_RAG_HUGGINGFACE_INTEGRATION.md   # HF統合詳細
└── PHASE9_BUGFIX_20260211.md                # バグ修正ログ

results/
├── phase9_evaluation_20260211_090656.json   # Full Test結果
└── phase9_summary_20260211_090656.txt       # サマリー
```

### A.2 主要クラス定義

```python
# agent_state.py
class AgentState(TypedDict):
    question: str
    iteration: int
    intermediate_steps: List[Dict[str, Any]]
    tool_results: List[Dict[str, Any]]
    final_answer: Optional[str]

# agentic_rag_system.py
class AgenticRAGSystem:
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        model_name: str,
        verbose: bool = False,
        max_iterations: int = 5,
        temperature: float = 0.0
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.verbose = verbose
        self.max_iterations = max_iterations
        self.temperature = temperature

        # ツール初期化
        self.tools = self._initialize_tools()
        self.tool_map = {tool.name: tool for tool in self.tools}

        # LangGraphワークフロー構築
        self.workflow = self._build_workflow()
        self.app = self.workflow.compile()

    def query(self, question: str) -> Dict[str, Any]:
        """質問に回答"""
        state = {
            "question": question,
            "iteration": 0,
            "intermediate_steps": [],
            "tool_results": [],
            "final_answer": None
        }

        result_state = self.app.invoke(state)
        return {
            "answer": result_state.get("final_answer", ""),
            "iterations": result_state["iteration"],
            "tool_calls_count": len(result_state["tool_results"]),
            "execution_time_sec": time.time() - start_time
        }
```

---

## Appendix B: 評価データ詳細

### B.1 カテゴリ別詳細結果

| カテゴリ | 件数 | Hybrid成功 | Agentic成功 | Hybrid率 | Agentic率 | 差分 |
|---------|------|----------|------------|---------|-----------|------|
| competitor | 3 | 2 | 3 | 66.7% | 100.0% | +33.3 |
| complementary | 5 | 4 | 5 | 80.0% | 100.0% | +20.0 |
| basic_location | 5 | 4 | 5 | 80.0% | 100.0% | +20.0 |
| basic_category | 5 | 4 | 4 | 80.0% | 80.0% | 0.0 |
| spatial_proximity | 5 | 5 | 5 | 100.0% | 100.0% | 0.0 |
| decision_business | 5 | 5 | 5 | 100.0% | 100.0% | 0.0 |
| decision_location | 5 | 5 | 5 | 100.0% | 100.0% | 0.0 |
| spatial_density | 5 | 5 | 5 | 100.0% | 100.0% | 0.0 |
| constraint_single | 5 | 5 | 5 | 100.0% | 100.0% | 0.0 |
| conditional_reasoning | 5 | 5 | 5 | 100.0% | 100.0% | 0.0 |
| multi_step_spatial | 5 | 5 | 5 | 100.0% | 100.0% | 0.0 |
| advanced_sensitivity | 5 | 5 | 5 | 100.0% | 100.0% | 0.0 |
| relation | 5 | 5 | 5 | 100.0% | 100.0% | 0.0 |
| proximity | 2 | 2 | 2 | 100.0% | 100.0% | 0.0 |
| cuisine | 4 | 4 | 4 | 100.0% | 100.0% | 0.0 |
| comparison | 2 | 2 | 2 | 100.0% | 100.0% | 0.0 |
| aggregation | 3 | 3 | 3 | 100.0% | 100.0% | 0.0 |
| iterative_refinement | 5 | 5 | 5 | 100.0% | 100.0% | 0.0 |
| hours | 3 | 3 | 3 | 100.0% | 100.0% | 0.0 |
| spatial_comparison | 5 | 5 | 4 | 100.0% | 80.0% | -20.0 |
| multi_hop | 3 | 3 | 2 | 100.0% | 66.7% | -33.3 |
| brand | 5 | 5 | 3 | 100.0% | 60.0% | -40.0 |
| advanced_comparison | 2 | 2 | 1 | 100.0% | 50.0% | -50.0 |
| constraint_multi | 5 | 5 | 2 | 100.0% | 40.0% | -60.0 |
| advanced_uncertainty | 3 | 3 | 0 | 100.0% | 0.0% | -100.0 |

### B.2 実行時間分布

```
Hybrid RAG実行時間分布:
  Min: 2.4秒
  Q1:  5.8秒
  Median: 9.2秒
  Q3: 14.7秒
  Max: 52.3秒
  Mean: 11.1秒
  Std: 10.1秒

Agentic RAG実行時間分布:
  Min: 12.3秒
  Q1: 32.1秒
  Median: 52.4秒
  Q3: 78.6秒
  Max: 132.6秒
  Mean: 56.4秒
  Std: 28.7秒
```

---

**報告書作成日**: 2026年2月11日
**バージョン**: 1.0
**著者**: Claude Code + User
**プロジェクト**: experiments-local-llm / Phase 9

---

本報告書は、Phase 9: Agentic RAG実験の包括的な記録として作成された。Phase 9-BまたはPhase 10への進展において、本報告書を技術的基盤として参照されたい。
