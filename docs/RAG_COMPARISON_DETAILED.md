# RAGアプローチの本質的違い - 詳細解説

## 概要

Structured RAG、Adaptive RAG、Agentic RAGは、表面的には「ルールベースの構造的処理」という共通点がありますが、**決定のタイミングと主体**において本質的に異なります。

---

## 1. Structured RAG: 事前決定・並列実行

### アーキテクチャ図

```
[Question Input]
       ↓
┌─────────────────┐
│ Question        │ ← ルールベース分析（固定ロジック）
│ Analysis        │    ・正規表現マッチング
│ (Rule-based)    │    ・キーワード検出
└─────────────────┘
       ↓
  【一度に全部決定】
       ↓
┌──────────┬──────────┬──────────┐
│ Proximity│ Aggreg.  │ Compare  │ ← 並列実行
│ Context  │ Context  │ Context  │    （if文で共存）
└──────────┴──────────┴──────────┘
       ↓
┌─────────────────────────────────┐
│ Context Integration             │
│ (All contexts combined)         │
└─────────────────────────────────┘
       ↓
┌─────────────────────────────────┐
│ LLM Generation (1回のみ)        │
└─────────────────────────────────┘
       ↓
   [Answer]
```

### 実装例

```python
class StructuredRAG:
    def query(self, question: str):
        # ステップ1: 質問分析（ルールベース、固定）
        analysis = analyze_question(question)
        # analysis.requires_proximity = True
        # analysis.requires_aggregation = True
        # analysis.requires_comparison = False

        # ステップ2: 該当する処理を全て実行（並列的）
        contexts = []

        if analysis.requires_proximity:
            # 距離計算を実行
            contexts.append(self._execute_proximity())

        if analysis.requires_aggregation:
            # 集計処理を実行
            contexts.append(self._execute_aggregation())

        if analysis.requires_comparison:
            # 比較処理を実行
            contexts.append(self._execute_comparison())

        # ベクトル検索は常に実行（相補的）
        contexts.append(self._execute_vector_search())

        # ステップ3: コンテキスト統合
        final_context = "\n\n".join(contexts)

        # ステップ4: LLM生成（1回のみ）
        answer = self.llm.generate(final_context, question)
        return answer
```

### 特徴

**✅ 長所**:
- 高速（1回のLLM呼び出し）
- 予測可能（常に同じフロー）
- デバッグしやすい

**❌ 短所**:
- 柔軟性がない（新しいパターンに対応できない）
- 過剰な情報取得（必要以上のコンテキストを含む可能性）
- フィードバックループなし（結果を見て戦略変更できない）

### 実行例

**質問**: "渋谷駅から最も近いカフェで、東側にあるものは？"

```
[分析結果]
✓ requires_proximity = True  (「最も近い」を検出)
✓ requires_comparison = True  (「東側」を検出)
✓ requires_aggregation = False

[実行される処理]（全て並列）
1. proximity_search() → "Urth Caffé 56.7m"
2. east_west_comparison() → "東側: 45件, 西側: 52件"
3. vector_search() → 関連POI情報

[生成されるコンテキスト]
【最寄りカフェ（上位3件）】
  1. Urth Caffé - 56.7m（北東方向）
  2. スターバックス - 280m（北西方向）

【東西比較】
  東側: 45件
  西側: 52件

【検索結果】
  Urth Caffé カフェ おしゃれな...

[LLM生成]（1回）
→ "渋谷駅から最も近く東側にあるカフェはUrth Cafféです..."
```

---

## 2. Adaptive RAG: メタレベル決定・システム選択

### アーキテクチャ図

```
[Question Input]
       ↓
┌─────────────────┐
│ Query Complexity│ ← ルールベース分析（複雑度判定）
│ Analysis        │    ・クエリの複雑さを評価
│ (Meta-level)    │    ・必要なRAGシステムを決定
└─────────────────┘
       ↓
  【システム選択】
       ↓
    ┌─────┴─────┬─────┴─────┬─────┴─────┐
    ↓           ↓           ↓           ↓
┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
│ Naive   │ │Structured│ │ Graph   │ │ Hybrid  │
│ RAG     │ │ RAG     │ │ RAG     │ │(複数)   │
└─────────┘ └─────────┘ └─────────┘ └─────────┘
    ↓           ↓           ↓           ↓
    └───────────┴───────────┴───────────┘
                     ↓
              【結果統合】
                     ↓
                 [Answer]
```

### 実装例

```python
class AdaptiveRAG:
    def __init__(self):
        # 複数のRAGシステムを保持
        self.naive_rag = NaiveRAG()
        self.structured_rag = StructuredRAG()
        self.graph_rag = GraphRAG()

    def query(self, question: str):
        # ステップ1: クエリ複雑度分析（メタレベル）
        complexity = self._analyze_complexity(question)

        # ステップ2: 適切なシステムを選択
        if complexity.type == "simple":
            # シンプルなクエリ → NaiveRAG
            return self.naive_rag.query(question)

        elif complexity.type == "spatial":
            # 空間クエリ → StructuredRAG
            return self.structured_rag.query(question)

        elif complexity.type == "relationship":
            # 関係性クエリ → GraphRAG
            return self.graph_rag.query(question)

        elif complexity.type == "hybrid":
            # 複雑なクエリ → 複数システムを併用
            results = []
            results.append(self.structured_rag.query(question))
            results.append(self.graph_rag.query(question))

            # 結果をマージ
            return self._merge_results(results, question)

    def _analyze_complexity(self, question):
        """クエリの複雑度を分析（ルールベース）"""
        complexity = QueryComplexity()

        # ルール1: 関係性キーワード検出
        if any(word in question for word in ["同じエリア", "近くの", "ブランド"]):
            complexity.type = "relationship"
            complexity.score = 0.8

        # ルール2: 空間キーワード検出
        elif any(word in question for word in ["最も近い", "距離", "方向"]):
            complexity.type = "spatial"
            complexity.score = 0.6

        # ルール3: シンプルなクエリ
        else:
            complexity.type = "simple"
            complexity.score = 0.3

        return complexity
```

### 特徴

**✅ 長所**:
- クエリに応じた最適なシステム選択
- 各システムの強みを活かせる
- オーバーヘッドを最小化（シンプルなクエリは軽量処理）

**❌ 短所**:
- 選択ロジックが複雑になる
- システム間の境界が曖昧な場合に判断が難しい
- 各システムは依然として単一パス（フィードバックなし）

### 実行例

**質問1**: "渋谷のカフェを教えて"（Simple）

```
[複雑度分析]
type = "simple"
score = 0.3

[選択されたシステム]
→ NaiveRAG

[実行]
NaiveRAG.query("渋谷のカフェを教えて")
→ ベクトル検索のみ → LLM生成 → 回答
```

**質問2**: "渋谷駅から最も近いカフェは？"（Spatial）

```
[複雑度分析]
type = "spatial"
score = 0.6

[選択されたシステム]
→ StructuredRAG

[実行]
StructuredRAG.query("渋谷駅から最も近いカフェは？")
→ 距離計算 + ベクトル検索 → LLM生成 → 回答
```

**質問3**: "渋谷駅周辺でスターバックスと同じエリアにあるカフェは？"（Hybrid）

```
[複雑度分析]
type = "hybrid" (spatial + relationship)
score = 0.9

[選択されたシステム]
→ StructuredRAG + GraphRAG（両方）

[実行]
result1 = StructuredRAG.query(...) → "駅周辺のカフェリスト"
result2 = GraphRAG.query(...) → "同一エリアの関係性"

merge_results([result1, result2]) → 統合回答
```

---

## 3. Agentic RAG: 動的決定・逐次実行・反復改善

### アーキテクチャ図

```
[Question Input]
       ↓
┌─────────────────────────────┐
│ Agent Loop (Iteration 0)    │
├─────────────────────────────┤
│ LLM: "What should I do?"    │ ← LLMが判断
│   Thought: カフェ情報が必要  │
│   Action: tool_get_nearest   │
└─────────────────────────────┘
       ↓
┌─────────────────────────────┐
│ Tool Execution              │
│   → {"name": "Urth Caffé"}  │
└─────────────────────────────┘
       ↓
┌─────────────────────────────┐
│ Agent Loop (Iteration 1)    │
├─────────────────────────────┤
│ LLM: "Now I have café info" │ ← 結果を見て再判断
│   Observation: Urth Caffé   │
│   Thought: 東側か確認必要    │
│   Action: tool_check_direction│
└─────────────────────────────┘
       ↓
┌─────────────────────────────┐
│ Tool Execution              │
│   → {"direction": "北東"}   │
└─────────────────────────────┘
       ↓
┌─────────────────────────────┐
│ Agent Loop (Iteration 2)    │
├─────────────────────────────┤
│ LLM: "I have enough info"   │ ← 十分な情報と判断
│   Final Answer: ...         │
└─────────────────────────────┘
       ↓
   [Answer]
```

### 実装例

```python
class AgenticRAG:
    def query(self, question: str, max_iterations: int = 5):
        # 初期状態
        state = {
            "question": question,
            "iteration": 0,
            "tool_results": [],
            "intermediate_steps": []
        }

        # 反復ループ
        while state["iteration"] < max_iterations:
            # ステップ1: LLMに「次に何をすべきか」を尋ねる
            prompt = self._build_agent_prompt(state)
            llm_response = self.llm.generate(prompt)

            # ステップ2: LLMの応答を解析
            parsed = self._parse_response(llm_response)

            if parsed["type"] == "tool_call":
                # LLMがツール使用を決定
                tool_name = parsed["tool_name"]
                tool_args = parsed["tool_args"]

                # ステップ3: 指定されたツールを実行（1つのみ）
                tool_result = self._execute_tool(tool_name, tool_args)

                # ステップ4: 結果を状態に追加
                state["tool_results"].append({
                    "tool": tool_name,
                    "input": tool_args,
                    "output": tool_result
                })

                # ステップ5: 次のイテレーションへ
                state["iteration"] += 1
                # → ループの先頭に戻る（LLMが次の行動を決定）

            elif parsed["type"] == "final_answer":
                # LLMが最終回答を決定
                return parsed["answer"]

    def _build_agent_prompt(self, state):
        """現在の状態に基づいてプロンプトを構築"""
        prompt_parts = [
            f"Question: {state['question']}\n",
            "\nAvailable Tools:",
            "- tool_get_nearest_pois",
            "- tool_check_direction",
            "- tool_aggregate_by_category",
            "..."
        ]

        # これまでのツール実行結果を追加
        if state["tool_results"]:
            prompt_parts.append("\n\nPrevious Tool Executions:")
            for result in state["tool_results"]:
                prompt_parts.append(f"\nTool: {result['tool']}")
                prompt_parts.append(f"Output: {result['output']}")

        prompt_parts.append("\n\nWhat should you do next?")
        prompt_parts.append("Use ReAct format:")
        prompt_parts.append("Thought: [your reasoning]")
        prompt_parts.append("Action: [tool_name] OR")
        prompt_parts.append("Final Answer: [answer]")

        return "\n".join(prompt_parts)
```

### 特徴

**✅ 長所**:
- 動的な戦略変更（結果を見て次の行動を決定）
- 複雑な推論タスクに対応
- 予期しないクエリにも柔軟に対応

**❌ 短所**:
- 実行時間が長い（複数回のLLM呼び出し）
- 予測不可能（LLMの判断に依存）
- デバッグが困難（なぜそのツールを選んだのか不明瞭）

### 実行例（詳細）

**質問**: "渋谷駅から最も近く、東側にあるカフェは？"

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Iteration 0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[LLMへのプロンプト]
Question: 渋谷駅から最も近く、東側にあるカフェは？

Available Tools:
- tool_get_nearest_pois: 最寄りPOIを検索
- tool_check_direction: 方角を確認
- tool_filter_east_west: 東西でフィルタ

What should you do next?

[LLMの応答]
Thought: まず最も近いカフェを見つける必要がある
Action: tool_get_nearest_pois
Action Input: {"category": "カフェ", "top_n": 5}

[実行]
tool_get_nearest_pois({"category": "カフェ", "top_n": 5})
→ {
    "pois": [
      {"name": "Urth Caffé", "distance": 56.7},
      {"name": "スターバックス", "distance": 280.4},
      {"name": "ドトール", "distance": 300.1},
      ...
    ]
  }

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Iteration 1
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[LLMへのプロンプト]
Question: 渋谷駅から最も近く、東側にあるカフェは？

Available Tools: ...

Previous Tool Executions:
Tool: tool_get_nearest_pois
Output: {"pois": [{"name": "Urth Caffé", "distance": 56.7}, ...]}

What should you do next?

[LLMの応答]
Thought: 最寄りのカフェは見つかった。次は東側かどうか確認する
Action: tool_check_direction
Action Input: {"poi_name": "Urth Caffé"}

[実行]
tool_check_direction({"poi_name": "Urth Caffé"})
→ {"name": "Urth Caffé", "direction": "北東"}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Iteration 2
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[LLMへのプロンプト]
Question: 渋谷駅から最も近く、東側にあるカフェは？

Previous Tool Executions:
Tool: tool_get_nearest_pois
Output: {"pois": [{"name": "Urth Caffé", "distance": 56.7}, ...]}

Tool: tool_check_direction
Output: {"name": "Urth Caffé", "direction": "北東"}

What should you do next?

[LLMの応答]
Thought: Urth Cafféは北東なので東側に該当する。
        距離56.7mで最も近い。回答に十分な情報が揃った。
Final Answer: 渋谷駅から最も近く東側にあるカフェは、
             北東方向56.7mにある「Urth Caffé」です。

[終了]
```

**重要な観察**:
- LLMが各ステップで「次に何をすべきか」を判断
- 前回の結果を見て戦略を変更
- 十分な情報が揃ったと判断したら終了

---

## 4. 本質的な違いの要約

### 決定のタイミングと主体

| システム | 誰が決める？ | いつ決める？ | どう実行？ |
|---------|------------|------------|----------|
| **Structured RAG** | ルール（固定） | 事前（質問分析時） | 並列実行（全て同時） |
| **Adaptive RAG** | ルール（メタ） | 事前（システム選択時） | システム選択（1つまたは複数） |
| **Agentic RAG** | **LLM（動的）** | **実行中（各イテレーション）** | **逐次実行（1つずつ）** |

### 情報フロー

#### Structured RAG: 一方向フロー
```
Question → Analysis → [Parallel Execution] → Context → LLM → Answer
         ↑
      (決定は1回)                                    (生成も1回)
```

#### Adaptive RAG: 分岐フロー
```
Question → Complexity Analysis → System Selection
                                      ↓
                         ┌────────────┼────────────┐
                         ↓            ↓            ↓
                    NaiveRAG    StructuredRAG   GraphRAG
                         ↓            ↓            ↓
                         └────────────┴────────────┘
                                      ↓
                                   Answer
```

#### Agentic RAG: 反復フロー（フィードバックループ）
```
Question → ┌─────────────────────────┐
           │ LLM Decision            │ ← 状態を見て判断
           └────────┬────────────────┘
                    ↓
              Tool Execution
                    ↓
              Update State ────┐
                    ↓           │
              Enough Info?      │
                 No ────────────┘ (ループ)
                 Yes
                    ↓
              Final Answer
```

### 類似性の誤解を解く

**なぜ混同されるか**:
- 全て「ルール」を使用している
- 全て「構造化された処理」を行う
- 全て「ツール」を使用できる

**本質的な違い**:

1. **Structured RAG**:
   - ルールは「何を実行するか」を決める
   - 決定は事前に一度だけ
   - 例: "近接性キーワードあり → 距離計算を実行"

2. **Adaptive RAG**:
   - ルールは「どのシステムを使うか」を決める
   - 決定はメタレベル（システム選択）
   - 例: "空間クエリ → StructuredRAGを使用"

3. **Agentic RAG**:
   - LLMが「次に何をすべきか」を決める
   - 決定は実行中に複数回
   - 例: "結果を見たら追加情報が必要 → 別のツールを実行"

---

## 5. 実世界の例え

### Structured RAG = 料理レシピ

```
材料を見る（質問分析）
  ↓
必要な調理器具を全部出す（並列準備）
  - フライパン
  - 鍋
  - ボウル
  ↓
レシピ通りに調理（固定フロー）
  ↓
完成
```

**特徴**: 事前に全て決まっている、途中で変更不可

### Adaptive RAG = 料理人の選択

```
料理の種類を判断（和食 or 洋食 or 中華）
  ↓
担当する料理人を決定
  - 和食 → 和食専門シェフ
  - 洋食 → 洋食専門シェフ
  - 複雑な融合料理 → 複数のシェフで協力
  ↓
選ばれたシェフが調理（各シェフは通常の調理フロー）
  ↓
完成
```

**特徴**: メタレベルで選択、選択後は固定フロー

### Agentic RAG = 料理中の臨機応変な判断

```
材料を見る
  ↓
味見する → まだ足りない
  ↓
調味料を追加
  ↓
また味見する → もう少し
  ↓
別の調味料を追加
  ↓
また味見する → OK
  ↓
完成
```

**特徴**: 結果を見ながら次の行動を決定、フィードバックループ

---

## 6. パフォーマンス特性

### 実行時間比較（105ケース平均）

```
Structured RAG:  11.1秒/クエリ  ████░░░░░░░░░░░░░░░░
Adaptive RAG:    12.5秒/クエリ  █████░░░░░░░░░░░░░░░
Agentic RAG:     56.4秒/クエリ  ████████████████████ (5.1倍)
```

**なぜAgenticが遅いか**:
- Structured: LLM呼び出し 1回
- Adaptive: LLM呼び出し 1-2回（システム選択 + 生成）
- **Agentic: LLM呼び出し 2-5回**（各イテレーション + 最終生成）

### 精度比較（Phase 9 Full Test結果）

```
Structured RAG:  96.2% ████████████████████
Adaptive RAG:    86.1% █████████████████░░░
Agentic RAG:     87.6% █████████████████░░░
```

**なぜAgenticが期待より低いか**:
- 中国語混入問題（7.6%）
- 空回答（1.9%）
- 本来の性能ではない（Phase 9-Bで検証予定）

---

## 7. 適用場面

### Structured RAG が最適な場合

✅ クエリパターンが既知で分類可能
✅ 高速なレスポンスが必要
✅ 予測可能な動作が重要
✅ コスト効率を重視

**例**:
- "最寄りのカフェは？" → 近接性検索
- "カフェは何件？" → 集計
- "東西の比較は？" → 比較処理

### Adaptive RAG が最適な場合

✅ クエリの複雑度が多様
✅ 複数の専門システムを持っている
✅ システム選択のオーバーヘッドが許容できる
✅ 各クエリに最適なシステムを使いたい

**例**:
- シンプルなクエリ → 軽量なNaiveRAG
- 空間クエリ → StructuredRAG
- 関係性クエリ → GraphRAG

### Agentic RAG が最適な場合

✅ 予期しないクエリパターンに対応が必要
✅ 複雑な多段階推論が必要
✅ 実行時間のオーバーヘッドが許容できる
✅ 動的な戦略変更が価値を持つ

**例**:
- "最寄りのカフェで、かつ今営業中で、かつレビュー評価が高いものは？"
  → 状況を見ながら段階的にフィルタリング
- "渋谷で一番人気のカフェの近くにある本屋を教えて"
  → 中間結果を使って次の検索を実行

---

## 8. まとめ

### 決定的な違い

| 観点 | Structured | Adaptive | Agentic |
|------|-----------|----------|---------|
| **制御主体** | 固定ルール | メタルール | **LLM** |
| **決定回数** | 1回 | 1回 | **複数回** |
| **フィードバック** | なし | なし | **あり** |
| **実行パターン** | 並列 | 選択 | **逐次** |
| **柔軟性** | 低 | 中 | **高** |
| **速度** | **高速** | 中速 | 低速 |
| **予測可能性** | **高** | 中 | 低 |

### Phase 9で学んだこと

1. **Agentic RAGは万能ではない**
   - 適用場面が限定的
   - 実行時間とのトレードオフ

2. **Structured RAGの優位性**
   - 既知のパターンには最適
   - 高速で予測可能

3. **適切なアプローチの選択が重要**
   - Adaptive RAGのようなメタレベルの判断が現実的
   - または、Phase 9-Bでモデル変更後のAgentic RAGを再評価

---

**文書バージョン**: 1.0
**作成日**: 2026-02-11
**関連**: Phase 9実験、Issue #6
