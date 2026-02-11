"""
agent_prompts.py - Agentic RAG用プロンプトテンプレート

Phase 9: エージェントのシステムプロンプト、ツール説明、ReActスタイルプロンプト

作成日: 2026-02-03
プロジェクト: experiments-local-llm
"""

from typing import List, Dict, Any


# =============================================================================
# システムプロンプト
# =============================================================================

AGENT_SYSTEM_PROMPT = """あなたは渋谷駅周辺の地理空間POI（Point of Interest）情報に精通したエージェントです。
ユーザーの質問に答えるため、利用可能なツールを駆使して正確な情報を収集し、回答を生成してください。

**重要: すべての回答は必ず日本語で行ってください。中国語や英語で回答してはいけません。**

# あなたの役割

1. **質問理解**: ユーザーの質問を分析し、何を求めているか正確に理解する
2. **計画立案**: 質問に答えるために必要なツールと実行順序を決定する
3. **ツール実行**: 適切なツールを選択し、正しいパラメータで実行する
4. **結果統合**: 複数のツール実行結果を統合し、包括的な回答を生成する
5. **検証**: 回答が質問に適切に答えているか確認し、必要に応じて追加検索する

# 利用可能なツール群

## 空間計算ツール
- `tool_get_nearest_pois`: 渋谷駅から最も近いPOIを検索
- `tool_count_pois_in_radius`: 指定半径内のPOI件数を集計
- `tool_compare_radius`: 異なる半径での件数比較（感度分析）
- `tool_analyze_sensitivity`: 複数半径での詳細な感度分析
- `tool_filter_by_area`: 指定半径内のPOIリストを取得
- `tool_calculate_distance`: 2つのPOI間の距離を計算

## 比較・集計ツール
- `tool_compare_east_west`: 東西でPOI件数を比較
- `tool_compare_north_south`: 南北でPOI件数を比較
- `tool_count_by_category`: カテゴリ別にPOI件数を集計
- `tool_get_top_categories`: 件数上位のカテゴリランキング
- `tool_analyze_category_by_direction`: カテゴリの方向別詳細分析

## 検索ツール
- `tool_vector_search`: セマンティック検索でPOIを検索
- `tool_find_pois_by_keyword`: キーワードでPOIを検索

# 実行ガイドライン

1. **段階的思考**: 一度に全てを実行せず、段階的にツールを実行して情報を収集する
2. **適切なツール選択**: 質問のタイプに応じて最も適したツールを選択する
   - 「最も近い」「最寄り」→ `tool_get_nearest_pois`
   - 「何件」「いくつ」→ `tool_count_pois_in_radius` または `tool_count_by_category`
   - 「東西比較」→ `tool_compare_east_west`
   - 「半径を変えると」「範囲を広げると」→ `tool_compare_radius` または `tool_analyze_sensitivity`
3. **パラメータ検証**: ツールを呼び出す前にパラメータが適切か確認する
4. **結果検証**: ツール実行結果が期待通りか確認し、必要に応じて再実行する
5. **エラーハンドリング**: エラーが発生した場合は代替手段を試みる

# 回答形式

回答は以下の形式で**必ず日本語で**生成してください：

1. **簡潔な答え**: 質問に対する直接的な回答（1-2文、日本語）
2. **詳細情報**: 必要に応じて具体的な数値、POI名、距離などを提示（日本語）
3. **根拠**: どのツールを使ってどのような情報を得たか簡潔に説明（日本語）
4. **追加情報**: 質問に直接関係しないが有用な情報があれば追加（日本語）

# 注意事項

- 渋谷駅の座標は固定 (35.658034, 139.701636)
- 距離はメートル単位で表示
- カテゴリは部分一致で検索可能（例: "カフェ"で"飲食店/カフェ"が該当）
- 最大10回のツール実行まで（無限ループ防止）
- 確信が持てない場合は推測ではなく「不明」と回答する

それでは、ユーザーの質問に答えてください。
"""


REACT_PROMPT_TEMPLATE = """Question: {question}

あなたは以下のフォーマットで思考と行動を繰り返してください：

Thought: 質問に答えるために何をすべきか考える
Action: 実行するツール名
Action Input: ツールへの入力（JSON形式）
Observation: ツールからの出力

... (このThought/Action/Action Input/Observationを必要なだけ繰り返す)

Thought: 十分な情報が集まったので最終回答を生成できる
Final Answer: ユーザーへの最終回答

それでは始めてください。

Question: {question}
Thought:"""


PLANNING_PROMPT_TEMPLATE = """以下の質問に答えるための実行計画を立ててください。

Question: {question}

計画には以下を含めてください：
1. 質問の分析（何を求めているか）
2. 必要なツールのリスト
3. ツールの実行順序
4. 各ステップで得られる情報

計画をステップバイステップで記述してください：

Plan:
"""


VALIDATION_PROMPT_TEMPLATE = """以下のツール実行結果が質問に適切に答えているか検証してください。

Question: {question}

Tool Results:
{tool_results}

検証項目：
1. 質問に直接答える情報が含まれているか
2. 情報は十分か（追加のツール実行が必要か）
3. 情報に矛盾がないか
4. 情報は正確か

検証結果をJSON形式で返してください：
{{
  "is_sufficient": true/false,
  "is_accurate": true/false,
  "needs_additional_info": true/false,
  "additional_tools": ["tool_name1", "tool_name2"],
  "issues": ["issue1", "issue2"]
}}

Validation:
"""


ANSWER_GENERATION_PROMPT_TEMPLATE = """以下の質問に対して、ツール実行結果を元に最終回答を生成してください。

**重要: 回答は必ず日本語で行ってください。**

Question: {question}

Tool Results:
{tool_results}

回答生成ガイドライン：
1. 質問に直接答える簡潔な文から始める（日本語で）
2. 具体的な数値、POI名、距離などを提示する（日本語で）
3. 必要に応じて補足情報を追加する（日本語で）
4. 推測や不確かな情報は含めない
5. 中国語や英語ではなく、必ず日本語で回答する

Final Answer:
"""


# =============================================================================
# プロンプト生成ヘルパー関数
# =============================================================================

def format_react_prompt(question: str) -> str:
    """
    ReActスタイルのプロンプトを生成

    Args:
        question: ユーザーからの質問

    Returns:
        フォーマットされたプロンプト
    """
    return REACT_PROMPT_TEMPLATE.format(question=question)


def format_planning_prompt(question: str) -> str:
    """
    プランニング用プロンプトを生成

    Args:
        question: ユーザーからの質問

    Returns:
        フォーマットされたプロンプト
    """
    return PLANNING_PROMPT_TEMPLATE.format(question=question)


def format_validation_prompt(question: str, tool_results: List[Dict[str, Any]]) -> str:
    """
    検証用プロンプトを生成

    Args:
        question: ユーザーからの質問
        tool_results: ツール実行結果のリスト

    Returns:
        フォーマットされたプロンプト
    """
    # ツール実行結果を文字列化
    results_str = "\n\n".join([
        f"Tool: {result['tool']}\nInput: {result['input']}\nOutput: {result['output']}"
        for result in tool_results
    ])

    return VALIDATION_PROMPT_TEMPLATE.format(
        question=question,
        tool_results=results_str
    )


def format_answer_generation_prompt(question: str, tool_results: List[Dict[str, Any]]) -> str:
    """
    回答生成用プロンプトを生成

    Args:
        question: ユーザーからの質問
        tool_results: ツール実行結果のリスト

    Returns:
        フォーマットされたプロンプト
    """
    # ツール実行結果を文字列化
    results_str = "\n\n".join([
        f"Tool: {result['tool']}\nOutput: {result['output']}"
        for result in tool_results
    ])

    return ANSWER_GENERATION_PROMPT_TEMPLATE.format(
        question=question,
        tool_results=results_str
    )


# =============================================================================
# 質問タイプ別プロンプト
# =============================================================================

PROXIMITY_QUESTION_PROMPT = """この質問は「最寄り検索」タイプです。

推奨アプローチ：
1. `tool_get_nearest_pois` を使用してカテゴリで最寄りPOIを検索
2. 結果から距離と方向情報を抽出
3. 上位3件程度を提示

Question: {question}
"""


AGGREGATION_QUESTION_PROMPT = """この質問は「集計」タイプです。

推奨アプローチ：
1. カテゴリが指定されている場合は `tool_count_pois_in_radius` または `tool_count_by_category`
2. カテゴリ別ランキングの場合は `tool_get_top_categories`
3. 結果を数値で明確に提示

Question: {question}
"""


COMPARISON_QUESTION_PROMPT = """この質問は「比較」タイプです。

推奨アプローチ：
1. 東西比較の場合は `tool_compare_east_west`
2. 南北比較の場合は `tool_compare_north_south`
3. どちらが多いか、差は何件かを明確に提示

Question: {question}
"""


SENSITIVITY_QUESTION_PROMPT = """この質問は「感度分析」タイプです。

推奨アプローチ：
1. 半径の変化による件数変化を見る場合は `tool_compare_radius`
2. 詳細な分析が必要な場合は `tool_analyze_sensitivity`
3. 半径ごとの件数と変化率を提示

Question: {question}
"""


def get_question_type_prompt(question_type: str, question: str) -> str:
    """
    質問タイプに応じたプロンプトを取得

    Args:
        question_type: 質問のタイプ（"proximity", "aggregation", "comparison", "sensitivity"）
        question: ユーザーからの質問

    Returns:
        質問タイプに応じたプロンプト
    """
    prompts = {
        "proximity": PROXIMITY_QUESTION_PROMPT,
        "aggregation": AGGREGATION_QUESTION_PROMPT,
        "comparison": COMPARISON_QUESTION_PROMPT,
        "sensitivity": SENSITIVITY_QUESTION_PROMPT,
    }

    template = prompts.get(question_type, "")
    if template:
        return template.format(question=question)
    return ""


# =============================================================================
# ツール説明生成
# =============================================================================

def generate_tools_description() -> str:
    """
    利用可能なツールの説明を生成

    Returns:
        ツール説明の文字列
    """
    return """
利用可能なツール：

【空間計算ツール】
- tool_get_nearest_pois(category, top_n): 最寄りPOI検索
- tool_count_pois_in_radius(radius_m, category): 半径内POI件数
- tool_compare_radius(radius1_m, radius2_m, category): 半径比較
- tool_analyze_sensitivity(category, radii): 詳細感度分析
- tool_filter_by_area(radius_m, category): エリア内POIリスト
- tool_calculate_distance(poi_name1, poi_name2): 2点間距離

【比較・集計ツール】
- tool_compare_east_west(category): 東西比較
- tool_compare_north_south(category): 南北比較
- tool_count_by_category(categories): カテゴリ別集計
- tool_get_top_categories(top_n): カテゴリランキング
- tool_analyze_category_by_direction(category): 方向別分析

【検索ツール】
- tool_vector_search(query, k): セマンティック検索
- tool_find_pois_by_keyword(keyword): キーワード検索
"""


# =============================================================================
# テスト・デバッグ用
# =============================================================================

def test_prompts():
    """プロンプト生成のテスト"""
    print("=" * 60)
    print("agent_prompts.py 動作確認")
    print("=" * 60)

    question = "渋谷駅から最も近いカフェは？"

    print("\n--- ReActプロンプト ---")
    print(format_react_prompt(question))

    print("\n--- プランニングプロンプト ---")
    print(format_planning_prompt(question))

    print("\n--- 質問タイプ別プロンプト（proximity） ---")
    print(get_question_type_prompt("proximity", question))

    print("\n--- ツール説明 ---")
    print(generate_tools_description())

    print("\n✅ agent_prompts.py 動作確認完了")


if __name__ == "__main__":
    test_prompts()
