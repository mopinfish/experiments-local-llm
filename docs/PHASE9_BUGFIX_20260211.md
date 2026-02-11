# Phase 9 評価バグ修正レポート

**日付**: 2026-02-11
**対象**: Phase 9 Agentic RAG評価の重大バグ修正

---

## 問題の概要

2026-02-04に実施されたPhase 9評価で以下の重大な問題が発生：

| システム | 成功率 | 主な問題 |
|---------|--------|----------|
| Structured RAG | 0% (0/10) | `AttributeError: 'query' method not found` |
| Agentic RAG | 10% (1/10) | `AttributeError: 'NoneType' has no attribute 'lower'` |

唯一成功したケースでも中国語で回答するという言語問題が発生。

---

## 修正内容

### 修正1: StructuredRAGSystemに`query()`メソッドを追加

**ファイル**: `src/structured_rag_system.py`

**問題**:
- 実装されているメソッド: `query_with_structured_rag()`
- 評価スクリプトが呼び出すメソッド: `query()` ← 存在しない

**修正**:
```python
def query(self, question: str) -> Dict[str, Any]:
    """
    質問に回答（評価用エイリアスメソッド）

    query_with_structured_rag()のエイリアス。
    評価スクリプトとの互換性のために提供。

    Args:
        question: 質問文

    Returns:
        回答と付随情報の辞書
    """
    return self.query_with_structured_rag(question)
```

**効果**: StructuredRAGSystemが正常に呼び出し可能に

---

### 修正2: 評価関数でNone値を処理

**ファイル**: `notebooks/phase9_agentic_rag_evaluation.ipynb` (Cell 13)

**問題**:
- `answer`がNoneの場合、`answer.lower()`でエラー
- ツール実行失敗時に`answer`が空またはNoneになる

**修正**:
```python
def evaluate_keyword_hit_rate(answer: str, expected_keywords: List[str]) -> float:
    """キーワードヒット率を計算"""
    if not expected_keywords:
        return 1.0
    # None または空文字列のチェック ← 追加
    if not answer:
        return 0.0
    answer_lower = answer.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    return hits / len(expected_keywords)

def evaluate_single_case(system, test_case, system_name: str) -> Dict[str, Any]:
    try:
        # ...
        answer = result.get('answer', '') or ''  # ← None対策追加
        # ...
```

**効果**: Noneエラーが発生しなくなり、失敗ケースも正しく記録

---

### 修正3: 日本語回答を強制

**ファイル**: `src/agent_prompts.py`

**問題**:
- 唯一成功したケースで中国語回答: "渋谷駅的坐标是东经139.701636..."
- プロンプトに日本語指示が不足

**修正**:

#### AGENT_SYSTEM_PROMPT の冒頭に追加:
```python
AGENT_SYSTEM_PROMPT = """あなたは渋谷駅周辺の地理空間POI（Point of Interest）情報に精通したエージェントです。
ユーザーの質問に答えるため、利用可能なツールを駆使して正確な情報を収集し、回答を生成してください。

**重要: すべての回答は必ず日本語で行ってください。中国語や英語で回答してはいけません。**

# あなたの役割
...
```

#### 回答形式セクションを強化:
```python
# 回答形式

回答は以下の形式で**必ず日本語で**生成してください：

1. **簡潔な答え**: 質問に対する直接的な回答（1-2文、日本語）
2. **詳細情報**: 必要に応じて具体的な数値、POI名、距離などを提示（日本語）
3. **根拠**: どのツールを使ってどのような情報を得たか簡潔に説明（日本語）
4. **追加情報**: 質問に直接関係しないが有用な情報があれば追加（日本語）
```

#### ANSWER_GENERATION_PROMPT_TEMPLATE に追加:
```python
ANSWER_GENERATION_PROMPT_TEMPLATE = """以下の質問に対して、ツール実行結果を元に最終回答を生成してください。

**重要: 回答は必ず日本語で行ってください。**

Question: {question}
...

回答生成ガイドライン：
1. 質問に直接答える簡潔な文から始める（日本語で）
2. 具体的な数値、POI名、距離などを提示する（日本語で）
3. 必要に応じて補足情報を追加する（日本語で）
4. 推測や不確かな情報は含めない
5. 中国語や英語ではなく、必ず日本語で回答する

Final Answer:
"""
```

**効果**: Qwen2.5モデルが日本語で回答するよう強制

---

## 修正後の期待結果

### 修正前（2026-02-04実行結果）
```
Structured RAG: 0.0% (全失敗)
Agentic RAG: 10.0% (9/10失敗、1件は中国語)
```

### 修正後（期待値）
```
Structured RAG: 85-90% (Phase 6実績ベース)
Agentic RAG: 85-92% (Phase 9目標値)
言語: 100%日本語
```

---

## 再評価手順

### 1. ファイルのアップロード

以下の修正済みファイルをGoogle Driveに再アップロード：
- `src/structured_rag_system.py`
- `src/agent_prompts.py`
- `notebooks/phase9_agentic_rag_evaluation.ipynb`

### 2. Colabで実行

1. ランタイムを再起動
2. Cell 2から順番に実行
3. Cell 11でシステム初期化確認
4. Cell 15でQuick test (10ケース)実行

### 3. 結果確認

Quick testで以下を確認：
- ✅ Structured RAGが正常動作（エラー0件）
- ✅ Agentic RAGが正常動作（エラー0-2件程度）
- ✅ すべての回答が日本語

問題なければFull test (105ケース)を実行。

---

## 技術的詳細

### なぜquery()メソッドがなかったのか

StructuredRAGSystemは当初`query_with_structured_rag()`という明示的な名前で実装されました。これはRAG有無を明確にするためでしたが、評価スクリプトとの互換性を考慮していませんでした。

AgenticRAGSystemは`query()`メソッドを持っていたため、評価スクリプトは両システムが同じインターフェースを持つと仮定していました。

### なぜ中国語で回答したのか

Qwen2.5-7B-Instructは多言語モデルで、中国語、英語、日本語をサポートしています。プロンプトに明示的な言語指示がない場合、モデルの学習データで最も多い中国語で回答する傾向があります。

日本語入力でも中国語で回答するのは、プロンプトエンジニアリングが不十分な場合に起こる既知の問題です。

---

## 学んだこと

1. **インターフェース統一の重要性**: 複数システムを比較評価する場合、共通インターフェース（`query()`）を事前に定義すべき

2. **防御的プログラミング**: 評価関数はNone、空文字列、予期しない型を常に考慮すべき

3. **多言語モデルの制御**: 出力言語は複数箇所で明示的に指定する必要がある
   - システムプロンプト
   - 回答生成プロンプト
   - 例示（Few-shot）

4. **早期テスト**: 1ケースでも実行して基本動作を確認してから本評価を実施すべき

---

## 次のステップ

1. ✅ 修正完了
2. ⏳ ファイル再アップロード
3. ⏳ Quick test実行
4. ⏳ Full test実行
5. ⏳ 結果分析とレポート作成
6. ⏳ Phase 9完了判定

---

**修正者**: Claude Code
**承認**: 待機中
**ステータス**: 修正完了、再評価待ち
