# =============================================================================
# 修正版 Cell 13 - 評価関数の定義（属性名の互換性対応）
# =============================================================================
# このコードをCell 13に上書きしてください

def evaluate_keyword_hit_rate(answer: str, expected_keywords: List[str]) -> float:
    """キーワードヒット率を計算"""
    if not expected_keywords:
        return 1.0
    # None または空文字列のチェック
    if not answer:
        return 0.0
    answer_lower = answer.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    return hits / len(expected_keywords)

def evaluate_single_case(system, test_case, system_name: str) -> Dict[str, Any]:
    """単一テストケースの評価

    注: 異なるテストケースクラスの属性名に対応:
    - TestCaseV2, AgenticTestCase: .prompt
    - GraphRAGTestCase: .question
    """
    try:
        # 質問文を取得（promptまたはquestion属性）
        question = getattr(test_case, 'prompt', None) or getattr(test_case, 'question', '')

        start_time = time.time()

        # システム実行
        result = system.query(question)
        answer = result.get('answer', '') or ''  # None対策
        execution_time = time.time() - start_time

        # キーワードヒット率
        keyword_hit_rate = evaluate_keyword_hit_rate(answer, test_case.expected_keywords)

        # 成功判定（50%以上）
        success = keyword_hit_rate >= 0.5

        return {
            'test_id': test_case.id,
            'category': getattr(test_case, 'subcategory', test_case.category),
            'question': question,
            'answer': answer,
            'execution_time': execution_time,
            'keyword_hit_rate': keyword_hit_rate,
            'success': success,
            'error': None
        }
    except Exception as e:
        # エラー時も質問文を取得
        question = getattr(test_case, 'prompt', None) or getattr(test_case, 'question', '')

        return {
            'test_id': test_case.id,
            'category': getattr(test_case, 'subcategory', test_case.category),
            'question': question,
            'answer': '',
            'execution_time': 0,
            'keyword_hit_rate': 0.0,
            'success': False,
            'error': str(e)
        }

print("✓ Evaluation functions defined (with multi-format test case support)")
