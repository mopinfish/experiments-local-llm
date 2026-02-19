# ========================================
# Debug Cell v3 - Test add_tool_result() Fix
# ========================================
# このセルをノートブックに貼り付けて実行してください
# 修正: add_tool_result()の署名を修正（4引数に変更）

# モジュールリロード
import importlib
import sys

# キャッシュクリア
if 'src.agentic_rag_system' in sys.modules:
    importlib.reload(sys.modules['src.agentic_rag_system'])
if 'src.agent_state' in sys.modules:
    importlib.reload(sys.modules['src.agent_state'])
if 'src.agent_tools' in sys.modules:
    importlib.reload(sys.modules['src.agent_tools'])
if 'src.agent_prompts' in sys.modules:
    importlib.reload(sys.modules['src.agent_prompts'])

print("✅ モジュールリロード完了\n")

# システム再初期化
from src.agentic_rag_system import AgenticRAGSystem

print("🔧 Agentic RAG System初期化中（verbose=True）...")
agentic_system = AgenticRAGSystem(
    chroma_path="chroma_db",
    verbose=True  # 詳細ログ有効
)
print("✅ 初期化完了\n")

# テストケース
test_question = "渋谷駅から最も近いカフェは？"
print(f"📝 テスト質問: {test_question}\n")
print("=" * 60)

# 実行
try:
    result = agentic_system.query(test_question)

    print("\n" + "=" * 60)
    print("✅ 実行成功!")
    print("=" * 60)
    print(f"\n【最終回答】")
    print(result.get('answer', 'No answer'))
    print(f"\n【実行時間】 {result.get('execution_time_sec', 0):.2f}秒")
    print(f"【ツール呼び出し回数】 {result.get('tool_calls_count', 0)}回")
    print(f"【イテレーション】 {result.get('iterations', 0)}回")

except Exception as e:
    print("\n" + "=" * 60)
    print("❌ エラー発生")
    print("=" * 60)
    print(f"エラー: {str(e)}")
    import traceback
    print("\nスタックトレース:")
    traceback.print_exc()
