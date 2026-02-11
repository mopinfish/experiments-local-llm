#!/usr/bin/env python3
"""
test_agentic_huggingface.py - HuggingFace統合のテスト

Phase 9: Agentic RAGのHuggingFace Transformers統合を検証

作成日: 2026-02-03
"""

import sys
sys.path.insert(0, 'src')

from agentic_rag_system import initialize_system

def main():
    """メイン実行"""
    print("=" * 60)
    print("Agentic RAG HuggingFace Integration Test")
    print("=" * 60)
    print()

    # システム初期化（モデルは自動ロード）
    print("Initializing Agentic RAG system...")
    print("Note: This will download Qwen2.5-7B-Instruct (~4GB 4-bit model)")
    print("      and load it with 4-bit quantization.")
    print()

    system = initialize_system(
        poi_file="poi_documents.json",
        model_name="Qwen/Qwen2.5-7B-Instruct",
        verbose=True,
        load_in_4bit=True
    )

    print("\n" + "=" * 60)
    print("Testing with sample questions")
    print("=" * 60)

    # テスト質問
    test_questions = [
        "渋谷駅から最も近いカフェは？",
        "カフェは東側と西側でどちらが多いですか？",
    ]

    for i, question in enumerate(test_questions, 1):
        print(f"\n[Test {i}/{len(test_questions)}]")
        print(f"Question: {question}")
        print("-" * 60)

        result = system.query(question)

        print(f"Answer: {result['answer']}")
        print(f"Iterations: {result['iterations']}")
        print(f"Execution time: {result['execution_time']}s")
        print(f"Tools used: {len(result['tool_results'])}")

        if result.get('error'):
            print(f"⚠ Error: {result['error']}")

    print("\n" + "=" * 60)
    print("✓ Test completed successfully")
    print("=" * 60)


if __name__ == "__main__":
    main()
