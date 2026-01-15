#!/usr/bin/env python3
"""
rag_system.py - OSM POI用RAGシステム (LangChain v1.x対応)

改善版:
- langchain-chromaへの移行
- デバッグ出力追加
- プロンプト強化（ハルシネーション抑制）
- RAGなし比較機能
"""
import json
import sys
from pathlib import Path
from typing import Optional

from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 設定
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen2.5:3b"
CHROMA_PERSIST_DIR = "./chroma_db"
POI_DOCUMENTS_PATH = "./poi_documents.json"

# デバッグモード
DEBUG = True


def load_documents(file_path: str) -> list:
    """POIドキュメントを読み込み"""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_vector_store(documents: list, persist_dir: str):
    """ベクトルストアを作成"""
    print("Embeddingモデルを初期化中...")
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    # テキストとメタデータを分離
    texts = [doc["content"] for doc in documents]
    metadatas = [doc["metadata"] for doc in documents]
    ids = [doc["id"] for doc in documents]

    print(f"ベクトルストアを構築中... ({len(texts)}件)")
    vectorstore = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        ids=ids,
        persist_directory=persist_dir
    )

    return vectorstore


def load_vector_store(persist_dir: str):
    """既存のベクトルストアを読み込み"""
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    return Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )


def format_docs(docs):
    """検索結果をフォーマット"""
    if not docs:
        return "関連するPOI情報が見つかりませんでした。"
    
    formatted = []
    for i, doc in enumerate(docs, 1):
        formatted.append(f"[POI {i}]\n{doc.page_content}")
    return "\n\n".join(formatted)


class POI_RAG_System:
    """POI RAGシステムクラス"""

    def __init__(self, rebuild: bool = False, debug: bool = DEBUG):
        self.debug = debug
        
        print("=" * 50)
        print("POI RAGシステム初期化")
        print("=" * 50)

        chroma_path = Path(CHROMA_PERSIST_DIR)

        if rebuild or not chroma_path.exists():
            # 新規構築
            print("\nドキュメントを読み込み中...")
            documents = load_documents(POI_DOCUMENTS_PATH)
            print(f"読み込んだドキュメント数: {len(documents)}")

            # 既存のDBを削除
            if chroma_path.exists():
                import shutil
                shutil.rmtree(chroma_path)
                print("既存のベクトルストアを削除しました")

            print("\nベクトルストアを構築中...")
            self.vectorstore = create_vector_store(documents, CHROMA_PERSIST_DIR)
            print("ベクトルストア構築完了!")
        else:
            # 既存を読み込み
            print("\n既存のベクトルストアを読み込み中...")
            self.vectorstore = load_vector_store(CHROMA_PERSIST_DIR)

        # コレクション内のドキュメント数を確認
        collection = self.vectorstore._collection
        doc_count = collection.count()
        print(f"ベクトルストア内のドキュメント数: {doc_count}")

        # Retriever
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )

        # LLM
        print("\nLLMを初期化中...")
        self.llm = OllamaLLM(
            model=LLM_MODEL, 
            temperature=0.0,  # より確定的な回答
            num_predict=512   # 回答の最大トークン数
        )

        # プロンプトテンプレート（ハルシネーション抑制強化版）
        prompt_template = """あなたは渋谷エリアのPOI（施設）情報を検索するアシスタントです。

【重要なルール】
1. 必ず以下の「検索結果」に含まれる情報のみを使用して回答してください
2. 検索結果にない情報は絶対に作成しないでください
3. 店舗名、住所、座標は検索結果からそのまま引用してください
4. 検索結果に該当する情報がない場合は「検索結果に該当するPOIが見つかりませんでした」と回答してください

【検索結果】
{context}

【ユーザーの質問】
{question}

【回答】
上記の検索結果に基づいて回答します："""

        self.prompt = PromptTemplate.from_template(prompt_template)

        # RAGチェーン (LCEL形式)
        self.rag_chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

        print("\n初期化完了!")
        print("=" * 50)

    def query_with_rag(self, question: str) -> dict:
        """RAGを使用して回答"""
        # まず検索結果を取得
        docs = self.retriever.invoke(question)
        
        if self.debug:
            print("\n" + "-" * 40)
            print(f"【デバッグ】検索結果 ({len(docs)}件):")
            print("-" * 40)
            for i, doc in enumerate(docs, 1):
                name = doc.metadata.get("name", "不明")
                category = doc.metadata.get("category", "不明")
                print(f"  [{i}] {name} ({category})")
            print("-" * 40)

        # コンテキストを作成
        context = format_docs(docs)
        
        if self.debug:
            print("\n【デバッグ】LLMに渡すコンテキスト（先頭500文字）:")
            print("-" * 40)
            print(context[:500] + "..." if len(context) > 500 else context)
            print("-" * 40)

        # プロンプトを作成して回答生成
        formatted_prompt = self.prompt.format(context=context, question=question)
        
        if self.debug:
            print("\n【デバッグ】実際のプロンプト（先頭800文字）:")
            print("-" * 40)
            print(formatted_prompt[:800] + "..." if len(formatted_prompt) > 800 else formatted_prompt)
            print("-" * 40)

        # 回答生成
        answer = self.llm.invoke(formatted_prompt)

        # ソース情報を整形
        sources = [
            {
                "name": doc.metadata.get("name", "不明"),
                "category": doc.metadata.get("category", "不明"),
                "lat": doc.metadata.get("lat"),
                "lon": doc.metadata.get("lon")
            }
            for doc in docs
        ]

        return {
            "answer": answer,
            "sources": sources,
            "context": context
        }

    def query_without_rag(self, question: str) -> dict:
        """RAGなしで回答（比較用）"""
        prompt = f"""あなたは渋谷エリアの情報に詳しいアシスタントです。
以下の質問に日本語で回答してください。

質問: {question}

回答:"""
        answer = self.llm.invoke(prompt)
        return {
            "answer": answer,
            "sources": []
        }

    def search_only(self, question: str, k: int = 5) -> list:
        """検索のみ実行（デバッグ用）"""
        docs = self.vectorstore.similarity_search(question, k=k)
        return [
            {
                "name": doc.metadata.get("name", "不明"),
                "category": doc.metadata.get("category", "不明"),
                "lat": doc.metadata.get("lat"),
                "lon": doc.metadata.get("lon"),
                "content": doc.page_content[:200] + "..."
            }
            for doc in docs
        ]


def interactive_mode(rag_system: POI_RAG_System):
    """対話モード"""
    print("\n" + "=" * 50)
    print("対話モード")
    print("=" * 50)
    print("コマンド:")
    print("  quit/exit/q - 終了")
    print("  compare <質問> - RAGあり/なしを比較")
    print("  search <質問> - 検索のみ実行")
    print("  debug on/off - デバッグモード切替")
    print("=" * 50)

    while True:
        try:
            user_input = input("\n質問: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ["quit", "exit", "q"]:
                print("終了します。")
                break

            # デバッグモード切替
            if user_input.lower() == "debug on":
                rag_system.debug = True
                print("デバッグモード: ON")
                continue
            elif user_input.lower() == "debug off":
                rag_system.debug = False
                print("デバッグモード: OFF")
                continue

            # 検索のみ
            if user_input.lower().startswith("search "):
                query = user_input[7:].strip()
                print(f"\n【検索のみ】「{query}」")
                results = rag_system.search_only(query)
                print(f"\n検索結果 ({len(results)}件):")
                for i, r in enumerate(results, 1):
                    print(f"\n[{i}] {r['name']}")
                    print(f"    カテゴリ: {r['category']}")
                    print(f"    座標: {r['lat']}, {r['lon']}")
                continue

            # 比較モード
            if user_input.lower().startswith("compare "):
                query = user_input[8:].strip()
                print(f"\n【比較モード】「{query}」")
                
                print("\n" + "=" * 40)
                print("【RAGなし】処理中...")
                no_rag_result = rag_system.query_without_rag(query)
                print(f"\n回答:\n{no_rag_result['answer']}")
                
                print("\n" + "=" * 40)
                print("【RAGあり】処理中...")
                rag_result = rag_system.query_with_rag(query)
                print(f"\n回答:\n{rag_result['answer']}")
                print(f"\n参照したPOI ({len(rag_result['sources'])}件):")
                for src in rag_result["sources"]:
                    print(f"  - {src['name']} ({src['category']}) [{src['lat']}, {src['lon']}]")
                continue

            # 通常のRAG質問
            print("\n【RAGあり】処理中...")
            result = rag_system.query_with_rag(user_input)
            print(f"\n回答:\n{result['answer']}")
            print(f"\n参照したPOI ({len(result['sources'])}件):")
            for src in result["sources"]:
                print(f"  - {src['name']} ({src['category']}) [{src['lat']}, {src['lon']}]")

        except KeyboardInterrupt:
            print("\n終了します。")
            break
        except Exception as e:
            print(f"\nエラー: {e}")
            if rag_system.debug:
                import traceback
                traceback.print_exc()


def main():
    """メイン関数"""
    # 引数解析
    rebuild = "--rebuild" in sys.argv
    no_debug = "--no-debug" in sys.argv
    debug = not no_debug

    # システム初期化
    rag_system = POI_RAG_System(rebuild=rebuild, debug=debug)

    # 対話モード
    interactive_mode(rag_system)


if __name__ == "__main__":
    main()
