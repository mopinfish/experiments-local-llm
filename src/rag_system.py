#!/usr/bin/env python3
"""
rag_system.py - POI検索用RAGシステム

Google Colab上でHugging Face Transformersを使用したRAGシステム実装
"""
import time
from typing import List, Dict, Optional, Tuple
import torch

from .utils import CATEGORY_KEYWORDS, detect_category


class POI_RAG_System:
    """
    POI検索用RAGシステム（Google Colab版）
    
    Attributes:
        model: Transformersモデル
        tokenizer: トークナイザー
        vectorstore: ChromaDBベクトルストア
        debug: デバッグモード
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        vectorstore,
        debug: bool = False
    ):
        """
        RAGシステムを初期化
        
        Args:
            model: Hugging Face Transformersモデル
            tokenizer: トークナイザー
            vectorstore: LangChain ChromaDBベクトルストア
            debug: デバッグ出力を有効にする
        """
        self.model = model
        self.tokenizer = tokenizer
        self.vectorstore = vectorstore
        self.debug = debug
        
        # システムプロンプト
        self.system_prompt = """あなたは渋谷エリアの地理情報に詳しいアシスタントです。
提供された情報に基づいて、正確かつ簡潔に回答してください。
座標情報がある場合は必ず含めてください。
情報がない場合は「情報がありません」と正直に回答してください。"""
    
    def _generate_response(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.1
    ) -> str:
        """
        LLMでレスポンスを生成
        
        Args:
            prompt: 入力プロンプト
            max_new_tokens: 最大生成トークン数
            temperature: サンプリング温度
            
        Returns:
            生成されたテキスト
        """
        # チャット形式でプロンプト構築
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        # chat_template適用
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # トークナイズ
        inputs = self.tokenizer(text, return_tensors="pt").to("cuda")
        
        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # デコード
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # アシスタント応答部分を抽出
        if "assistant" in response.lower():
            parts = response.split("assistant")
            if len(parts) > 1:
                response = parts[-1].strip()
        
        return response
    
    def search_only(
        self,
        query: str,
        k: int = 5,
        use_category_filter: bool = True
    ) -> List[Dict]:
        """
        ベクトル検索のみ実行
        
        Args:
            query: 検索クエリ
            k: 取得件数
            use_category_filter: カテゴリフィルタリングを使用するか
            
        Returns:
            検索結果のリスト
        """
        # カテゴリ検出
        detected_categories, matched_keywords = detect_category(query)
        
        if self.debug and detected_categories:
            print(f"  検出カテゴリ: {detected_categories}")
            print(f"  マッチキーワード: {matched_keywords}")
        
        # 検索実行
        if use_category_filter and detected_categories:
            # 多めに取得してフィルタリング
            results = self.vectorstore.similarity_search(query, k=k * 2)
            filtered = [
                r for r in results 
                if r.metadata.get("category") in detected_categories
            ][:k]
            
            if filtered:
                results = filtered
            else:
                results = results[:k]
        else:
            results = self.vectorstore.similarity_search(query, k=k)
        
        # 結果を辞書形式に変換
        return [
            {
                "name": r.metadata.get("name", "不明"),
                "category": r.metadata.get("category", "不明"),
                "lat": r.metadata.get("lat"),
                "lon": r.metadata.get("lon"),
                "content": r.page_content,
                "metadata": r.metadata
            }
            for r in results
        ]
    
    def query_with_rag(
        self,
        question: str,
        k: int = 5,
        use_category_filter: bool = True
    ) -> Dict:
        """
        RAGを使用して質問に回答
        
        Args:
            question: 質問文
            k: 検索で取得する件数
            use_category_filter: カテゴリフィルタリングを使用するか
            
        Returns:
            回答と付随情報の辞書
        """
        start_time = time.time()
        
        # 検索実行
        search_results = self.search_only(question, k=k, use_category_filter=use_category_filter)
        
        # コンテキスト構築
        context = "\n\n".join([r["content"] for r in search_results])
        
        # プロンプト構築
        prompt = f"""以下の情報を参考にして、質問に回答してください。

【参考情報】
{context}

【質問】
{question}

【回答】
上記の情報を基に回答します。座標情報がある場合は必ず含めてください。"""
        
        # 回答生成
        answer = self._generate_response(prompt)
        
        elapsed_time = time.time() - start_time
        
        return {
            "answer": answer,
            "sources": search_results,
            "context": context,
            "time_ms": int(elapsed_time * 1000)
        }
    
    def query_without_rag(self, question: str) -> Dict:
        """
        RAGを使用せずに質問に回答（比較用）
        
        Args:
            question: 質問文
            
        Returns:
            回答と付随情報の辞書
        """
        start_time = time.time()
        
        prompt = f"""以下の質問に回答してください。

【質問】
{question}

【回答】"""
        
        answer = self._generate_response(prompt)
        
        elapsed_time = time.time() - start_time
        
        return {
            "answer": answer,
            "sources": [],
            "time_ms": int(elapsed_time * 1000)
        }
    
    def compare(self, question: str, verbose: bool = True) -> Dict:
        """
        RAGあり/なしの回答を比較
        
        Args:
            question: 質問文
            verbose: 詳細出力
            
        Returns:
            比較結果の辞書
        """
        if verbose:
            print(f"\n質問: {question}")
            print("=" * 50)
        
        # RAGなし
        if verbose:
            print("\n【RAGなし】処理中...")
        no_rag_result = self.query_without_rag(question)
        
        if verbose:
            print(f"回答: {no_rag_result['answer'][:300]}...")
            print(f"処理時間: {no_rag_result['time_ms']}ms")
        
        # RAGあり
        if verbose:
            print("\n【RAGあり】処理中...")
        rag_result = self.query_with_rag(question)
        
        if verbose:
            print(f"回答: {rag_result['answer'][:300]}...")
            print(f"処理時間: {rag_result['time_ms']}ms")
            print(f"参照POI数: {len(rag_result['sources'])}件")
        
        return {
            "question": question,
            "rag": rag_result,
            "no_rag": no_rag_result
        }
    
    def list_categories(self) -> Dict[str, int]:
        """
        ベクトルストア内のカテゴリ一覧を取得
        
        Returns:
            カテゴリと件数の辞書
        """
        # 全件取得（大規模データでは注意）
        all_docs = self.vectorstore.similarity_search("", k=10000)
        
        categories = {}
        for doc in all_docs:
            cat = doc.metadata.get("category", "不明")
            categories[cat] = categories.get(cat, 0) + 1
        
        return dict(sorted(categories.items(), key=lambda x: -x[1]))
    
    def search_by_category(self, category: str, limit: int = 10) -> List[Dict]:
        """
        カテゴリ指定で検索
        
        Args:
            category: カテゴリ名（部分一致）
            limit: 取得件数
            
        Returns:
            検索結果のリスト
        """
        # カテゴリに関連するクエリで検索
        results = self.vectorstore.similarity_search(category, k=limit * 3)
        
        # カテゴリでフィルタリング
        filtered = [
            {
                "name": r.metadata.get("name", "不明"),
                "category": r.metadata.get("category", "不明"),
                "lat": r.metadata.get("lat"),
                "lon": r.metadata.get("lon")
            }
            for r in results
            if category.lower() in r.metadata.get("category", "").lower()
        ][:limit]
        
        return filtered


def create_rag_system(
    model,
    tokenizer,
    poi_documents: List[Dict],
    embeddings,
    collection_name: str = "poi_collection",
    debug: bool = False
) -> POI_RAG_System:
    """
    RAGシステムを構築するヘルパー関数
    
    Args:
        model: Transformersモデル
        tokenizer: トークナイザー
        poi_documents: POIドキュメントのリスト
        embeddings: Embeddingモデル
        collection_name: ChromaDBコレクション名
        debug: デバッグモード
        
    Returns:
        初期化されたPOI_RAG_Systemインスタンス
    """
    from langchain_chroma import Chroma
    from langchain_core.documents import Document
    
    # LangChain Documentに変換
    documents = [
        Document(
            page_content=poi["content"],
            metadata=poi["metadata"]
        )
        for poi in poi_documents
    ]
    
    print(f"ベクトルストア構築中... ({len(documents)}件)")
    
    # ベクトルストア構築
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=collection_name
    )
    
    print(f"✅ ベクトルストア構築完了")
    
    # RAGシステム初期化
    rag_system = POI_RAG_System(
        model=model,
        tokenizer=tokenizer,
        vectorstore=vectorstore,
        debug=debug
    )
    
    print(f"✅ RAGシステム初期化完了")
    
    return rag_system


if __name__ == "__main__":
    print("RAGシステムモジュール")
    print("このモジュールはGoogle Colab上で使用することを想定しています。")
    print("単体での実行にはGPU環境とモデルのロードが必要です。")
