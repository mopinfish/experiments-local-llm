#!/usr/bin/env python3
"""
adaptive_rag_system.py - Adaptive RAG System

質問タイプに応じてGraphRAGと構造化RAGを動的に切り替えるシステム。
拡張GraphRAG比較評価の結果に基づき、各システムの得意分野に応じてルーティングを行う。

適性マップ（90テストケース評価結果より）:
- GraphRAG優位: comparison (+25%), proximity (+16.6%), hours (+11.1%), aggregation (+8.3%)
- 構造化RAG優位: relation (-23.4%), multi_hop (-22.2%), brand (-20%), basic_retrieval (-14.2%), cuisine (-8.3%)
- 同等: advanced_reasoning, competitor, complementary, constraint_satisfaction, decision_support, spatial_reasoning
"""

from dataclasses import dataclass
from typing import Optional, Literal, Dict, Any, List
import time

# Import both RAG systems
try:
    from .structured_rag_system import StructuredRAGSystem, QuestionAnalysis, analyze_question
    from .graph_rag_system import GraphRAGSystem
    from .graph_builder import POIGraphBuilder
except ImportError:
    from structured_rag_system import StructuredRAGSystem, QuestionAnalysis, analyze_question
    from graph_rag_system import GraphRAGSystem
    from graph_builder import POIGraphBuilder


@dataclass
class AdaptiveRAGResult:
    """Adaptive RAGの応答結果"""
    response: str
    context: str
    selected_system: Literal["GraphRAG", "StructuredRAG"]
    selection_reason: str
    question_analysis: Dict[str, Any]
    query_time: float
    llm_time: float
    total_time: float


class AdaptiveRAGSystem:
    """
    質問タイプに応じてGraphRAGと構造化RAGを動的に切り替えるシステム
    """

    # GraphRAGが優位なクエリタイプ（評価結果より）
    GRAPHRAG_PREFERRED_TYPES = {
        "comparison": 25.0,      # 東西比較など
        "proximity": 16.6,       # 近接性検索
        "hours": 11.1,           # 営業時間検索
        "aggregation": 8.3,      # 集計クエリ
    }

    # 構造化RAGが優位なクエリタイプ
    STRUCTURED_PREFERRED_TYPES = {
        "relation": 23.4,        # POI間関係
        "multi_hop": 22.2,       # 複数ステップ推論
        "brand": 20.0,           # ブランド検索
        "basic_retrieval": 14.2, # 基本検索
        "cuisine": 8.3,          # 料理ジャンル検索
    }

    def __init__(
        self,
        model,
        tokenizer,
        vectorstore,
        all_pois: List[Dict[str, Any]],
        include_extended_edges: bool = True,
        verbose: bool = False
    ):
        """
        Args:
            model: Hugging Face Transformersモデル
            tokenizer: トークナイザー
            vectorstore: LangChain ChromaDBベクトルストア
            all_pois: 全POIデータ
            include_extended_edges: 拡張エッジを含むか（GraphRAG用）
            verbose: 詳細ログを出力するか
        """
        self.model = model
        self.tokenizer = tokenizer
        self.vectorstore = vectorstore
        self.all_pois = all_pois
        self.include_extended_edges = include_extended_edges
        self.verbose = verbose

        self._log("Adaptive RAG System 初期化中...")

        # 両システムの初期化（遅延ロード用にNoneで初期化）
        self._structured_rag: Optional[StructuredRAGSystem] = None
        self._graph_rag: Optional[GraphRAGSystem] = None

        self._log("Adaptive RAG System 初期化完了")

    def _log(self, message: str):
        """詳細ログ出力"""
        if self.verbose:
            print(f"[AdaptiveRAG] {message}")

    @property
    def structured_rag(self) -> StructuredRAGSystem:
        """構造化RAGシステム（遅延ロード）"""
        if self._structured_rag is None:
            self._log("構造化RAGシステムを初期化中...")
            self._structured_rag = StructuredRAGSystem(
                model=self.model,
                tokenizer=self.tokenizer,
                vectorstore=self.vectorstore,
                all_pois=self.all_pois,
                debug=self.verbose
            )
        return self._structured_rag

    @property
    def graph_rag(self) -> GraphRAGSystem:
        """GraphRAGシステム（遅延ロード）"""
        if self._graph_rag is None:
            self._log("GraphRAGシステムを初期化中...")
            # POIGraphBuilderでグラフを構築
            builder = POIGraphBuilder()
            graph = builder.build_graph(
                self.all_pois,
                include_extended_edges=self.include_extended_edges,
                verbose=self.verbose
            )
            # 構築済みグラフをGraphRAGSystemに渡す
            self._graph_rag = GraphRAGSystem(graph_or_pois=graph)
            # GraphRAGにもモデル・トークナイザーを設定
            self._graph_rag.model = self.model
            self._graph_rag.tokenizer = self.tokenizer
        return self._graph_rag

    def select_system(self, question: str, analysis: Optional[QuestionAnalysis] = None) -> tuple[Literal["GraphRAG", "StructuredRAG"], str]:
        """
        質問に最適なRAGシステムを選択

        Args:
            question: 質問文
            analysis: 質問分析結果（省略時は自動分析）

        Returns:
            (選択されたシステム名, 選択理由)
        """
        if analysis is None:
            analysis = analyze_question(question)

        # GraphRAG優位な条件をチェック
        graphrag_reasons = []

        # comparison: 東西比較など
        if analysis.requires_comparison:
            graphrag_reasons.append("東西/方向比較クエリ (+25%)")

        # proximity: 近接性検索
        if analysis.requires_proximity:
            graphrag_reasons.append("近接性検索クエリ (+16.6%)")

        # hours: 営業時間検索（キーワードベース）
        hours_keywords = ["24時間", "深夜", "早朝", "営業時間", "何時まで", "何時から"]
        if any(kw in question for kw in hours_keywords):
            graphrag_reasons.append("営業時間クエリ (+11.1%)")

        # aggregation: 集計クエリ
        if analysis.requires_aggregation:
            graphrag_reasons.append("集計クエリ (+8.3%)")

        # GraphRAG優位な条件が1つでもあれば GraphRAG を選択
        if graphrag_reasons:
            return "GraphRAG", "、".join(graphrag_reasons)

        # それ以外は構造化RAGを使用（デフォルト）
        structured_reasons = []

        # brand: ブランド検索
        brand_keywords = ["チェーン", "ブランド", "店舗数", "何店舗"]
        if any(kw in question for kw in brand_keywords):
            structured_reasons.append("ブランド検索 (+20%)")

        # multi_hop: 複数ステップ推論（キーワードベース）
        multi_hop_keywords = ["から徒歩", "を経由", "から最寄り", "の近くの"]
        if any(kw in question for kw in multi_hop_keywords):
            structured_reasons.append("複数ステップ推論 (+22.2%)")

        # cuisine: 料理ジャンル検索
        cuisine_keywords = ["料理", "ジャンル", "和食", "洋食", "中華", "イタリアン", "フレンチ"]
        if any(kw in question for kw in cuisine_keywords):
            structured_reasons.append("料理ジャンル検索 (+8.3%)")

        if structured_reasons:
            return "StructuredRAG", "、".join(structured_reasons)

        # デフォルト: 構造化RAG（全体的に安定した性能）
        return "StructuredRAG", "デフォルト選択（安定した性能）"

    def query(self, question: str) -> AdaptiveRAGResult:
        """
        質問に回答

        Args:
            question: 質問文

        Returns:
            AdaptiveRAGResult
        """
        start_time = time.time()

        # 質問分析
        analysis = analyze_question(question)
        analysis_dict = {
            "question_type": analysis.question_type,
            "categories": analysis.categories,
            "requires_proximity": analysis.requires_proximity,
            "requires_comparison": analysis.requires_comparison,
            "requires_aggregation": analysis.requires_aggregation,
            "requires_sensitivity": analysis.requires_sensitivity,
        }

        # システム選択
        selected_system, selection_reason = self.select_system(question, analysis)
        self._log(f"選択システム: {selected_system} ({selection_reason})")

        # 選択されたシステムで回答
        query_start = time.time()

        if selected_system == "GraphRAG":
            result = self.graph_rag.query(question)
            response = result.response
            context = result.context
            query_time = result.query_time
            llm_time = result.llm_time
        else:
            result = self.structured_rag.query(question)
            response = result.response
            context = result.context
            query_time = result.query_time
            llm_time = result.llm_time

        total_time = time.time() - start_time

        return AdaptiveRAGResult(
            response=response,
            context=context,
            selected_system=selected_system,
            selection_reason=selection_reason,
            question_analysis=analysis_dict,
            query_time=query_time,
            llm_time=llm_time,
            total_time=total_time
        )

    def query_with_system(
        self,
        question: str,
        system: Literal["GraphRAG", "StructuredRAG", "Adaptive"]
    ) -> AdaptiveRAGResult:
        """
        指定されたシステムで回答（比較評価用）

        Args:
            question: 質問文
            system: 使用するシステム

        Returns:
            AdaptiveRAGResult
        """
        start_time = time.time()

        # 質問分析
        analysis = analyze_question(question)
        analysis_dict = {
            "question_type": analysis.question_type,
            "categories": analysis.categories,
            "requires_proximity": analysis.requires_proximity,
            "requires_comparison": analysis.requires_comparison,
            "requires_aggregation": analysis.requires_aggregation,
            "requires_sensitivity": analysis.requires_sensitivity,
        }

        # システム選択
        if system == "Adaptive":
            selected_system, selection_reason = self.select_system(question, analysis)
        else:
            selected_system = system
            selection_reason = "手動指定"

        # 選択されたシステムで回答
        if selected_system == "GraphRAG":
            result = self.graph_rag.query(question)
        else:
            result = self.structured_rag.query(question)

        total_time = time.time() - start_time

        return AdaptiveRAGResult(
            response=result.response,
            context=result.context,
            selected_system=selected_system,
            selection_reason=selection_reason,
            question_analysis=analysis_dict,
            query_time=result.query_time,
            llm_time=result.llm_time,
            total_time=total_time
        )

    def get_system_selection_stats(self, questions: List[str]) -> Dict[str, Any]:
        """
        質問リストに対するシステム選択統計を取得

        Args:
            questions: 質問リスト

        Returns:
            選択統計
        """
        stats = {
            "total": len(questions),
            "graphrag_selected": 0,
            "structured_selected": 0,
            "selection_reasons": {}
        }

        for question in questions:
            selected, reason = self.select_system(question)
            if selected == "GraphRAG":
                stats["graphrag_selected"] += 1
            else:
                stats["structured_selected"] += 1

            if reason not in stats["selection_reasons"]:
                stats["selection_reasons"][reason] = 0
            stats["selection_reasons"][reason] += 1

        stats["graphrag_ratio"] = stats["graphrag_selected"] / stats["total"] * 100
        stats["structured_ratio"] = stats["structured_selected"] / stats["total"] * 100

        return stats


def main():
    """テスト実行"""
    print("=" * 60)
    print("Adaptive RAG System テスト")
    print("=" * 60)

    # テスト質問
    test_questions = [
        # GraphRAG優位が期待されるクエリ
        ("渋谷駅の東側と西側で、レストランが多いのはどちらですか？", "comparison"),
        ("渋谷駅に最も近いカフェはどこですか？", "proximity"),
        ("渋谷で24時間営業の店舗を教えてください", "hours"),
        ("渋谷の飲食店は全部で何件ありますか？", "aggregation"),

        # 構造化RAG優位が期待されるクエリ
        ("渋谷にあるスターバックスは何店舗ありますか？", "brand"),
        ("渋谷で日本料理のレストランを探しています", "cuisine"),
        ("渋谷駅周辺の観光スポットを教えてください", "basic_retrieval"),
    ]

    # システム選択のみテスト（実際のクエリは実行しない）
    print("\n【システム選択テスト】")
    print("-" * 60)

    for question, expected_type in test_questions:
        analysis = analyze_question(question)

        # 仮のシステムでselect_systemを呼び出し
        graphrag_reasons = []

        if analysis.requires_comparison:
            graphrag_reasons.append("comparison")
        if analysis.requires_proximity:
            graphrag_reasons.append("proximity")
        if analysis.requires_aggregation:
            graphrag_reasons.append("aggregation")

        hours_keywords = ["24時間", "深夜", "早朝", "営業時間"]
        if any(kw in question for kw in hours_keywords):
            graphrag_reasons.append("hours")

        if graphrag_reasons:
            selected = "GraphRAG"
            reason = ", ".join(graphrag_reasons)
        else:
            selected = "StructuredRAG"
            reason = "default"

        print(f"質問: {question[:40]}...")
        print(f"  期待: {expected_type} → 選択: {selected} ({reason})")
        print()


if __name__ == "__main__":
    main()
