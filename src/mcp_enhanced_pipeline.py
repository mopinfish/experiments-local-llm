"""
mcp_enhanced_pipeline.py - System A: Enhanced+Pipeline

構造化ツール付き MCP サーバーを使ったパイプライン方式。
質問分析に基づいてツールを決定的に呼び出し、構造化コンテキストを生成した上で
LLM に回答を生成させる。

Phase 10-A: MCP サーバー構造化ツール拡張実験
作成日: 2026-03-03
"""

import json
import time
from typing import Any, Dict, List, Optional

try:
    from .mcp_client import MCPClientWrapper
except ImportError:
    from mcp_client import MCPClientWrapper


# システムプロンプト
SYSTEM_PROMPT = """あなたは東京都内の主要駅周辺エリアの地理情報に詳しいアシスタントです。
提供されたデータに基づいて、以下の構造で回答してください。

# 回答の構造
1. **結論**: 質問への直接的な回答を最初に述べる
2. **根拠**: データから得られた具体的な証拠を引用する
3. **補足**: 注意点や不確実な点があれば述べる

# 回答ルール
- 推論過程を明示する: 「したがって」「比較すると」「分析すると」「なぜなら」等の論理接続詞を使い、結論に至る過程を示す
- 根拠を具体的に引用する: POI名、座標(緯度, 経度)、距離(m)、件数を提供データから引用し、「データから」「検索結果に基づき」等で出典を明記する
- 数値は単位付きで示す: 距離はm、件数は件、座標は(35.xxx, 139.xxx)の形式で記載する
- 比較表現を使う: 「より多い」「最も近い」「〜倍」等の比較表現で差異を明確にする
- 不確実性を正直に示す: データで確認できない点は「ただし」「データの限界として」「可能性があります」「データからは確認できません」等で明記する
- 情報がない場合は「提供データからは確認できません」と正直に回答する"""


class EnhancedMCPPipeline:
    """
    System A: Enhanced+Pipeline

    構造化ツール（geo_*）を質問分析に基づいて決定的に呼び出し、
    取得したコンテキストで LLM に回答生成させるパイプライン。
    """

    def __init__(
        self,
        mcp_client: MCPClientWrapper,
        model=None,
        tokenizer=None,
        debug: bool = False,
    ):
        """
        Args:
            mcp_client: MCP サーバーへの接続ラッパー
            model: Hugging Face モデル（Colab で設定）
            tokenizer: トークナイザー（Colab で設定）
            debug: デバッグ出力を有効化
        """
        self.mcp = mcp_client
        self.model = model
        self.tokenizer = tokenizer
        self.debug = debug

    def set_model(self, model, tokenizer):
        """モデルとトークナイザーを後から設定。"""
        self.model = model
        self.tokenizer = tokenizer

    async def query(self, question: str) -> Dict[str, Any]:
        """
        質問に回答する（パイプライン方式）。

        1. geo_analyze_question で質問分析
        2. 分析結果に基づいてツール呼び出し（決定的）
        3. LLM でコンテキストから回答生成

        Returns:
            {
                "answer": str,
                "context": str,
                "analysis": dict,
                "tool_calls": list,
                "time_sec": float,
            }
        """
        start = time.time()
        tool_calls = []

        # 1. 質問分析
        analysis_raw = await self.mcp.call_tool(
            "geo_analyze_question", {"question": question}
        )
        tool_calls.append({"tool": "geo_analyze_question", "result_len": len(analysis_raw)})

        try:
            analysis = json.loads(analysis_raw)
        except json.JSONDecodeError:
            analysis = {"question_type": "simple"}

        if self.debug:
            print(f"[分析] type={analysis.get('question_type')}, "
                  f"station={analysis.get('detected_station')}, "
                  f"category={analysis.get('detected_category')}")

        # 2. 分析結果に基づいてツール選択（決定的パイプライン）
        context_parts = []
        station_name = analysis.get("detected_station") or "渋谷駅"
        category = analysis.get("detected_category")

        # 近接性
        if analysis.get("requires_proximity") and category:
            try:
                radius = int(analysis.get("distance_constraint") or 1000)
                result = await self.mcp.call_tool("geo_nearest_pois", {
                    "station_name": station_name,
                    "category": category,
                    "radius": radius,
                    "top_n": 5,
                })
                context_parts.append(result)
                tool_calls.append({"tool": "geo_nearest_pois", "result_len": len(result)})
            except Exception as e:
                if self.debug:
                    print(f"  ツールエラー (geo_nearest_pois): {e}")

        # 感度分析
        if analysis.get("requires_sensitivity") and category:
            try:
                radii = analysis.get("sensitivity_radii") or [300, 500]
                result = await self.mcp.call_tool("geo_sensitivity_analysis", {
                    "station_name": station_name,
                    "category": category,
                    "radius1": radii[0],
                    "radius2": radii[1],
                })
                context_parts.append(result)
                tool_calls.append({"tool": "geo_sensitivity_analysis", "result_len": len(result)})
            except Exception as e:
                if self.debug:
                    print(f"  ツールエラー (geo_sensitivity_analysis): {e}")

        # 方角比較
        if analysis.get("requires_comparison") and analysis.get("detected_directions") and category:
            try:
                result = await self.mcp.call_tool("geo_compare_directions", {
                    "station_name": station_name,
                    "category": category,
                    "radius": int(analysis.get("distance_constraint") or 1000),
                })
                context_parts.append(result)
                tool_calls.append({"tool": "geo_compare_directions", "result_len": len(result)})
            except Exception as e:
                if self.debug:
                    print(f"  ツールエラー (geo_compare_directions): {e}")

        # 集約
        if analysis.get("requires_aggregation"):
            try:
                result = await self.mcp.call_tool("geo_count_by_category", {
                    "station_name": station_name,
                    "radius": int(analysis.get("distance_constraint") or 1000),
                })
                context_parts.append(result)
                tool_calls.append({"tool": "geo_count_by_category", "result_len": len(result)})
            except Exception as e:
                if self.debug:
                    print(f"  ツールエラー (geo_count_by_category): {e}")

        # 3. フォールバック: 基本周辺検索
        if not context_parts:
            try:
                coords = _StationsCoords.get(station_name)
                if not coords:
                    coords = {"lat": 35.658034, "lon": 139.701636}  # 渋谷デフォルト

                search_args = {
                    "lon": coords["lon"],
                    "lat": coords["lat"],
                    "radius": int(analysis.get("distance_constraint") or 1000),
                    "num_results": 20,
                }
                if category:
                    search_args["genre_name"] = category

                result = await self.mcp.call_tool(
                    "mapfan_search_spot_area", search_args
                )
                context_parts.append(result)
                tool_calls.append({"tool": "mapfan_search_spot_area", "result_len": len(result)})
            except Exception as e:
                if self.debug:
                    print(f"  ツールエラー (mapfan_search_spot_area): {e}")

        # 4. LLM 応答生成
        context = "\n\n".join(context_parts)
        answer = self._generate_response(question, context)

        elapsed = time.time() - start
        return {
            "answer": answer,
            "context": context,
            "analysis": analysis,
            "tool_calls": tool_calls,
            "time_sec": round(elapsed, 2),
        }

    def _generate_response(self, question: str, context: str) -> str:
        """LLM で回答を生成。"""
        if self.model is None or self.tokenizer is None:
            return f"[モデル未設定] コンテキスト:\n{context}"

        prompt = f"""以下のデータに基づいて質問に回答してください。

## 検索データ
{context}

## 質問
{question}

## 回答"""

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        try:
            import torch

            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer(text, return_tensors="pt").to("cuda")

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # アシスタント応答部分を抽出
            if "assistant" in response.lower():
                parts = response.split("assistant")
                if len(parts) > 1:
                    response = parts[-1].strip()

            return response

        except Exception as e:
            return f"[LLM生成エラー: {e}]\nコンテキスト:\n{context}"


# フォールバック用の駅座標（MCP サーバーが接続できない場合のローカル参照）
class _StationsCoords:
    """フォールバック用駅座標。"""

    COORDS = {
        "渋谷駅": {"lat": 35.658034, "lon": 139.701636},
        "新宿駅": {"lat": 35.689607, "lon": 139.700571},
        "池袋駅": {"lat": 35.729503, "lon": 139.710999},
        "東京駅": {"lat": 35.681236, "lon": 139.767125},
    }

    @classmethod
    def get(cls, name: str) -> Optional[Dict[str, float]]:
        if name in cls.COORDS:
            return cls.COORDS[name]
        for k, v in cls.COORDS.items():
            if k.replace("駅", "") in name or name in k:
                return v
        return None
