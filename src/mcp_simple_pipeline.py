"""
mcp_simple_pipeline.py - System B: Simple+Pipeline

既存 MCP ツール（mapfan_search_spot_area 等）のみを使うシンプルパイプライン。
構造化ツール（geo_*）を使わず、基本的な周辺検索のみでコンテキストを構築し
LLM に回答生成させる。System A との比較用ベースライン。

Phase 10-A: MCP サーバー構造化ツール拡張実験
作成日: 2026-03-03
"""

import re
import time
from typing import Any, Dict, List, Optional

try:
    from .mcp_client import MCPClientWrapper
except ImportError:
    from mcp_client import MCPClientWrapper


# システムプロンプト（Enhanced と同一）
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


# 駅座標（ローカル定数、MCP サーバーに依存しない）
STATIONS = {
    "渋谷駅": {"lat": 35.658034, "lon": 139.701636},
    "渋谷": {"lat": 35.658034, "lon": 139.701636},
    "新宿駅": {"lat": 35.689607, "lon": 139.700571},
    "新宿": {"lat": 35.689607, "lon": 139.700571},
    "池袋駅": {"lat": 35.729503, "lon": 139.710999},
    "池袋": {"lat": 35.729503, "lon": 139.710999},
    "東京駅": {"lat": 35.681236, "lon": 139.767125},
    "東京": {"lat": 35.681236, "lon": 139.767125},
}

# 簡易ジャンルマッピング（主要カテゴリのみ）
SIMPLE_GENRE_MAP = {
    "カフェ": "喫茶店・カフェ",
    "コーヒー": "喫茶店・カフェ",
    "喫茶店": "喫茶店・カフェ",
    "コンビニ": "コンビニエンスストア",
    "レストラン": "ファミリーレストラン",
    "ファストフード": "ファストフード",
    "ラーメン": "ラーメン店",
    "寿司": "寿司屋",
    "中華": "中華料理店",
    "焼肉": "焼肉・韓国料理店",
    "居酒屋": "居酒屋",
    "薬局": "薬局",
    "ドラッグストア": "薬局",
    "銀行": "銀行",
    "ATM": "ATM",
    "病院": "病院",
    "公園": "公園",
    "ホテル": "宿泊施設",
    "映画館": "映画館",
    "カラオケ": "カラオケボックス",
    "スーパー": "スーパーマーケット",
    "駐車場": "駐車場",
}


class SimpleMCPPipeline:
    """
    System B: Simple+Pipeline

    既存ツール（mapfan_search_spot_area）のみを使うベースライン。
    """

    def __init__(
        self,
        mcp_client: MCPClientWrapper,
        model=None,
        tokenizer=None,
        debug: bool = False,
    ):
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
        質問に回答する（シンプルパイプライン方式）。

        1. 駅名キーワードで駅座標を特定
        2. ジャンル名を推定
        3. mapfan_search_spot_area で基本検索
        4. LLM で回答生成

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

        # 1. 簡易エリア検出
        station_name, coords = self._detect_station(question)

        # 2. 簡易ジャンル推定
        genre_name = self._detect_genre(question)

        # 3. 距離制約の抽出
        radius = self._detect_radius(question) or 1000

        analysis = {
            "question_type": "simple",
            "detected_station": station_name,
            "detected_genre": genre_name,
            "radius": radius,
        }

        if self.debug:
            print(f"[Simple] station={station_name}, genre={genre_name}, radius={radius}")

        # 4. 基本検索
        search_args: Dict[str, Any] = {
            "lon": coords["lon"],
            "lat": coords["lat"],
            "radius": radius,
            "num_results": 20,
        }
        if genre_name:
            search_args["genre_name"] = genre_name

        try:
            result = await self.mcp.call_tool("mapfan_search_spot_area", search_args)
            tool_calls.append({
                "tool": "mapfan_search_spot_area",
                "result_len": len(result),
            })
        except Exception as e:
            if self.debug:
                print(f"  ツールエラー (mapfan_search_spot_area): {e}")
            result = ""

        # 5. LLM 応答生成
        context = result
        answer = self._generate_response(question, context)

        elapsed = time.time() - start
        return {
            "answer": answer,
            "context": context,
            "analysis": analysis,
            "tool_calls": tool_calls,
            "time_sec": round(elapsed, 2),
        }

    def _detect_station(self, question: str) -> tuple:
        """質問から駅を検出。"""
        for name, coords in STATIONS.items():
            if name in question:
                return (name, coords)
        # デフォルト: 渋谷駅
        return ("渋谷駅", STATIONS["渋谷駅"])

    def _detect_genre(self, question: str) -> Optional[str]:
        """質問からジャンル名を推定。"""
        # 長いキーワードから優先マッチ
        for keyword in sorted(SIMPLE_GENRE_MAP.keys(), key=len, reverse=True):
            if keyword in question:
                return SIMPLE_GENRE_MAP[keyword]
        return None

    def _detect_radius(self, question: str) -> Optional[int]:
        """質問から距離制約を抽出。"""
        for pattern, multiplier in [
            (r'(\d+)\s*(m|メートル)', 1),
            (r'(\d+)\s*(km|キロ)', 1000),
            (r'徒歩\s*(\d+)\s*分', 80),
        ]:
            m = re.search(pattern, question)
            if m:
                return int(m.group(1)) * multiplier
        return None

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

            if "assistant" in response.lower():
                parts = response.split("assistant")
                if len(parts) > 1:
                    response = parts[-1].strip()

            return response

        except Exception as e:
            return f"[LLM生成エラー: {e}]\nコンテキスト:\n{context}"
