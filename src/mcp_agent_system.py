"""
mcp_agent_system.py - System C/D: LLM Agent 方式

Qwen3 の function calling で MCP ツールを自律的に呼び出すエージェント方式。
tool_filter パラメータで利用可能ツールをフィルタリングし、
System C（全ツール）と System D（基本ツールのみ）を切り替える。

Phase 10-A: MCP サーバー構造化ツール拡張実験
作成日: 2026-03-03
"""

import copy
import json
import re
import time
from typing import Any, Dict, List, Optional

try:
    from .mcp_client import MCPClientWrapper
except ImportError:
    from mcp_client import MCPClientWrapper


# エージェント用システムプロンプト
AGENT_SYSTEM_PROMPT = """あなたは東京都内の主要駅周辺エリアの地理情報に詳しいアシスタントです。
ユーザーの質問に回答するために、利用可能なツールを使って情報を検索してください。

# 回答の構造
1. まず必要な情報をツールで検索する
2. 検索結果に基づいて回答を生成する
3. 結論→根拠→補足の順で回答する

# 回答ルール
- POI名、座標(緯度, 経度)、距離(m)、件数を具体的に引用する
- 数値は単位付きで示す
- 不確実な点は正直に示す
- 情報がない場合は「確認できません」と回答する"""

MAX_AGENT_ITERATIONS = 5


class MCPAgentSystem:
    """
    System C/D: LLM Agent 方式

    LLM が function calling で MCP ツールを自律的に選択・呼び出す。
    - System C: tool_filter=None (全ツール = 既存 + 構造化)
    - System D: tool_filter="simple" (geo_* ツールを除外)
    """

    def __init__(
        self,
        mcp_client: MCPClientWrapper,
        model=None,
        tokenizer=None,
        tool_filter: Optional[str] = None,
        debug: bool = False,
    ):
        """
        Args:
            mcp_client: MCP サーバーへの接続ラッパー
            model: Hugging Face モデル
            tokenizer: トークナイザー
            tool_filter: "simple" で geo_* を除外、None で全ツール
            debug: デバッグ出力を有効化
        """
        self.mcp = mcp_client
        self.model = model
        self.tokenizer = tokenizer
        self.tool_filter = tool_filter
        self.debug = debug
        self._tool_defs: Optional[List[Dict[str, Any]]] = None

    def set_model(self, model, tokenizer):
        """モデルとトークナイザーを後から設定。"""
        self.model = model
        self.tokenizer = tokenizer

    async def _get_tool_definitions(self) -> List[Dict[str, Any]]:
        """MCP からツール一覧を取得し、フィルタリング。"""
        if self._tool_defs is not None:
            return self._tool_defs

        tools = await self.mcp.list_tools()

        if self.tool_filter == "simple":
            # geo_* ツールを除外
            tools = [t for t in tools if not t["name"].startswith("geo_")]

        self._tool_defs = tools
        return tools

    def _convert_to_qwen_tools(
        self, tools: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        MCP ツール定義を Qwen3 function calling 形式に変換。

        Qwen3 の tool calling は OpenAI 形式:
        [{"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}]
        """
        qwen_tools = []
        for tool in tools:
            schema = tool.get("inputSchema", {})
            # MCP の inputSchema から properties を抽出
            parameters = {
                "type": "object",
                "properties": schema.get("properties", {}),
                "required": schema.get("required", []),
            }

            qwen_tools.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": parameters,
                },
            })

        return qwen_tools

    async def query(self, question: str) -> Dict[str, Any]:
        """
        エージェントループで質問に回答。

        LLM がツールを選択→呼び出し→結果を受け取り→回答生成（最大 5 反復）。

        Returns:
            {
                "answer": str,
                "tool_calls": list,
                "iterations": int,
                "time_sec": float,
            }
        """
        if self.model is None or self.tokenizer is None:
            return {
                "answer": "[モデル未設定]",
                "tool_calls": [],
                "iterations": 0,
                "time_sec": 0,
            }

        start = time.time()
        tool_calls_log = []

        # ツール定義取得
        mcp_tools = await self._get_tool_definitions()
        qwen_tools = self._convert_to_qwen_tools(mcp_tools)

        if self.debug:
            print(f"[Agent] 利用可能ツール: {[t['function']['name'] for t in qwen_tools]}")

        # メッセージ履歴
        messages = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]

        # エージェントループ
        for iteration in range(MAX_AGENT_ITERATIONS):
            if self.debug:
                print(f"[Agent] Iteration {iteration + 1}/{MAX_AGENT_ITERATIONS}")

            # LLM 生成
            response_text, tool_call_results = self._generate_with_tools(
                messages, qwen_tools
            )

            if tool_call_results:
                # ツール呼び出しがある場合
                # LLM の応答をアシスタントメッセージとして追加
                messages.append({
                    "role": "assistant",
                    "content": response_text,
                    "tool_calls": tool_call_results,
                })

                # 各ツールを実行
                for tc in tool_call_results:
                    tool_name = tc["function"]["name"]
                    try:
                        args = json.loads(tc["function"]["arguments"])
                    except json.JSONDecodeError:
                        args = {}

                    if self.debug:
                        print(f"  ツール呼び出し: {tool_name}({args})")

                    try:
                        result = await self.mcp.call_tool(tool_name, args)
                    except Exception as e:
                        if self.debug:
                            print(f"  ツールエラー ({tool_name}): {e}")
                        result = f"ツール実行エラー: {e}"

                    tool_calls_log.append({
                        "tool": tool_name,
                        "args": args,
                        "result_len": len(result),
                        "iteration": iteration + 1,
                    })

                    messages.append({
                        "role": "tool",
                        "name": tool_name,
                        "content": result,
                    })
            else:
                # ツール呼び出しなし = 最終回答
                elapsed = time.time() - start
                return {
                    "answer": response_text,
                    "tool_calls": tool_calls_log,
                    "iterations": iteration + 1,
                    "time_sec": round(elapsed, 2),
                }

        # 最大反復到達 — 最後のメッセージを回答とする
        elapsed = time.time() - start
        final_answer = messages[-1].get("content", "") if messages else ""
        return {
            "answer": final_answer,
            "tool_calls": tool_calls_log,
            "iterations": MAX_AGENT_ITERATIONS,
            "time_sec": round(elapsed, 2),
        }

    def _generate_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
    ) -> tuple:
        """
        Qwen3 で function calling 付き生成。

        Returns:
            (response_text: str, tool_calls: list | None)
            tool_calls が None なら最終回答。
        """
        import torch

        # Qwen3 の apply_chat_template は tools パラメータをサポート
        try:
            text = self.tokenizer.apply_chat_template(
                messages,
                tools=tools,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            # tools パラメータ非対応の場合（旧モデル）
            # ツール定義をシステムプロンプトに埋め込む
            tool_desc = self._format_tools_as_text(tools)
            augmented = copy.deepcopy(messages)
            augmented[0] = {
                "role": "system",
                "content": messages[0]["content"] + "\n\n" + tool_desc,
            }
            text = self.tokenizer.apply_chat_template(
                augmented, tokenize=False, add_generation_prompt=True
            )

        inputs = self.tokenizer(text, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=False,
        )

        # ツール呼び出しの解析
        tool_calls = self._parse_tool_calls(response)

        # テキスト部分を抽出
        clean_text = self._extract_text_response(response)

        return (clean_text, tool_calls if tool_calls else None)

    def _parse_tool_calls(self, response: str) -> List[Dict[str, Any]]:
        """
        LLM 出力からツール呼び出しを解析。

        Qwen3 の function calling 形式:
        <tool_call>{"name": "...", "arguments": {...}}</tool_call>
        """
        tool_calls = []

        # Qwen3 形式: <tool_call>...</tool_call>
        pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
        matches = re.findall(pattern, response, re.DOTALL)

        for match in matches:
            try:
                parsed = json.loads(match)
                name = parsed.get("name", "")
                arguments = parsed.get("arguments", {})
                tool_calls.append({
                    "id": f"call_{len(tool_calls)}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(arguments, ensure_ascii=False),
                    },
                })
            except json.JSONDecodeError:
                continue

        return tool_calls

    def _extract_text_response(self, response: str) -> str:
        """ツール呼び出しタグを除去したテキスト部分を抽出。"""
        # <tool_call>...</tool_call> を除去
        clean = re.sub(
            r'<tool_call>.*?</tool_call>', '', response, flags=re.DOTALL
        )
        # 特殊トークンを除去
        for token in ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]:
            clean = clean.replace(token, "")
        return clean.strip()

    def _format_tools_as_text(self, tools: List[Dict[str, Any]]) -> str:
        """ツール定義をテキスト形式に変換（apply_chat_template 非対応時のフォールバック）。"""
        lines = ["# 利用可能なツール", ""]
        for tool in tools:
            func = tool["function"]
            lines.append(f"## {func['name']}")
            lines.append(f"{func.get('description', '')}")
            params = func.get("parameters", {}).get("properties", {})
            if params:
                lines.append("パラメータ:")
                for pname, pinfo in params.items():
                    desc = pinfo.get("description", "")
                    ptype = pinfo.get("type", "")
                    lines.append(f"  - {pname} ({ptype}): {desc}")
            lines.append("")

        lines.append(
            "ツールを呼び出すには <tool_call>{\"name\": \"ツール名\", "
            "\"arguments\": {パラメータ}}</tool_call> の形式を使ってください。"
        )
        return "\n".join(lines)
