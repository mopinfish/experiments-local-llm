"""
mcp_client.py - MCP サーバー接続ラッパー

Colab から ngrok トンネル経由で MCP サーバーに接続するクライアント。
httpx で JSON-RPC over HTTP を直接送信する方式。
（MCP SDK の streamablehttp_client は anyio TaskGroup の問題で
 Colab の nest_asyncio 環境と互換性がないため不使用）

Phase 10-A: MCP サーバー構造化ツール拡張実験
作成日: 2026-03-03
"""

import json
import time
from typing import Any, Dict, List, Optional

try:
    import httpx
except ImportError:
    raise ImportError("httpx が必要です。`pip install httpx` でインストールしてください。")


class MCPClientWrapper:
    """MCP サーバーへの接続ラッパー（httpx JSON-RPC 方式）。"""

    def __init__(self, server_url: str, timeout: float = 120.0):
        """
        Args:
            server_url: MCP サーバーの URL（例: "https://xxxx.ngrok.io/mcp"）
            timeout: HTTP タイムアウト（秒）
        """
        self.server_url = server_url.rstrip("/")
        if not self.server_url.endswith("/mcp"):
            self.server_url += "/mcp"
        self.timeout = timeout
        self._tools_cache: Optional[List[Dict[str, Any]]] = None
        self._request_id = 0
        # セッション ID（MCP streamable-http の場合、initialize で返される）
        self._session_id: Optional[str] = None

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    async def _send_jsonrpc(
        self, method: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """JSON-RPC リクエストを MCP サーバーに送信。"""
        payload = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": method,
        }
        if params is not None:
            payload["params"] = params

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                self.server_url,
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()

            # セッション ID を保存
            sid = resp.headers.get("Mcp-Session-Id")
            if sid:
                self._session_id = sid

            content_type = resp.headers.get("content-type", "")

            # SSE (text/event-stream) レスポンスの場合
            if "text/event-stream" in content_type:
                return self._parse_sse_response(resp.text)

            # 通常の JSON レスポンス
            return resp.json()

    def _parse_sse_response(self, body: str) -> Dict[str, Any]:
        """SSE レスポンスから最後の JSON-RPC メッセージを抽出。"""
        last_data = None
        for line in body.split("\n"):
            line = line.strip()
            if line.startswith("data: "):
                data_str = line[6:]
                try:
                    parsed = json.loads(data_str)
                    # id のあるレスポンス（通知ではない）を優先
                    if "id" in parsed:
                        last_data = parsed
                    elif last_data is None:
                        last_data = parsed
                except json.JSONDecodeError:
                    continue
        return last_data or {}

    async def _ensure_initialized(self):
        """初回呼び出し時に initialize を送信。"""
        if self._session_id is not None:
            return
        result = await self._send_jsonrpc("initialize", {
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": {"name": "mcp-eval-client", "version": "1.0.0"},
        })
        # initialized 通知を送信
        await self._send_notification("notifications/initialized")

    async def _send_notification(self, method: str, params: Optional[Dict] = None):
        """JSON-RPC 通知（id なし）を送信。"""
        payload: Dict[str, Any] = {
            "jsonrpc": "2.0",
            "method": method,
        }
        if params:
            payload["params"] = params

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            await client.post(
                self.server_url,
                json=payload,
                headers=headers,
            )

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """
        MCP ツールを呼び出して結果テキストを返す。

        Args:
            tool_name: ツール名（例: "mapfan_search_spot_area", "geo_nearest_pois"）
            arguments: ツール引数の辞書

        Returns:
            ツール実行結果のテキスト
        """
        start = time.time()
        try:
            await self._ensure_initialized()
            resp = await self._send_jsonrpc("tools/call", {
                "name": tool_name,
                "arguments": arguments,
            })

            result = resp.get("result", {})
            content = result.get("content", [])

            texts = []
            for block in content:
                if isinstance(block, dict):
                    texts.append(block.get("text", str(block)))
                else:
                    texts.append(str(block))

            return "\n".join(texts)

        except Exception as e:
            elapsed = time.time() - start
            return f"MCP ツール呼び出しエラー ({tool_name}, {elapsed:.1f}s): {e}"

    async def call_tool_with_timing(
        self, tool_name: str, arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        ツール呼び出し + タイミング情報付き。

        Returns:
            {"result": str, "elapsed_sec": float, "tool_name": str, "success": bool}
        """
        start = time.time()
        result = await self.call_tool(tool_name, arguments)
        elapsed = time.time() - start
        return {
            "result": result,
            "elapsed_sec": round(elapsed, 2),
            "tool_name": tool_name,
            "success": not result.startswith("MCP ツール呼び出しエラー"),
        }

    async def list_tools(self) -> List[Dict[str, Any]]:
        """
        利用可能なツール一覧を取得。

        Returns:
            ツール定義のリスト。各要素は
            {"name": str, "description": str, "inputSchema": dict}
        """
        await self._ensure_initialized()
        resp = await self._send_jsonrpc("tools/list")

        result = resp.get("result", {})
        raw_tools = result.get("tools", [])

        tools = []
        for tool in raw_tools:
            tools.append({
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "inputSchema": tool.get("inputSchema", {}),
            })

        self._tools_cache = tools
        return tools

    async def ping(self) -> Dict[str, Any]:
        """
        接続テスト。ツール一覧を取得して接続確認。

        Returns:
            {"connected": bool, "tool_count": int, "tools": list[str], "elapsed_sec": float}
        """
        start = time.time()
        try:
            tools = await self.list_tools()
            elapsed = time.time() - start
            return {
                "connected": True,
                "tool_count": len(tools),
                "tools": [t["name"] for t in tools],
                "elapsed_sec": round(elapsed, 2),
            }
        except Exception as e:
            elapsed = time.time() - start
            return {
                "connected": False,
                "tool_count": 0,
                "tools": [],
                "elapsed_sec": round(elapsed, 2),
                "error": str(e),
            }

    def get_cached_tools(self) -> Optional[List[Dict[str, Any]]]:
        """キャッシュ済みツール一覧を返す（list_tools 未実行なら None）。"""
        return self._tools_cache

    def get_tool_names(self, prefix: Optional[str] = None) -> List[str]:
        """
        キャッシュ済みツール名リストを返す。

        Args:
            prefix: フィルタ用プレフィックス（例: "geo_" で構造化ツールのみ）
        """
        if not self._tools_cache:
            return []
        names = [t["name"] for t in self._tools_cache]
        if prefix:
            names = [n for n in names if n.startswith(prefix)]
        return names
