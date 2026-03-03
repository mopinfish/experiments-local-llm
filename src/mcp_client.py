"""
mcp_client.py - MCP サーバー接続ラッパー

Colab から ngrok トンネル経由で MCP サーバーに接続するクライアント。
MCP Python SDK の streamablehttp_client を使用。

Phase 10-A: MCP サーバー構造化ツール拡張実験
作成日: 2026-03-03
"""

import json
import asyncio
import time
from typing import Any, Dict, List, Optional

try:
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client
except ImportError:
    raise ImportError(
        "mcp パッケージが必要です。`pip install 'mcp[cli]>=1.9.0'` でインストールしてください。"
    )


class MCPClientWrapper:
    """MCP サーバーへの接続ラッパー。"""

    def __init__(self, server_url: str, timeout: float = 60.0):
        """
        Args:
            server_url: MCP サーバーの URL（例: "https://xxxx.ngrok.io/mcp"）
            timeout: ツール呼び出しタイムアウト（秒）
        """
        self.server_url = server_url.rstrip("/")
        if not self.server_url.endswith("/mcp"):
            self.server_url += "/mcp"
        self.timeout = timeout
        self._tools_cache: Optional[List[Dict[str, Any]]] = None

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
            async with streamablehttp_client(self.server_url) as (
                read_stream,
                write_stream,
                _,
            ):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    result = await session.call_tool(tool_name, arguments)

                    # result.content はリスト。テキスト部分を連結して返す
                    texts = []
                    for block in result.content:
                        if hasattr(block, "text"):
                            texts.append(block.text)
                        else:
                            texts.append(str(block))

                    elapsed = time.time() - start
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
        try:
            result = await self.call_tool(tool_name, arguments)
            elapsed = time.time() - start
            return {
                "result": result,
                "elapsed_sec": round(elapsed, 2),
                "tool_name": tool_name,
                "success": not result.startswith("MCP ツール呼び出しエラー"),
            }
        except Exception as e:
            elapsed = time.time() - start
            return {
                "result": str(e),
                "elapsed_sec": round(elapsed, 2),
                "tool_name": tool_name,
                "success": False,
            }

    async def list_tools(self) -> List[Dict[str, Any]]:
        """
        利用可能なツール一覧を取得。

        Returns:
            ツール定義のリスト。各要素は
            {"name": str, "description": str, "inputSchema": dict}
        """
        async with streamablehttp_client(self.server_url) as (
            read_stream,
            write_stream,
            _,
        ):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.list_tools()

                tools = []
                for tool in result.tools:
                    tools.append({
                        "name": tool.name,
                        "description": tool.description or "",
                        "inputSchema": (
                            tool.inputSchema if hasattr(tool, "inputSchema") else {}
                        ),
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
