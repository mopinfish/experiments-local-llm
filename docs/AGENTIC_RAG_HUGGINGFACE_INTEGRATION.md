# Agentic RAG HuggingFace Integration

**作成日**: 2026-02-03
**プロジェクト**: experiments-local-llm
**Phase**: 9

## 概要

Agentic RAGシステムをChatOllama（Ollama専用）からHuggingFace Transformers（Google Colab対応）へ完全移行しました。

## 変更内容

### 1. LLM統合の変更

**変更前（ChatOllama）**:
```python
from langchain_ollama import ChatOllama

self.llm = ChatOllama(model="qwen2.5:7b-instruct", temperature=0.0)
self.llm_with_tools = self.llm.bind_tools(self.tools)

# LangChainのbind_tools()を使用
response = self.llm_with_tools.invoke(messages)
```

**変更後（HuggingFace Transformers）**:
```python
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# 4bit量子化設定
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

# モデルロード
self.model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    quantization_config=bnb_config,
    device_map="auto"
)

# 直接model.generate()を呼び出し
def _generate_llm_response(self, prompt: str) -> str:
    messages = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]
    text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
    outputs = self.model.generate(**inputs, max_new_tokens=512, ...)
    return self.tokenizer.decode(outputs[0][...], skip_special_tokens=True)
```

### 2. ツール呼び出しの変更

**変更前（LangChain bind_tools）**:
```python
self.llm_with_tools = self.llm.bind_tools(self.tools)

# ツール呼び出しは自動パース
if hasattr(response, 'tool_calls') and response.tool_calls:
    # LangChainが自動でパース
    tool_calls = response.tool_calls
```

**変更後（ReActスタイルプロンプト）**:
```python
# プロンプトでツールを説明
prompt = f"""
Available Tools:
{generate_tools_description()}

Use the ReAct format:
Thought: [your reasoning]
Action: [tool_name]
Action Input: {{"arg1": "value1", ...}}
"""

# モデルの出力を正規表現でパース
def _parse_tool_calls(self, response: str) -> List[Dict[str, Any]]:
    pattern = r'Action:\s*(\w+)\s*\n\s*Action Input:\s*(\{[^}]+\})'
    matches = re.finditer(pattern, response, re.MULTILINE)
    tool_calls = []
    for match in matches:
        tool_name = match.group(1)
        args = json.loads(match.group(2))
        tool_calls.append({"tool": tool_name, "args": args})
    return tool_calls
```

### 3. グラフノードの変更

**変更前**:
```python
from langgraph.prebuilt import ToolNode

workflow.add_node("tools", ToolNode(self.tools))
```

**変更後**:
```python
# カスタムツール実行ノード
def _tool_execution_node(self, state: AgentState) -> AgentState:
    last_step = state["intermediate_steps"][-1]
    tool_calls = last_step.get("tool_calls", [])

    for tool_call in tool_calls:
        tool_name = tool_call["tool"]
        args = tool_call["args"]
        tool = self.tool_map[tool_name]
        output = tool.invoke(args)
        state = add_tool_result(state, {
            "tool": tool_name,
            "args": args,
            "output": output
        })

    return state

workflow.add_node("tools", self._tool_execution_node)
```

### 4. 状態管理の変更

**変更前**:
```python
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
```

**変更後**:
```python
# LangChainメッセージ依存を削除
class AgentState(TypedDict):
    messages: List[Dict[str, str]]  # シンプルなdict
```

## 主要な利点

### 1. Google Colab互換性
- ローカルにOllamaをインストール不要
- T4 GPU（無料枠）で実行可能
- 4bit量子化でVRAM効率的

### 2. 柔軟な初期化
```python
# 事前ロード済みモデルを使用（Colab推奨）
system = AgenticRAGSystem(
    model=model,
    tokenizer=tokenizer,
    verbose=False
)

# またはモデルを自動ロード（ローカル）
system = AgenticRAGSystem(
    model_name="Qwen/Qwen2.5-7B-Instruct",
    load_in_4bit=True
)
```

### 3. Phase 8との一貫性
- StructuredRAG、GraphRAG、AdaptiveRAGと同じモデルロード方式
- すべてのシステムがTransformersベース

## ファイル変更一覧

### 修正ファイル

1. **src/agentic_rag_system.py**
   - ChatOllama → HuggingFace Transformers
   - bind_tools() → ReActプロンプト
   - ToolNode → カスタム_tool_execution_node
   - 新メソッド: `_generate_llm_response()`, `_parse_tool_calls()`

2. **src/agent_state.py**
   - LangChain messages削除
   - シンプルなList[Dict]に変更

3. **pyproject.toml**
   - 新依存関係追加:
     - transformers>=4.36.0
     - torch>=2.1.0
     - bitsandbytes>=0.41.0
     - accelerate>=0.25.0
     - sentencepiece>=0.1.99

4. **notebooks/phase9_agentic_rag_evaluation.ipynb**
   - Agentic RAG初期化コード更新
   - Structured vs Agentic比較評価を追加
   - Ollamaインストールコード削除

### 新規ファイル

5. **test_agentic_huggingface.py**
   - HuggingFace統合のローカルテストスクリプト
   - 2つのサンプル質問で動作確認

6. **docs/AGENTIC_RAG_HUGGINGFACE_INTEGRATION.md** (このファイル)
   - 統合内容の詳細ドキュメント

## 使用方法

### ローカル実行

```bash
# 依存関係インストール
uv sync

# POIデータがあることを確認
ls poi_documents.json

# テスト実行（初回はモデルダウンロード）
uv run python test_agentic_huggingface.py
```

### Google Colab実行

```python
# notebooks/phase9_agentic_rag_evaluation.ipynb を開く
# すべてのセルを順番に実行

# モデルロード（自動）
# システム初期化（自動）
# 評価実行

# クイックテスト: quick_test = True
# フルテスト: quick_test = False
```

## 技術的詳細

### ReActプロンプト形式

エージェントは以下の形式で応答します：

```
Thought: 渋谷駅から最も近いカフェを見つける必要があります
Action: tool_get_nearest_pois
Action Input: {"category": "カフェ", "top_n": 1}

Observation: [ツール実行結果]

Thought: 結果から答えをまとめます
Final Answer: 渋谷駅から最も近いカフェは○○です（距離50m）
```

### ツール実行フロー

```
User Question
    ↓
Agent Node
    ├→ プロンプト構築（質問 + ツール説明 + 履歴）
    ├→ model.generate()
    ├→ レスポンスパース（_parse_tool_calls）
    └→ ツール呼び出し検出
        ↓
Tool Execution Node
    ├→ tool_map から該当ツール取得
    ├→ tool.invoke(args)
    └→ 結果を state に保存
        ↓
Agent Node（再帰）
    ├→ 前回の結果を含めて再度推論
    └→ 必要に応じてさらにツール実行 or 回答生成
        ↓
Finalize Node
    ├→ すべてのツール結果を集約
    └→ 最終回答生成
```

### エラーハンドリング

- **ツールパースエラー**: JSONパース失敗時はスキップ
- **ツール実行エラー**: エラーメッセージをツール結果として記録
- **最大イテレーション**: 5回でループを強制終了し回答生成

## パフォーマンス

### 想定スペック

- **メモリ**: 最小8GB RAM（4bit量子化時）
- **GPU**: CUDA対応GPU推奨（CPU可だが遅い）
- **ストレージ**: モデルファイル約4GB

### 実行時間（参考値）

- **初回実行**: モデルダウンロード（約5分）
- **クエリ1回**: 3-10秒（イテレーション数に依存）
- **10ケース評価**: 約3-5分（GPU）

## トラブルシューティング

### Q: メモリ不足エラーが出る

A: `load_in_4bit=True` が有効か確認してください。さらに軽量化が必要な場合は8bitを検討してください。

### Q: ツール呼び出しが認識されない

A: `agent_prompts.py` の ReActプロンプトを確認し、モデルが正しい形式で出力しているかデバッグしてください。

### Q: Colab で CUDA out of memory

A: ランタイムをリスタートして、他のノートブックを閉じてください。T4 GPU（無料枠）で4bit量子化なら動作します。

## 今後の改善

1. **プロンプト最適化**: ReActプロンプトのチューニング
2. **Few-shot例の追加**: ツール呼び出しの精度向上
3. **ストリーミング対応**: 長時間実行時のUX改善
4. **Self-correction**: エラー検出と自動修正ループ

## 参考資料

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/)
- [ReAct Paper](https://arxiv.org/abs/2210.03629)
- [Qwen2.5 Model Card](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
