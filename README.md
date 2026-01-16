# Raspberry Pi ローカルLLM + OSM POI RAG システム

Raspberry Pi 4B上でローカルLLM（Ollama）とOpenStreetMapのPOIデータを組み合わせたRAG（Retrieval-Augmented Generation）システムの検証プロジェクトです。

## 📋 概要

### 目的

LLMがPOI（Point of Interest）に関する質問に回答する際の課題を、RAGによって解決することを検証します。

| 課題 | RAGによる解決 |
|------|--------------|
| ハルシネーション（架空の施設名を回答） | 検索結果に基づく回答で抑制 |
| 座標情報の欠如 | OSMデータから正確な座標を提供 |
| 情報の古さ | 最新のOSMデータを参照 |

### システム構成

```
┌─────────────────────────────────────────────────────────────────┐
│                    Raspberry Pi 4B (8GB)                        │
├─────────────────────────────────────────────────────────────────┤
│  ┌───────────────┐    ┌───────────────┐    ┌───────────────┐   │
│  │    Ollama     │    │   ChromaDB    │    │  OSM POI Data │   │
│  │ (LLM Server)  │◄──►│(Vector Store) │◄───│   (JSON)      │   │
│  │               │    │               │    │               │   │
│  │ - qwen2.5:3b  │    │ - 1,046 docs  │    │ - Overpass    │   │
│  │ - nomic-embed │    │ - 768次元     │    │   API取得     │   │
│  └───────┬───────┘    └───────────────┘    └───────────────┘   │
│          │                    ▲                                 │
│          ▼                    │                                 │
│  ┌───────────────┐    ┌───────┴───────┐                        │
│  │   REST API    │    │  RAG System   │                        │
│  │  (11434)      │◄──►│  (Python)     │                        │
│  └───────────────┘    └───────────────┘                        │
└─────────────────────────────────────────────────────────────────┘
```

### 検証結果

| 指標 | RAGあり | RAGなし | 改善率 |
|------|---------|---------|--------|
| 総合スコア | 68.1 | 61.4 | **+10.9%** |
| 座標情報含有率 | 53.3% | 0.0% | **+100%** |
| キーワードヒット率 | 69.4% | 72.8% | -4.6% |

---

## 🔧 必要要件

### ハードウェア

| 項目 | 要件 |
|------|------|
| デバイス | Raspberry Pi 4 Model B |
| RAM | **8GB** （必須） |
| ストレージ | 32GB以上（SSD推奨） |
| OS | **64-bit** Raspberry Pi OS または Debian |

### ソフトウェア

| ソフトウェア | バージョン | 用途 |
|-------------|-----------|------|
| Python | 3.11+ | 実行環境 |
| uv | 0.9+ | パッケージ管理 |
| Ollama | 0.14+ | LLMランタイム |

---

## 🚀 セットアップ手順

### 1. Ollamaのインストール

```bash
# Ollamaのインストール
curl -fsSL https://ollama.com/install.sh | sh

# サービスの起動確認
systemctl status ollama

# 起動していない場合
sudo systemctl start ollama
sudo systemctl enable ollama
```

### 2. LLMモデルのダウンロード

```bash
# LLMモデル（日本語対応、約1.9GB）
ollama pull qwen2.5:3b

# Embeddingモデル（約274MB）
ollama pull nomic-embed-text

# モデル一覧の確認
ollama list
```

### 3. プロジェクトのクローン

```bash
# リポジトリのクローン
git clone git@github.com:mopinfish/experiments-local-llm.git
cd experiments-local-llm
```

### 4. Python環境のセットアップ（uv使用）

```bash
# uvのインストール（未インストールの場合）
curl -LsSf https://astral.sh/uv/install.sh | sh

# シェルの再起動または設定の読み込み
source ~/.bashrc  # または source ~/.config/fish/config.fish

# プロジェクトの同期（依存関係のインストール）
uv sync
```

### 5. POIデータの取得

```bash
# OSMからPOIデータを取得（渋谷駅周辺）
uv run python osm_poi_fetcher.py

# 取得結果の確認
cat poi_documents.json | head -100
```

### 6. RAGシステムの初期化

```bash
# ベクトルストアの構築（初回のみ、約5-10分）
uv run python -c "from rag_system import POI_RAG_System; POI_RAG_System(rebuild=True)"
```

---

## 📖 使用方法

### 対話モードの起動

```bash
uv run python rag_system.py
```

### 対話モードのコマンド

| コマンド | 説明 | 例 |
|---------|------|-----|
| `search <質問>` | RAGあり検索 | `search 渋谷の映画館を教えて` |
| `compare <質問>` | RAGあり/なし比較 | `compare 渋谷駅の場所は？` |
| `categories` | カテゴリ一覧表示 | `categories` |
| `category <名前>` | カテゴリ指定検索 | `category 娯楽/映画館` |
| `quit` / `exit` | 終了 | `quit` |

### 使用例

```
============================================================
POI RAG システム 対話モード
============================================================
コマンド:
  search <質問>   - RAGを使って検索
  compare <質問>  - RAGあり/なしを比較
  categories      - カテゴリ一覧を表示
  category <名前> - 指定カテゴリのPOIを表示
  quit/exit       - 終了
============================================================

質問: search 渋谷の映画館を教えてください

========== RAGあり回答 ==========
以下の施設が渋谷で映画館として営業しています：

1. Cinema Vera Shibuya
   - 座標: 緯度 35.659351, 経度 139.695385
   - 電話番号: 03-3461-7703

2. シアター･イメージフォーラム
   - 座標: 緯度 35.660303, 経度 139.707022
   ...
```

---

## 🧪 テストの実行

### クイックテスト（5件）

```bash
uv run python test_runner.py --quick
```

### フルテスト（30件）

```bash
# バックグラウンド実行（SSH切断対策）
nohup uv run python test_runner.py > test_output.log 2>&1 &

# 進捗確認
tail -f test_output.log

# または screen/tmux を使用
screen -S rag_test
uv run python test_runner.py
# Ctrl+A → D で切り離し
# screen -r rag_test で再接続
```

### テスト結果の確認

```bash
# Markdownレポート
cat test_report_latest.md

# JSONレポート（サマリー）
cat test_report_latest.json | jq '.summary'
```

---

## 📁 ファイル構成

```
experiments-local-llm/
├── README.md                 # このファイル
├── pyproject.toml            # uv プロジェクト設定
├── uv.lock                   # 依存関係ロックファイル
│
├── osm_poi_fetcher.py        # OSM POIデータ取得スクリプト
├── rag_system.py             # RAGシステム本体
├── test_cases.py             # テストケース定義
├── test_runner.py            # テストランナー
│
├── poi_documents.json        # 取得したPOIデータ
├── chroma_db/                # ベクトルストア（永続化）
│
├── test_report_*.json        # テスト結果（JSON）
├── test_report_*.md          # テスト結果（Markdown）
│
├── PHASE1-3_REPORT.md        # Phase 1-3 作業報告書
├── PHASE4_TEST_REPORT.md     # Phase 4 テスト結果報告書
└── TEST_PROMPT_IMPROVEMENT_DESIGN.md  # テスト改良設計書
```

---

## ⚙️ 設定・カスタマイズ

### POI取得エリアの変更

`osm_poi_fetcher.py` の以下の部分を編集：

```python
# Bounding Box (南緯, 西経, 北緯, 東経)
BBOX = "35.655,139.695,35.665,139.710"  # 渋谷駅周辺

# 例: 東京駅周辺に変更
BBOX = "35.676,139.760,35.686,139.775"
```

### LLMモデルの変更

`rag_system.py` の定数を編集：

```python
LLM_MODEL = "qwen2.5:3b"           # LLMモデル
EMBEDDING_MODEL = "nomic-embed-text"  # Embeddingモデル
```

### 検索パラメータの調整

```python
# 検索結果の上限数
DEFAULT_K = 5  # デフォルト5件

# LLMのtemperature（0.0 = 確定的、1.0 = ランダム）
temperature = 0.0
```

---

## 🐛 トラブルシューティング

### Ollamaが起動しない

```bash
# ステータス確認
systemctl status ollama

# ログ確認
journalctl -u ollama -f

# 手動起動
ollama serve
```

### メモリ不足エラー

```bash
# メモリ使用量の確認
free -h

# 不要なプロセスの停止
sudo systemctl stop bluetooth
sudo systemctl stop cups
```

### ベクトルストアのエラー

```bash
# ベクトルストアの再構築
rm -rf chroma_db/
uv run python -c "from rag_system import POI_RAG_System; POI_RAG_System(rebuild=True)"
```

### 応答が遅い

- Raspberry Pi 4Bでは1回の質問に約3-5分かかることがあります
- より高速な応答が必要な場合は、軽量モデル（`phi3:mini`等）の使用を検討してください

---

## 📊 性能特性

### 応答時間（Raspberry Pi 4B）

| 処理 | 時間 |
|------|------|
| RAGあり（平均） | 約5分 |
| RAGなし（平均） | 約1.5分 |
| ベクトルストア初期構築 | 約10分 |

### メモリ使用量

| 状態 | メモリ |
|------|--------|
| アイドル時 | 約1GB |
| LLM推論時 | 約4-5GB |
| ピーク時 | 約6GB |

---

## 📚 関連ドキュメント

| ドキュメント | 内容 |
|-------------|------|
| [PHASE1-3_REPORT.md](./PHASE1-3_REPORT.md) | 環境構築〜RAGシステム構築の報告 |
| [PHASE4_TEST_REPORT.md](./PHASE4_TEST_REPORT.md) | テスト結果の詳細分析 |
| [TEST_PROMPT_IMPROVEMENT_DESIGN.md](./TEST_PROMPT_IMPROVEMENT_DESIGN.md) | テスト改良の設計書 |

---

## 🔗 参考リンク

- [Ollama公式](https://ollama.ai/)
- [LangChain ドキュメント](https://python.langchain.com/docs/)
- [ChromaDB ドキュメント](https://docs.trychroma.com/)
- [Overpass API](https://wiki.openstreetmap.org/wiki/Overpass_API)
- [OpenStreetMap](https://www.openstreetmap.org/)

---

## 📝 ライセンス

MIT License

---

## 👤 作成者

- プロジェクト: GeoTechAgent / experiments-local-llm
- 作成日: 2026-01-16

