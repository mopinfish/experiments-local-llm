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
├── CLAUDE.md                 # Claude Code用ガイダンス
├── pyproject.toml            # uv プロジェクト設定
├── uv.lock                   # 依存関係ロックファイル
│
├── osm_poi_fetcher.py        # OSM POIデータ取得スクリプト
├── rag_system.py             # RAGシステム本体（Phase 1-4）
├── test_cases.py             # テストケース定義（Phase 1-4）
├── test_runner.py            # テストランナー
│
├── src/                      # Phase 5-6 構造化RAGモジュール
│   ├── __init__.py           # バージョン管理
│   ├── geo_utils.py          # 空間処理（距離計算、方角、近接性、感度分析）
│   ├── aggregator.py         # 集計処理（東西比較、カテゴリ集計）
│   ├── structured_rag_system.py  # 質問分析、コンテキスト構築
│   ├── test_cases_v2.py      # 55件の階層化テストケース（L1-L5）
│   └── evaluators_v2.py      # 評価用スコアリングシステム
│
├── notebooks/                # Jupyter Notebooks
│   ├── phase5_advanced_test_cases.ipynb   # Phase 5 テストケース設計
│   ├── phase6_full_evaluation.ipynb       # Phase 6 評価（Colab用）
│   ├── finetuning_01_data_preparation.ipynb  # FT: データ準備
│   ├── finetuning_02_training.ipynb       # FT: QLoRA学習
│   └── finetuning_03_evaluation.ipynb     # FT: 4モデル比較評価
│
├── docs/                     # ドキュメント
│   ├── STRUCTURED_RAG_RESEARCH_REPORT.md  # Phase 5-6 研究レポート
│   ├── PHASE6_IMPROVEMENT_REPORT.md       # Phase 6 改善詳細
│   ├── HANDOVER_PHASE6.md                 # Phase 6 引き継ぎ
│   ├── FINETUNING_EXPERIMENT_REPORT.md    # FT実験レポート
│   └── STRUCTURED_DATA_DESIGN_GUIDE.md    # 構造化データ設計ガイド
│
├── data/
│   └── poi_documents.json    # POIデータ（1,047件）
│
├── chroma_db/                # ベクトルストア（永続化）
├── results/                  # 評価結果
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

## 🏗️ Phase 5-6: 構造化RAGアーキテクチャ

### 概要

Phase 5-6では、従来のベクトル検索のみのRAGを拡張し、**構造化データ処理とベクトル検索を相補的に統合**するアーキテクチャを実装しました。

| Phase | 内容 | スコア |
|-------|------|--------|
| Phase 5 | 階層化テストフレームワーク（55件） | 60.3pt（ベースライン） |
| Phase 6.1 | 空間エンリッチメント・集計・比較機能 | 69.6pt (+9.3pt) |
| Phase 6.2 | 近接性検索・感度分析機能 | 64.1pt (-5.5pt) ※一時悪化 |
| Phase 6.2.1 | 統合最適化（if文への修正） | **91.6pt (+31.3pt)** |

### アーキテクチャ

```
質問入力
    ↓
┌─────────────────────────────────────┐
│          質問分析モジュール           │
│  (カテゴリ抽出・質問タイプ判定)        │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│      並列コンテキスト構築（if文）      │
│  ┌────────┐ ┌────────┐ ┌────────┐  │
│  │構造化処理│ │構造化処理│ │ベクトル │  │
│  │  (集計) │ │ (比較) │ │ 検索  │  │
│  └────────┘ └────────┘ └────────┘  │
└─────────────────────────────────────┘
    ↓
統合コンテキスト → LLM回答生成
```

### 質問分析モジュール

質問分析モジュール（`src/structured_rag_system.py`）は、ユーザーの質問文を解析し、適切な検索・処理戦略を決定するコンポーネントです。

#### QuestionAnalysis クラス

```python
class QuestionAnalysis:
    question_type: str          # simple, comparison, aggregation, spatial, proximity, sensitivity
    categories: List[str]       # 検出されたカテゴリ（例: ["飲食店/カフェ"]）
    subcategories: List[str]    # サブカテゴリ（例: ["カフェ"]）
    directions: List[str]       # 方向キーワード（例: ["east", "west"]）

    # 処理フラグ
    requires_aggregation: bool  # 集計が必要か
    requires_comparison: bool   # 比較が必要か
    requires_spatial: bool      # 空間処理が必要か
    requires_proximity: bool    # 近接性検索が必要か
    requires_sensitivity: bool  # 感度分析が必要か

    # 制約パラメータ
    distance_constraint: float  # 距離制約（メートル）
    sensitivity_radii: Tuple    # 感度分析用の半径ペア
```

#### キーワードベース判定

質問タイプは事前定義されたキーワード辞書で判定します：

```python
# カテゴリキーワード（30種類以上）
CATEGORY_KEYWORDS = {
    "カフェ": ["飲食店/カフェ"],
    "コンビニ": ["商店/コンビニ"],
    "映画館": ["娯楽/映画館"],
    # ...
}

# 質問タイプ判定キーワード
COMPARISON_KEYWORDS = ["比較", "どちら", "違い", "どっち"]
AGGREGATION_KEYWORDS = ["いくつ", "何件", "多い", "少ない", "ランキング"]
PROXIMITY_KEYWORDS = ["最も近い", "一番近い", "最寄り"]
SENSITIVITY_KEYWORDS = ["変えても", "範囲を", "半径を", "成立"]
```

#### 分析処理フロー

```
入力: 「渋谷駅の東側と西側、どちらにカフェが多いですか？」
   │
   ├─ Step 1: カテゴリ抽出
   │    「カフェ」→ ["飲食店/カフェ"]
   │
   ├─ Step 2: 方向キーワード抽出
   │    「東」「西」→ ["east", "west"]
   │
   ├─ Step 3: 処理フラグ設定
   │    「どちら」→ requires_comparison = True
   │    「多い」  → requires_aggregation = True
   │
   └─ Step 4: 質問タイプ決定
        → question_type = "comparison"

出力: QuestionAnalysis オブジェクト
```

#### 距離制約の抽出（正規表現）

```python
# パターンマッチング例
「500m以内」   → distance_constraint = 500
「徒歩5分以内」→ distance_constraint = 400  # (5分 × 80m/分)
「1km圏内」    → distance_constraint = 1000
```

#### 具体例

**例1: 東西比較クエリ**
```
入力: 「渋谷駅の東側と西側、どちらにカフェが多いですか？」

分析結果:
{
    "question_type": "comparison",
    "categories": ["飲食店/カフェ"],
    "directions": ["east", "west"],
    "requires_comparison": true,
    "requires_aggregation": true
}
```

**例2: 近接性クエリ**
```
入力: 「渋谷駅に最も近いコンビニはどれですか？」

分析結果:
{
    "question_type": "proximity",
    "categories": ["商店/コンビニ"],
    "requires_proximity": true
}
```

**例3: 感度分析クエリ**
```
入力: 「カフェが多いという結論は、半径を500mから300mに変えても成立する？」

分析結果:
{
    "question_type": "sensitivity",
    "categories": ["飲食店/カフェ"],
    "requires_sensitivity": true,
    "sensitivity_radii": [300, 500]
}
```

#### 設計上のポイント

1. **キーワードベースの軽量実装**: LLMを使わずルールベースで高速に判定
2. **複数フラグの同時有効化**: `if`文（`elif`ではない）により、複数処理が並列実行可能
3. **正規表現による数値抽出**: 距離制約や半径ペアを柔軟に抽出
4. **拡張性**: 新しいキーワードや質問タイプを追加しやすい構造

### 4つの構造化処理コンポーネント

| コンポーネント | 機能 | 主な寄与 |
|---------------|------|---------|
| 空間情報エンリッチメント | 全POIに距離・方角を付与 | basic_location (+25pt) |
| 集計・比較機能 | 東西比較、カテゴリ集計 | spatial_comparison (+41pt) |
| 近接性検索 | 距離ソート、最近傍取得 | spatial_proximity (+34pt) |
| 感度分析 | 半径比較、結論の堅牢性評価 | advanced_sensitivity (+40pt) |

### Google Colab評価

主要な評価は `notebooks/phase6_full_evaluation.ipynb` でGoogle Colab（T4 GPUランタイム）上で実行されます。

```bash
# 評価実行（Colab上）
# 1. ノートブックを開く: notebooks/phase6_full_evaluation.ipynb
# 2. ランタイム → GPUを選択（T4推奨）
# 3. セルを順次実行（約30-40分）
```

---

## 📚 関連ドキュメント

| ドキュメント | 内容 |
|-------------|------|
| [PHASE1-3_REPORT.md](./PHASE1-3_REPORT.md) | 環境構築〜RAGシステム構築の報告 |
| [PHASE4_TEST_REPORT.md](./PHASE4_TEST_REPORT.md) | テスト結果の詳細分析 |
| [TEST_PROMPT_IMPROVEMENT_DESIGN.md](./TEST_PROMPT_IMPROVEMENT_DESIGN.md) | テスト改良の設計書 |
| [docs/STRUCTURED_RAG_RESEARCH_REPORT.md](./docs/STRUCTURED_RAG_RESEARCH_REPORT.md) | Phase 5-6 構造化RAG研究レポート |
| [docs/PHASE6_IMPROVEMENT_REPORT.md](./docs/PHASE6_IMPROVEMENT_REPORT.md) | Phase 6 改善詳細レポート |
| [docs/HANDOVER_PHASE6.md](./docs/HANDOVER_PHASE6.md) | Phase 6 技術引き継ぎドキュメント |
| [docs/FINETUNING_EXPERIMENT_REPORT.md](./docs/FINETUNING_EXPERIMENT_REPORT.md) | ファインチューニング実験レポート |

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

