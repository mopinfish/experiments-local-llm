# Phase 9-B: 複数エリア対応RAG比較実験計画

**作成日**: 2026-02-18
**プロジェクト**: experiments-local-llm
**ステータス**: 計画策定中

---

## 目次

- [1. 実験概要](#1-実験概要)
- [2. 対象エリア定義](#2-対象エリア定義)
- [3. POIデータ取得の広域対応](#3-poiデータ取得の広域対応)
- [4. コアモジュールの広域対応](#4-コアモジュールの広域対応)
- [5. RAGシステムの広域対応](#5-ragシステムの広域対応)
- [6. テストケースの設計](#6-テストケースの設計)
- [7. 評価モジュールの設計](#7-評価モジュールの設計)
- [8. 評価ノートブックの設計](#8-評価ノートブックの設計)
- [9. 実装タスクと工数見積もり](#9-実装タスクと工数見積もり)
- [10. リスクと対策](#10-リスクと対策)
- [11. 成果物一覧](#11-成果物一覧)

---

## 1. 実験概要

### 1.1 背景と目的

Phase 6〜9では渋谷駅周辺（1,047 POI）に限定してRAG比較実験を行い、以下の成果を得た：

| Phase | アプローチ | 成功率 | 特徴 |
|-------|----------|--------|------|
| Phase 6 | Hybrid RAG（構造化RAG） | 96.2% | ルールベース質問分析 + 計算処理 + ベクトル検索 |
| Phase 8 | Graph RAG | 76.7% | NetworkXナレッジグラフ + グラフトラバーサル |
| Phase 8 | Adaptive RAG | 86.1% | クエリ複雑度に基づくシステム動的選択 |
| Phase 9 | Agentic RAG | 87.6% | LangGraph + ReActパターン + 16ツール |

しかし、これらの結果は**渋谷駅周辺という単一エリア**に最適化された環境での評価であり、以下の疑問が残る：

1. **汎化性能**: 渋谷以外のエリアでも同等の精度が出るのか？
2. **エリア横断クエリ**: 「新宿と渋谷でカフェが多いのはどちら？」のような比較は可能か？
3. **スケーラビリティ**: POI数が増加（1,047→4,000〜5,000件）した場合の性能劣化はあるか？
4. **エリア特定能力**: 質問文からターゲットエリアを正しく特定できるか？

本実験では**渋谷・新宿・池袋・東京の4エリア**に対象範囲を拡大し、既存の4つのRAGシステムを広域対応させた上で公平に比較評価する。

### 1.2 実験の位置づけ

```
Phase 6-9（渋谷単一エリア）
    ↓
Phase 9-B（4エリア拡大）  ← 本計画
    ↓
Phase 10（全国展開・PostGIS）
```

Phase 10の全国展開前に、**中間スケール**での検証を行うことで：
- 広域対応に必要なアーキテクチャ変更を特定する
- Phase 10で採用すべきRAGアプローチの最終選定を行う
- 渋谷固有のハードコーディングを排除し汎用化の基盤を整える

### 1.3 比較対象RAGシステム（4システム）

| # | システム | 既存実装 | 広域対応工数 |
|---|---------|---------|------------|
| 1 | **Hybrid RAG**（構造化RAG） | `structured_rag_system.py` | 中 |
| 2 | **Graph RAG** | `graph_rag_system.py` + `graph_builder.py` | 中 |
| 3 | **Adaptive RAG** | `adaptive_rag_system.py` | 小（上位2つに依存） |
| 4 | **Agentic RAG** | `agentic_rag_system.py` + `agent_tools.py` | 大 |

### 1.4 使用モデル・環境

- **LLM**: Qwen2.5-7B-Instruct（4bit量子化） ※Phase 9と同一で公平比較
- **埋め込みモデル**: multilingual-e5-base
- **実行環境**: Google Colab T4 GPU
- **ベクトルDB**: ChromaDB

> **Note**: モデル変更（Llama 3.1等）は本実験のスコープ外とする。モデル変数を固定することで、広域対応のアーキテクチャ影響のみを評価する。

---

## 2. 対象エリア定義

### 2.1 4エリアの選定理由

| エリア | 選定理由 | 期待POI数 | エリア特性 |
|--------|---------|----------|-----------|
| **渋谷駅** | 既存ベースライン、比較の基準 | ~1,000 | 商業・娯楽中心、若年層向け |
| **新宿駅** | 日本最大の乗降客数、多様なPOI | ~1,200 | ビジネス＋歓楽街＋商業 |
| **池袋駅** | 副都心、渋谷との比較対象 | ~800 | 商業施設集中、サンシャイン |
| **東京駅** | 官庁街・ビジネス街、特性が異なる | ~600 | オフィス・観光・交通ハブ |

### 2.2 エリア座標とバウンディングボックス

```python
AREAS = {
    "shibuya": {
        "name": "渋谷駅周辺",
        "station": {"name": "渋谷駅", "lat": 35.658034, "lon": 139.701636},
        "bbox": "35.655,139.695,35.665,139.710"  # 既存
    },
    "shinjuku": {
        "name": "新宿駅周辺",
        "station": {"name": "新宿駅", "lat": 35.689607, "lon": 139.700571},
        "bbox": "35.685,139.693,35.697,139.710"
    },
    "ikebukuro": {
        "name": "池袋駅周辺",
        "station": {"name": "池袋駅", "lat": 35.729503, "lon": 139.710999},
        "bbox": "35.725,139.704,35.736,139.718"
    },
    "tokyo": {
        "name": "東京駅周辺",
        "station": {"name": "東京駅", "lat": 35.681236, "lon": 139.767125},
        "bbox": "35.676,139.760,35.687,139.775"
    }
}
```

> **bbox設計方針**: 各駅を中心に約500m四方（南北約1km、東西約1km）をカバー。渋谷の既存bboxサイズ（南北約1.1km、東西約1.7km）に合わせる。

### 2.3 エリア間の距離

| From → To | 直線距離 | 特徴 |
|-----------|---------|------|
| 渋谷 → 新宿 | 約3.5km | 山手線隣接 |
| 渋谷 → 池袋 | 約8.0km | 副都心ライン |
| 渋谷 → 東京 | 約7.2km | 山手線経由 |
| 新宿 → 池袋 | 約4.8km | 山手線隣接 |
| 新宿 → 東京 | 約7.0km | 中央線 |
| 池袋 → 東京 | 約8.5km | 丸ノ内線 |

→ エリア間は十分に離れており、POIの重複は発生しない。

---

## 3. POIデータ取得の広域対応

### 3.1 `osm_poi_fetcher.py` の変更

#### 変更方針
既存の`AREAS`辞書を4エリアに拡張し、各エリアにstation座標を持たせる。

#### 変更内容

```python
# === 変更前 ===
AREAS = {
    "shibuya": {
        "name": "渋谷駅周辺",
        "bbox": "35.655,139.695,35.665,139.710"
    }
}

# === 変更後 ===
AREAS = {
    "shibuya": {
        "name": "渋谷駅周辺",
        "station": {"name": "渋谷駅", "lat": 35.658034, "lon": 139.701636},
        "bbox": "35.655,139.695,35.665,139.710"
    },
    "shinjuku": {
        "name": "新宿駅周辺",
        "station": {"name": "新宿駅", "lat": 35.689607, "lon": 139.700571},
        "bbox": "35.685,139.693,35.697,139.710"
    },
    "ikebukuro": {
        "name": "池袋駅周辺",
        "station": {"name": "池袋駅", "lat": 35.729503, "lon": 139.710999},
        "bbox": "35.725,139.704,35.736,139.718"
    },
    "tokyo": {
        "name": "東京駅周辺",
        "station": {"name": "東京駅", "lat": 35.681236, "lon": 139.767125},
        "bbox": "35.676,139.760,35.687,139.775"
    }
}
```

#### 出力ファイルの変更

```python
# 変更前
output_path = Path("./poi_documents.json")       # 全POI一括

# 変更後
output_dir = Path("./data")
output_dir.mkdir(exist_ok=True)
# エリア別ファイル
for area_key in AREAS:
    save(f"data/poi_{area_key}.json")             # エリア別
# 統合ファイル
save("data/poi_all_areas.json")                   # 全エリア統合
# 既存互換
save("poi_documents.json")                        # 渋谷のみ（後方互換）
```

#### POIメタデータにエリアキー追加

```python
# metadataに追加するフィールド
"area_key": "shinjuku",                # エリアキー（プログラム用）
"area": "新宿駅周辺",                   # エリア名（表示用）※既存フィールド
"station_name": "新宿駅",              # 基準駅名
"station_lat": 35.689607,             # 基準駅の緯度
"station_lon": 139.700571,            # 基準駅の経度
```

### 3.2 データ量の見積もり

| エリア | 推定POI数 | 根拠 |
|--------|----------|------|
| 渋谷 | ~1,047 | 実績値 |
| 新宿 | ~1,200 | 渋谷より広域・高密度 |
| 池袋 | ~800 | 渋谷よりやや小規模 |
| 東京 | ~600 | ビジネス街のためPOI種別は偏り |
| **合計** | **~3,600** | |

> ChromaDBのメモリ制限（Colab T4: RAM 12.7GB）で問題ないレベル。

### 3.3 Overpass API制約の考慮

- **レートリミット**: 連続リクエストは10秒間隔を空ける
- **タイムアウト**: 各クエリ120秒
- **実装**: `time.sleep(10)` をエリア間に挿入

---

## 4. コアモジュールの広域対応

### 4.1 `geo_utils.py` の変更

#### 4.1.1 SHIBUYA_STATION定数の扱い

**方針**: `SHIBUYA_STATION`定数は**後方互換のため残す**が、全関数のstation引数のデフォルト値として使用しないようにする。新たに`STATIONS`辞書を追加する。

```python
# === 追加 ===
STATIONS = {
    "渋谷駅": {"name": "渋谷駅", "lat": 35.658034, "lon": 139.701636},
    "新宿駅": {"name": "新宿駅", "lat": 35.689607, "lon": 139.700571},
    "池袋駅": {"name": "池袋駅", "lat": 35.729503, "lon": 139.710999},
    "東京駅": {"name": "東京駅", "lat": 35.681236, "lon": 139.767125},
}

# 後方互換
SHIBUYA_STATION = STATIONS["渋谷駅"]

def resolve_station(name_or_dict):
    """駅名文字列またはstation辞書からstation辞書を解決"""
    if isinstance(name_or_dict, str):
        return STATIONS.get(name_or_dict, STATIONS["渋谷駅"])
    return name_or_dict
```

#### 4.1.2 関数シグネチャの変更

既存関数のstation引数はオプショナル（デフォルト=None→SHIBUYA_STATION）のままとし、後方互換を維持する。
新しい広域対応関数は明示的なstation引数を**必須**とする。

```python
# 既存（後方互換維持）
def enrich_all_pois(pois, station=None):
    station = station or SHIBUYA_STATION
    ...

# 広域対応（新規追加）
def enrich_pois_for_area(pois, area_key, areas_config):
    """指定エリアのPOIに空間情報を付与"""
    station = areas_config[area_key]["station"]
    area_pois = [p for p in pois if p.get("metadata", {}).get("area_key") == area_key]
    return enrich_all_pois(area_pois, station=station)

def enrich_all_areas(pois, areas_config):
    """全エリアのPOIに各エリアの基準駅からの空間情報を付与"""
    enriched = []
    for area_key, area_info in areas_config.items():
        station = area_info["station"]
        area_pois = [p for p in pois if p.get("metadata", {}).get("area_key") == area_key]
        enriched.extend(enrich_all_pois(area_pois, station=station))
    return enriched
```

#### 4.1.3 エリア特定関数（新規）

```python
def detect_target_area(question: str, areas_config: dict) -> Optional[str]:
    """質問文からターゲットエリアを推定する

    Returns:
        area_key (str or None): "shibuya", "shinjuku", "ikebukuro", "tokyo", None
    """
    # 駅名キーワードマッチング
    station_keywords = {
        "shibuya": ["渋谷"],
        "shinjuku": ["新宿"],
        "ikebukuro": ["池袋"],
        "tokyo": ["東京駅", "丸の内", "大手町"],
    }
    matched = []
    for area_key, keywords in station_keywords.items():
        for kw in keywords:
            if kw in question:
                matched.append(area_key)
                break

    if len(matched) == 1:
        return matched[0]       # 単一エリア特定
    elif len(matched) > 1:
        return None             # 複数エリア → クロスエリアクエリ
    else:
        return None             # エリア不明

def detect_cross_area_query(question: str, areas_config: dict) -> List[str]:
    """クロスエリアクエリの対象エリアリストを返す"""
    ...
```

### 4.1.4 ランドマーク座標テーブル（新規）

テストケースでランドマークを起点とした問いに対応するため、各エリアの主要ランドマークの座標をハードコードで定義する。将来的にはGeocodingツール（Nominatim等）に置き換える。

```python
# geo_utils.py に追加

LANDMARKS = {
    # 渋谷エリア
    "渋谷109": {"lat": 35.659517, "lon": 139.698471, "area_key": "shibuya"},
    "ハチ公像": {"lat": 35.659020, "lon": 139.700464, "area_key": "shibuya"},
    "渋谷ヒカリエ": {"lat": 35.659098, "lon": 139.703628, "area_key": "shibuya"},
    "渋谷スクランブルスクエア": {"lat": 35.658580, "lon": 139.702095, "area_key": "shibuya"},
    # 新宿エリア
    "東京都庁": {"lat": 35.689634, "lon": 139.691577, "area_key": "shinjuku"},
    "新宿御苑": {"lat": 35.685175, "lon": 139.710052, "area_key": "shinjuku"},
    "歌舞伎町": {"lat": 35.694003, "lon": 139.703506, "area_key": "shinjuku"},
    "新宿アルタ": {"lat": 35.692854, "lon": 139.701238, "area_key": "shinjuku"},
    # 池袋エリア
    "サンシャインシティ": {"lat": 35.729185, "lon": 139.718611, "area_key": "ikebukuro"},
    "池袋西口公園": {"lat": 35.730070, "lon": 139.709747, "area_key": "ikebukuro"},
    "東武百貨店池袋店": {"lat": 35.729641, "lon": 139.710815, "area_key": "ikebukuro"},
    # 東京エリア
    "東京国際フォーラム": {"lat": 35.676987, "lon": 139.763489, "area_key": "tokyo"},
    "KITTE": {"lat": 35.679032, "lon": 139.765650, "area_key": "tokyo"},
    "丸ビル": {"lat": 35.681461, "lon": 139.763641, "area_key": "tokyo"},
    "皇居前広場": {"lat": 35.680959, "lon": 139.757280, "area_key": "tokyo"},
}

def resolve_landmark(name: str) -> Optional[Dict]:
    """ランドマーク名から座標情報を解決する

    Returns:
        {"lat": float, "lon": float, "area_key": str} or None
    """
    # 完全一致
    if name in LANDMARKS:
        return LANDMARKS[name]
    # 部分一致
    for lm_name, lm_info in LANDMARKS.items():
        if lm_name in name or name in lm_name:
            return lm_info
    return None
```

> **Note**: ランドマーク座標は暫定的なハードコード。Phase 10ではGeocodingツール（Nominatim API等）に置き換え予定。

### 4.2 `aggregator.py` の変更

#### 変更方針
集計関数自体はPOIリストを受け取る設計のため、入力データを正しくフィルタリングすれば既存関数はそのまま使える。

#### 追加関数

```python
def compare_across_areas(pois: List[Dict], areas_config: dict, category: str = None) -> Dict:
    """エリア間のPOI分布比較"""
    result = {}
    for area_key, area_info in areas_config.items():
        area_pois = [p for p in pois if p.get("area_key") == area_key]
        if category:
            area_pois = [p for p in area_pois if category in p.get("category", "")]
        result[area_info["name"]] = {
            "count": len(area_pois),
            "categories": count_by_category(area_pois)
        }
    return result

def aggregate_by_area(pois: List[Dict]) -> Dict[str, List[Dict]]:
    """POIリストをエリア別に分割"""
    ...
```

### 4.3 `graph_builder.py` の変更

#### 変更方針
`SHIBUYA_STATION`定数を`geo_utils.STATIONS`から取得するように変更。グラフ構築時にエリア単位またはエリア横断で構築可能にする。

```python
# 変更前
SHIBUYA_STATION = {"name": "渋谷駅", "lat": 35.658034, "lon": 139.701636}

# 変更後
from .geo_utils import STATIONS, SHIBUYA_STATION  # 後方互換

class POIGraphBuilder:
    def __init__(self, pois, station=None):
        self.station = station or SHIBUYA_STATION
        ...
```

---

## 5. RAGシステムの広域対応

### 5.1 共通設計方針

全RAGシステムに共通する広域対応の設計方針：

1. **質問文からのエリア特定**: `detect_target_area()` を使い、対象エリアのPOIのみを検索・計算対象にする
2. **エリア別ベクトルストア**: ChromaDBのcollectionをエリア別に分離し、ベクトル検索の精度を維持する。加えて全エリア統合のcollectionも作成する
3. **システムプロンプトの汎用化**: 「渋谷エリア」→「指定エリア」に変更
4. **エリア不明時のフォールバック**: エリアが特定できない場合は全エリア統合のcollectionで検索

### 5.2 ベクトルストアの設計

```python
# エリア別collection + 統合collection
collections = {
    "shibuya": chroma_client.get_or_create_collection("pois_shibuya"),
    "shinjuku": chroma_client.get_or_create_collection("pois_shinjuku"),
    "ikebukuro": chroma_client.get_or_create_collection("pois_ikebukuro"),
    "tokyo": chroma_client.get_or_create_collection("pois_tokyo"),
    "all": chroma_client.get_or_create_collection("pois_all"),  # 全統合
}
```

**検索フロー**:
```
質問 → detect_target_area()
  ├─ 単一エリア特定 → 該当エリアのcollectionで検索
  ├─ 複数エリア（クロス） → "all" collectionで検索 + area_keyでフィルタ
  └─ エリア不明 → "all" collectionで検索
```

### 5.3 Hybrid RAG（`structured_rag_system.py`）の変更

#### コンストラクタ

```python
class StructuredRAGSystem:
    def __init__(self, model, tokenizer, vectorstore, all_pois,
                 areas_config=None, debug=False):
        self.areas_config = areas_config or {
            "shibuya": {"name": "渋谷駅周辺", "station": SHIBUYA_STATION}
        }
        # エリア別にPOIを空間情報付与
        if areas_config and len(areas_config) > 1:
            self.all_pois = enrich_all_areas(all_pois, areas_config)
        else:
            self.all_pois = enrich_all_pois(all_pois)  # 後方互換

        # エリア別POIインデックス
        self.pois_by_area = {}
        for poi in self.all_pois:
            area_key = poi.get("metadata", {}).get("area_key", "shibuya")
            self.pois_by_area.setdefault(area_key, []).append(poi)
```

#### システムプロンプト

```python
# 変更前
self.system_prompt = """あなたは渋谷エリアの地理情報に詳しいアシスタントです。..."""

# 変更後
self.system_prompt = """あなたは東京都内の主要駅周辺エリア（渋谷、新宿、池袋、東京）の
地理情報に詳しいアシスタントです。
提供された情報に基づいて、正確かつ簡潔に回答してください。
座標情報がある場合は必ず含めてください。
数値データがある場合は具体的な数字を使って回答してください。
情報がない場合は「情報がありません」と正直に回答してください。"""
```

#### クエリ処理フロー

```python
def query_with_structured_rag(self, question):
    # 1. エリア特定（新規）
    target_area = detect_target_area(question, self.areas_config)
    target_pois = self.pois_by_area.get(target_area, self.all_pois)
    target_station = self.areas_config.get(target_area, {}).get("station", SHIBUYA_STATION)

    # 2. 質問分析（既存ロジック）
    analysis = analyze_question(question)

    # 3. 構造化処理（target_poisとtarget_stationを使用）
    context = self._build_context(question, analysis,
                                   pois=target_pois,
                                   station=target_station)

    # 4. ベクトル検索（エリア別collection使用）
    # 5. LLM生成
    ...
```

### 5.4 Graph RAG（`graph_rag_system.py`）の変更

#### エリア別グラフ構築

```python
class GraphRAGSystem:
    def __init__(self, ..., areas_config=None):
        self.graphs = {}
        if areas_config:
            for area_key, area_info in areas_config.items():
                area_pois = [p for p in all_pois if p["metadata"]["area_key"] == area_key]
                builder = POIGraphBuilder(area_pois, station=area_info["station"])
                self.graphs[area_key] = builder.build_graph()
        else:
            # 後方互換
            builder = POIGraphBuilder(all_pois)
            self.graphs["shibuya"] = builder.build_graph()
```

### 5.5 Adaptive RAG（`adaptive_rag_system.py`）の変更

Adaptive RAGはHybrid RAGとGraph RAGのルーターのため、下位システムが広域対応すれば自動的に対応する。変更は最小限。

### 5.6 Agentic RAG（`agentic_rag_system.py`）の変更

#### ツールの広域対応

```python
# agent_tools.py

# グローバルPOIデータをエリア別に保持
_global_pois_by_area = {}
_global_areas_config = {}

def set_global_pois_multi_area(pois, areas_config):
    """複数エリアのPOIデータを設定"""
    global _global_pois_by_area, _global_areas_config
    _global_areas_config = areas_config
    for poi in pois:
        area_key = poi.get("metadata", {}).get("area_key", "shibuya")
        _global_pois_by_area.setdefault(area_key, []).append(poi)

@tool
def tool_get_nearest_pois(area: str, category: str, top_n: int = 3) -> str:
    """指定エリアの指定カテゴリで最寄りのPOIを検索する

    Args:
        area: エリアキー（"shibuya", "shinjuku", "ikebukuro", "tokyo"）
        category: POIカテゴリ（例: "カフェ", "コンビニ"）
        top_n: 取得件数
    """
    pois = _global_pois_by_area.get(area, [])
    station = _global_areas_config.get(area, {}).get("station")
    ...
```

#### 中国語混入対策: ツール出力の日本語自然文化

Phase 9でJSON形式のツール出力がQwenの中国語モード（寄与度50%）をトリガーする問題が判明した。本実験ではこの問題に対処するため、ツール出力をJSON→日本語自然文に変換する処理を追加する。

```python
# agent_tools.py に追加

def format_tool_output_japanese(tool_name: str, output: dict) -> str:
    """ツール出力をJSON形式から日本語自然文に変換する

    Phase 9の中国語混入問題への対策。LLMにはJSON構造ではなく
    日本語テキストとしてツール結果を渡す。
    """
    if tool_name == "tool_get_nearest_pois":
        lines = [f"{output['category']}の検索結果（{output['count']}件）:"]
        for poi in output.get("pois", []):
            lines.append(
                f"  - {poi['name']}: 駅から{poi['direction']}方向に"
                f"{poi['distance_m']:.1f}メートル"
            )
        return "\n".join(lines)

    elif tool_name == "tool_count_pois_in_radius":
        return (f"半径{output['radius_m']}メートル以内の"
                f"{output['category']}は{output['count']}件です。")

    # ... 他のツールも同様に日本語テキスト化
    else:
        # フォールバック: JSONを簡易テキスト化
        return json.dumps(output, ensure_ascii=False, indent=2)
```

#### ReActメタ言語の日本語化

```python
# agent_prompts.py の変更

# 変更前（英語メタ言語 → 中国語モードの寄与度30%）
REACT_PROMPT_TEMPLATE = """
Thought: ...
Action: ...
Observation: ...
Final Answer: ...
"""

# 変更後（日本語メタ言語）
REACT_PROMPT_TEMPLATE = """
思考: 質問に答えるために何をすべきか考える
行動: 実行するツール名
行動入力: ツールへの入力（JSON形式）
観察結果: ツールからの出力（日本語テキスト）

... (この思考/行動/行動入力/観察結果を必要なだけ繰り返す)

思考: 十分な情報が集まったので最終回答を生成できる
最終回答: ユーザーへの最終回答（必ず日本語で）
"""
```

#### エージェントプロンプトの変更

```python
# 変更前
AGENT_SYSTEM_PROMPT = """あなたは渋谷駅周辺の地理空間POI情報に精通したエージェントです。"""

# 変更後
AGENT_SYSTEM_PROMPT = """あなたは東京都内の主要駅周辺（渋谷、新宿、池袋、東京）の
地理空間POI情報に精通したエージェントです。
ユーザーの質問からまず対象エリアを特定し、適切なツールを使って情報を収集してください。

利用可能なエリア:
- shibuya: 渋谷駅周辺
- shinjuku: 新宿駅周辺
- ikebukuro: 池袋駅周辺
- tokyo: 東京駅周辺
"""
```

---

## 6. テストケースの設計

### 6.1 テストケース構造の拡張

```python
@dataclass
class MultiAreaTestCase:
    """複数エリア対応テストケース"""
    id: str                          # "MA-L1-01" 形式
    level: int                       # 1-5
    category: str                    # テストカテゴリ
    subcategory: str                 # サブカテゴリ
    prompt: str                      # 質問文
    expected_keywords: List[str]     # 期待されるキーワード
    difficulty: str                  # easy, medium, hard, expert
    description: str                 # テストの説明
    evaluation_points: List[str]     # 評価ポイント

    # 広域対応フィールド（新規）
    target_area: Optional[str] = None        # 対象エリアキー（None=クロスエリア）
    target_areas: Optional[List[str]] = None # クロスエリアの場合の対象エリアリスト
    query_type: str = "single_area"          # "single_area" or "cross_area"

    # 既存フィールド
    expected_poi_category: Optional[str] = None
    requires_coordinate: bool = False
    requires_reasoning: bool = False
    requires_evidence: bool = False
    constraints: Optional[List[str]] = None
```

### 6.2 subcategory体系

Phase 6-9の既存テストケースで使われたsubcategoryパターンを本実験でも継承する。各エリア20件は以下のsubcategory体系を網羅するように設計する。

#### Phase 6由来（test_cases_v2.py）

| subcategory | 説明 | 対応レベル |
|-------------|------|----------|
| `basic_location` | 基本的な位置情報検索 | L1 |
| `proximity` | 最寄りPOI検索、距離ソート | L2 |
| `aggregation` | カテゴリ別件数、集計 | L2 |
| `comparison` | 東西比較、方角分析 | L2 |
| `constraint_single` | 単一制約条件（24h営業、Wi-Fi等） | L3 |
| `constraint_multi` | 複数制約条件の組み合わせ | L3 |
| `brand` | ブランド・チェーン店検索 | L3 |
| `decision_support` | おすすめ、ランキング | L4 |
| `sensitivity` | 半径変更時の件数変化分析 | L5 |
| `multi_hop` | 複数ステップ推論 | L5 |

#### Phase 8由来（test_cases_graphrag.py）

| subcategory | 説明 | 対応レベル |
|-------------|------|----------|
| `relation` | POI間の空間的関係性 | L4 |
| `competitor` | 同カテゴリ競合分析 | L5 |
| `complementary` | 異カテゴリの補完関係 | L5 |

#### Phase 9-B 新規

| subcategory | 説明 | 対応レベル |
|-------------|------|----------|
| `landmark_origin` | ランドマーク起点の空間推論 | L2-L4 |
| `cross_area_comparison` | エリア間比較 | L4-L5 |
| `area_detection` | エリア特定能力 | L1-L3 |

### 6.3 テストケースの分類と件数

#### 全体構成: 合計130件

```
A. エリア内クエリ（各エリア × 各レベル）:  80件
B. クロスエリアクエリ:                     20件
C. ランドマーク起点クエリ:                  15件
D. エリア特定テスト:                       15件
                                     ──────
                                      130件
```

> Phase 9-B計画時の120件からランドマーク起点クエリ15件を追加し130件に拡大。代わりにクロスエリアクエリを25→20件に調整し、総量を抑制。

### 6.4 A. エリア内クエリ（80件）

各エリア20件 × 4エリア = 80件。渋谷の既存テストケースをベースに各エリアに適応する。
**各エリア20件は上記subcategory体系を均等にカバーする設計**とする。

#### エリア内20件のsubcategory配分

| レベル | subcategory | 件数 |
|--------|-------------|------|
| L1（4件） | `basic_location` ×2, `brand` ×1, `area_detection` ×1 | 4 |
| L2（4件） | `proximity` ×1, `aggregation` ×1, `comparison` ×1, `landmark_origin` ×1 | 4 |
| L3（4件） | `constraint_single` ×1, `constraint_multi` ×1, `brand` ×1, `landmark_origin` ×1 | 4 |
| L4（4件） | `decision_support` ×1, `relation` ×1, `landmark_origin` ×1, `sensitivity` ×1 | 4 |
| L5（4件） | `multi_hop` ×1, `competitor` ×1, `complementary` ×1, `sensitivity` ×1 | 4 |

> 各エリアで同一のsubcategory配分を適用し、エリア間のスコア比較を公平に行えるようにする。

#### 6.4.1 渋谷エリア（20件）- 既存テストから厳選

既存の55件から代表的な20件を選定（各レベル4件、上記subcategory配分に従う）。既存テストの`test_id`との対応を記録する。

```python
# 例
MultiAreaTestCase(
    id="MA-SBY-L1-01",
    level=1,
    subcategory="basic_location",
    target_area="shibuya",
    query_type="single_area",
    prompt="渋谷駅の場所を教えてください",
    expected_keywords=["渋谷", "駅", "35.", "139."],
    # ... 既存L1-01と同等
)
```

#### 6.4.2 新宿エリア（20件）

```python
# L1: 基礎検索（4件）
"新宿駅の場所を教えてください"                          # basic_location
"新宿駅周辺のコンビニを教えてください"                    # basic_location
"新宿駅周辺のスターバックスはありますか？"                 # brand
"新宿駅近くのカフェはありますか？"                        # area_detection

# L2: 空間推論（4件）
"新宿駅に最も近いコンビニはどれですか？"                   # proximity
"新宿駅から500m以内にカフェは何件ありますか？"              # aggregation
"新宿駅の東側と西側、どちらに飲食店が多いですか？"           # comparison
"新宿御苑から最も近いカフェはどこですか？"                  # landmark_origin

# L3: 制約充足（4件）
"新宿駅周辺で24時間営業のコンビニは？"                    # constraint_single
"新宿駅から300m以内でWi-Fiが使えるカフェは？"              # constraint_multi
"新宿駅周辺のドトールを全て教えてください"                  # brand
"東京都庁の近くでランチができるレストランは？"               # landmark_origin

# L4: 意思決定支援（4件）
"新宿駅から近い順にカフェを3つ教えてください"               # decision_support
"新宿駅周辺でカフェとコンビニが両方ある場所は？"             # relation
"歌舞伎町の周辺300mにある飲食店を教えてください"            # landmark_origin
"新宿駅500m圏と1km圏でカフェの件数はどう変わりますか？"     # sensitivity

# L5: 高度推論（4件）
"新宿駅から最も近いカフェと、そこから300m以内の他カフェ数は？"  # multi_hop
"新宿駅周辺でコンビニの競合状況を分析してください"           # competitor
"新宿駅周辺でカフェの近くにある書店を教えてください"          # complementary
"新宿駅周辺のカフェの平均距離と最寄り・最遠の距離差は？"      # sensitivity
```

#### 6.4.3 池袋・東京エリア（各20件）

新宿と同一のsubcategory配分で、エリア固有のランドマーク・POIを反映したテストケースを作成。

**池袋のランドマーク起点例**:
- `"サンシャインシティから最も近いカフェはどこですか？"` (L2, landmark_origin)
- `"池袋西口公園の周辺300mにある飲食店は？"` (L3, landmark_origin)
- `"東武百貨店の近くでランチできる場所は？"` (L4, landmark_origin)

**東京のランドマーク起点例**:
- `"KITTEから最も近いカフェはどこですか？"` (L2, landmark_origin)
- `"東京国際フォーラムの周辺300mの飲食店は？"` (L3, landmark_origin)
- `"丸ビルの近くで朝食が取れるカフェは？"` (L4, landmark_origin)

### 6.5 B. クロスエリアクエリ（20件）

Phase 9-Bの**核心的な新要素**。複数エリアをまたがるクエリで各システムの対応力を評価。

```python
# B1: エリア間比較（8件） subcategory: cross_area_comparison
"渋谷駅と新宿駅の周辺、カフェが多いのはどちらですか？"
"池袋と渋谷で、コンビニの密度が高いのはどちらですか？"
"東京駅と新宿駅の周辺で、飲食店の種類が豊富なのはどちらですか？"
"4エリアの中で最もカフェが多い駅はどこですか？"
"渋谷と池袋、24時間営業の店舗が多いのはどちらですか？"
"スターバックスが最も多いエリアはどこですか？"
"各エリアで最もPOI密度が高いカテゴリは同じですか？"
"4エリアの中で駅から最も近いカフェがあるのはどこですか？"

# B2: エリア間参照（5件） subcategory: cross_area_comparison
"渋谷駅周辺にあるスターバックスは、新宿駅周辺と比べて何店舗差がありますか？"
"東京駅の最寄りカフェと渋谷駅の最寄りカフェ、駅からの距離が近いのはどちらですか？"
"新宿と池袋でマクドナルドの店舗数を比較してください"
"渋谷駅500m圏のコンビニ数と東京駅500m圏のコンビニ数、どちらが多い？"
"各エリアの最寄りコンビニの距離を比較してください"

# B3: 全エリア集計（4件） subcategory: cross_area_comparison
"4エリア全体でコンビニは合計何件ありますか？"
"全エリアで最も多いPOIカテゴリは何ですか？"
"4エリアの総POI数を教えてください"
"全エリアのカフェを合計すると何件ですか？"

# B4: 条件付きクロスエリア（3件） subcategory: cross_area_comparison
"ラーメン店が3件以上ある駅はどこですか？"
"駅から200m以内にカフェが最も多いエリアはどこですか？"
"24時間営業のコンビニが10件以上あるエリアはどこですか？"
```

### 6.6 C. ランドマーク起点クエリ（15件）

駅ではなくランドマークを基準点とした問い。座標はLANDMARKSテーブル（セクション4.1.4）からハードコードで解決する。将来的にはGeocodingツールに置き換え予定。

```python
# C1: ランドマーク起点の空間推論（5件） subcategory: landmark_origin
"渋谷109から最も近いカフェはどこですか？"                    # → 渋谷
"サンシャインシティから500m以内にコンビニは何件ありますか？"     # → 池袋
"東京国際フォーラムの最寄りのレストランを教えてください"        # → 東京
"新宿アルタから一番近い銀行はどこですか？"                   # → 新宿
"KITTEから300m以内のカフェを教えてください"                 # → 東京

# C2: ランドマーク起点の制約充足（5件） subcategory: landmark_origin
"渋谷ヒカリエの近くで24時間営業のコンビニはありますか？"       # → 渋谷
"皇居前広場の近くで食事できるレストランは？"                 # → 東京
"池袋西口公園の周辺でWi-Fiが使えるカフェは？"               # → 池袋
"丸ビルの近くでスターバックスはありますか？"                 # → 東京
"歌舞伎町の近くで深夜営業しているバーは？"                  # → 新宿

# C3: ランドマーク起点の複合推論（5件） subcategory: landmark_origin
"東京都庁から最も近いカフェと、その駅からの距離は？"           # → 新宿, multi_hop
"サンシャインシティの近くにあるカフェとコンビニの数は？"        # → 池袋, aggregation
"渋谷スクランブルスクエアの周辺で飲食店のカテゴリ別件数は？"    # → 渋谷, aggregation
"新宿御苑から最寄りのカフェまでの距離と方角は？"              # → 新宿, proximity
"ハチ公像から見て東側にあるカフェを教えてください"             # → 渋谷, comparison
```

### 6.7 D. エリア特定テスト（15件）

質問文からのエリア特定能力を評価するテストケース。

```python
# D1: 明示的エリア指定（5件） subcategory: area_detection
"池袋駅周辺のホテルを教えてください"           # → 池袋
"東京駅近くの銀行はありますか？"                # → 東京
"渋谷109の近くのカフェを教えてください"          # → 渋谷
"新宿駅西口付近のコンビニはどこですか？"         # → 新宿
"池袋駅から最も近い薬局は？"                   # → 池袋

# D2: 暗黙的エリア推定（5件） subcategory: area_detection
"サンシャインシティの近くのレストランは？"        # → 池袋（ランドマークから推定）
"歌舞伎町の近くで食事できる場所は？"            # → 新宿（地名から推定）
"丸の内で朝食が取れるカフェは？"               # → 東京（地名から推定）
"ハチ公像の周りにコンビニはある？"              # → 渋谷（ランドマークから推定）
"都庁前でランチができる場所は？"               # → 新宿（ランドマークから推定）

# D3: エリア不明（5件） subcategory: area_detection
"おすすめのカフェを教えてください"              # → 全エリア検索
"24時間営業のコンビニはどこにありますか？"       # → 全エリア検索
"美味しいラーメン屋を探しています"              # → 全エリア検索
"Wi-Fiが使える場所を教えてください"            # → 全エリア検索
"近くに薬局はありますか？"                    # → 全エリア検索
```

### 6.6 テストケースファイル構成

```
src/
├── test_cases_multi_area.py      # 広域対応テストケース（120件）
│   ├── MultiAreaTestCase          # データクラス定義
│   ├── SHIBUYA_TESTS = [...]      # 渋谷20件
│   ├── SHINJUKU_TESTS = [...]     # 新宿20件
│   ├── IKEBUKURO_TESTS = [...]    # 池袋20件
│   ├── TOKYO_TESTS = [...]        # 東京20件
│   ├── CROSS_AREA_TESTS = [...]   # クロスエリア25件
│   ├── AREA_DETECTION_TESTS = [...] # エリア特定15件
│   └── ALL_MULTI_AREA_TESTS       # 全120件統合リスト
└── test_cases_v2.py               # 既存（変更なし、後方互換）
```

---

## 7. 評価モジュールの設計

### 7.1 評価指標

#### 7.1.1 既存指標（継続使用）

| 指標 | 説明 | 対象 |
|------|------|------|
| **成功率** | キーワードヒット率 > 0 のケース割合 | 全テスト |
| **キーワードヒット率** | 期待キーワードの平均出現率 | 全テスト |
| **平均実行時間** | クエリあたりの平均実行時間（秒） | 全テスト |
| **エラー率** | クラッシュ・例外の割合 | 全テスト |

#### 7.1.2 新規指標（広域対応）

| 指標 | 説明 | 対象 |
|------|------|------|
| **エリア特定精度** | 質問文からの対象エリア正確な特定率 | エリア特定テスト |
| **クロスエリア成功率** | エリア間比較・参照クエリの成功率 | クロスエリアテスト |
| **エリア別成功率** | エリアごとの成功率（渋谷バイアスの検出） | エリア内テスト |
| **エリア一貫性** | 同じ質問パターンの各エリアでのスコア分散 | エリア内テスト |

### 7.2 評価モジュールの実装

#### ファイル: `src/evaluators_multi_area.py`

```python
@dataclass
class MultiAreaEvalResult:
    """広域対応評価結果"""
    test_id: str
    system_name: str              # "hybrid_rag", "graph_rag", etc.
    target_area: Optional[str]    # 対象エリア
    query_type: str               # "single_area" or "cross_area"

    # 基本指標
    answer: str
    time_sec: float
    keyword_hit_rate: float
    success: bool

    # 広域対応指標
    area_detected: Optional[str]   # システムが特定したエリア
    area_detection_correct: bool   # エリア特定が正しいか

    # エラー情報
    error: Optional[str] = None
    language_issue: bool = False   # 中国語混入等

class MultiAreaEvaluator:
    """広域対応評価クラス"""

    def evaluate_single_case(self, system, test_case: MultiAreaTestCase) -> MultiAreaEvalResult:
        """1テストケースを評価"""
        ...

    def evaluate_all(self, system, test_cases: List[MultiAreaTestCase]) -> List[MultiAreaEvalResult]:
        """全テストケースを評価"""
        ...

    def generate_summary(self, results: List[MultiAreaEvalResult]) -> Dict:
        """評価結果サマリーを生成"""
        summary = {
            "overall": self._calc_overall_metrics(results),
            "by_area": self._calc_by_area(results),
            "by_level": self._calc_by_level(results),
            "by_query_type": self._calc_by_query_type(results),
            "area_detection": self._calc_area_detection(results),
            "cross_area": self._calc_cross_area(results),
        }
        return summary

    def compare_systems(self, all_results: Dict[str, List[MultiAreaEvalResult]]) -> Dict:
        """複数システムの比較分析"""
        comparison = {}
        for system_name, results in all_results.items():
            comparison[system_name] = self.generate_summary(results)

        # システム間の優劣分析
        comparison["rankings"] = self._calc_rankings(comparison)
        comparison["area_consistency"] = self._calc_area_consistency(comparison)
        return comparison
```

### 7.3 エリア特定精度の評価方法

```python
def evaluate_area_detection(self, system, test_case):
    """エリア特定の正確性を評価"""
    # テストケースの期待エリアと、システムが実際に使ったエリアを比較
    expected_area = test_case.target_area

    # システムの内部ログからエリア特定結果を取得
    result = system.query(test_case.prompt)
    detected_area = result.get("detected_area")

    return {
        "expected": expected_area,
        "detected": detected_area,
        "correct": expected_area == detected_area,
    }
```

### 7.4 レポート出力フォーマット

```
results/
├── phase9b_evaluation_{timestamp}.json    # 全結果（JSON）
├── phase9b_summary_{timestamp}.txt        # テキストサマリー
└── phase9b_comparison_{timestamp}.json    # システム間比較
```

---

## 8. 評価ノートブックの設計

### 8.1 ノートブック構成

**ファイル**: `notebooks/phase9b_multi_area_evaluation.ipynb`

```
Cell 1:  環境セットアップ（pip install, GPU確認）
Cell 2:  LLMモデルロード（Qwen2.5-7B-Instruct, 4bit）
Cell 3:  埋め込みモデルロード（multilingual-e5-base）
Cell 4:  srcモジュールのインポート
Cell 5:  POIデータ読み込み（4エリア統合）
Cell 6:  エリア別ベクトルストア構築（ChromaDB × 5 collections）
Cell 7:  POI空間情報付与（エリア別enrich）
Cell 8:  テストケース読み込み（120件）
Cell 9:  RAGシステム初期化（4システム）
Cell 10: Quick Test実行（各エリア5件 = 20件 × 4システム）
Cell 11: Full Test実行（120件 × 4システム）
Cell 12: 結果分析 - 全体比較
Cell 13: 結果分析 - エリア別比較
Cell 14: 結果分析 - クロスエリアクエリ分析
Cell 15: 結果分析 - エリア特定精度
Cell 16: 可視化（グラフ作成）
Cell 17: 結果保存（JSON + テキスト）
```

### 8.2 実行時間の見積もり

| テスト種別 | ケース数 | システム数 | 平均時間/ケース | 合計時間 |
|-----------|---------|-----------|---------------|---------|
| Quick Test | 20 | 4 | ~30秒 | ~40分 |
| Full Test | 120 | 4 | ~30秒 | ~4時間 |

> Colab T4の連続実行上限（約12時間）内に収まる。ただし、Full Testは途中保存を入れる設計にする。

### 8.3 途中保存・再開機能

```python
# 途中結果の自動保存
def run_evaluation_with_checkpoint(system, test_cases, checkpoint_file):
    """チェックポイント付き評価実行"""
    # 既存結果を読み込み
    completed = load_checkpoint(checkpoint_file)

    for tc in test_cases:
        if tc.id in completed:
            continue  # スキップ

        result = evaluate_single_case(system, tc)
        completed[tc.id] = result

        # 10件ごとに自動保存
        if len(completed) % 10 == 0:
            save_checkpoint(checkpoint_file, completed)

    return completed
```

---

## 9. 実装タスクと工数見積もり

### 9.1 タスク一覧

| # | タスク | 優先度 | 推定工数 | 依存 |
|---|--------|--------|---------|------|
| **T1** | `osm_poi_fetcher.py` の広域対応 | 高 | 2h | - |
| **T2** | POIデータ取得実行（4エリア） | 高 | 1h | T1 |
| **T3** | `geo_utils.py` にSTATIONS辞書追加・関数拡張 | 高 | 3h | - |
| **T4** | `aggregator.py` にエリア間比較関数追加 | 中 | 2h | T3 |
| **T5** | `graph_builder.py` の広域対応 | 中 | 2h | T3 |
| **T6** | `structured_rag_system.py` の広域対応 | 高 | 4h | T3, T4 |
| **T7** | `graph_rag_system.py` の広域対応 | 中 | 3h | T5 |
| **T8** | `adaptive_rag_system.py` の広域対応 | 低 | 1h | T6, T7 |
| **T9** | `agent_tools.py` + `agentic_rag_system.py` の広域対応 | 高 | 4h | T3, T4 |
| **T10** | `test_cases_multi_area.py` 作成（120件） | 高 | 6h | T2 |
| **T11** | `evaluators_multi_area.py` 作成 | 高 | 4h | T10 |
| **T12** | 評価ノートブック作成 | 高 | 4h | T6-T11 |
| **T13** | Quick Test実行・デバッグ | 高 | 3h | T12 |
| **T14** | Full Test実行 | 高 | 5h | T13 |
| **T15** | 結果分析・レポート作成 | 高 | 4h | T14 |
| | **合計** | | **~48h** | |

### 9.2 実装順序

```
Phase A: データ基盤（T1→T2→T3→T4→T5）     ~10h
    ↓
Phase B: RAGシステム広域対応（T6→T7→T8→T9）  ~12h
    ↓
Phase C: テスト・評価基盤（T10→T11→T12）      ~14h
    ↓
Phase D: 実行・分析（T13→T14→T15）            ~12h
```

### 9.3 マイルストーン

| マイルストーン | 達成条件 | 目標日（着手日起算） |
|--------------|---------|-------------------|
| **M1: データ準備完了** | 4エリアPOIデータ取得、統合ファイル生成 | 着手+2日 |
| **M2: コアモジュール完了** | geo_utils, aggregator, graph_builder広域対応 | 着手+4日 |
| **M3: RAGシステム完了** | 4システム全ての広域対応完了 | 着手+7日 |
| **M4: テスト基盤完了** | 120テストケース＋評価モジュール完成 | 着手+9日 |
| **M5: Quick Test通過** | 20件Quick Testで全システム動作確認 | 着手+10日 |
| **M6: Full Test完了** | 120件×4システムの評価完了 | 着手+12日 |
| **M7: レポート完成** | 分析レポート作成・Phase 10への提言 | 着手+14日 |

---

## 10. リスクと対策

### 10.1 技術リスク

| リスク | 影響度 | 発生確率 | 対策 |
|--------|-------|---------|------|
| **Overpass APIレートリミット** | 中 | 中 | エリア間に10秒待機、失敗時リトライ |
| **ChromaDBメモリ不足**（~3,600 POI） | 中 | 低 | collection分割、メモリ使用量モニタリング |
| **Colab T4タイムアウト**（Full Test ~4h） | 高 | 中 | チェックポイント機能、バッチ分割実行 |
| **中国語混入の増加**（新エリアで） | 中 | 高 | Phase 9で判明済み、評価指標として計測 |
| **新エリアのPOIデータ品質** | 中 | 中 | データ取得後に手動サンプリング確認 |

### 10.2 実験設計リスク

| リスク | 影響度 | 発生確率 | 対策 |
|--------|-------|---------|------|
| **テストケースの地域バイアス** | 高 | 中 | 各エリア同数・同構造のテストケースで統制 |
| **渋谷のテストケースとの整合性** | 中 | 低 | 渋谷20件は既存55件からの厳選で対応付け記録 |
| **クロスエリアの期待回答作成が困難** | 高 | 高 | POIデータ取得後に期待キーワードを確定 |

### 10.3 リスク緩和のための段階的実行

```
Step 1: 渋谷＋新宿の2エリアでプロトタイプ評価
    → 動作確認、エリア特定ロジックの検証
    → 問題なければStep 2へ

Step 2: 池袋＋東京を追加して4エリアで評価
    → Full Testの実行

Step 3: 結果分析・レポート
```

---

## 11. 成果物一覧

### 11.1 ソースコード

| ファイル | 種別 | 説明 |
|---------|------|------|
| `osm_poi_fetcher.py` | 変更 | 4エリア対応 |
| `src/geo_utils.py` | 変更 | STATIONS辞書、エリア特定関数 |
| `src/aggregator.py` | 変更 | エリア間比較関数追加 |
| `src/graph_builder.py` | 変更 | 動的station対応 |
| `src/structured_rag_system.py` | 変更 | 広域対応 |
| `src/graph_rag_system.py` | 変更 | エリア別グラフ |
| `src/adaptive_rag_system.py` | 変更 | 下位システム連動 |
| `src/agentic_rag_system.py` | 変更 | 広域対応 |
| `src/agent_tools.py` | 変更 | エリア別ツール |
| `src/agent_prompts.py` | 変更 | プロンプト汎用化 |
| `src/test_cases_multi_area.py` | **新規** | 120テストケース |
| `src/evaluators_multi_area.py` | **新規** | 広域対応評価モジュール |

### 11.2 データ

| ファイル | 説明 |
|---------|------|
| `data/poi_shibuya.json` | 渋谷POIデータ |
| `data/poi_shinjuku.json` | 新宿POIデータ |
| `data/poi_ikebukuro.json` | 池袋POIデータ |
| `data/poi_tokyo.json` | 東京POIデータ |
| `data/poi_all_areas.json` | 全エリア統合POIデータ |
| `poi_documents.json` | 渋谷のみ（後方互換） |

### 11.3 評価

| ファイル | 説明 |
|---------|------|
| `notebooks/phase9b_multi_area_evaluation.ipynb` | 評価ノートブック |
| `results/phase9b_evaluation_{timestamp}.json` | 評価結果 |
| `results/phase9b_summary_{timestamp}.txt` | テキストサマリー |

### 11.4 ドキュメント

| ファイル | 説明 |
|---------|------|
| `docs/plans/PHASE9B_MULTI_AREA_EXPERIMENT_PLAN.md` | 本計画書 |
| `docs/reports/PHASE9B_MULTI_AREA_EXPERIMENT_REPORT.md` | 実験結果レポート（実験後作成） |

---

## 12. 期待される成果と判断基準

### 12.1 成功基準

| 指標 | 目標 |
|------|------|
| 全テスト完走率 | 100%（クラッシュなし） |
| 渋谷エリアのスコア維持 | 既存スコアの±5%以内 |
| 新エリアのスコア | 渋谷スコアの80%以上 |
| エリア特定精度 | 明示的指定: 95%以上、暗黙的推定: 70%以上 |
| 4システム全ての評価完了 | 全120件 × 4システム |

### 12.2 Phase 10への判断基準

| 結果パターン | Phase 10への推奨 |
|-------------|----------------|
| Hybrid RAGが全エリアで安定（90%+） | Hybrid RAGベースで全国展開 |
| Agentic RAGがクロスエリアで優位 | Hybrid + Agentic RAGのハイブリッド |
| 全システムで新エリアスコアが大幅低下 | エリア特定ロジック強化が先 |
| Graph RAGがエリア間関係で優位 | Graph RAG要素をHybridに統合 |

---

## 13. 期待される結果と得られる知見

### 13.1 各RAGシステムの予測パフォーマンス

Phase 9までの結果と広域化による影響を踏まえ、各システムの予測を以下に示す。

#### Hybrid RAG（構造化RAG）

| 指標 | 渋谷単一（Phase 9実績） | 広域予測 | 予測根拠 |
|------|----------------------|---------|---------|
| エリア内成功率 | 96.2% | 85〜92% | ルールベース質問分析はエリア非依存だが、ベクトル検索の精度がPOI数増加で希薄化する可能性 |
| クロスエリア成功率 | - | 50〜70% | 現行の単一エリア前提設計では、エリア間比較の構造化処理が未実装 |
| 実行時間 | 11.1秒 | 12〜15秒 | POI数増加（約3.5倍）の影響は限定的（空間計算はエリア別フィルタ後に実行） |

**鍵となる予測**: Hybrid RAGの強みである「ルールベース質問分析 → 構造化計算 → 日本語自然文コンテキスト」のパイプラインは本質的にエリア非依存であり、エリア特定さえ正しく行えれば高いパフォーマンスを維持すると予想する。一方、クロスエリアクエリは既存のアーキテクチャが想定していないため、成功率は低くなる見込み。

#### Graph RAG

| 指標 | 渋谷単一（Phase 8実績） | 広域予測 | 予測根拠 |
|------|----------------------|---------|---------|
| エリア内成功率 | 76.7% | 70〜78% | グラフ構造はエリア内完結のため劣化は少ないが、元の精度が低い |
| クロスエリア成功率 | - | 30〜50% | エリア別に独立したグラフになるため、エリア間トラバーサルが困難 |
| 実行時間 | 未計測 | 15〜25秒 | エリア別グラフは小規模のため高速 |

**鍵となる予測**: Graph RAGはエリア内のPOI間関係（近接・同カテゴリ・同ブランド）の推論に強みがあるが、エリア間をまたぐ関係の表現が構造的に困難。ただし「池袋にもある同じブランドの店舗」のようなブランドノードを介したクロスエリア推論が可能であれば、予想を上回る可能性がある。

#### Adaptive RAG

| 指標 | 渋谷単一（Phase 8実績） | 広域予測 | 予測根拠 |
|------|----------------------|---------|---------|
| エリア内成功率 | 86.1% | 78〜88% | 下位システムの広域対応品質に依存 |
| クロスエリア成功率 | - | 40〜60% | ルーティングが適切なら下位システムの強みを活かせる |
| 実行時間 | 未計測 | 15〜30秒 | ルーティング判定のオーバーヘッド + 下位システムの実行時間 |

**鍵となる予測**: Adaptive RAGのルーティングロジック（クエリ複雑度によるHybrid/Graph切り替え）に「クロスエリアクエリ」という新しい軸が加わる。既存のルーティングテーブルがエリア横断の状況で適切に機能するかが焦点。クロスエリアクエリをHybrid RAGに回すルールを追加できれば、全体として安定した成績が見込める。

#### Agentic RAG

| 指標 | 渋谷単一（Phase 9実績） | 広域予測 | 予測根拠 |
|------|----------------------|---------|---------|
| エリア内成功率 | 87.6% | 75〜85% | ツールにarea引数が追加されるため、LLMが正しいエリアを指定できるかが課題 |
| クロスエリア成功率 | - | 55〜75% | エージェントの自律的な複数ツール呼び出しでクロスエリア対応が可能 |
| 実行時間 | 56.4秒 | 60〜90秒 | クロスエリアではツール呼び出し回数が増加 |

**鍵となる予測**: Agentic RAGにとって広域対応は**最も挑戦的かつ最もポテンシャルがある**変更になる。ツールのarea引数をLLMが正しく選択する必要があり、エリア内クエリでは難易度が上がる。しかしクロスエリアクエリでは、複数エリアのツールを自律的に呼び分ける能力が他のシステムにない強みとなりうる。ただし、中国語混入問題がツール引数増加（area引数の追加）で悪化する可能性がある。

### 13.2 検証する仮説

本実験で検証する主要仮説を以下に定義する。各仮説には、実験結果からの判定基準を付記する。

#### H1: 構造化パイプラインのエリア汎化性

> **仮説**: Hybrid RAGの「質問分析 → 構造化計算 → 日本語自然文コンテキスト」パイプラインは、エリア特定が正しければ渋谷以外でも同等の精度を維持する。

| 判定 | 条件 |
|------|------|
| **支持** | 新エリア（新宿・池袋・東京）のエリア内成功率が渋谷の90%以上 |
| **部分的支持** | 新エリアのエリア内成功率が渋谷の80〜90% |
| **棄却** | 新エリアのエリア内成功率が渋谷の80%未満 |

**この仮説が重要な理由**: Phase 10の全国展開でHybrid RAGアプローチを採用するか否かの判断に直結する。渋谷でのみ高精度なのか、アーキテクチャとして汎用性があるのかを確認する。

#### H2: エージェント型のクロスエリア優位性

> **仮説**: Agentic RAGは、複数エリアのツールを自律的に呼び分けることで、クロスエリアクエリにおいて他のシステムより高い成功率を達成する。

| 判定 | 条件 |
|------|------|
| **支持** | Agentic RAGのクロスエリア成功率が他システムの最高値を10%以上上回る |
| **部分的支持** | Agentic RAGのクロスエリア成功率が最高だが差は10%未満 |
| **棄却** | Agentic RAGのクロスエリア成功率が他システム以下 |

**この仮説が重要な理由**: Phase 9ではAgentic RAGの優位性が限定的（3カテゴリのみ）だったが、クロスエリアという新しいタスク次元がその真価を発揮する場面になりうる。クロスエリアで明確な優位性があれば、Phase 10でHybrid + Agenticのハイブリッド構成を検討する根拠になる。

#### H3: エリア特定がボトルネックになる

> **仮説**: 広域対応において、RAGシステム自体の推論能力よりも「質問文からのエリア特定」の精度が全体の成功率を律速する。

| 判定 | 条件 |
|------|------|
| **支持** | エリア特定の誤りが失敗ケース全体の50%以上を占める |
| **部分的支持** | エリア特定の誤りが失敗ケースの30〜50%を占める |
| **棄却** | エリア特定の誤りが失敗ケースの30%未満 |

**この仮説が重要な理由**: もしエリア特定がボトルネックなら、Phase 10ではRAGアーキテクチャよりもエリア特定ロジック（NER、ジオコーディングAPI連携等）への投資が優先される。

#### H4: POI数の増加が検索精度に影響する

> **仮説**: POI総数が約3.5倍（1,047→~3,600）になることで、ベクトル検索の精度が低下し、特にカテゴリが多様なエリア（新宿）でノイズが増加する。

| 判定 | 条件 |
|------|------|
| **支持** | 全エリア統合collectionでの検索がエリア別collectionより成功率10%以上低い |
| **部分的支持** | 差が5〜10% |
| **棄却** | 差が5%未満 |

**この仮説が重要な理由**: Phase 10では500万POI規模を扱う。エリア別インデックス分割がどの程度有効かの知見は、PostGISのインデックス設計に直結する。

#### H5: 中国語混入がエリアによって変動する

> **仮説**: Agentic RAGの中国語混入率はエリアによって変動しない（QwenモデルとJSON出力形式の構造的問題であり、入力データの地名に依存しない）。

| 判定 | 条件 |
|------|------|
| **支持** | 4エリア間の中国語混入率の差が±3%以内 |
| **棄却** | 特定エリアで混入率が突出（例: 東京駅の英語表記"Tokyo"が英語→中国語をトリガー） |

**この仮説が重要な理由**: 中国語混入の原因がモデル固有なのかデータ依存なのかを切り分けることで、Phase 10でのモデル選定判断の材料になる。

### 13.3 得られると期待される知見

#### 13.3.1 アーキテクチャに関する知見

**A1: エリア特定戦略の最適解**

キーワードマッチングによるエリア特定（`detect_target_area()`）の精度と限界が明らかになる。具体的には：
- 明示的な駅名指定ではほぼ100%の精度が出るはず
- ランドマーク名（サンシャインシティ→池袋）による暗黙的推定の実用可能性
- エリア不明クエリ（全エリア検索フォールバック）の品質
- Phase 10でNERやジオコーディングAPIがどの程度必要かの判断材料

**A2: ベクトルストア分割戦略の有効性**

エリア別collection vs 全統合collectionの検索精度の差が定量化される。これにより：
- PostGISでのパーティション設計（エリア別テーブル vs 単一テーブル＋空間インデックス）の指針
- 検索時のフィルタリング戦略（事前フィルタ vs 事後フィルタ）の選択基準

**A3: 構造化パイプラインの汎用性評価**

Phase 6で確立した「質問分析→構造化計算→日本語自然文→LLM生成」のパイプラインが、データを差し替えるだけで新エリアに適用できるかが実証される。これはPhase 10のアーキテクチャ設計の根幹に関わる知見。

#### 13.3.2 RAGアプローチの適性に関する知見

**B1: タスクタイプとRAGアプローチの適性マトリクス拡張**

Phase 8-9で作成した「カテゴリ別のRAGシステム適性マップ」に、クロスエリアという新しい次元が加わる。

```
                  エリア内  クロスエリア
Hybrid RAG:        ◎         △ or ○?
Graph RAG:         ○         △?
Adaptive RAG:      ○         △ or ○?
Agentic RAG:       ○         ○ or ◎?
```

この適性マトリクスが埋まることで、Phase 10で「どのクエリタイプにどのRAGアプローチを適用するか」の設計指針が得られる。

**B2: Agentic RAGの真の評価**

Phase 9ではAgentic RAGが全体で8.6%劣後したが、その主因は中国語混入問題であった。本実験では以下が明確になる：
- 中国語混入を除外した場合の「純粋な推論能力」の比較
- クロスエリアという新しいタスクでのエージェント的アプローチの真価
- ツール引数の複雑化（area追加）がLLMのツール選択精度に与える影響

**B3: Graph RAGのスケール特性**

エリア別に独立したグラフを構築した場合の精度と、Phase 8（渋谷単一グラフ）との比較により：
- グラフの規模と精度の関係（小さいグラフの方が精度が高いのか）
- エリア間関係のグラフ表現の限界

#### 13.3.3 Phase 10設計に直結する知見

**C1: 全国展開のアーキテクチャ選定**

4エリアの結果から、以下の設計判断を行う根拠データが得られる：

| 判断項目 | 本実験で得られるデータ |
|---------|-------------------|
| ベースRAGアプローチ | 4システムのエリア間一貫性と成功率 |
| エリア特定の実装方式 | キーワードマッチの限界値 → NER/ジオコーディング必要性 |
| データベース分割戦略 | collection分割の効果 → PostGISパーティション設計 |
| クロスエリア対応の必要性 | クロスエリアクエリの実需と各システムの対応力 |
| モデル変更の必要性 | 中国語混入のエリア依存性 → Qwen継続 or Llama等への移行 |

**C2: スケーラビリティの限界点の特定**

1,047 POI → ~3,600 POI への拡大で性能がどう変化するかの観測データにより、500万POIへのスケール時に発生する問題を事前に予測できる。特に：
- ベクトル検索のレイテンシ増加曲線
- 空間計算のPOI数に対するスケーリング特性
- LLMコンテキストに含めるべきPOI数の上限

**C3: テストケース設計パターンの確立**

120件のテストケース設計を通じて、POI RAGシステムの評価に必要なテストパターンの体系が確立される。Phase 10の全国展開時には47都道府県対応のテストケースを作成する必要があり、本実験のテンプレートが再利用可能になる。

### 13.4 予測される結果シナリオ

最も蓋然性の高いシナリオとそこから導かれるアクションを3パターン想定する。

#### シナリオ A: Hybrid RAGの汎化成功（確率: 50%）

```
結果:
  Hybrid RAG エリア内: 88〜93% (渋谷以外でも安定)
  Hybrid RAG クロスエリア: 50〜65%
  Agentic RAG クロスエリア: 60〜75%
  エリア特定精度: 85%以上

→ 知見: 構造化パイプラインは汎用的。クロスエリアではAgenticに一定の優位性。
→ Phase 10アクション: Hybrid RAGベースで全国展開。
   クロスエリア機能はオプショナルとしてAgentic要素を統合検討。
```

#### シナリオ B: エリア特定がボトルネック（確率: 30%）

```
結果:
  全システム エリア内: 70〜82% (渋谷より大幅低下)
  失敗原因の50%以上がエリア誤特定
  渋谷エリアのみ: 90%以上維持

→ 知見: RAGアプローチより前段のエリア特定が律速。
→ Phase 10アクション: ジオコーディングAPI統合を優先実装。
   RAGアプローチの選定は本実験では確定せず、エリア特定改善後に再評価。
```

#### シナリオ C: 全体的な精度低下（確率: 20%）

```
結果:
  全システムで渋谷を含め精度低下
  POI数増加によるベクトル検索ノイズが主因
  エリア別collectionの方が統合より10%以上高精度

→ 知見: POI数の増加がベクトル検索の精度を圧迫。
→ Phase 10アクション: PostGISの空間インデックスによる事前フィルタリングが必須。
   ベクトル検索の前段にSQLベースの空間フィルタを置くアーキテクチャが必要。
```

---

**文書バージョン**: 1.1
**最終更新**: 2026-02-18
**次のアクション**: Issue #9 から実装着手
