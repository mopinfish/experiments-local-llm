# 構造化RAG データ設計ガイド

**作成日**: 2026年1月23日  
**対象**: experiments-local-llm Phase 6

---

## 1. 構造化データとテストカテゴリの対応表

### 1.1 空間エンリッチメント（事前付与）

| 付与データ | データ型 | サンプル値 | 処理タイミング |
|-----------|---------|-----------|---------------|
| `distance_from_station` | float | `245.3` (メートル) | RAG構築時 |
| `direction_from_station` | string | `"northeast"` | RAG構築時 |

**データサンプル（エンリッチメント前後）**:

```python
# Before
{
    "name": "スターバックス渋谷店",
    "category": "飲食店/カフェ",
    "lat": 35.659782,
    "lon": 139.703415
}

# After（エンリッチメント後）
{
    "name": "スターバックス渋谷店",
    "category": "飲食店/カフェ",
    "lat": 35.659782,
    "lon": 139.703415,
    "distance_from_station": 245.3,      # ← 追加
    "direction_from_station": "northeast" # ← 追加
}
```

**寄与するテストカテゴリ**:

| テストカテゴリ | 改善幅 | テストケースサンプル | なぜ寄与するか |
|---------------|--------|---------------------|---------------|
| `basic_location` | +25.1pt | 「渋谷駅の場所を教えてください」 | 座標情報がコンテキストに含まれる |
| `spatial_comparison` | +40.6pt | 「渋谷駅の東側と西側、どちらにカフェが多い？」 | 方角情報で東西の分類が可能 |
| `spatial_density` | +15.8pt | 「渋谷駅周辺で飲食店が集中しているのはどこ？」 | 距離・方角で分布を分析可能 |

---

### 1.2 集計・比較機能（クエリ時実行）

| 機能 | 出力データ | サンプル出力 | 処理タイミング |
|-----|-----------|-------------|---------------|
| `compare_east_west()` | ComparisonResult | `{east: 51, west: 58, winner: "west"}` | クエリ時 |
| `get_top_categories()` | List[CategoryCount] | `[{category: "カフェ", count: 149}, ...]` | クエリ時 |
| `filter_by_category()` | List[POI] | `[{name: "スタバ", ...}, ...]` | クエリ時 |

**出力サンプル（東西比較）**:

```python
# compare_east_west(pois, category="カフェ") の出力
ComparisonResult(
    east_count=51,
    west_count=58,
    winner="west",
    difference=7,
    east_percentage=46.8,
    west_percentage=53.2
)

# LLMに渡されるコンテキスト
"""
【東西比較: カフェ】
東側: 51件 (46.8%)
西側: 58件 (53.2%)
→ 西側が7件多い
"""
```

**寄与するテストカテゴリ**:

| テストカテゴリ | 改善幅 | テストケースサンプル | なぜ寄与するか |
|---------------|--------|---------------------|---------------|
| `spatial_comparison` | +40.6pt | 「渋谷駅の東側と西側、どちらにカフェが多い？」 | 東西の件数を正確に集計・比較 |
| `basic_category` | +23.0pt | 「渋谷にあるカフェを3つ教えて」 | カテゴリフィルタで正確に抽出 |
| `decision_business` | +34.0pt | 「新規カフェの出店余地はある？」 | 競合数の正確な集計 |
| `advanced_comparison` | +31.3pt | 「映画館と劇場、どちらが多い？」 | 複数カテゴリの比較集計 |

---

### 1.3 近接性検索（クエリ時実行）

| 機能 | 出力データ | サンプル出力 | 処理タイミング |
|-----|-----------|-------------|---------------|
| `get_nearest_pois()` | List[POI] (距離順) | `[{name: "ローソン", distance: 102}, ...]` | クエリ時 |
| `filter_by_radius()` | List[POI] | 半径内のPOIリスト | クエリ時 |

**出力サンプル（最近傍検索）**:

```python
# get_nearest_pois(pois, category="コンビニ", top_n=3) の出力
[
    {"name": "ローソン渋谷駅前", "distance_from_station": 102, "direction": "south"},
    {"name": "セブンイレブン道玄坂", "distance_from_station": 156, "direction": "west"},
    {"name": "ファミマ渋谷センター街", "distance_from_station": 189, "direction": "northwest"}
]

# LLMに渡されるコンテキスト
"""
【コンビニの最寄り3件】
1. ローソン渋谷駅前 - 102m (南方向)
2. セブンイレブン道玄坂 - 156m (西方向)
3. ファミマ渋谷センター街 - 189m (北西方向)
"""
```

**寄与するテストカテゴリ**:

| テストカテゴリ | 改善幅 | テストケースサンプル | なぜ寄与するか |
|---------------|--------|---------------------|---------------|
| `spatial_proximity` | +34.0pt | 「渋谷駅に最も近いコンビニは？距離も推定して」 | 距離順ソートで最近傍を特定 |
| `constraint_single` | +32.0pt | 「渋谷駅から徒歩5分以内のカフェを教えて」 | 半径フィルタで距離制約を満たす |
| `constraint_multi` | +44.7pt | 「駅から近く、映画館の近くにあるカフェは？」 | 複数基準点からの距離計算 |
| `decision_location` | +35.1pt | 「保育園の最適な場所は？公園の近さを考慮」 | 複数POIとの距離関係を分析 |

---

### 1.4 感度分析（クエリ時実行）

| 機能 | 出力データ | サンプル出力 | 処理タイミング |
|-----|-----------|-------------|---------------|
| `compare_by_radius()` | RadiusComparisonResult | `{r1: 300, r2: 500, count1: 37, count2: 86}` | クエリ時 |

**出力サンプル（感度分析）**:

```python
# compare_by_radius(pois, 300, 500, category="カフェ") の出力
RadiusComparisonResult(
    radius1_m=300,
    radius2_m=500,
    count1=37,
    count2=86,
    category="カフェ",
    difference=49,
    ratio=2.32
)

# LLMに渡されるコンテキスト
"""
【感度分析: カフェ】
半径300m: 37件
半径500m: 86件 (+49件、2.32倍)

【結論】
半径を変えると件数が大きく変化するため、結論は条件に依存します。
"""
```

**寄与するテストカテゴリ**:

| テストカテゴリ | 改善幅 | テストケースサンプル | なぜ寄与するか |
|---------------|--------|---------------------|---------------|
| `advanced_sensitivity` | +40.0pt | 「カフェが多いという結論は、半径を500mから300mに変えても成立する？」 | 複数半径での件数比較と結論生成 |
| `advanced_uncertainty` | +15.3pt | 「コンビニ密度が十分と言える最小半径は？」 | 半径ごとの件数変化を分析 |

---

## 2. 対応表サマリー

| 構造化データ/機能 | 付与タイミング | 主な寄与カテゴリ | 改善幅合計 |
|------------------|---------------|-----------------|-----------|
| **空間エンリッチメント** | RAG構築時 | basic_location, spatial_comparison, spatial_density | +81.5pt |
| **集計・比較機能** | クエリ時 | spatial_comparison, basic_category, decision_business | +129.5pt |
| **近接性検索** | クエリ時 | spatial_proximity, constraint_*, decision_location | +145.8pt |
| **感度分析** | クエリ時 | advanced_sensitivity, advanced_uncertainty | +55.3pt |

---

## 3. 展開段階別の実装注意点

### 3.1 複数エリア対応時（渋谷 + 新宿 + 横浜など）

#### 変更が必要な箇所

| 現在の実装 | 問題点 | 改善案 |
|-----------|--------|--------|
| `SHIBUYA_STATION = (35.658, 139.701)` | 基準点が固定 | エリアごとの基準点マッピング |
| 質問分析で「渋谷」をハードコード | 他エリア名に対応しない | パターンマッチングに変更 |
| テストケースが渋谷のみ | 他エリアの検証不可 | 複数エリアのテストケース追加 |

#### 具体的な実装変更

```python
# 改善案1: エリア別基準点マッピング
AREA_STATIONS = {
    "渋谷": (35.658034, 139.701636),
    "新宿": (35.689607, 139.700571),
    "横浜": (35.465786, 139.622313),
    "池袋": (35.729503, 139.710999),
}

def resolve_reference_point(question: str) -> tuple:
    """質問文からエリアを特定し、基準点を返す"""
    for area_name, coords in AREA_STATIONS.items():
        if area_name in question:
            return coords
    return None  # エリア特定不可

# 改善案2: エリア別POIデータの管理
class MultiAreaPOIManager:
    def __init__(self):
        self.pois_by_area = {
            "渋谷": load_pois("shibuya_pois.json"),
            "新宿": load_pois("shinjuku_pois.json"),
            # ...
        }
    
    def get_pois(self, area: str) -> List[POI]:
        return self.pois_by_area.get(area, [])
```

#### 注意点

1. **エリア間のPOI重複**: 渋谷と原宿の境界付近のPOIをどう扱うか
2. **基準点の選定**: 各エリアの「中心」をどう定義するか（駅？エリア重心？）
3. **テストケースの網羅性**: 各エリアで同等のテストカテゴリをカバーする
4. **エンリッチメントの再実行**: 新エリア追加時に事前計算が必要

---

### 3.2 全国展開対応時（数百万POI）

#### パフォーマンス課題

| 処理 | 現在（1,047件） | 全国（500万件） | 対策 |
|------|----------------|----------------|------|
| 空間エンリッチメント | 10ms | **50秒** ❌ | 事前計算廃止、PostGIS |
| 最近傍検索 | 10ms | **50秒** ❌ | 空間インデックス |
| カテゴリ集計 | 5ms | **25秒** ❌ | DB集計 + キャッシュ |

#### 推奨アーキテクチャ

```
┌─────────────────────────────────────────────────────────────┐
│                    質問分析                                  │
│         「新宿駅に最も近いカフェは？」                        │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│              基準点解決                                      │
│         「新宿駅」→ (35.6896, 139.6917)                     │
│         ※ ジオコーディングAPI or POI検索                     │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│              PostGIS 空間検索（10-50ms）                     │
│                                                             │
│  SELECT *, ST_Distance(geom, ref_point) as distance         │
│  FROM pois                                                  │
│  WHERE ST_DWithin(geom, ref_point, 1000)                    │
│    AND category LIKE '%カフェ%'                              │
│  ORDER BY distance                                          │
│  LIMIT 50;                                                  │
│                                                             │
│  → 500万件から50件に絞り込み                                 │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│              構造化RAG処理（現在のロジックを適用）            │
│                                                             │
│  # 候補50件に対してのみ実行                                  │
│  nearest = get_nearest_pois(candidates, ref_point)          │
│  comparison = compare_east_west(candidates)                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                           ↓
                    LLM回答生成
```

#### 具体的な実装変更

```python
# 改善案: PostGIS統合版の構造化RAG
class NationwideStructuredRAG:
    def __init__(self, db_connection):
        self.db = db_connection  # PostGIS対応DB
    
    def query(self, question: str):
        # 1. 質問分析
        analysis = analyze_question(question)
        
        # 2. 基準点解決
        reference_point = self._resolve_reference(question)
        if not reference_point:
            return self._fallback_response(question)
        
        # 3. PostGISで候補取得（ここで500万→50件に）
        candidates = self._spatial_search(
            reference_point=reference_point,
            radius_m=analysis.distance_constraint or 1000,
            category=analysis.subcategories[0] if analysis.subcategories else None,
            limit=50
        )
        
        # 4. 候補に対して構造化処理（現在のロジック）
        context = self._build_context(question, analysis, candidates, reference_point)
        
        # 5. LLM回答生成
        return self._generate(context, question)
    
    def _spatial_search(self, reference_point, radius_m, category, limit):
        """PostGISによる空間検索"""
        query = """
            SELECT 
                name, category, lat, lon,
                ST_Distance(
                    geom::geography, 
                    ST_MakePoint(%s, %s)::geography
                ) as distance
            FROM pois
            WHERE ST_DWithin(
                geom::geography, 
                ST_MakePoint(%s, %s)::geography, 
                %s
            )
        """
        params = [reference_point[1], reference_point[0],  # lon, lat
                  reference_point[1], reference_point[0], radius_m]
        
        if category:
            query += " AND category LIKE %s"
            params.append(f"%{category}%")
        
        query += " ORDER BY distance LIMIT %s"
        params.append(limit)
        
        return self.db.execute(query, params)
```

#### 注意点

1. **事前エンリッチメントの廃止**: 基準点が動的なため、距離・方角は都度計算
2. **空間インデックスの作成**: `CREATE INDEX idx_pois_geom ON pois USING GIST(geom);`
3. **クエリ最適化**: 半径を段階的に広げる戦略（まず500m、結果が少なければ1km）
4. **カテゴリマッピング**: 全国で統一されたカテゴリ体系の整備
5. **キャッシュ戦略**: 人気エリア（東京駅周辺など）の結果をキャッシュ

---

### 3.3 MCPサーバとして実装する際

#### アーキテクチャの違い

| 観点 | 現在のRAG | MCPサーバ |
|------|----------|----------|
| データソース | ローカルJSON + ChromaDB | MapFan API（リアルタイム） |
| 処理の主体 | RAGシステム | LLM（ツール呼び出し） |
| コンテキスト構築 | RAGが一括構築 | LLMが必要に応じてツール呼び出し |
| エンリッチメント | 事前処理 | APIレスポンスに含まれる or 都度計算 |

#### MCPツール設計

```typescript
// 推奨するMCPツール構成
const tools = [
  // 基本検索
  {
    name: "search_poi",
    description: "指定条件でPOIを検索",
    parameters: {
      keyword: "検索キーワード",
      category: "カテゴリ",
      lat: "中心緯度",
      lon: "中心経度",
      radius: "検索半径(m)"
    }
  },
  
  // 近接性検索（Phase 6.2の成果）
  {
    name: "get_nearest_pois",
    description: "指定地点に最も近いPOIを取得",
    parameters: {
      lat: "基準点緯度",
      lon: "基準点経度",
      category: "カテゴリ",
      limit: "取得件数"
    }
  },
  
  // 比較分析（Phase 6.1の成果）
  {
    name: "compare_areas",
    description: "2つのエリアのPOI数を比較",
    parameters: {
      lat: "中心緯度",
      lon: "中心経度",
      category: "カテゴリ",
      compare_type: "east_west | north_south"
    }
  },
  
  // 感度分析（Phase 6.2の成果）
  {
    name: "analyze_radius_sensitivity",
    description: "半径を変えた場合のPOI数の変化を分析",
    parameters: {
      lat: "中心緯度",
      lon: "中心経度",
      category: "カテゴリ",
      radius1: "比較半径1(m)",
      radius2: "比較半径2(m)"
    }
  },
  
  // 集計
  {
    name: "get_category_stats",
    description: "指定エリアのカテゴリ別POI数を集計",
    parameters: {
      lat: "中心緯度",
      lon: "中心経度",
      radius: "集計半径(m)"
    }
  }
];
```

#### 共通ライブラリの切り出し

```
geo-common/                        # 共通ライブラリ
├── src/
│   ├── distance.ts               # 距離計算
│   │   ├── calculateDistance()
│   │   └── calculateBearing()
│   │
│   ├── direction.ts              # 方角計算
│   │   ├── getDirection8()       # 8方位
│   │   └── classifyEastWest()    # 東西判定
│   │
│   ├── aggregation.ts            # 集計処理
│   │   ├── compareEastWest()
│   │   ├── getTopCategories()
│   │   └── filterByCategory()
│   │
│   ├── proximity.ts              # 近接性処理
│   │   ├── getNearestPois()
│   │   └── filterByRadius()
│   │
│   └── sensitivity.ts            # 感度分析
│       ├── compareByRadius()
│       └── analyzeRadiusSensitivity()
│
└── package.json

mapfan-mcp-server/                 # MCPサーバ
├── src/
│   └── tools/
│       ├── search_poi.ts         # MapFan API呼び出し
│       ├── nearest_poi.ts        # geo-common.getNearestPois使用
│       ├── compare_areas.ts      # geo-common.compareEastWest使用
│       └── sensitivity.ts        # geo-common.compareByRadius使用
└── package.json                   # geo-common を依存に追加
```

#### MCPサーバ実装時の注意点

1. **ツールの粒度設計**
   - 細かすぎる: LLMが複数回呼び出す必要あり → レイテンシ増
   - 粗すぎる: 柔軟性が失われる
   - **推奨**: 「1つの質問タイプ = 1つのツール」の粒度

2. **レスポンス形式の設計**
   ```typescript
   // LLMが解釈しやすい形式
   {
     "summary": "渋谷駅に最も近いコンビニはローソン（102m）です",
     "data": [
       {"name": "ローソン", "distance": 102, "direction": "south"},
       // ...
     ],
     "metadata": {
       "total_count": 15,
       "search_radius": 500
     }
   }
   ```

3. **エラーハンドリング**
   - 基準点が特定できない場合
   - 該当POIが0件の場合
   - API制限に達した場合

4. **MapFan APIとの統合**
   ```typescript
   async function getNearestPois(lat, lon, category, limit) {
     // 1. MapFan APIでPOI取得
     const pois = await mapfanApi.searchPoi({
       lat, lon, 
       category,
       radius: 1000,
       limit: limit * 2  // 余裕を持って取得
     });
     
     // 2. 共通ライブラリで距離計算・ソート
     const enriched = pois.map(poi => ({
       ...poi,
       distance: calculateDistance([lat, lon], [poi.lat, poi.lon]),
       direction: getDirection8([lat, lon], [poi.lat, poi.lon])
     }));
     
     // 3. 距離順でソートして返却
     return enriched
       .sort((a, b) => a.distance - b.distance)
       .slice(0, limit);
   }
   ```

5. **キャッシュ戦略**
   - POI検索結果を短時間キャッシュ（同一セッション内の再利用）
   - カテゴリ集計結果を長時間キャッシュ（変化が少ない）

---

## 4. 実装ロードマップ

```
Phase 6（完了）
    │
    ├─ 渋谷駅固定の構造化RAG
    ├─ 91.6pt達成
    │
    ▼
Phase 7: 複数エリア対応
    │
    ├─ エリア別基準点マッピング
    ├─ 新宿・横浜でのテスト
    ├─ geo-common ライブラリ切り出し
    │
    ▼
Phase 8: 全国展開
    │
    ├─ PostGIS/Supabase導入
    ├─ 動的基準点解決
    ├─ パフォーマンス検証（500万件で50ms以下）
    │
    ▼
Phase 9: MCPサーバ統合
    │
    ├─ MapFan API統合
    ├─ MCPツール実装（nearest_poi, compare_areas等）
    ├─ geo-common からの関数利用
    └─ 本番デプロイ
```

---

## 5. チェックリスト

### 複数エリア対応時
- [ ] エリア別基準点マッピングの実装
- [ ] 質問分析のパターンマッチング化
- [ ] 新宿・横浜のPOIデータ準備
- [ ] 複数エリアテストケースの作成
- [ ] エリア間のPOI重複処理の決定

### 全国展開時
- [ ] PostGIS/Supabaseの環境構築
- [ ] 空間インデックスの作成
- [ ] 事前エンリッチメントの廃止・都度計算への移行
- [ ] パフォーマンステスト（500万件）
- [ ] 全国カテゴリマッピングの整備

### MCPサーバ統合時
- [ ] geo-commonライブラリの切り出し
- [ ] MCPツール設計・実装
- [ ] MapFan API統合
- [ ] レスポンス形式の標準化
- [ ] エラーハンドリングの実装
- [ ] キャッシュ戦略の実装
