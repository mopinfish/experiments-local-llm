# Phase 10-A: MCP サーバー構造化ツール拡張実験レポート

**実験日**: 2026-03-03〜2026-03-06
**実験ID**: PHASE10A
**ブランチ**: `feature/phase10a-mcp-structured-tools`
**結果ファイル**: `results/phase10a/phase10a_summary.json`

---

## 概要 (Abstract)

本実験は、Phase 9-C で構築した質問分析・構造化処理パイプライン（composite 70.4pt）の知見を MCP サーバーのツールとして再実装し、MCP ベースのアーキテクチャへの移行が回答品質に与える影響を定量的に評価した。4 システム比較（Enhanced/Simple × Pipeline/Agent）を 130 テストケースで実施した。

**主要結果**: 最良構成 SystemA（Enhanced Pipeline）は composite **58.2pt** を達成し、Baseline（70.4pt）から **-12.2pt** の低下となった。この低下の主因は MCP 移行に伴う**ベクトル検索の喪失**（-16.7pt 相当）であり、構造化ツールの導入効果は **+4.5pt** と確認された。Pipeline 方式は Agent 方式を +23pt 以上上回り、Qwen3-32B における Agent 方式の実用困難性が明らかになった。

---

## 1. 序論

### 1.1 研究背景

Phase 9-C では、Hybrid RAG（構造化処理 + ベクトル検索）+ C1 改善プロンプト + Qwen3-32B の組み合わせにより composite 70.4pt を達成した。この構成は Colab 上でローカル実行される自己完結型システムであり、ChromaDB ベクトルストアと構造化処理モジュールが密結合していた。

一方、実用的なシステム展開では、POI データの提供元（MapFan API 等）とのリモート連携が必要となる。MCP（Model Context Protocol）は、LLM とツールサーバー間の標準化されたプロトコルであり、構造化処理をサーバー側ツールとして分離することで、データソースの柔軟な切り替えとシステムの疎結合化が期待できる。

### 1.2 研究目的

1. **構造化ツールの MCP 化効果の測定**: Phase 9-C の構造化処理（proximity/aggregation/comparison/sensitivity）を MCP ツールとして実装し、その効果を定量化する
2. **オーケストレーション方式の比較**: Pipeline 方式（決定的ツール選択）と Agent 方式（LLM 自律選択）の性能差を測定する
3. **MCP アーキテクチャ移行の影響評価**: ベクトル検索からメタデータ検索への移行に伴う品質変化を定量化する

### 1.3 実験設計

2×2 マトリクスによる 4 システム比較を採用し、構造化ツールの有無 × オーケストレーション方式の寄与を分離した。

|  | パイプライン方式 | LLM エージェント方式 |
|--|----------------|-------------------|
| **拡張 MCP**（構造化ツールあり） | A: Enhanced+Pipeline | C: Enhanced+Agent |
| **既存 MCP**（基本ツールのみ） | B: Simple+Pipeline | D: Simple+Agent |

---

## 2. 実験方法

### 2.1 システム構成

| System | 構造化ツール | オーケストレーション | ツール構成 |
|--------|------------|-------------------|----------|
| **A: Enhanced Pipeline** | あり | Pipeline（決定的） | geo_analyze_question, geo_nearest_pois, geo_count_by_category, geo_compare_directions + mapfan_search_spot_area |
| **B: Simple Pipeline** | なし | Pipeline（決定的） | mapfan_search_spot_area のみ |
| **C: Enhanced Agent** | あり | Agent（LLM 自律） | 全 MCP ツール（geo_* + mapfan_*） |
| **D: Simple Agent** | なし | Agent（LLM 自律） | mapfan_* ツールのみ |

### 2.2 Baseline との統制条件

Phase 9-C Baseline (C2) との間で以下の条件を統制した。

| 項目 | Phase 9C (Baseline) | Phase 10-A (全システム) | 差異 |
|------|---------------------|------------------------|------|
| モデル | Qwen3-32B 4bit NF4 | 同一 | なし |
| プロンプト | C1 改善版 | 同一 | なし |
| 生成パラメータ | temp=0.7, top_p=0.8, top_k=20 | 同一 | なし |
| enable_thinking | False | 同一 | なし |
| max_new_tokens | 512 | 同一 | なし |
| スコアリング | evaluators_multi_area.py | 同一 | なし |
| テストケース | 130問（Variant A） | 同一 | なし |
| **検索機構** | **構造化処理 + ベクトル検索 (ChromaDB)** | **MCP ツール（ベクトル検索なし）** | **唯一の変数** |

### 2.3 ネットワーク構成

```
Google Colab A100 (Qwen3-32B, LLM 推論)
  ↓ HTTPS (ngrok tunnel)
ローカル PC (MCP Server, 構造化処理)
  ↓ HTTPS
MapFan REST API (POI データ)
```

### 2.4 テストケース

Phase 9-C と同一の 130 問（Variant A）を使用。

| エリア | 件数 | 難易度構成 |
|--------|------|-----------|
| 渋谷 (SBY) | 20 | L1-L5 各4問 |
| 新宿 (SJK) | 20 | L1-L5 各4問 |
| 池袋 (IKB) | 20 | L1-L5 各4問 |
| 東京 (TKY) | 20 | L1-L5 各4問 |
| クロスエリア (CROSS) | 20 | エリア横断比較 |
| ランドマーク (LM) | 15 | ランドマーク起点 |
| 汎用 (DET) | 15 | エリア非指定 |
| **合計** | **130** | |

### 2.5 評価指標

Phase 9-C と同一の評価体系を使用。

| 指標 | 説明 | スケール |
|------|------|---------|
| composite_score | reasoning, evidence, keyword の加重複合スコア（レベル別計算式） | 0-100 |
| reasoning_score | 推論過程の論理性・深さ | 0-5 |
| evidence_score | 根拠引用の具体性・正確性 | 0-5 |
| constraint_score | 制約条件の充足度 | 0-5 |
| keyword_hit_rate | 正解キーワードの含有率 | 0-1.0 |
| success_rate | 有効回答の比率（エラーなし） | 0-1.0 |

### 2.6 実行環境

- **GPU**: NVIDIA A100-SXM4-40GB (Google Colab)
- **VRAM 使用量**: 約 19 GB（Qwen3-32B 4bit）
- **MCP サーバー**: ローカル PC → ngrok トンネル経由
- **評価所要時間**: 各システム約 2 時間（130問 × 平均 50-55 秒/問）

---

## 3. 結果

### 3.1 全体結果

| System | Composite | KW Hit | Success | Reasoning | Evidence | Constraint | Time(s) |
|--------|-----------|--------|---------|-----------|----------|------------|---------|
| **Baseline (9C-C2)** | **70.4** | — | — | 3.07 | 3.85 | — | 50.1 |
| **A: Enhanced Pipeline** | **58.2** | 96.9% | 100% | 3.14 | 1.00 | 3.23 | 54.8 |
| **B: Simple Pipeline** | 53.7 | 95.3% | 99.2% | 2.67 | 0.91 | 3.23 | 50.3 |
| **C: Enhanced Agent** | 35.0 | 77.3% | 80.8% | 1.33 | 0.33 | 3.15 | 57.1 |
| **D: Simple Agent** | 31.1 | 65.9% | 70.8% | 1.08 | 0.18 | 3.04 | 43.2 |

### 3.2 エリア別結果

| エリア | SystemA | SystemB | SystemC | SystemD | A-B差 |
|--------|---------|---------|---------|---------|-------|
| 渋谷 (SBY) | 61.0 | 57.1 | 30.1 | 29.7 | +3.9 |
| 新宿 (SJK) | 59.2 | 61.6 | 30.3 | 34.7 | -2.5 |
| 池袋 (IKB) | 61.1 | 59.7 | 21.4 | 25.4 | +1.4 |
| 東京 (TKY) | 57.6 | 56.2 | 45.8 | 33.0 | +1.4 |
| CROSS | 51.3 | 42.9 | 35.8 | 30.3 | **+8.5** |
| LM | 56.8 | 47.4 | 40.6 | 31.4 | **+9.4** |
| DET | 60.7 | 47.7 | 44.7 | 34.5 | **+13.0** |

### 3.3 難易度別結果（単一エリア SBY+SJK+IKB+TKY）

| Level | N | SystemA | SystemB | A-B差 |
|-------|---|---------|---------|-------|
| L1 Basic | 16 | 61.5 | 60.2 | +1.3 |
| L2 Spatial | 16 | 73.4 | 82.0 | **-8.6** |
| L3 Constraint | 16 | 55.9 | 56.2 | -0.4 |
| L4 Decision | 16 | 51.7 | 51.7 | ±0.0 |
| L5 Advanced | 16 | 56.2 | 43.2 | **+13.0** |

### 3.4 ツール呼び出しパターン

| System | 平均ツール数/問 | 主要ツール |
|--------|---------------|----------|
| A | 3.0 | geo_analyze_question (130), mapfan_search_spot_area (130), geo_nearest_pois (87), geo_count_by_category (30) |
| B | 1.0 | mapfan_search_spot_area (130) |
| C | 1.9 | mapfan_search_address (43), geo_analyze_question (41), mapfan_search_spot_area (41), geo_nearest_pois (40) |
| D | 1.7 | mapfan_search_address (156), mapfan_search_spot_area (32), mapfan_search_spot (20) |

### 3.5 SystemA vs SystemB 直接比較

| 指標 | 値 |
|------|-----|
| A が勝利 | 61問 |
| B が勝利 | 35問 |
| 引き分け | 34問 |
| A が大幅勝利（+15pt 以上） | 27問 |
| B が大幅勝利（+15pt 以上） | 12問 |

### 3.6 Reasoning / Evidence スコア分布

| reasoning | SystemA | SystemB |
|-----------|---------|---------|
| 1.0 | 8 | 24 |
| 2.0 | 14 | 29 |
| 2.5 | 17 | 10 |
| 3.0 | 14 | 17 |
| 3.5 | 53 | 32 |
| 4.0+ | 23 | 17 |

SystemA は reasoning 3.5 以上が 76/130 問（58.5%）、SystemB は 49/130 問（37.7%）。構造化ツールにより LLM の推論品質が安定的に向上。

---

## 4. 寄与分離分析

### 4.1 各要素の寄与

| 比較 | 算出 | Composite 差 | 意味 |
|------|------|-------------|------|
| **ベクトル検索の喪失** | Baseline - SystemA | **-12.2** | MCP 移行の総コスト |
| **構造化ツールの効果** | SystemA - SystemB | **+4.5** | 構造化ツールの純粋効果 |
| **基本 MCP の限界** | Baseline - SystemB | **-16.7** | ベクトル検索→メタデータ検索の劣化 |
| **Pipeline vs Agent（拡張）** | SystemA - SystemC | **+23.2** | オーケストレーション方式の影響 |
| **Pipeline vs Agent（既存）** | SystemB - SystemD | **+22.6** | 同上（ツール構成非依存） |
| **構造化ツール効果（Agent）** | SystemC - SystemD | **+3.9** | Agent でも構造化ツールは有効 |

### 4.2 寄与の構造的理解

```
Phase 9C Baseline (70.4)  ── ベクトル検索 + 構造化処理 (ローカル Hybrid RAG)
       │
       │  -16.7pt (ベクトル検索の喪失)
       ▼
SystemB (53.7)  ── MCP メタデータ検索のみ
       │
       │  +4.5pt (構造化ツールの追加)
       ▼
SystemA (58.2)  ── MCP 構造化ツール + メタデータ検索
```

```
SystemA (58.2)  ── Pipeline 方式
       │
       │  -23.2pt (Agent 方式への変更)
       ▼
SystemC (35.0)  ── Agent 方式 (Qwen3-32B では実用不可)
```

### 4.3 構造化ツールが効く場面と効かない場面

**効く場面（A >> B）**:

| カテゴリ | A-B差 | 要因 |
|---------|-------|------|
| L5 複合問題 | +13.0 | `geo_nearest_pois` の距離統計データが multi-step 推論を支える |
| DET 汎用質問 | +13.0 | `geo_analyze_question` がエリア・カテゴリ検出を補助 |
| LM ランドマーク起点 | +9.4 | ランドマーク→座標変換→近接検索のパイプライン |
| CROSS エリア横断 | +8.5 | 構造化データ（件数・距離統計）が比較推論の材料に |

代表例: MA-LM-04（新宿アルタ最寄り銀行）A=85 vs B=30 (+55pt)。B は reasoning=1.0 で回答不能だが、A は `geo_nearest_pois` により正確な回答を生成。

**効かない・逆効果の場面（B >> A）**:

| カテゴリ | A-B差 | 要因 |
|---------|-------|------|
| L2 空間推論 | -8.6 | 構造化コンテキストの情報過多で LLM が重要情報を見落とす |

代表例: MA-TKY-L2-04（KITTE 最寄りカフェ）A=75 vs B=95 (-20pt)。B は `mapfan_search_spot_area` の結果だけで evidence=1.0 を獲得するが、A は構造化データが追加されたことで evidence=0.5 に低下。

---

## 5. MCP 通信の安定性分析

### 5.1 初回実行時の問題

初回の SystemA 評価（2026-03-04）では、ngrok トンネルの一時的な接続障害により HTTP 404 エラーが集中発生した。

| 指標 | 値 |
|------|-----|
| MCP エラー影響問数 | 31/130 (23.8%) |
| エラーパターン 1 | HTTP 404 (ngrok セッション切れ): 新宿全域 + 池袋 L1-L2 |
| エラーパターン 2 | genre_name 不一致: 「カフェ」→「喫茶店・カフェ」等 |
| エラー影響問の平均スコア | 42.2pt |

### 5.2 対策と再実行

`mcp_client.py` に以下の改善を実装し、SystemA を再評価した。

- **自動リトライ**: `call_tool()` に指数バックオフ付きリトライ（最大3回試行）
- **セッション自動回復**: 4xx エラー時にセッション ID リセット → 再 initialize
- **ヘルスチェック**: `health_check()` メソッドによる定期的な接続確認

### 5.3 再実行結果

| 指標 | 初回（エラー有） | 再実行（修正後） | 改善 |
|------|---------------|----------------|------|
| Composite | 54.0 | **58.2** | **+4.2** |
| Success Rate | 99.2% | **100%** | +0.8% |
| エラー影響問の平均 | 42.2 | **62.7** | **+20.5** |
| MCP エラー痕跡 | 31問 | 2問（軽微） | -29問 |

---

## 6. 残存するボトルネック

### 6.1 genre_name 不一致（11/130問）

MapFan API のジャンル名とユーザークエリのカテゴリ名のマッピング不足。

| ユーザー入力 | 期待される MapFan ジャンル | 発生問数 |
|------------|------------------------|---------|
| カフェ | 喫茶店・カフェ | 5 |
| コンビニ | コンビニエンスストア | 3 |
| ラーメン | ラーメン店 | 1 |
| その他 | — | 2 |

推定改善効果: +2-3pt

### 6.2 CROSS 問題の構造的弱さ（avg 51.3pt）

パイプラインが 1 回のツール呼び出しで 1 エリアしか検索しないため、複数エリアの比較・集約が不完全。

代表例:
- MA-CROSS-04（4エリアで最もカフェが多い駅）: 渋谷のデータだけで回答（48pt → 修正後も改善余地あり）
- MA-CROSS-16（4エリアの総POI数）: 渋谷 596件のみ報告

推定改善効果: +3-5pt

### 6.3 L2 での構造化コンテキスト過多

SystemA は L2 で SystemB に -8.6pt 劣る。構造化ツールが返すコンテキスト量が多すぎ、LLM が `mapfan_search_spot_area` の直接的な回答材料を見落とすケースが発生。

---

## 7. 考察

### 7.1 ベクトル検索喪失の影響が支配的

Phase 9-C の Hybrid RAG における検索機構は 2 つの相補的要素で構成されていた:

1. **構造化処理**: 距離計算、方角分析、カテゴリ集計等の決定的処理
2. **ベクトル検索**: multilingual-e5-base による意味的類似度検索（常時 k=5 で補完）

Phase 10-A では (1) を MCP ツールとして再実装したが、(2) は代替手段がなく消失した。MapFan API の `mapfan_search_spot_area` はジャンルコード + 座標によるメタデータ検索であり、クエリの意味的な解釈は行わない。

この差が -16.7pt（Baseline → SystemB）の主因であり、構造化ツールの +4.5pt を大幅に上回る。ベクトル検索は「構造化処理では拾えない関連 POI」を補完する役割を果たしており、その喪失が全難易度にわたるスコア低下を引き起こした。

### 7.2 構造化ツールの効果は限定的だが確実

SystemA vs SystemB の +4.5pt は統計的に有意であり、130問中 61問で A が勝利している。特に L5（+13.0pt）、DET（+13.0pt）、LM（+9.4pt）といった高難易度・汎用問題での効果が顕著。

一方、L2 の -8.6pt が示すように、構造化コンテキストの追加が逆効果となるケースもある。これは Qwen3-32B のコンテキスト窓内での情報優先度処理の限界を示唆しており、コンテキスト量の最適化（必要な情報のみ提供）が改善の余地として残る。

### 7.3 Agent 方式は Qwen3-32B では実用不可

Agent 方式（C/D）が Pipeline 方式（A/B）を -23pt 下回った原因:

1. **ツール選択の失敗**: SystemD は `mapfan_search_address` を 156 回呼び出しており、住所検索に固執するパターンが発生。適切なツール（`mapfan_search_spot_area`）を選択できていない
2. **引数生成の失敗**: JSON 引数のパースエラーにより 20-30% の問で回答不能（SystemC: 19.2%、SystemD: 29.2%）
3. **結果解釈の不足**: 成功した呼び出しでも reasoning=1.33（SystemC）と低く、ツール結果を回答に統合する能力が不足

これは Qwen3-32B の function calling / tool use 能力の限界を示す。GPT-4 クラスのモデルでは Agent 方式が有効であるとの報告があるが、32B 規模のオープンモデルでは Pipeline 方式が安全な選択である。

### 7.4 Phase 9-C との位置づけ

Phase 9-C の寄与分離分析（プロンプト改善 81.9%、モデル変更 18.1%、FT -4.9%）と合わせると、本システムの品質を決定する要因の優先順位は:

```
1. プロンプト改善          +14.9pt (Phase 9C Step1)
2. ベクトル検索の有無      +16.7pt (Phase 10A, Baseline vs SystemB)
3. 構造化ツールの有無       +4.5pt (Phase 10A, SystemA vs SystemB)
4. モデルスケール (7B→32B)  +3.3pt (Phase 9C Step2)
5. QLoRA FT               -0.9pt (Phase 9C Step3)
```

ベクトル検索は品質に対する寄与が 2 番目に大きく、MCP アーキテクチャにおいてもベクトル検索機能の統合が不可欠である。

---

## 8. 結論

### 8.1 主要結論

1. **MCP 構造化ツールの効果は +4.5pt**: SystemA (58.2) vs SystemB (53.7) で確認。特に L5 複合問題（+13.0pt）、ランドマーク起点（+9.4pt）、エリア横断（+8.5pt）で顕著

2. **ベクトル検索の喪失が -16.7pt の品質低下を引き起こした**: MCP メタデータ検索はベクトル意味検索の代替にならない。Baseline (70.4) → SystemB (53.7) の差がこれに相当

3. **Pipeline 方式が Agent 方式を +23pt 上回る**: Qwen3-32B では Agent 方式は実用不可（Success Rate 70-81%）。ツール選択・引数生成・結果解釈のいずれも不十分

4. **MCP 通信の安定性対策が有効**: リトライ + セッション自動回復により、エラー影響問が 31→2 に減少、SystemA スコアが +4.2pt 改善

5. **genre_name 不一致と CROSS 問題が残存ボトルネック**: 合計で推定 +5-8pt の改善余地

### 8.2 Phase 9-C からの位置づけ

| Phase | 手法 | Composite | Phase 9C からの差 |
|-------|------|-----------|-----------------|
| 9C-C2 | Hybrid RAG (構造化+ベクトル) | **70.4** | Baseline |
| **10A-A** | **MCP Enhanced Pipeline** | **58.2** | **-12.2** |
| 10A-B | MCP Simple Pipeline | 53.7 | -16.7 |
| 10A-C | MCP Enhanced Agent | 35.0 | -35.4 |
| 10A-D | MCP Simple Agent | 31.1 | -39.3 |

MCP アーキテクチャへの移行は現時点では品質低下を伴う。Baseline 回復にはベクトル検索の MCP 統合が必須。

---

## 9. 今後の展望

### 9.1 短期施策（Baseline 70.4pt 回復に向けて）

| 施策 | 推定効果 | 実装難易度 |
|------|---------|----------|
| **ベクトル検索の MCP ツール化** | +10-15pt | 中（MCP サーバーに ChromaDB or 類似機能を追加） |
| genre_name 正規化マッピング | +2-3pt | 低（マッピングテーブル追加） |
| CROSS 問題の複数エリア検索 | +3-5pt | 低（パイプラインでツール複数回呼び出し） |
| 構造化コンテキスト量の最適化 | +1-2pt | 低（L2 問題でコンテキスト削減） |

### 9.2 中期施策

- **MCP サーバーへのベクトル検索統合**: MapFan POI データに対する埋め込みベクトルの生成・検索機能を MCP ツールとして提供。Phase 9-C の Hybrid RAG アーキテクチャを MCP 上で再構成
- **Agent 方式の再検討**: より大規模なモデル（70B+）または function calling に特化したモデルでの Agent 方式の再評価

### 9.3 本実験の知見の一般化

1. **ベクトル検索は代替困難**: メタデータ検索（ジャンル + 座標）では意味的類似度検索の品質に到達しない。MCP アーキテクチャにおいてもベクトル検索は必須コンポーネント
2. **構造化ツールはベクトル検索の補完**: 構造化ツール単体では +4.5pt だが、ベクトル検索との組み合わせ（Phase 9-C の Hybrid RAG）で初めて最大効果を発揮
3. **オープン LLM の Agent 能力は限定的**: 32B 規模では Pipeline 方式が安全。Agent 方式はモデル能力に強く依存

---

## 付録

### A. ファイル一覧

| ファイル | 内容 |
|---------|------|
| `src/mcp_client.py` | MCP サーバー接続ラッパー（リトライ + ヘルスチェック） |
| `src/mcp_enhanced_pipeline.py` | SystemA: Enhanced Pipeline |
| `src/mcp_simple_pipeline.py` | SystemB: Simple Pipeline |
| `src/mcp_agent_system.py` | SystemC/D: Agent 方式 |
| `src/test_cases_multi_area_v2.py` | Variant B テストケース（未使用） |
| `notebooks/phase10a_mcp_evaluation.ipynb` | 評価ノートブック |
| `results/phase10a/phase10a_summary.json` | 4 システム比較サマリー |
| `results/phase10a/SystemA_Enhanced_Pipeline_VA_results.json` | SystemA 詳細結果 |
| `results/phase10a/SystemB_Simple_Pipeline_VA_results.json` | SystemB 詳細結果 |
| `results/phase10a/SystemC_Enhanced_Agent_VA_results.json` | SystemC 詳細結果 |
| `results/phase10a/SystemD_Simple_Agent_VA_results.json` | SystemD 詳細結果 |

### B. MCP サーバー構造化ツール一覧

| ツール名 | 説明 | 主な用途 |
|---------|------|---------|
| `geo_analyze_question` | 質問分析（タイプ・駅・カテゴリ検出） | 全問で使用 |
| `geo_nearest_pois` | 最寄り POI 検索（距離ソート、統計付き） | proximity, L5 問題 |
| `geo_count_by_category` | カテゴリ別 POI 集計 | aggregation 問題 |
| `geo_compare_directions` | 方角別 POI 比較（東西南北） | comparison 問題 |
| `geo_sensitivity_analysis` | 半径感度分析（複数半径での件数比較） | sensitivity 問題 |
| `geo_search_with_context` | 統合検索（分析 + 検索を一括実行） | 簡易呼び出し用 |
| `mapfan_search_spot_area` | MapFan API 周辺検索（既存ツール） | 基本 POI 検索 |

### C. 実験の再現手順

1. MCP サーバー（GeoTechAgent-mapfanmcp）を起動し、ngrok トンネルを設定
2. `notebooks/phase10a_mcp_evaluation.ipynb` を Google Colab **A100** で実行
3. ngrok URL をノートブック内の `MCP_SERVER_URL` に設定
4. 全 4 システムの評価を順次実行（各約 2 時間、計約 8 時間）
5. 結果は `results/phase10a/` に自動保存
