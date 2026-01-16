#!/usr/bin/env python3
"""
test_cases.py - RAGシステム評価用テストケース定義

テストカテゴリ:
- location: POI位置検索
- nearby: 周辺施設検索
- category_search: カテゴリ別検索
- business_info: 営業情報検索
- complex: 複合検索
"""
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class TestCase:
    """テストケース定義"""
    id: int
    category: str  # テストカテゴリ
    prompt: str  # 質問文
    expected_keywords: List[str]  # 期待されるキーワード
    expected_data_type: str  # 期待されるデータタイプ (coordinate, name, time, address)
    difficulty: str  # 難易度 (easy, medium, hard)
    expected_category: Optional[str] = None  # 期待されるPOIカテゴリ（フィルター検証用）
    description: str = ""  # テストの説明


# =============================================================================
# テストケース定義
# =============================================================================

TEST_CASES: List[TestCase] = [
    # -------------------------------------------------------------------------
    # カテゴリ1: POI位置検索 (location) - 10件
    # -------------------------------------------------------------------------
    TestCase(
        id=1,
        category="location",
        prompt="渋谷駅の場所を教えてください",
        expected_keywords=["渋谷", "駅", "35.", "139."],
        expected_data_type="coordinate",
        difficulty="easy",
        expected_category="交通/鉄道駅",
        description="主要駅の位置検索"
    ),
    TestCase(
        id=2,
        category="location",
        prompt="東宝シネマの座標は？",
        expected_keywords=["東宝", "シネマ", "座標", "緯度", "経度"],
        expected_data_type="coordinate",
        difficulty="easy",
        expected_category="娯楽/映画館",
        description="映画館の座標検索"
    ),
    TestCase(
        id=3,
        category="location",
        prompt="渋谷東武ホテルはどこにありますか？",
        expected_keywords=["東武", "ホテル", "渋谷"],
        expected_data_type="coordinate",
        difficulty="easy",
        expected_category="宿泊/ホテル",
        description="ホテルの位置検索"
    ),
    TestCase(
        id=4,
        category="location",
        prompt="渋谷神南郵便局の場所を教えて",
        expected_keywords=["神南", "郵便局", "渋谷"],
        expected_data_type="coordinate",
        difficulty="easy",
        expected_category="公共/郵便局",
        description="郵便局の位置検索"
    ),
    TestCase(
        id=5,
        category="location",
        prompt="マクドナルドの位置情報を教えてください",
        expected_keywords=["マクドナルド", "座標"],
        expected_data_type="coordinate",
        difficulty="easy",
        expected_category="飲食店/ファストフード",
        description="ファストフード店の位置検索"
    ),
    TestCase(
        id=6,
        category="location",
        prompt="ローソンはどこにありますか？",
        expected_keywords=["ローソン", "座標"],
        expected_data_type="coordinate",
        difficulty="easy",
        expected_category="商店/コンビニ",
        description="コンビニの位置検索"
    ),
    TestCase(
        id=7,
        category="location",
        prompt="ヒューマントラストシネマ渋谷の場所",
        expected_keywords=["ヒューマントラスト", "シネマ", "渋谷"],
        expected_data_type="coordinate",
        difficulty="medium",
        expected_category="娯楽/映画館",
        description="映画館の位置検索（正式名称）"
    ),
    TestCase(
        id=8,
        category="location",
        prompt="渋谷警察署渋谷駅前交番の位置",
        expected_keywords=["警察", "交番", "渋谷"],
        expected_data_type="coordinate",
        difficulty="medium",
        expected_category="公共/警察",
        description="交番の位置検索"
    ),
    TestCase(
        id=9,
        category="location",
        prompt="パルコ劇場はどこですか？",
        expected_keywords=["パルコ", "劇場"],
        expected_data_type="coordinate",
        difficulty="medium",
        expected_category="娯楽/劇場",
        description="劇場の位置検索"
    ),
    TestCase(
        id=10,
        category="location",
        prompt="渋谷の三菱UFJ銀行の場所",
        expected_keywords=["三菱", "UFJ", "銀行", "渋谷"],
        expected_data_type="coordinate",
        difficulty="medium",
        expected_category="金融/銀行",
        description="銀行の位置検索"
    ),

    # -------------------------------------------------------------------------
    # カテゴリ2: 周辺施設検索 (nearby) - 10件
    # -------------------------------------------------------------------------
    TestCase(
        id=11,
        category="nearby",
        prompt="渋谷駅周辺のコンビニを教えてください",
        expected_keywords=["コンビニ", "ローソン", "ファミリーマート", "セブン", "ミニストップ"],
        expected_data_type="name",
        difficulty="medium",
        expected_category="商店/コンビニ",
        description="コンビニの周辺検索"
    ),
    TestCase(
        id=12,
        category="nearby",
        prompt="渋谷にあるカフェを教えて",
        expected_keywords=["カフェ", "コーヒー", "cafe"],
        expected_data_type="name",
        difficulty="medium",
        expected_category="飲食店/カフェ",
        description="カフェの検索"
    ),
    TestCase(
        id=13,
        category="nearby",
        prompt="渋谷周辺のレストランを3つ挙げてください",
        expected_keywords=["レストラン", "店"],
        expected_data_type="name",
        difficulty="medium",
        expected_category="飲食店/レストラン",
        description="レストランの検索（件数指定）"
    ),
    TestCase(
        id=14,
        category="nearby",
        prompt="渋谷の映画館を全部教えて",
        expected_keywords=["映画館", "シネマ", "cinema"],
        expected_data_type="name",
        difficulty="medium",
        expected_category="娯楽/映画館",
        description="映画館の全件検索"
    ),
    TestCase(
        id=15,
        category="nearby",
        prompt="渋谷にある薬局はどこですか？",
        expected_keywords=["薬局", "ドラッグ"],
        expected_data_type="name",
        difficulty="medium",
        expected_category="医療/薬局",
        description="薬局の検索"
    ),
    TestCase(
        id=16,
        category="nearby",
        prompt="渋谷周辺の銀行を教えてください",
        expected_keywords=["銀行", "ATM"],
        expected_data_type="name",
        difficulty="medium",
        expected_category="金融/銀行",
        description="銀行の検索"
    ),
    TestCase(
        id=17,
        category="nearby",
        prompt="渋谷駅近くのホテルはありますか？",
        expected_keywords=["ホテル", "宿泊"],
        expected_data_type="name",
        difficulty="medium",
        expected_category="宿泊/ホテル",
        description="ホテルの検索"
    ),
    TestCase(
        id=18,
        category="nearby",
        prompt="渋谷のファストフード店を教えて",
        expected_keywords=["ファストフード", "マクドナルド", "ロッテリア", "バーガー"],
        expected_data_type="name",
        difficulty="medium",
        expected_category="飲食店/ファストフード",
        description="ファストフード店の検索"
    ),
    TestCase(
        id=19,
        category="nearby",
        prompt="渋谷にバーはありますか？",
        expected_keywords=["バー", "bar"],
        expected_data_type="name",
        difficulty="medium",
        expected_category="飲食店/バー",
        description="バーの検索"
    ),
    TestCase(
        id=20,
        category="nearby",
        prompt="渋谷の本屋を教えてください",
        expected_keywords=["本屋", "書店", "ブック"],
        expected_data_type="name",
        difficulty="medium",
        expected_category="商店/書店",
        description="書店の検索"
    ),

    # -------------------------------------------------------------------------
    # カテゴリ3: カテゴリ別検索 (category_search) - 5件
    # -------------------------------------------------------------------------
    TestCase(
        id=21,
        category="category_search",
        prompt="渋谷にある観光案内所を教えてください",
        expected_keywords=["観光", "案内所", "information"],
        expected_data_type="name",
        difficulty="medium",
        expected_category="観光/案内所",
        description="観光案内所の検索"
    ),
    TestCase(
        id=22,
        category="category_search",
        prompt="渋谷の劇場を全部教えて",
        expected_keywords=["劇場", "シアター", "theatre"],
        expected_data_type="name",
        difficulty="medium",
        expected_category="娯楽/劇場",
        description="劇場の検索"
    ),
    TestCase(
        id=23,
        category="category_search",
        prompt="渋谷周辺のクリニックはどこにありますか？",
        expected_keywords=["クリニック", "医院", "病院"],
        expected_data_type="name",
        difficulty="medium",
        expected_category="医療/クリニック",
        description="クリニックの検索"
    ),
    TestCase(
        id=24,
        category="category_search",
        prompt="渋谷のパン屋を教えて",
        expected_keywords=["パン", "ベーカリー", "bakery"],
        expected_data_type="name",
        difficulty="medium",
        expected_category="商店/パン屋",
        description="パン屋の検索"
    ),
    TestCase(
        id=25,
        category="category_search",
        prompt="渋谷にスーパーマーケットはありますか？",
        expected_keywords=["スーパー", "マーケット"],
        expected_data_type="name",
        difficulty="medium",
        expected_category="商店/スーパー",
        description="スーパーの検索"
    ),

    # -------------------------------------------------------------------------
    # カテゴリ4: 複合検索 (complex) - 5件
    # -------------------------------------------------------------------------
    TestCase(
        id=26,
        category="complex",
        prompt="渋谷駅周辺でコンビニの座標を教えてください",
        expected_keywords=["コンビニ", "座標", "緯度", "経度"],
        expected_data_type="coordinate",
        difficulty="hard",
        expected_category="商店/コンビニ",
        description="コンビニの座標を含む検索"
    ),
    TestCase(
        id=27,
        category="complex",
        prompt="渋谷の映画館の電話番号を教えて",
        expected_keywords=["映画館", "電話", "シネマ"],
        expected_data_type="phone",
        difficulty="hard",
        expected_category="娯楽/映画館",
        description="映画館の連絡先検索"
    ),
    TestCase(
        id=28,
        category="complex",
        prompt="渋谷でウェブサイトがあるカフェを教えて",
        expected_keywords=["カフェ", "ウェブ", "サイト", "http"],
        expected_data_type="url",
        difficulty="hard",
        expected_category="飲食店/カフェ",
        description="ウェブサイト付きカフェの検索"
    ),
    TestCase(
        id=29,
        category="complex",
        prompt="渋谷駅から近いホテルの名前と座標を教えて",
        expected_keywords=["ホテル", "座標", "渋谷"],
        expected_data_type="coordinate",
        difficulty="hard",
        expected_category="宿泊/ホテル",
        description="ホテルの名前と座標の複合検索"
    ),
    TestCase(
        id=30,
        category="complex",
        prompt="渋谷にある郵便局の場所と名前を全部教えて",
        expected_keywords=["郵便局", "場所", "座標"],
        expected_data_type="coordinate",
        difficulty="hard",
        expected_category="公共/郵便局",
        description="郵便局の全件検索（名前と座標）"
    ),
]


def get_test_cases_by_category(category: str) -> List[TestCase]:
    """カテゴリでテストケースをフィルタ"""
    return [tc for tc in TEST_CASES if tc.category == category]


def get_test_cases_by_difficulty(difficulty: str) -> List[TestCase]:
    """難易度でテストケースをフィルタ"""
    return [tc for tc in TEST_CASES if tc.difficulty == difficulty]


def get_all_test_cases() -> List[TestCase]:
    """全テストケースを取得"""
    return TEST_CASES


if __name__ == "__main__":
    # テストケースのサマリー表示
    print("=" * 60)
    print("テストケース サマリー")
    print("=" * 60)
    
    categories = {}
    difficulties = {}
    
    for tc in TEST_CASES:
        categories[tc.category] = categories.get(tc.category, 0) + 1
        difficulties[tc.difficulty] = difficulties.get(tc.difficulty, 0) + 1
    
    print(f"\n総テストケース数: {len(TEST_CASES)}")
    
    print("\nカテゴリ別:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}件")
    
    print("\n難易度別:")
    for diff, count in sorted(difficulties.items()):
        print(f"  {diff}: {count}件")
    
    print("\n" + "=" * 60)
    print("テストケース一覧")
    print("=" * 60)
    for tc in TEST_CASES:
        print(f"\n[{tc.id}] {tc.category} ({tc.difficulty})")
        print(f"    Q: {tc.prompt}")
        print(f"    期待キーワード: {tc.expected_keywords}")
