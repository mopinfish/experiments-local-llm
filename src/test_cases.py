#!/usr/bin/env python3
"""
test_cases.py - RAGシステム評価用テストケース定義

テストカテゴリ:
- location: POI位置検索
- nearby: 周辺施設検索
- category_search: カテゴリ別検索
- complex: 複合検索

難易度:
- easy: 単純な位置検索
- medium: 条件付き検索
- hard: 複合条件・空間推論
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
# テストケース定義（30件）
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
        prompt="渋谷にファストフード店はありますか？",
        expected_keywords=["ファストフード", "マクドナルド", "モス", "ケンタッキー"],
        expected_data_type="name",
        difficulty="medium",
        expected_category="飲食店/ファストフード",
        description="ファストフードの検索"
    ),
    TestCase(
        id=19,
        category="nearby",
        prompt="渋谷の郵便局を教えて",
        expected_keywords=["郵便局"],
        expected_data_type="name",
        difficulty="medium",
        expected_category="公共/郵便局",
        description="郵便局の検索"
    ),
    TestCase(
        id=20,
        category="nearby",
        prompt="渋谷にバーはありますか？",
        expected_keywords=["バー", "Bar", "居酒屋"],
        expected_data_type="name",
        difficulty="medium",
        expected_category="飲食店/バー",
        description="バーの検索"
    ),

    # -------------------------------------------------------------------------
    # カテゴリ3: カテゴリ別検索 (category_search) - 5件
    # -------------------------------------------------------------------------
    TestCase(
        id=21,
        category="category_search",
        prompt="渋谷で食事できる場所を教えて",
        expected_keywords=["レストラン", "カフェ", "食堂", "飲食"],
        expected_data_type="name",
        difficulty="medium",
        description="飲食カテゴリの検索"
    ),
    TestCase(
        id=22,
        category="category_search",
        prompt="渋谷の娯楽施設を教えて",
        expected_keywords=["映画館", "劇場", "シネマ"],
        expected_data_type="name",
        difficulty="medium",
        description="娯楽カテゴリの検索"
    ),
    TestCase(
        id=23,
        category="category_search",
        prompt="渋谷の金融機関を教えてください",
        expected_keywords=["銀行", "信用金庫", "ATM"],
        expected_data_type="name",
        difficulty="medium",
        description="金融カテゴリの検索"
    ),
    TestCase(
        id=24,
        category="category_search",
        prompt="渋谷の公共施設を教えて",
        expected_keywords=["郵便局", "警察", "交番"],
        expected_data_type="name",
        difficulty="medium",
        description="公共施設カテゴリの検索"
    ),
    TestCase(
        id=25,
        category="category_search",
        prompt="渋谷で買い物できる場所は？",
        expected_keywords=["コンビニ", "スーパー", "店"],
        expected_data_type="name",
        difficulty="medium",
        description="商店カテゴリの検索"
    ),

    # -------------------------------------------------------------------------
    # カテゴリ4: 複合検索 (complex) - 5件
    # -------------------------------------------------------------------------
    TestCase(
        id=26,
        category="complex",
        prompt="渋谷駅から一番近いコンビニの座標を教えて",
        expected_keywords=["コンビニ", "座標", "緯度", "経度"],
        expected_data_type="coordinate",
        difficulty="hard",
        description="距離条件付き検索"
    ),
    TestCase(
        id=27,
        category="complex",
        prompt="渋谷で映画を見た後に食事できる場所を教えて",
        expected_keywords=["映画館", "レストラン", "カフェ"],
        expected_data_type="name",
        difficulty="hard",
        description="複数カテゴリの組み合わせ検索"
    ),
    TestCase(
        id=28,
        category="complex",
        prompt="渋谷で朝食をとれる場所とその座標を教えて",
        expected_keywords=["カフェ", "ファストフード", "座標"],
        expected_data_type="coordinate",
        difficulty="hard",
        description="時間帯条件付き検索"
    ),
    TestCase(
        id=29,
        category="complex",
        prompt="渋谷駅周辺でATMと郵便局の両方がある場所",
        expected_keywords=["ATM", "銀行", "郵便局"],
        expected_data_type="name",
        difficulty="hard",
        description="複数施設の近接検索"
    ),
    TestCase(
        id=30,
        category="complex",
        prompt="渋谷のホテルとその周辺のコンビニを教えて",
        expected_keywords=["ホテル", "コンビニ", "ローソン", "ファミリーマート"],
        expected_data_type="name",
        difficulty="hard",
        description="施設とその周辺施設の検索"
    ),
]


# =============================================================================
# ユーティリティ関数
# =============================================================================

def get_test_cases_by_category(category: str) -> List[TestCase]:
    """指定カテゴリのテストケースを取得"""
    return [tc for tc in TEST_CASES if tc.category == category]


def get_test_cases_by_difficulty(difficulty: str) -> List[TestCase]:
    """指定難易度のテストケースを取得"""
    return [tc for tc in TEST_CASES if tc.difficulty == difficulty]


def get_test_case_by_id(test_id: int) -> Optional[TestCase]:
    """IDでテストケースを取得"""
    for tc in TEST_CASES:
        if tc.id == test_id:
            return tc
    return None


def get_test_case_stats() -> dict:
    """テストケースの統計情報を取得"""
    categories = {}
    difficulties = {}
    
    for tc in TEST_CASES:
        categories[tc.category] = categories.get(tc.category, 0) + 1
        difficulties[tc.difficulty] = difficulties.get(tc.difficulty, 0) + 1
    
    return {
        "total": len(TEST_CASES),
        "by_category": categories,
        "by_difficulty": difficulties
    }


if __name__ == "__main__":
    # テストケース統計を表示
    stats = get_test_case_stats()
    print(f"テストケース総数: {stats['total']}件")
    print("\nカテゴリ別:")
    for cat, count in stats["by_category"].items():
        print(f"  {cat}: {count}件")
    print("\n難易度別:")
    for diff, count in stats["by_difficulty"].items():
        print(f"  {diff}: {count}件")
