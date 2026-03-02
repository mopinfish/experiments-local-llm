#!/usr/bin/env python3
"""
prepare_training_data.py - Phase 9-C Step 3: QLoRA学習データ準備

C2結果から高品質な回答を抽出し、Alpaca形式の学習データを作成する。

使用方法:
    python scripts/prepare_training_data.py

出力:
    data/phase9c_training_data.json
"""

import json
import random
import re
import sys
from pathlib import Path

# プロジェクトルート
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from test_cases_multi_area import ALL_MULTI_AREA_TEST_CASES


def clean_answer(answer: str) -> str:
    """<think>タグを除去し、回答テキストをクリーンアップ"""
    # <think>...</think> を除去（閉じタグのみも対応）
    cleaned = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL).strip()
    # 閉じタグのみ残るケース
    cleaned = re.sub(r'</think>', '', cleaned).strip()
    # 先頭の空白行を除去
    cleaned = cleaned.lstrip('\n')
    return cleaned


def main():
    # --- 1. C2結果読み込み ---
    results_file = PROJECT_ROOT / "results" / "phase9c_step2_20260227_071313.json"
    print(f"Loading C2 results from: {results_file}")

    with open(results_file, encoding="utf-8") as f:
        data = json.load(f)

    results = data["systems"]["hybrid_rag"]["results"]
    print(f"Total results: {len(results)}")

    # --- 2. テストケースのプロンプト取得 ---
    tc_map = {tc.id: tc for tc in ALL_MULTI_AREA_TEST_CASES}

    # --- 3. 高品質データ選定 ---
    COMPOSITE_THRESHOLD = 70
    REASONING_THRESHOLD = 3

    high_quality = []
    for r in results:
        composite = r.get("composite_score", 0)
        reasoning = r.get("reasoning_score", 0)

        if composite >= COMPOSITE_THRESHOLD and reasoning >= REASONING_THRESHOLD:
            test_id = r["test_id"]
            tc = tc_map.get(test_id)
            if tc is None:
                print(f"  WARNING: test_id {test_id} not found in test cases, skipping")
                continue

            answer = clean_answer(r["answer"])
            if not answer:
                print(f"  WARNING: empty answer for {test_id}, skipping")
                continue

            high_quality.append({
                "instruction": tc.prompt,
                "input": "",
                "output": answer,
                "metadata": {
                    "test_id": test_id,
                    "composite_score": composite,
                    "reasoning_score": reasoning,
                    "evidence_score": r.get("evidence_score", 0),
                    "level": r["level"],
                    "subcategory": r["subcategory"],
                    "area": r.get("target_area", "cross_area"),
                }
            })

    print(f"High quality samples (composite>={COMPOSITE_THRESHOLD} AND reasoning>={REASONING_THRESHOLD}): {len(high_quality)}")

    # --- 4. 統計表示 ---
    from collections import Counter
    level_dist = Counter(s["metadata"]["level"] for s in high_quality)
    subcat_dist = Counter(s["metadata"]["subcategory"] for s in high_quality)
    area_dist = Counter(s["metadata"]["area"] for s in high_quality)

    print(f"\nLevel distribution:")
    for level in sorted(level_dist):
        print(f"  L{level}: {level_dist[level]}")

    print(f"\nSubcategory distribution:")
    for subcat in sorted(subcat_dist):
        print(f"  {subcat}: {subcat_dist[subcat]}")

    print(f"\nArea distribution:")
    for area in sorted(area_dist, key=lambda x: x or ""):
        print(f"  {area or 'cross_area'}: {area_dist[area]}")

    # --- 5. Train/Validation分割 (80/20) ---
    random.seed(42)
    indices = list(range(len(high_quality)))
    random.shuffle(indices)

    split_point = int(len(high_quality) * 0.8)
    train_indices = sorted(indices[:split_point])
    valid_indices = sorted(indices[split_point:])

    train_data = [high_quality[i] for i in train_indices]
    valid_data = [high_quality[i] for i in valid_indices]

    print(f"\nTrain/Validation split:")
    print(f"  Train: {len(train_data)}")
    print(f"  Validation: {len(valid_data)}")

    # --- 6. 保存 ---
    output = {
        "metadata": {
            "source": str(results_file.name),
            "selection_criteria": {
                "composite_score_min": COMPOSITE_THRESHOLD,
                "reasoning_score_min": REASONING_THRESHOLD,
            },
            "total_results": len(results),
            "selected_count": len(high_quality),
            "train_count": len(train_data),
            "valid_count": len(valid_data),
            "random_seed": 42,
            "level_distribution": dict(sorted(level_dist.items())),
            "subcategory_distribution": dict(sorted(subcat_dist.items())),
            "area_distribution": {(k or "cross_area"): v for k, v in sorted(area_dist.items(), key=lambda x: x[0] or "")},
        },
        "train": train_data,
        "validation": valid_data,
    }

    output_file = PROJECT_ROOT / "data" / "phase9c_training_data.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nSaved to: {output_file}")
    print(f"File size: {output_file.stat().st_size / 1024:.1f} KB")

    # --- 7. サンプル表示 ---
    print(f"\n{'='*60}")
    print("Sample training data:")
    print(f"{'='*60}")
    for sample in train_data[:3]:
        print(f"\n[{sample['metadata']['test_id']}] L{sample['metadata']['level']} {sample['metadata']['subcategory']}")
        print(f"  instruction: {sample['instruction'][:80]}...")
        print(f"  output: {sample['output'][:100]}...")
        print(f"  composite: {sample['metadata']['composite_score']}, reasoning: {sample['metadata']['reasoning_score']}")


if __name__ == "__main__":
    main()
