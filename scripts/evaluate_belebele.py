import csv
import json
import random
from pathlib import Path
from time import perf_counter

import mlx.core as mx
from datasets import load_dataset
from mlx_lm import load

# ==========================================
# 1. 実験設定
# ==========================================

MODEL_ID = "Qwen/Qwen3-8B-MLX-4bit"

# 元モデルは baseline、軽量化後は compressed にする
RUN_NAME = "baseline"

# 動作確認は10問、本評価は900問
QUESTIONS_PER_LANGUAGE = 10
SEED = 42

# 保持対象5言語 + 削減対象5言語
LANGUAGES = {
    "日本語": "jpn_Jpan",
    "英語": "eng_Latn",
    "ロシア語": "rus_Cyrl",
    "韓国語": "kor_Hang",
    "中国語（簡体字）": "zho_Hans",
    "アラビア語": "arb_Arab",
    "フランス語": "fra_Latn",
    "スペイン語": "spa_Latn",
    "ヒンディー語": "hin_Deva",
    "タイ語": "tha_Thai",
}

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "evaluations" / RUN_NAME
CHOICES = "ABCD"


# ==========================================
# 2. 評価問題の識別
# ==========================================

def question_key(row):
    """言語間で同じ問題を対応付ける識別子"""
    return (
        f"{row['link']}|"
        f"{row['split']}|"
        f"{row['question_number']}"
    )


# ==========================================
# 3. 評価プロンプトの作成
# ==========================================

def make_prompt(row):
    """Belebeleの文章と4択問題を構築"""
    return (
        "Read the passage and answer the multiple-choice "
        "question. Reply with only the correct letter: "
        "A, B, C, or D.\n\n"
        f"Passage:\n{row['flores_passage']}\n\n"
        f"Question:\n{row['question']}\n"
        f"A. {row['mc_answer1']}\n"
        f"B. {row['mc_answer2']}\n"
        f"C. {row['mc_answer3']}\n"
        f"D. {row['mc_answer4']}\n"
    )


# ==========================================
# 4. モデルによる回答予測
# ==========================================

def predict(model, tokenizer, row, letter_ids):
    messages = [
        {"role": "user", "content": make_prompt(row)}
    ]

    token_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    if not isinstance(token_ids, list):
        raise TypeError("Tokenizer returned unexpected data")

    # 次のトークンのスコアを計算
    logits = model(mx.array([token_ids]))
    final_logits = logits[0, -1]
    mx.eval(final_logits)

    # A、B、C、Dの相対スコアで選択
    scores = [
        float(final_logits[token_id].item())
        for token_id in letter_ids
    ]

    best_index = max(range(4), key=lambda i: scores[i])

    return CHOICES[best_index], len(token_ids), scores


# ==========================================
# 5. 評価実行
# ==========================================

def main():
    if not (1 <= QUESTIONS_PER_LANGUAGE <= 900):
        raise ValueError("問題数は1〜900の範囲で指定")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("モデルを読み込みます:", MODEL_ID)
    model, tokenizer = load(MODEL_ID)

    letter_ids = []

    for letter in CHOICES:
        ids = tokenizer.encode(
            letter, add_special_tokens=False
        )
        if len(ids) != 1:
            raise ValueError(
                f"{letter} が単一トークンではありません: {ids}"
            )
        letter_ids.append(ids[0])

    # 英語版から評価対象の問題を選定
    english = load_dataset(
        "facebook/belebele",
        "eng_Latn",
        split="test",
    )

    indexes = random.Random(SEED).sample(
        range(len(english)),
        QUESTIONS_PER_LANGUAGE,
    )
    chosen_keys = [
        question_key(english[i]) for i in indexes
    ]

    detail_fields = [
        "language", "language_code", "question_id",
        "predicted", "correct", "is_correct",
        "prompt_tokens", "seconds",
        "logit_A", "logit_B", "logit_C", "logit_D",
    ]
    summary_fields = [
        "language", "language_code",
        "total", "correct", "accuracy_percent",
    ]

    details_file = OUTPUT_DIR / "details.csv"
    summary_file = OUTPUT_DIR / "summary.csv"

    with (
        details_file.open(
            "w", encoding="utf-8-sig", newline=""
        ) as dfile,
        summary_file.open(
            "w", encoding="utf-8-sig", newline=""
        ) as sfile,
    ):
        detail_writer = csv.DictWriter(
            dfile, fieldnames=detail_fields
        )
        summary_writer = csv.DictWriter(
            sfile, fieldnames=summary_fields
        )

        detail_writer.writeheader()
        summary_writer.writeheader()

        for language, code in LANGUAGES.items():
            print(f"\n評価中: {language} ({code})")

            dataset = load_dataset(
                "facebook/belebele",
                code,
                split="test",
            )

            rows_by_key = {
                question_key(row): row for row in dataset
            }

            missing = [
                key for key in chosen_keys
                if key not in rows_by_key
            ]
            if missing:
                raise RuntimeError(
                    f"{code} に共通問題がありません: {missing[:3]}"
                )

            correct_count = 0

            for index, key in enumerate(
                chosen_keys, start=1
            ):
                row = rows_by_key[key]

                start = perf_counter()
                predicted, length, scores = predict(
                    model, tokenizer, row, letter_ids
                )
                elapsed = perf_counter() - start

                answer = CHOICES[
                    int(row["correct_answer_num"]) - 1
                ]
                is_correct = int(predicted == answer)
                correct_count += is_correct

                detail_writer.writerow({
                    "language": language,
                    "language_code": code,
                    "question_id": key,
                    "predicted": predicted,
                    "correct": answer,
                    "is_correct": is_correct,
                    "prompt_tokens": length,
                    "seconds": round(elapsed, 3),
                    "logit_A": scores[0],
                    "logit_B": scores[1],
                    "logit_C": scores[2],
                    "logit_D": scores[3],
                })

                dfile.flush()

                if index % 10 == 0 or index == len(chosen_keys):
                    print(
                        f"  {index}/{len(chosen_keys)} "
                        f"| 正解数: {correct_count}"
                    )

            accuracy = (
                100 * correct_count / len(chosen_keys)
            )

            summary_writer.writerow({
                "language": language,
                "language_code": code,
                "total": len(chosen_keys),
                "correct": correct_count,
                "accuracy_percent": round(accuracy, 2),
            })
            sfile.flush()

            print(f"正答率: {accuracy:.2f}%")
            mx.clear_cache()

    # 実験条件も保存
    config = {
        "model": MODEL_ID,
        "run": RUN_NAME,
        "dataset": "facebook/belebele",
        "questions_per_language": QUESTIONS_PER_LANGUAGE,
        "seed": SEED,
        "thinking": False,
        "scoring": "next-token logits for A/B/C/D",
        "instruction_language": "English",
    }

    (OUTPUT_DIR / "config.json").write_text(
        json.dumps(
            config, ensure_ascii=False, indent=2
        ),
        encoding="utf-8",
    )

    print("\n評価終了")
    print("保存先:", OUTPUT_DIR)


if __name__ == "__main__":
    main()