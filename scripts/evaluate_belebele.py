"""Qwen3-8Bの多言語読解力（正答率）をBelebeleで測定するプログラム。

【このプログラムの役割】
1. Hugging Faceの公開ベンチマーク「facebook/belebele」を取得する。
2. 各言語で意味が対応する読解問題を、同じ問題IDで選択する。
3. モデルに文章・質問・4つの選択肢を提示する。
4. A～Dの「次トークンのスコア」を比較して選択肢を決定する。
5. 正解ラベルと比較して正答率を計算し、CSVに保存する。

【重要】
- これは「文章読解の正答率」の評価であり、チャットの文章生成速度の測定ではない。
- details.csvのsecondsは、4択のスコア算出処理にかかった時間。
  応答の生成開始時間（TTFT）や生成トークン/秒ではない。
- 問題文の指示は英語で固定し、読解文・選択肢を各言語に切り替える。
- モデルの重みは変更せず、学習もしない。

保存場所：five-lang-llm/scripts/evaluate_belebele.py
VS Codeで .venv/bin/python を選択し、ファイルを実行する。
"""

# 評価結果をCSV・JSONで保存し、問題を再現可能に抽出するための標準ライブラリ。
import csv
import json
import random
from pathlib import Path
from time import perf_counter

# MLX：MacのGPU上でモデルのスコア（logits）を計算する。
import mlx.core as mx
# 【Belebeleの読み込み】Hugging Faceから「正解付きの読解問題」を取得する。
from datasets import load_dataset
# MLX LM：ダウンロード済みのQwen3モデルとTokenizerを読み込む。
from mlx_lm import load
# ---- 評価条件：軽量化前後で固定すること ----
# 評価対象モデル。語彙削減後は、圧縮モデルのフォルダパスに変更する。
MODEL_ID = "Qwen/Qwen3-8B-MLX-4bit"
# CSV保存先を分ける名前。同じ名前で再実行すると既存の結果を上書きする。
RUN_NAME = "baseline_full"  # 動作確認後、正式評価時は "baseline" に変更する。
# 1言語あたりの評価問題数：最初は10問、正式評価は900問。
QUESTIONS_PER_LANGUAGE = 900
# 問題抽出の乱数を固定し、別モデルでも同じ問題を使えるようにする。
SEED = 42

# 保持対象5言語と、能力低下を観察する削減対象5言語。値はBelebeleの言語識別子。
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

# このファイルの1つ上の親フォルダがプロジェクトのルートディレクトリ。
ROOT = Path(__file__).resolve().parents[1]
# 評価結果：evaluations/baseline/ や evaluations/compressed/ に分離して保存する。
OUTPUT_DIR = ROOT / "evaluations" / RUN_NAME
# 4択のラベルはすべての言語でA/B/C/Dに統一。
CHOICES = "ABCD"


def question_key(row):
    """翻訳されても共通の問題を対応付ける識別子を作る。"""
    # Hugging Face版Belebeleの列には split が存在しない。
    # 対応する問題の link と question_number で言語間のIDを揃える。
    # 同一IDが重複した場合は、main() 側で検出して停止する。
    return f"{row['link']}|{row['question_number']}"


def make_prompt(row):
    """Belebeleの読解文・質問・4択を、AIへ渡すプロンプトへ整形する。"""
    # 評価指示は英語で統一。文章や各選択肢は各言語のデータをそのまま使用する。
    # 指示の言語によるバイアスがあり得るため、圧縮前後では必ず同条件にする。
    return (
        "Read the passage and answer the multiple-choice question. "
        "Reply with only the correct letter: A, B, C, or D.\n\n"
        f"Passage:\n{row['flores_passage']}\n\n"
        f"Question:\n{row['question']}\n"
        f"A. {row['mc_answer1']}\n"
        f"B. {row['mc_answer2']}\n"
        f"C. {row['mc_answer3']}\n"
        f"D. {row['mc_answer4']}\n"
    )


def predict(model, tokenizer, row, letter_ids):
    """【正答選択】モデルの次トークンのスコアからA～Dを選ぶ。"""
    # Qwen3が想定するチャット形式に読解問題を変換する。
    messages = [{"role": "user", "content": make_prompt(row)}]
    token_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,  # 長い思考文は生成させず、選択肢の採点に統一する。
    )
    if not isinstance(token_ids, list):
        raise TypeError("Expected a list of token IDs from apply_chat_template")

    # 【正答率の測定・核心部分】
    # 文章は生成しない。1回の順伝播で「次にどのトークンが出やすいか」を計算する。
    logits = model(mx.array([token_ids]))
    final_logits = logits[0, -1]  # 回答開始位置の、全語彙に対するスコア。
    mx.eval(final_logits)  # MLXの遅延計算をここで確定し、計算を実行する。
    # A/B/C/DそれぞれのトークンIDに対するスコアを抽出。
    scores = [float(final_logits[token_id].item()) for token_id in letter_ids]
    # 最も高いスコアの選択肢を予測結果とする（確率の大小順と等価）。
    best_index = max(range(4), key=lambda i: scores[i])
    return CHOICES[best_index], len(token_ids), scores


def main():
    """ベンチマークの取得 → 推論 → 自動採点 → 保存の全工程。"""
    # 誤って全件以上の問題数を指定した場合は停止する。
    if not (1 <= QUESTIONS_PER_LANGUAGE <= 900):
        raise ValueError("QUESTIONS_PER_LANGUAGE must be between 1 and 900")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 【モデル読込】初回はHugging Faceから必要なファイルを取得する場合がある。
    # 学習・重みの更新は行わない。
    print("Loading model:", MODEL_ID)
    model, tokenizer = load(MODEL_ID)
    # A/B/C/Dが1トークンで表されることを確認し、各トークンIDを取得。
    letter_ids = []
    for letter in CHOICES:
        ids = tokenizer.encode(letter, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"Choice {letter!r} is not a single token: {ids}")
        letter_ids.append(ids[0])

    # 【Belebeleを使っている場所①】
    # Hugging Face公開データセット facebook/belebele の英語版・testを読み込む。
    # 英語版から「全言語で評価する共通問題」を乱数固定で選ぶ。
    print("Loading aligned English question IDs...")
    english = load_dataset("facebook/belebele", "eng_Latn", split="test")
    indexes = random.Random(SEED).sample(range(len(english)), QUESTIONS_PER_LANGUAGE)
    chosen_keys = [question_key(english[index]) for index in indexes]
    if len(chosen_keys) != len(set(chosen_keys)):
        raise RuntimeError("抽出したBelebele問題IDが重複しています")

    # 【結果ファイル】details.csv＝問題ごとの予測と処理時間、
    # summary.csv＝言語ごとの正答数・正答率。
    detail_fields = [
        "language", "language_code", "question_id", "predicted", "correct",
        "is_correct", "prompt_tokens", "seconds", "logit_A", "logit_B",
        "logit_C", "logit_D",
    ]
    summary_fields = ["language", "language_code", "total", "correct", "accuracy_percent"]
    details_file = OUTPUT_DIR / "details.csv"
    summary_file = OUTPUT_DIR / "summary.csv"

    with details_file.open("w", encoding="utf-8-sig", newline="") as dfile, \
         summary_file.open("w", encoding="utf-8-sig", newline="") as sfile:
        detail_writer = csv.DictWriter(dfile, fieldnames=detail_fields)
        summary_writer = csv.DictWriter(sfile, fieldnames=summary_fields)
        detail_writer.writeheader()
        summary_writer.writeheader()

        for language, code in LANGUAGES.items():
            print(f"\nEvaluating {language} ({code})")
            # 【Belebeleを使っている場所②】
            # 指定言語の翻訳版900問を取得する（例：jpn_Jpan、arb_Arab）。
            dataset = load_dataset("facebook/belebele", code, split="test")
            # 英語版から抽出した問題と同じ元問題だけを言語横断で選ぶ。
            rows_by_key = {question_key(row): row for row in dataset}
            if len(rows_by_key) != len(dataset):
                raise RuntimeError(f"{code} のBelebele問題IDに重複があります")
            missing = [key for key in chosen_keys if key not in rows_by_key]
            if missing:
                raise RuntimeError(f"Missing parallel questions in {code}: {missing[:3]}")

            correct_count = 0
            for index, key in enumerate(chosen_keys, start=1):
                row = rows_by_key[key]
                # 【実行時間の測定】4択スコア算出の処理時間（秒）。
                # 文章を生成していないため「チャット応答速度」ではない。
                start = perf_counter()
                predicted, length, scores = predict(model, tokenizer, row, letter_ids)
                elapsed = perf_counter() - start
                # 【正解ラベル】Belebeleが提供する1～4の正解番号をA～Dに変換。
                answer = CHOICES[int(row["correct_answer_num"]) - 1]
                # 【自動採点】予測結果と正解ラベルを比較して1点／0点を付ける。
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
                dfile.flush()  # 途中で中断しても、採点済みの行が残るように書き込む。
                if index % 10 == 0 or index == len(chosen_keys):
                    print(f"  {index}/{len(chosen_keys)} | correct: {correct_count}")

            # 【正答率の計算】正答率（%）＝正解数 ÷ 評価問題数 × 100。
            accuracy = 100 * correct_count / len(chosen_keys)
            summary_writer.writerow({
                "language": language,
                "language_code": code,
                "total": len(chosen_keys),
                "correct": correct_count,
                "accuracy_percent": round(accuracy, 2),
            })
            sfile.flush()
            print(f"  Accuracy: {accuracy:.2f}%")
            mx.clear_cache()  # 言語を切り替える際に不要なMLXキャッシュを解放する。

    # 使用モデルや問題数・乱数シードを記録し、軽量化前後の条件を照合できるようにする。
    (OUTPUT_DIR / "config.json").write_text(
        json.dumps({
            "model": MODEL_ID,
            "run": RUN_NAME,
            "dataset": "facebook/belebele",
            "questions_per_language": QUESTIONS_PER_LANGUAGE,
            "seed": SEED,
            "thinking": False,
            "scoring": "next-token logits, greedy A/B/C/D choice",
            "instruction_language": "English",
            "question_id_columns": ["link", "question_number"],
            "note": "未学習モデルの相対logitsによる独自zero-shot評価。Belebele公式論文の全条件の再現ではない。",
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("\nSaved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
