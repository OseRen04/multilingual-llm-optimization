"""Qwen3-8Bの「文章生成時の応答速度」を10言語で実測するプログラム。

【このプログラムの役割】
1. 言語ごとに同じ意味の質問をモデルに入力する。
2. MLX LMの stream_generate() で、文章を1トークンずつ順次生成する。
3. 実際の経過時間、最初の出力までの時間、生成速度、メモリ使用量を記録する。
4. 複数回測定した値の中央値を言語ごとにCSVで保存する。

【何をどこで測っているか】
- run_one() の perf_counter()：経過時間の実測（秒）。
- run_one() の for response in stream_generate(...)：実際の文章生成と先頭出力時刻の測定。
- last.generation_tps / last.prompt_tps：MLX LMが報告する生成速度／入力処理速度。
- last.peak_memory：MLX LMが報告するピークメモリ。
- summary_for()：各言語の測定値を中央値で集計。

【重要】
- 正答率を測定するBelebeleとは別の速度ベンチマーク。
- モデルの読み込み時間とウォームアップ時間は測定に含めない。
- TTFTは、入力を受けてから最初のストリーム出力が返るまでの実測値。
- 端末負荷、温度、入力長、出力長によって測定値は変わる。
- モデルの学習や重みの変更は行わない。
- stream_generate() に make_sampler() を明示している。

保存場所：five-lang-llm/scripts/evaluate_speed.py
VS Codeで .venv/bin/python を選択して実行する。
出力：evaluations/<RUN_NAME>/speed_details.csv 等
"""

# 評価結果の保存・統計処理・環境情報の記録に使用する標準ライブラリ。
import csv
import json
import platform
import statistics
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

# MLX：GPU計算の同期、ピークメモリ計測、乱数固定に使用する。
import mlx.core as mx
# MLX LM：モデルの読込と「逐次的な文章生成」を行う。
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler  # 【サンプリング】生成条件を明示的に固定する。

# === 実験設定：圧縮前後でなるべく同じ値を維持する ===
MODEL_ID = "Qwen/Qwen3-8B-MLX-4bit"  # 評価対象モデル。
RUN_NAME = "baseline_smoke"  # 動作確認後、正式評価時は "baseline" に変更する。
REPEATS_PER_PROMPT = 3  # 1問を3回測り、偶然の測定誤差を減らす。
MAX_NEW_TOKENS = 128  # 1回で生成する最大トークン数。長さの上限を揃える。
SEED = 42  # MLXの乱数シードを固定して実験を再現しやすくする。
ENABLE_THINKING = False  # Thinkingモードを無効化。比較する全モデルで統一。
# Qwen3公式のNon-Thinking推奨値を使用。値を変更した場合は圧縮後の評価でも揃える。
SAMPLING_TEMPERATURE = 0.7
SAMPLING_TOP_P = 0.8
SAMPLING_TOP_K = 20

# 【速度測定用の質問】各言語で意味を揃えた2問を定義する。
# 圧縮前後でプロンプトを変更すると、速度差の原因を特定しづらくなる。
PROMPTS = {
    "日本語": [
        "標高が高い場所で水の沸点が下がる理由を、3文で説明してください。",
        "家庭で電力消費を減らす方法を、具体例を挙げて5つ説明してください。",
    ],
    "英語": [
        "Explain in three sentences why water boils at a lower temperature at high altitudes.",
        "Explain five concrete ways to reduce electricity use at home, with examples.",
    ],
    "ロシア語": [
        "Объясни тремя предложениями, почему на большой высоте вода кипит при более низкой температуре.",
        "Назови пять конкретных способов снизить расход электроэнергии дома с примерами.",
    ],
    "韓国語": [
        "고도가 높은 곳에서 물의 끓는점이 낮아지는 이유를 세 문장으로 설명해 주세요.",
        "가정에서 전기 사용량을 줄일 수 있는 구체적인 방법 다섯 가지를 예와 함께 설명해 주세요.",
    ],
    "中国語（簡体字）": [
        "请用三句话解释为什么海拔越高，水的沸点越低。",
        "请举例说明家庭节约用电的五种具体方法。",
    ],
    "アラビア語": [
        "اشرح في ثلاث جمل لماذا تنخفض درجة غليان الماء في المرتفعات.",
        "اذكر خمس طرق عملية لتقليل استهلاك الكهرباء في المنزل مع أمثلة.",
    ],
    "フランス語": [
        "Explique en trois phrases pourquoi l'eau bout à une température plus basse en altitude.",
        "Explique cinq façons concrètes de réduire la consommation d'électricité à la maison, avec des exemples.",
    ],
    "スペイン語": [
        "Explica en tres frases por qué el agua hierve a menor temperatura a gran altitud.",
        "Explica cinco formas concretas de reducir el consumo eléctrico en casa, con ejemplos.",
    ],
    "ヒンディー語": [
        "तीन वाक्यों में समझाइए कि अधिक ऊँचाई पर पानी का क्वथनांक कम क्यों होता है।",
        "उदाहरणों सहित घर में बिजली की खपत कम करने के पाँच व्यावहारिक तरीके बताइए।",
    ],
    "タイ語": [
        "อธิบายเป็นสามประโยคว่าทำไมน้ำจึงเดือดที่อุณหภูมิต่ำลงเมื่ออยู่บนที่สูง",
        "อธิบายวิธีลดการใช้ไฟฟ้าในบ้านห้าวิธีพร้อมตัวอย่างที่เป็นรูปธรรม",
    ],
}

# スクリプトの1つ上の親ディレクトリをプロジェクトのルートとする。
ROOT = Path(__file__).resolve().parents[1]
# baseline / compressed を保存先のフォルダ名で区別する。
OUTPUT_DIR = ROOT / "evaluations" / RUN_NAME

# 【CSV保存項目】1回の生成ごとの測定結果。
# ttft_s：最初の出力までの秒数、generation_tokens_per_s：生成速度、
# end_to_end_s：テンプレート化から生成完了までの秒数。
DETAIL_FIELDS = [
    "language", "prompt_id", "repeat", "prompt_tokens", "generated_tokens",
    "ttft_s", "generation_tokens_per_s", "prompt_tokens_per_s",
    "output_characters", "output_utf8_bytes", "characters_per_s",
    "utf8_bytes_per_s", "inference_total_s", "end_to_end_s", "peak_memory_gb",
    "finish_reason", "generated_text",
]
# 【CSV集計項目】1言語内の複数回測定を中央値などでまとめたもの。
SUMMARY_FIELDS = [
    "language", "samples", "median_ttft_s", "median_generation_tokens_per_s",
    "median_characters_per_s", "median_utf8_bytes_per_s",
    "median_inference_total_s", "median_end_to_end_s",
    "median_peak_memory_gb", "mean_prompt_tokens", "mean_generated_tokens",
]


def to_chat_prompt(tokenizer, user_message):
    """通常のチャットと同じQwen3のテンプレートに質問を変換する。"""
    # Thinkingの有効・無効もここで指定する。生成開始前の前処理。
    messages = [{"role": "user", "content": user_message}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=ENABLE_THINKING,
    )


def run_one(model, tokenizer, user_message):
    """【応答速度の実測・核心部分】1回の文章生成について時間と速度を記録。"""
    # 【実測①】入力のテンプレート整形を含めた開始時刻（高精度タイマー）。
    start_all = perf_counter()
    prompt = to_chat_prompt(tokenizer, user_message)
    # 【実測②】モデルによる推論の開始時刻。テンプレート処理時間を分離する。
    start_inference = perf_counter()

    # 【メモリ】前回の測定のピーク値をリセットする。
    mx.reset_peak_memory()
    last = None
    first_output_at = None  # 最初の生成結果が返った時刻を保存する。
    chunks = []
    # 【文章生成と速度実測】stream_generate() は生成途中のデータを順次返す。
    for response in stream_generate(
        model,
        tokenizer,
        prompt,
        max_tokens=MAX_NEW_TOKENS,
        sampler=make_sampler(
            temp=SAMPLING_TEMPERATURE,
            top_p=SAMPLING_TOP_P,
            top_k=SAMPLING_TOP_K,
        ),
    ):
        # 【実測③】最初のストリーム出力が届いた時刻を直接記録する。
        # 注意：画面描画や通信が発生しないローカルPython API内のTTFT。
        if first_output_at is None:
            first_output_at = perf_counter()
        last = response  # 最終要素にMLX LMの速度・ピークメモリ情報が入る。
        chunks.append(response.text)  # 生成された部分文字列を順次連結できるよう蓄積。
    mx.synchronize()  # GPUの未完了計算を完了させてから終了時刻を取る。
    # 【実測④】文章生成とGPU計算が終わった時刻。
    end = perf_counter()

    if last is None:
        raise RuntimeError("MLX LM returned no generation response")

    # 【TTFTの実測】入力の整形開始から最初の生成結果までの経過秒数。
    # 以前の計算式（prompt_tokens / prompt_tps）は近似値だったため使用しない。
    first_token_latency = first_output_at - start_all
    text = "".join(chunks)
    # 【生成処理時間の推定】MLX LMの生成トークン数 ÷ 生成tokens/s。
    # これを用いる文字数/秒は生成フェーズの値で、TTFTを含まない。
    generation_duration = (
        last.generation_tokens / last.generation_tps
        if last.generation_tps > 0 else None
    )
    # 【記録する指標】「実測した経過時間」と「MLX LMが報告する速度」を分ける。
    return {
        "prompt_tokens": int(last.prompt_tokens),  # 質問（テンプレート含む）のトークン数。
        "generated_tokens": int(last.generation_tokens),  # 実際に生成されたトークン数。
        "ttft_s": first_token_latency,  # perf_counterによる実測値（秒）。
        "generation_tokens_per_s": float(last.generation_tps),  # MLX LM報告値。
        "prompt_tokens_per_s": float(last.prompt_tps),  # MLX LM報告値。
        "output_characters": len(text),  # 日本語等ではtokens/sと文字数/sが異なる。
        "output_utf8_bytes": len(text.encode("utf-8")),  # UTF-8の出力バイト数。
        "characters_per_s": len(text) / generation_duration if generation_duration else 0.0,
        "utf8_bytes_per_s": len(text.encode("utf-8")) / generation_duration
        if generation_duration else 0.0,
        "inference_total_s": end - start_inference,  # モデル実行の総経過秒数。
        "end_to_end_s": end - start_all,  # テンプレート整形も含む総経過秒数。
        "peak_memory_gb": float(last.peak_memory),  # MLX LMが報告するメモリ（GB）。
        "finish_reason": last.finish_reason,  # 停止理由（長さ上限など）。
        "generated_text": text,  # 生成された文章そのもの。品質確認にも使える。
    }


def rounded(value):
    """CSVを読みやすくするため、小数点以下を4桁に丸める。"""
    return round(value, 4) if value is not None else ""


def summary_for(language, rows):
    """【速度の集計】同じ言語の複数回測定結果を中央値／平均値でまとめる。"""
    def med(key):
        # 速度には一時的な遅延や熱の影響が出るため、中央値を使う。
        return rounded(statistics.median(r[key] for r in rows if r[key] is not None))
    def avg(key):
        return rounded(statistics.mean(r[key] for r in rows))

    return {
        "language": language,
        "samples": len(rows),
        "median_ttft_s": med("ttft_s"),
        "median_generation_tokens_per_s": med("generation_tokens_per_s"),
        "median_characters_per_s": med("characters_per_s"),
        "median_utf8_bytes_per_s": med("utf8_bytes_per_s"),
        "median_inference_total_s": med("inference_total_s"),
        "median_end_to_end_s": med("end_to_end_s"),
        "median_peak_memory_gb": med("peak_memory_gb"),
        "mean_prompt_tokens": avg("prompt_tokens"),
        "mean_generated_tokens": avg("generated_tokens"),
    }


def main():
    """実験条件の記録 → モデル読込 → ウォームアップ → 速度測定 → CSV保存。"""
    if REPEATS_PER_PROMPT < 1 or MAX_NEW_TOKENS < 2:
        raise ValueError("Invalid benchmark configuration")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # 【実験条件の保存】モデル・測定回数・PC環境・質問文等をJSONで記録。
    # 注意：RUN_NAMEが同じ場合、以前の結果が上書きされる。
    (OUTPUT_DIR / "speed_config.json").write_text(
        json.dumps({
            "model_id": MODEL_ID,
            "run_name": RUN_NAME,
            "repeats_per_prompt": REPEATS_PER_PROMPT,
            "max_new_tokens": MAX_NEW_TOKENS,
            "thinking": ENABLE_THINKING,
            "seed": SEED,
            "temperature": SAMPLING_TEMPERATURE,
            "top_p": SAMPLING_TOP_P,
            "top_k": SAMPLING_TOP_K,
            "prompt_kv_cache_reuse": False,
            "measurement": "Local Python API / MLX generation; model load excluded",
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "os": platform.platform(),
            "machine": platform.machine(),
            "mlx_version": version("mlx"),
            "mlx_lm_version": version("mlx-lm"),
            "prompts": PROMPTS,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    mx.random.seed(SEED)
    # 【モデル読込】この時間は速度の測定対象から除外する。
    print("Loading model:", MODEL_ID)
    model, tokenizer = load(MODEL_ID)

    # 【ウォームアップ】GPU初回起動等の影響を減らすため、短く生成してから本計測する。
    # ウォームアップの時間はCSVには保存しない。
    print("Warming up GPU (not included in measurements)...")
    warmup_prompt = to_chat_prompt(tokenizer, PROMPTS["日本語"][0])
    for _ in stream_generate(
        model, tokenizer, warmup_prompt, max_tokens=16,
        sampler=make_sampler(
            temp=SAMPLING_TEMPERATURE,
            top_p=SAMPLING_TOP_P,
            top_k=SAMPLING_TOP_K,
        ),
    ):
        pass
    mx.synchronize()

    # 【出力先】1回ごとの結果と、言語別の集計結果を別CSVで保存する。
    details_path = OUTPUT_DIR / "speed_details.csv"
    summary_path = OUTPUT_DIR / "speed_summary.csv"
    with (
        details_path.open("w", encoding="utf-8-sig", newline="") as detail_f,
        summary_path.open("w", encoding="utf-8-sig", newline="") as summary_f,
    ):
        detail_writer = csv.DictWriter(detail_f, fieldnames=DETAIL_FIELDS)
        summary_writer = csv.DictWriter(summary_f, fieldnames=SUMMARY_FIELDS)
        detail_writer.writeheader()
        summary_writer.writeheader()
        for language, prompt_list in PROMPTS.items():
            language_rows = []
            print(f"\n{language}")
            for prompt_index, prompt in enumerate(prompt_list, start=1):
                for repeat in range(1, REPEATS_PER_PROMPT + 1):
                    # 【応答速度の実測を呼び出す場所】1問を設定回数だけ生成して計測する。
                    # 同じ問題の各反復で乱数を揃え、試行間の生成文の差を減らす。
                    mx.random.seed(SEED + prompt_index)
                    record = run_one(model, tokenizer, prompt)
                    record.update({
                        "language": language,
                        "prompt_id": prompt_index,
                        "repeat": repeat,
                    })
                    detail_writer.writerow({
                        field: rounded(record[field]) if isinstance(record[field], float)
                        else record[field]
                        for field in DETAIL_FIELDS
                    })
                    detail_f.flush()
                    language_rows.append(record)
                    print(
                        f"  prompt {prompt_index}, run {repeat}: "
                        f"{record['generation_tokens_per_s']:.1f} tok/s, "
                        f"TTFT {record['ttft_s']:.2f}s, "
                        f"total {record['end_to_end_s']:.2f}s"
                    )
            # 【言語別集計】TTFT・tokens/s・総時間・メモリの中央値を計算。
            result = summary_for(language, language_rows)
            summary_writer.writerow(result)
            summary_f.flush()
            print(
                f"  Median: {result['median_generation_tokens_per_s']} tok/s, "
                f"TTFT {result['median_ttft_s']}s"
            )

    print("\nFinished. Results saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
