# 評価スクリプト（`scripts/`）

Qwen3-8Bを対象に、**多言語読解の正答率**と**文章生成時の応答速度**を測定するためのスクリプトをまとめる。語彙削減・追加学習の前後で同じ条件を使い、モデルの品質と効率を比較する。

> **注意：** 両スクリプトは評価専用であり、学習やモデル重みの変更は行わない。ここに記載する設定は現在のコードに対応する。M5 MacBook Airでの通し実行・計測精度の検証は未実施。

## ファイル一覧

| ファイル | 目的 | 主な出力 |
| --- | --- | --- |
| [`evaluate_belebele.py`](./evaluate_belebele.py) | Belebeleの4択読解問題による正答率評価 | `details.csv`, `summary.csv`, `config.json` |
| [`evaluate_speed.py`](./evaluate_speed.py) | 実際の文章生成における時間・速度・メモリを測定 | `speed_details.csv`, `speed_summary.csv`, `speed_config.json` |

## 評価対象

- **保持対象（5言語）**：日本語、英語、ロシア語、韓国語、中国語（簡体字）
- **比較対象（5言語）**：アラビア語、フランス語、スペイン語、ヒンディー語、タイ語
- **元モデル**：`Qwen/Qwen3-8B-MLX-4bit`
- **実行環境**：Apple Silicon搭載Mac、Python 3.12、MLX / MLX LM

比較対象の5言語は、語彙削減による影響を観察するために測定する。これらの言語の性能低下自体を研究の成功条件とはしない。

## 事前準備とVS Codeでの実行

1. VS Codeでプロジェクトルート `five-lang-llm/` を開く。
2. Pythonインタープリターに `five-lang-llm/.venv/bin/python` を指定する。
3. 使用中の `mlx_lm.chat` を終了し、メモリの競合を避ける。
4. `scripts/evaluate_belebele.py` または `scripts/evaluate_speed.py` を開き、右上の **Run Python File** で実行する。
5. プロジェクトルートの `evaluations/<RUN_NAME>/` に作成されたCSV・JSONを確認する。

両スクリプトは `mlx`、`mlx-lm`、`datasets` などのパッケージを必要とする。現在の仮想環境には導入済み。環境を作り直す場合は、**対象の仮想環境を有効化した後**、次を実行する。

```bash
uv pip install "mlx-lm[train]" datasets
```

- `uv pip install`：現在の仮想環境へPythonパッケージをインストールする。
- `"mlx-lm[train]"`：MLX LMと学習用の追加依存関係を導入する。
- `datasets`：Hugging Faceの評価データセットを取得する。
- 初回はインターネット経由でパッケージやモデル・評価データがダウンロードされる場合がある。

## 1. `evaluate_belebele.py`：読解正答率

### 使用データ

[Belebele（`facebook/belebele`）](https://huggingface.co/datasets/facebook/belebele) の `test` 分割を使用する。10言語で対応する問題を選び、読解文・設問・4つの選択肢をモデルに入力する。

**処理の流れ**

1. `main()` の `load_dataset(...)` でBelebeleの英語版と各言語版を取得する。
2. `question_key()` で同じ元問題を対応付ける。`SEED` によって抽出対象を固定する。
3. `make_prompt()` で文章・質問・選択肢A〜Dを組み立てる。**指示文は英語で固定**する。
4. `predict()` でモデルの次トークンのスコア（logits）を計算し、A〜Dの最大スコアを予測として選ぶ。
5. `main()` で正解ラベルと突き合わせ、`正答率 = 正解数 / 問題数 × 100` を計算する。
6. 問題別結果と各言語の集計をCSVに保存する。

**主な設定（ファイル冒頭）**

```python
MODEL_ID = "Qwen/Qwen3-8B-MLX-4bit"
RUN_NAME = "baseline"
QUESTIONS_PER_LANGUAGE = 10  # 動作確認。全問評価は900
SEED = 42
```

- `QUESTIONS_PER_LANGUAGE` は**1言語あたり**の問題数。10なら計100問、900なら計9,000問。
- `details.csv`：`question_id`、モデル予測、正解、正誤、入力トークン数、選択肢スコア、スコア算出時間（秒）。
- `summary.csv`：言語別の問題数、正解数、正答率（%）。
- `config.json`：使用モデル、問題数、乱数シードなどの評価条件。

**測れないもの：** 日常会話の自然さや文章生成の速度は測定しない。`details.csv` の `seconds` は4択スコア算出時間であり、チャットの応答時間ではない。また、これは**A〜Dの次トークンスコアを使う独自評価方式**であり、Belebele論文の全評価手順を完全再現したものではない。

## 2. `evaluate_speed.py`：文章生成速度

### 使用データ・計測箇所

Belebeleではなく、スクリプト内の `PROMPTS` に記述した質問文を使用する。各言語**2問×3回＝6回**、10言語で合計**60回**測定する。

1. `main()` でモデルを読み込み、短い文章生成によるウォームアップを実行する。**モデル読み込みとウォームアップは計測から除外**する。
2. `run_one()` の `perf_counter()` で処理開始時刻を記録する。
3. `stream_generate(...)` により実際にテキストを生成し、最初の出力を受け取った時刻を記録する。
4. 生成完了までの経過時間を計算し、MLX LMが報告する生成速度・入力処理速度・ピークメモリを取得する。
5. `summary_for()` で言語ごとに中央値・平均値を集計し、CSVに保存する。

**主な設定（ファイル冒頭）**

```python
MODEL_ID = "Qwen/Qwen3-8B-MLX-4bit"
RUN_NAME = "baseline"
REPEATS_PER_PROMPT = 3
MAX_NEW_TOKENS = 128
SEED = 42
ENABLE_THINKING = False
```

| 指標 | CSV列 | 説明 |
| --- | --- | --- |
| 最初の出力までの時間 | `ttft_s` | 入力のテンプレート化開始から、最初のストリーム出力を受信するまでの実測秒数 |
| 生成速度 | `generation_tokens_per_s` | MLX LMが報告する生成速度（tokens/s） |
| 入力処理速度 | `prompt_tokens_per_s` | MLX LMが報告する入力処理速度（tokens/s） |
| 推論時間 | `inference_total_s` | 推論開始からGPU同期完了までの実測秒数 |
| 総処理時間 | `end_to_end_s` | チャットテンプレート化から生成完了までの実測秒数 |
| 出力の文字数／秒 | `characters_per_s` | 出力文字数と推定生成時間から算出 |
| ピークメモリ | `peak_memory_gb` | MLX LMが報告するピークメモリ（GB） |

- `speed_details.csv`：各試行の指標と生成テキスト。
- `speed_summary.csv`：言語ごとの中央値・平均値（例：`median_ttft_s`、`median_generation_tokens_per_s`）。
- `speed_config.json`：実行日時、OS、MLXのバージョン、質問文、設定値など。

**注意：** 現行コードの `speed_config.json` には `temperature: 0` と記録する処理があるが、`stream_generate()` 呼び出しでサンプリング方式を明示的に設定していない。したがって「Greedy固定」とはまだ保証できない。本格比較前にサンプラーの明示指定と動作検証が必要である。

## 評価結果の保存先

```text
five-lang-llm/
├── scripts/
│   ├── README.md
│   ├── evaluate_belebele.py
│   └── evaluate_speed.py
└── evaluations/
    ├── baseline/                 # 語彙削減前
    │   ├── details.csv
    │   ├── summary.csv
    │   ├── config.json
    │   ├── speed_details.csv
    │   ├── speed_summary.csv
    │   └── speed_config.json
    └── compressed/               # 語彙削減後（実行すると生成）
```

`RUN_NAME` を変えると保存先が変わる。**同じ `RUN_NAME` で再実行すると対応するCSV・JSONは上書きされる**ため、確定した結果は別名で保存する。

## 語彙削減前後を比較する際のルール

1. 元モデルの評価結果を `baseline` に保存する。
2. 軽量化した**モデルのコピー**を用意し、各スクリプトの `MODEL_ID` と `RUN_NAME = "compressed"` を変更する。
3. 問題抽出用の `SEED`、問題数、プロンプト、Thinking設定、生成長、測定回数を揃える。
4. 同一Mac・同一電源条件・できるだけ同じ端末負荷で速度を測り、複数回の中央値を比較する。
5. 正答率の差を**パーセントポイント**で示し、モデルファイル容量やメモリ使用量も比較する。
6. Tokenizer変更後は1文字あたりのトークン数が変わり得るため、`tokens/s` だけで高速化を判断しない。

## 評価上の限界・今後の改善

- Belebeleは読解評価であり、自由記述の生成品質や会話品質を直接測定しない。
- 正答選択は英語の指示文とA〜Dの次トークンスコアに依存する。プロンプト・チャットテンプレートの影響を検証する必要がある。
- 速度評価には出力長・発熱・GPU負荷・生成内容の変動が影響する。比較時は出力長の分布も確認する。
- 現在はモデルのバージョン（コミットID）やデータセットのリビジョンを固定していない。再現性を高めるなら固定する。
- ここでいうピークメモリはMLX LMが報告する値であり、macOS全体のメモリ消費量ではない。

## 参照資料

- [Qwen3-8B MLX 4bit（モデル）](https://huggingface.co/Qwen/Qwen3-8B-MLX-4bit)
- [Belebele（評価データセット）](https://huggingface.co/datasets/facebook/belebele)
- [MLX LM（公式リポジトリ）](https://github.com/ml-explore/mlx-lm)
