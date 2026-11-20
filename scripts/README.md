# ベースライン評価：Qwen3-8B（語彙削減・追加学習前）

## 目的

未変更の `Qwen/Qwen3-8B-MLX-4bit` を、Belebeleの読解正答率と、10言語における生成速度で評価し、後の軽量化モデルとの比較用の結果を保存する。**この工程ではモデルの学習・語彙削減は行わない。**

- 保持対象：日本語、英語、ロシア語、韓国語、中国語（簡体字）
- 比較対象：アラビア語、フランス語、スペイン語、ヒンディー語、タイ語
- 既定モデル：`Qwen/Qwen3-8B-MLX-4bit`

## 保存場所

このフォルダの `scripts/` にある2つの `.py` ファイルを、Macの `~/Desktop/five-lang-llm/scripts/` にコピーする。VS Codeでプロジェクト全体を開き、Pythonインタープリターを `five-lang-llm/.venv/bin/python` にする。

```text
five-lang-llm/
├── .venv/
├── scripts/
│   ├── evaluate_belebele.py
│   └── evaluate_speed.py
└── evaluations/
    ├── baseline_smoke/  # 動作確認（各10問）
    └── baseline/        # 正式な実験結果
```

## 実行順序（VS Code）

1. `mlx_lm.chat` を `q` で終了し、メモリの競合を避ける。
2. `scripts/evaluate_belebele.py` を開いて、`QUESTIONS_PER_LANGUAGE = 10`、`RUN_NAME = "baseline_smoke"` のまま、VS Codeの **Run Python File** から実行する。
3. `evaluations/baseline_smoke/details.csv` と `summary.csv` が生成されていることを確認する。
4. 読解評価コードの設定を `QUESTIONS_PER_LANGUAGE = 900`、`RUN_NAME = "baseline"` に変え、同じ方法で実行する。対象は10言語×900問=9,000問。
5. `scripts/evaluate_speed.py` で `RUN_NAME = "baseline"` に変えて、同じ方法で実行する。対象は10言語×2問×3回=60回。
6. `evaluations/baseline/` 内のCSV/JSONが揃ったら、それを学習前の基準値として保管する。

**注意：** 同じ `RUN_NAME` で再実行すると、同じ出力ファイルは上書きされる。正式評価が終わった時点で結果をバックアップすること。

## 結果ファイル

| ファイル | 内容 |
| --- | --- |
| `details.csv` | Belebele問題ごとの予測、正解、正誤、入力トークン数、選択肢スコア、処理秒数 |
| `summary.csv` | 10言語それぞれの正答率（%） |
| `config.json` | Belebeleの評価設定・モデル名・乱数シード |
| `speed_details.csv` | 各試行の生成トークン数、TTFT、tokens/s、出力文字数、時間、メモリ、生成文 |
| `speed_summary.csv` | 言語別の速度指標の中央値など |
| `speed_config.json` | 速度計測条件、実行日時、環境、プロンプト |

## スクリプト内の主な処理

### `evaluate_belebele.py`

- `load_dataset("facebook/belebele", ..., split="test")`：Belebele評価データの取得。
- `question_key()`：`link` と `question_number` で10言語間の同一問題を対応付ける。
- `make_prompt()`：文章、設問、A〜D選択肢を構築する。
- `predict()`：推論時にA〜Dに対応するトークンの次トークンlogitを比較する。文章の生成は行わない。
- `main()`：正解ラベルと比較してAccuracyを計算し、問題別・言語別CSVを出力する。

### `evaluate_speed.py`

- `PROMPTS`：各言語で意味を揃えた2問を定義する。
- `to_chat_prompt()`：QwenのチャットテンプレートでNon-Thinking設定にする。
- `run_one()`：`perf_counter()`と`stream_generate()`で生成時間、最初のトークン到着までの時間、速度、ピークメモリを計測する。
- `make_sampler(temp=0.7, top_p=0.8, top_k=20)`：生成条件を明示する。
- `summary_for()`：言語別に中央値を算出する。

## 評価条件・解釈の注意

- Belebeleは**文章読解のベンチマーク**。日常会話の流暢さを直接測らない。公開元は **テストデータ専用**としているので、学習データに混ぜてはならない。
- 今回の正答率は、A〜Dの**次トークンlogit**を使う独自のzero-shotプロトコルによる。公式論文の全条件を再現したものではなく、他論文の数値と無条件に比較してはいけない。
- `seconds`（読解評価）は4択スコア算出に要した時間。**チャット応答時間ではない。**
- `ttft_s`（速度評価）はPython APIが最初のトークンイベントを返すまでの時間であり、UI表示・ネットワーク遅延は含まない。
- Qwen3のThinkingモードは両方無効にする。速度測定はサンプラーを明示（温度0.7、Top-p0.8、Top-k20）。モデル読み込み・ウォームアップ時間は除外する。
- MLX報告の`peak_memory_gb`はOS全体のメモリ消費量ではない。
- 語彙削減後は**同じ問題、プロンプト、設定、実行端末、データセット**で評価する。モデルやTokenizerが変わるとトークン長も変わるため、tokens/sのみでは性能が判断できない。

## 公式資料

- https://huggingface.co/datasets/facebook/belebele
- https://github.com/facebookresearch/belebele
- https://huggingface.co/Qwen/Qwen3-8B-MLX-4bit
- https://github.com/ml-explore/mlx-lm/tree/v0.32.2
