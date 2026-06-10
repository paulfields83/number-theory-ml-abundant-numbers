# Number Theory Features x Machine Learning MVP

自然数 `n` の数論的な特徴量から、`n` が deficient / perfect / abundant のどれに分類されるかを予測するための研究 MVP です。まず小規模で再現可能に動かし、その後 `N=50,000,000` まで拡張できる設計にしています。

## Research Background

約数和を `sigma(n)` とすると、分類は次のように定義できます。

- `sigma(n) < 2n`: deficient
- `sigma(n) = 2n`: perfect
- `sigma(n) > 2n`: abundant

この MVP では、`sigma(n) / n` を特徴量に含める条件と除外する条件を比較します。`sigma_ratio` は分類規則そのものに非常に近いため高精度になりやすい一方、除外条件では「素因数構造だけで abundant 性をどこまで予測できるか」を見る研究上の意味があります。

## Project Structure

```text
README.md
requirements.txt
.gitignore
src/
  generate_dataset.py
  train_models.py
  analyze_distribution.py
  number_theory_features.py
  utils.py
data/
  chunks/
figures/
results/
notebooks/
```

## Features

生成される特徴量は次の通りです。

- `n`
- `log_n`
- `sigma_n`
- `sigma_ratio = sigma(n) / n`
- `tau_n`
- `omega_n`
- `Omega_n`
- `min_prime_factor`
- `max_prime_factor`
- `label`

`omega_n` は異なる素因数の個数、`Omega_n` は重複込みの素因数の個数です。

## Setup

```bash
pip install -r requirements.txt
```

SymPy の `factorint` / `divisor_sigma` / `divisor_count` を `n` ごとに呼び出す実装は禁止しています。このプロジェクトでは SPF sieve と numba による範囲因数分解を使います。

## Generate Data

小規模確認:

```bash
python src/generate_dataset.py --max-n 100000 --chunk-size 100000
```

100万件:

```bash
python src/generate_dataset.py --max-n 1000000 --chunk-size 1000000
```

5000万件:

```bash
python src/generate_dataset.py --max-n 50000000 --chunk-size 1000000
```

出力は `data/chunks/features_00000001_01000000.parquet` のように chunk ごとの Parquet として保存します。全データを巨大な CSV 1つにまとめる設計にはしていません。

## Optional Omega Loader

`data/omega_values.parquet` または `data/omega_values.csv` が存在し、列が `n, omega_n` の形式なら、生成時に optional に読み込みます。存在しない場合は Python 側で `omega_n` を計算します。現時点では Maple の `.m` ファイルを直接読む必要はありません。

## Analyze Distribution

```bash
python src/analyze_distribution.py --data-dir data/chunks
```

主な出力:

- `results/distribution_summary.csv`
- `results/omega_abundant_rate.csv`
- `figures/class_distribution.png`
- `figures/sigma_ratio_histogram.png`
- `figures/omega_abundant_rate.png`

## Train Models

```bash
python src/train_models.py --data-dir data/chunks --sample-size 100000
```

全 chunk を一度に読み込まず、各 chunk からサンプルを抽出して学習データを作ります。perfect number は非常に少ないため、サンプル作成時に可能な限り保持します。

比較条件:

- `with_sigma_ratio`: `sigma_ratio` を特徴量に含める
- `without_sigma_ratio`: `sigma_ratio` を特徴量から除外する

分類タスク:

- multi-class: deficient / perfect / abundant
- binary: abundant / non-abundant

モデル:

- Logistic Regression
- Random Forest
- HistGradientBoostingClassifier

主な出力:

- `results/metrics.csv`
- `results/classification_report_*.txt`
- `figures/confusion_matrix_*.png`
- `figures/feature_importance_*.png`

## Memory and Runtime Notes

`N=50,000,000` では SPF 配列だけで約 190.7 MiB を使います。`chunk-size=1,000,000` の特徴量配列は概算で 47.7 MiB 程度ですが、pandas DataFrame と Parquet 書き込み時の一時領域も加わります。実行環境によっては数百 MiB から 1 GiB 以上の余裕を見てください。

小規模実行で numba の初回コンパイル時間が乗ります。2回目以降は速くなります。5000万件では、まず `N=1,000,000` または `N=5,000,000` で時間とメモリを測ってから拡張してください。

## Research Interpretation Notes

`sigma_ratio` は分類規則に近いため、これを入れたモデルの評価は「答えに近い特徴量を使った場合」として解釈してください。より研究的に重要なのは、`sigma_ratio` を除外した条件で、`tau_n`、`omega_n`、`Omega_n`、最小・最大素因数などの素因数構造が abundant 性をどの程度説明できるかです。

perfect number は極端に少ないため、multi-class の precision / recall / F1 は不安定になりやすいです。binary の abundant / non-abundant も併せて確認してください。

5000万件すべてでモデル学習する必要はありません。全量は分布分析に使い、機械学習は chunk からの抽出サンプルで十分に研究できます。

## Suggested Verification

```bash
python src/generate_dataset.py --max-n 100000 --chunk-size 100000
python src/analyze_distribution.py --data-dir data/chunks
python src/train_models.py --data-dir data/chunks --sample-size 50000
```

必要に応じて追加確認:

```bash
python src/generate_dataset.py --max-n 1000000 --chunk-size 1000000
python src/train_models.py --data-dir data/chunks --sample-size 100000
```
