# Number Theory Features x Machine Learning

自然数 `n` の数論的特徴量から、`n` が abundant / non-abundant かを予測する研究用プロジェクトです。最初の MVP では `sigma(n) / n` や `omega(n)` と分類の関係を確認しましたが、本リポジトリではそこから一歩進めて、単純な数論的規則だけでは説明しきれない予測問題を設定します。

重要な方針:

- `N=50,000,000` は実行可能な設計にしていますが、通常の実験確認ではまだ実行しません。
- 大容量の chunk データは `data/chunks/` に保存し、Git には含めません。
- `sigma_ratio` は分類定義に近すぎるため、研究実験では原則としてモデル入力から除外します。

## Research Question

単に `omega(n)` が大きいほど abundant になりやすい、という確認だけでは研究として弱いです。これは数論的にも機械学習的にも比較的予想しやすい結果です。

そのため、このプロジェクトでは次のより難しい問いを扱います。

1. 単純な rule baseline と機械学習モデルの差はどれくらいあるか。
2. `omega(n)` を固定した条件でも abundant / non-abundant を予測できるか。
3. `sigma_ratio` や `tau_n` のような強い特徴量を除いても、素因数構造だけで予測できるか。
4. `sigma_ratio` が 2 に近い境界サンプルだけでも分類できるか。
5. 小さい範囲で学習したモデルは、より大きい範囲に一般化できるか。

## Definitions

約数和を `sigma(n)` とすると、分類は次の通りです。

- `sigma(n) < 2n`: deficient
- `sigma(n) = 2n`: perfect
- `sigma(n) > 2n`: abundant

機械学習実験では、perfect number は非常に少ないため、主に `abundant` と `non_abundant` の二値分類として扱います。

## Project Structure

```text
README.md
requirements.txt
.gitignore
src/
  generate_dataset.py
  analyze_distribution.py
  train_models.py
  rule_baselines.py
  feature_ablation.py
  train_fixed_omega.py
  train_boundary_cases.py
  train_cross_range.py
  number_theory_features.py
  utils.py
data/
  chunks/
figures/
results/
notebooks/
```

## Features

生成される特徴量:

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

SymPy の `factorint` / `divisor_sigma` / `divisor_count` を `n` ごとに呼ぶ実装は避けています。SPF sieve と numba による範囲因数分解を使い、`N=50,000,000` まで拡張しやすい構成にしています。

## Generate Data

100,000 件:

```bash
python src/generate_dataset.py --max-n 100000 --chunk-size 100000
python src/analyze_distribution.py --data-dir data/chunks
```

1,000,000 件:

```bash
python src/generate_dataset.py --max-n 1000000 --chunk-size 1000000
```

50,000,000 件用の想定コマンド:

```bash
python src/generate_dataset.py --max-n 50000000 --chunk-size 1000000
```

50,000,000 件は時間とメモリを使うため、まず 1,000,000 または 5,000,000 で確認してから実行してください。

## MVP Result Summary

100,000 件では次の分布でした。

| label | count | share |
|---|---:|---:|
| deficient | 75,201 | 0.75201 |
| perfect | 4 | 0.00004 |
| abundant | 24,795 | 0.24795 |

最初の分布分析では、`omega_n` が大きいほど abundant になりやすい傾向が確認できました。ただし、それだけでは「素因数が多い数は約数も多くなりやすい」という自然な観察に近く、研究としてはやや自明です。

## Additional Experiments

### 1. Rule Baselines

`src/rule_baselines.py` は単純な規則を評価します。

- `omega_n >= 4`
- `omega_n >= 5`
- `tau_n >= 12`
- `tau_n >= 24`
- `omega_n` と `log_n` の単純な組み合わせ

目的は、機械学習モデルが「omega が大きいなら abundant」という素朴な規則より本当に優れているかを確認することです。

100,000 件での主な結果:

| rule | accuracy | F1 abundant |
|---|---:|---:|
| tau_ge_12 | 0.8286 | 0.7273 |
| tau_ge_24 | 0.8696 | 0.6650 |
| omega_ge_4 | 0.8222 | 0.5816 |
| omega_ge_5 | 0.7695 | 0.1345 |

1,000,000 件では `tau_ge_24` が accuracy 0.8729 / F1 0.7090 で最良でした。単純 rule は一定の説明力を持ちますが、後述の ML 実験とは大きな差があります。

### 2. Feature Ablation

`src/feature_ablation.py` は `sigma_ratio` を除いたうえで、特徴量セットを比較します。

| set | features |
|---|---|
| full_without_sigma_ratio | `log_n`, `tau_n`, `omega_n`, `Omega_n`, `min_prime_factor`, `max_prime_factor` |
| no_sigma_no_tau | `log_n`, `omega_n`, `Omega_n`, `min_prime_factor`, `max_prime_factor` |
| prime_structure_only | `omega_n`, `Omega_n`, `min_prime_factor`, `max_prime_factor` |
| size_only | `log_n` |
| omega_only | `omega_n` |

100,000 件サンプルでの結果:

| feature set | accuracy | F1 abundant |
|---|---:|---:|
| full_without_sigma_ratio | 0.9838 | 0.9671 |
| no_sigma_no_tau | 0.9814 | 0.9623 |
| prime_structure_only | 0.9451 | 0.8854 |
| omega_only | 0.8276 | 0.5988 |
| size_only | 0.7507 | 0.0000 |

`tau_n` を除いても性能がほとんど落ちないため、モデルは単に約数個数だけを見ているわけではありません。`prime_structure_only` でも高い性能が残る点は、素因数構造に十分な予測情報があることを示しています。

1,000,000 件サンプルでも同じ傾向で、`no_sigma_no_tau` は accuracy 0.9742 / F1 0.9475 でした。

### 3. Fixed Omega Experiment

`src/train_fixed_omega.py` は `omega_n = 2, 3, 4, 5` の各グループ内だけで二値分類します。`omega_n` 自体を固定することで、「omega が大きいから abundant」という説明を取り除きます。

100,000 件での結果:

| omega_n | rows | accuracy | F1 abundant |
|---:|---:|---:|---:|
| 2 | 33,759 | 0.9998 | 0.9910 |
| 3 | 38,844 | 0.9936 | 0.9897 |
| 4 | 15,855 | 0.9198 | 0.9383 |
| 5 | 1,816 | 0.9912 | 0.9955 |

特に `omega_n = 4` は難しくなります。同じ omega の中に abundant と non-abundant が混在しており、単純な omega ルールでは説明できないためです。

1,000,000 件でも `omega_n = 4` は accuracy 0.9021 / F1 0.9088 と相対的に難しいままでした。

### 4. Boundary Case Experiment

`src/train_boundary_cases.py` は `sigma_ratio` が 2 に近い整数だけを取り出し、その中で分類します。`sigma_ratio` はサンプル抽出にのみ使い、モデル入力には入れません。

100,000 件での結果:

| sigma_ratio window | rows | accuracy | F1 abundant |
|---|---:|---:|---:|
| 1.80-2.20 | 19,036 | 0.9239 | 0.9199 |
| 1.90-2.10 | 9,278 | 0.9328 | 0.9420 |
| 1.95-2.05 | 5,541 | 0.9639 | 0.9739 |

境界サンプルだけでも、素因数構造から高い分類性能が残りました。ただし、窓を狭くするとサンプルの分布が変わるため、単純に「狭いほど難しい」とは限りません。ここは今後、サンプル数や class balance を制御してさらに検証する価値があります。

1,000,000 件でも `1.80-2.20` は accuracy 0.8912 / F1 0.8894、`1.95-2.05` は accuracy 0.9537 / F1 0.9658 でした。

### 5. Cross-Range Generalization

`src/train_cross_range.py` は `1..100000` で学習し、`100001..1000000` でテストします。範囲内分布を覚えているだけなのか、より大きい範囲にも一般化するのかを見るための実験です。

1,000,000 件データでの結果:

| model | test rows | accuracy | F1 abundant |
|---|---:|---:|---:|
| HistGradientBoosting | 200,000 | 0.9065 | 0.8389 |
| RandomForest | 200,000 | 0.9033 | 0.8348 |

同一範囲内の feature ablation と比べると性能は落ちます。これは、モデルが数論的構造をある程度学んでいる一方で、範囲外汎化はまだ難しいことを示します。

## Commands Used

100,000 件の必須実験:

```bash
python src/rule_baselines.py --data-dir data/chunks
python src/feature_ablation.py --data-dir data/chunks --sample-size 50000
python src/train_fixed_omega.py --data-dir data/chunks
python src/train_boundary_cases.py --data-dir data/chunks
```

1,000,000 件の追加確認:

```bash
python src/train_cross_range.py --data-dir work/chunks_1m
python src/rule_baselines.py --data-dir work/chunks_1m --results-dir work/research_1m/results --figures-dir work/research_1m/figures
python src/feature_ablation.py --data-dir work/chunks_1m --sample-size 100000 --results-dir work/research_1m/results --figures-dir work/research_1m/figures
python src/train_fixed_omega.py --data-dir work/chunks_1m --results-dir work/research_1m/results --figures-dir work/research_1m/figures
python src/train_boundary_cases.py --data-dir work/chunks_1m --results-dir work/research_1m/results --figures-dir work/research_1m/figures
```

## Outputs

主な結果ファイル:

- `results/rule_baselines.csv`
- `results/feature_ablation_metrics.csv`
- `results/fixed_omega_metrics.csv`
- `results/boundary_case_metrics.csv`
- `results/cross_range_metrics.csv`

主な図:

- `figures/rule_baseline_comparison.png`
- `figures/feature_ablation_accuracy.png`
- `figures/feature_ablation_f1.png`
- `figures/fixed_omega_accuracy.png`
- `figures/boundary_case_accuracy.png`
- `figures/cross_range_comparison.png`
- `figures/confusion_matrix_fixed_omega_*.png`
- `figures/confusion_matrix_boundary_*.png`

## Interpretation

追加実験から見えること:

- 単純な `omega_n` rule は説明力を持つが、ML モデルには大きく劣る。
- `omega_n` を固定しても分類できるため、モデルは omega の大小だけを使っていない。
- `tau_n` を除いても高性能が残るため、素因数構造そのものが強い情報を持つ。
- 境界サンプルでも分類性能が残るが、サンプル窓の取り方によって class balance が変わる点に注意が必要。
- cross-range では性能が落ちるため、範囲外汎化は今後の重要な検証対象である。

## Limitations

- perfect number は極端に少ないため、multi-class 評価は安定しにくい。
- boundary case は `sigma_ratio` で抽出しているため、窓ごとの分布差をさらに制御する必要がある。
- 1,000,000 件実験は十分に軽量だが、50,000,000 件での全量実験はまだ未実行。
- モデルは予測に有用な構造を捉えているが、数論的な定理や因果的説明を直接与えるものではない。

## Future Work

- `omega_n = 4` に絞った詳細分析。
- 境界サンプルで class balance を揃えた再実験。
- prime exponent pattern の明示的特徴量追加。
- `min_prime_factor` / `max_prime_factor` の比や分布特徴量の追加。
- cross-range を `1e5 -> 1e6`, `1e6 -> 5e6` のように段階化。
- 50,000,000 件で分布分析を行い、ML は抽出サンプルで検証。

## Large-scale experiments

The project is designed to scale in stages on a local Windows machine. Large raw feature chunks are written under `data/chunks/` as Parquet files and are intentionally not committed to GitHub. Sampled training files under `data/samples/` are also ignored. This keeps the repository small while allowing local experiments to grow from 1,000,000 to 10,000,000 and eventually 50,000,000 integers.

The large-scale workflow uses chunk scanning and sampling because loading all rows into memory is not practical at 10M or 50M. Distribution analysis reads one chunk at a time. Machine-learning experiments train on lower ranges and test on sampled higher ranges, while still avoiding `sigma_ratio` as an input feature.

Recommended step 1: verify 1,000,000 rows.

```powershell
python src/generate_dataset.py --max-n 1000000 --chunk-size 1000000
python src/analyze_large_distribution.py --data-dir data/chunks
python src/train_large_cross_range.py --data-dir data/chunks --train-max-n 100000 --test-ranges "100001:1000000" --sample-per-range 200000
```

Recommended step 2: extend to 10,000,000 rows.

```powershell
python src/generate_dataset.py --max-n 10000000 --chunk-size 1000000
python src/analyze_large_distribution.py --data-dir data/chunks
python src/sample_large_dataset.py --data-dir data/chunks --output-path data/samples/sample_10m.parquet --sample-size 500000 --random-state 42
python src/train_large_cross_range.py --data-dir data/chunks --train-max-n 100000 --test-ranges "100001:1000000,1000001:10000000" --sample-per-range 200000
```

Optional advanced step: extend to 50,000,000 rows only when disk space and runtime are acceptable.

```powershell
python src/generate_dataset.py --max-n 50000000 --chunk-size 1000000
python src/analyze_large_distribution.py --data-dir data/chunks
```

Do not run the 50M command casually. The generator prints estimated SPF memory, per-chunk working memory, and rough Parquet disk usage before generation. It also skips chunks that already exist or are already covered by existing chunk ranges, so staged runs can continue without overwriting previous data.

Large-scale outputs:

- `results/large_distribution_summary.csv`
- `results/large_omega_abundant_rate.csv`
- `results/large_Omega_abundant_rate.csv`
- `results/large_range_summary.csv`
- `results/large_cross_range_metrics.csv`
- `figures/large_omega_abundant_rate.png`
- `figures/large_range_abundant_share.png`
- `figures/large_cross_range_f1.png`
