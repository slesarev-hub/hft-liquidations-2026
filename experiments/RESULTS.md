# Feature experiments — results (exp 1 & 2)

Harness: `experiments/feature_eval.py` (DuckDB + numpy, no ML deps).
Flow = prefix-sum + searchsorted (O(n)); BBO/markout = ASOF at sampled points only.
Strictly causal (features use ts < t_i). Train = Dec 1–22 (3w), Val = Feb 1–15 (2w),
1M sampled trades each. Benchmark to beat: BTC +0.445 (61.5% kept), ETH +1.085 (64.4% kept).

## Exp 1 — weighted Spearman markout-IC (train)

| feature | BTC IC30 | BTC IC120 | ETH IC30 | ETH IC120 | note |
|---|---|---|---|---|---|
| flow_same_30s (OFI) | +0.056 | +0.044 | **+0.079** | **+0.064** | strongest; missing from benchmark |
| flow_same_5s | +0.042 | +0.031 | +0.072 | +0.052 | |
| flow_imb_30s | +0.044 | +0.040 | +0.065 | +0.053 | |
| mid_vel_5s_same | +0.041 | +0.031 | +0.069 | +0.045 | reversion |
| mp_dev_same | -0.024 | -0.012 | -0.007 | -0.006 | adverse-selection (BTC only) |
| imb_same | -0.027 | -0.012 | -0.012 | -0.009 | adverse-selection (BTC only) |
| depth_ratio | -0.013 | -0.006 | -0.005 | +0.002 | weak / FLIP — drop |
| spread_bps | +0.006 | +0.002 | +0.009 | +0.011 | weak |

Key findings:
- **Order-flow (OFI) is the dominant signal on both symbols** — and the benchmark had no flow features at all.
- Sign is **positive**: same-side taker flow predicts **reversion** (maker earns on uninformed flow), not continuation.
- Instantaneous book lean (mp_dev/imb) is **negative** = adverse selection, but only material on BTC.
- depth_ratio / spread are weak/unstable → drop.

## Exp 2 — Score(keep-rate), fair train→val (Feb, out-of-sample)

Linear-IC factor fit on TRAIN, applied to VAL. Score = PnL_kept − PnL_all (bps).

| sym | tau | keep 61% (bench op) | keep 40% | keep 20% | turnover @20% |
|---|---|---|---|---|---|
| BTC | 120 | +0.320 | +0.491 | +1.108 | $28.9M/d |
| BTC | 300 | +0.428 | +0.595 | +0.790 | $28.9M/d |
| ETH | 120 | +0.992 | +1.784 | **+4.815** | $15.1M/d |
| ETH | 300 | +0.975 | +1.862 | +4.778 | $15.1M/d |

Key findings:
- At the benchmark's 61% keep, the simple factor already lands near benchmark
  (BTC +0.43 vs +0.445; ETH +0.99 vs +1.085) — with no liq features and no ML.
- **Filtering more aggressively ~3–5× the Score, and it holds out-of-sample** (val ≥ train at matched keep).
- Turnover at keep=20% is $15–29M/day = **30–58× the $500k/day floor** → the constraint is nowhere near binding.

Caveats: val (Feb) is one regime; absolute keep=20% numbers (esp. ETH +4.8) are regime-favorable and
not a hidden-test guarantee. The robust, reproducible conclusions are the two **bolded** ones above.

## Variant 3 — add liquidation-cascade features to the factor

New causal liq features (EWMA per venue, direction-relative; bybit +200ms; x-venue):
`liq_bin_same_{5,30}s`, `liq_byb_same_30s`, `liq_all_same_30s`, `liq_cnt_30s`.

IC: liq is strong on ETH (`liq_bin_same_5s` IC30 +0.074 ≈ flow) and moderate on BTC (+0.052);
sign is **positive** like flow (reversion). Strongest at long horizon (τ=300).

Score(keep-rate) on VAL, micro vs micro+liq:

| sym | τ | micro @61% | +liq @61% | micro @20% | +liq @20% |
|---|---|---|---|---|---|
| BTC | 120 | +0.319 | +0.386 | +1.118 | +1.481 |
| BTC | 300 | +0.428 | +0.486 | +0.799 | +1.080 |
| ETH | 120 | +0.993 | +1.050 | +4.82 | +5.15 |
| ETH | 300 | +0.977 | +1.069 | +4.78 | +4.96 |
| ETH | 30 | +0.945 | +0.876 | +3.74 | +3.81 |

Conclusion: liq **adds out-of-sample**, best at τ=120/300; gain is modest because flow and liq are
**correlated** (same cascades) and the naive IC-weighted sum double-counts them. A proper model
(variant 4, LightGBM with decorrelation) should extract liq's non-redundant part better.

## Variant 4 — LightGBM on (micro + liq), train fit -> Feb val

3 scorers compared OOS: linear-IC factor (v3 baseline), LGBM (Huber, weighted),
LGBM + monotone constraints (sign = sign(train IC)). Early stop on a time-ordered
inner holdout (last 20% of train). Score(keep-rate):

| sym | τ | linear @61% / @20% | lgbm @61% / @20% | lgbm+mono @61% / @20% |
|---|---|---|---|---|
| BTC | 30  | +0.296 / +0.950 | +0.313 / +0.989 | +0.270 / **+1.151** |
| BTC | 120 | +0.387 / +1.489 | −0.072 / +0.059 | **+0.579 / +1.746** |
| BTC | 300 | +0.501 / +1.045 | +0.519 / +0.001 | +0.493 / **+1.546** |
| ETH | 30  | **+0.856 / +3.816** | +0.562 / +2.444 | +0.718 / +3.099 |
| ETH | 120 | **+1.050 / +5.145** | +0.758 / +2.339 | +0.738 / +2.827 |
| ETH | 300 | +1.065 / +5.178 | +0.365 / +2.359 | **+1.479** / +4.321 |

Conclusions:
- **Unconstrained LGBM overfits the noisy markout target** → worst on both symbols
  (BTC τ=120 goes negative; τ=300@20% collapses to ~0). Do not use it here.
- **Monotone constraints are essential** and make LGBM the best on **BTC** (weaker, more
  nonlinear signal): τ=120@20% +1.75 vs +1.49, τ=300@20% +1.55 vs +1.05.
- On **ETH** the signal is strong and near-linear (flow reversion) → the **simple linear
  factor wins** at τ=30/120; monotone GBM only edges it at τ=300@61%.
- Practical: **per-symbol model** — linear factor for ETH, monotone GBM for BTC; never
  unconstrained GBM. Next: vol-normalization (v5) for regime transfer to the hidden test.
