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
