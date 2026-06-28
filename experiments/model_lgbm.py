"""
Variant 4 — LightGBM model on the full (micro + liq) feature set.

Compares, fairly out-of-sample (train fit -> Feb val), three scorers:
  1. linear-IC factor (micro+liq)        — the variant-3 baseline
  2. LightGBM regressor (Huber, weighted) — unconstrained
  3. LightGBM + monotone constraints      — sign = sign(train IC) per feature

A model should combine the correlated flow+liq signals better than the naive
IC-weighted sum (which double-counts them).

Causality / leakage handling is inherited from feature_eval (features use ts<t_i,
markout at t_i+tau). Early stopping uses a TIME-ORDERED holdout inside train
(last 20% by timestamp) so no future leaks into model selection.
"""

from __future__ import annotations

import argparse

import duckdb
import numpy as np
import lightgbm as lgb

from feature_eval import (
    build_scored, FEATURES, FEATURES_MICRO, TAUS, US, to_us,
    fit_factor, factor_score, keep_curve, weighted_ic,
)

FEAT_COLS = [n for n, _, _ in FEATURES]


def time_mask(df, frac=0.2):
    """True = earlier rows (model-train); False = latest `frac` (early-stop eval)."""
    ts = df["ts"].to_numpy()
    cut = np.quantile(ts, 1.0 - frac)
    return ts < cut


def monotone_from_ic(train, tau, thresh=0.004):
    cons = []
    w = train["w"].to_numpy()
    pnl = train[f"pnl_{tau}"].to_numpy()
    for name in FEAT_COLS:
        ic = weighted_ic(train[name].to_numpy().astype(float), pnl, w)
        cons.append(int(np.sign(ic)) if np.isfinite(ic) and abs(ic) >= thresh else 0)
    return cons


def train_lgbm(train, tau, monotone=None, seed=42):
    m = np.isfinite(train[f"pnl_{tau}"].to_numpy())
    d = train.loc[m]
    X = d[FEAT_COLS].astype(float).fillna(0.0)
    y = d[f"pnl_{tau}"].to_numpy()
    w = d["w"].to_numpy()
    tr = time_mask(d, 0.2)
    params = dict(
        objective="huber", n_estimators=800, learning_rate=0.03,
        num_leaves=31, min_child_samples=500, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, reg_lambda=5.0, max_depth=-1,
        n_jobs=4, random_state=seed, verbose=-1,
    )
    if monotone is not None:
        params["monotone_constraints"] = monotone
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X[tr], y[tr], sample_weight=w[tr],
        eval_set=[(X[~tr], y[~tr])], eval_sample_weight=[w[~tr]],
        eval_metric="l2", callbacks=[lgb.early_stopping(60, verbose=False)],
    )
    return model


def score_at(df, score, tau, num_days, keeps=(0.2, 0.61)):
    frac, sc, pk, turn, pnl_all = keep_curve(df, score, tau, num_days)
    out = {}
    for kr in keeps:
        i = min(int(np.searchsorted(frac, kr)), len(frac) - 1)
        out[kr] = sc[i]
    return out, pnl_all


def run_symbol(con, sym, tr, va, n_sample, seed):
    print(f"\n{'='*78}\n{sym.upper()}  LightGBM vs linear factor  (train fit -> Feb val)\n{'='*78}")
    t0, t1 = tr
    train = build_scored(con, sym, t0, t1, n_sample, seed)
    v0, v1 = va
    va_days = (v1 - v0) / (US * 86400)
    val = build_scored(con, sym, v0, v1, n_sample, seed)

    # baseline: linear-IC factor on micro+liq
    zp, wt = fit_factor(train, FEATURES)

    print(f"\n{'tau':>4} {'model':<16} {'Score@61%':>10} {'Score@20%':>10}   (PnL_all val)")
    for tau in TAUS:
        lin = factor_score(val, zp, wt[tau], FEATURES)
        m_unc = train_lgbm(train, tau, monotone=None, seed=seed)
        m_mon = train_lgbm(train, tau, monotone=monotone_from_ic(train, tau), seed=seed)
        Xval = val[FEAT_COLS].astype(float).fillna(0.0)
        scorers = {
            "linear micro+liq": lin,
            "lgbm": m_unc.predict(Xval),
            "lgbm+monotone": m_mon.predict(Xval),
        }
        first = True
        for nm, sc in scorers.items():
            res, pnl_all = score_at(val, sc, tau, va_days)
            tag = f"  (PnL_all={pnl_all:+.3f})" if first else ""
            print(f"{tau:>4} {nm:<16} {res[0.61]:>+10.4f} {res[0.2]:>+10.4f}{tag}")
            first = False
        print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--syms", default="btc,eth")
    ap.add_argument("--train-start", default="2025-12-01")
    ap.add_argument("--train-end", default="2025-12-22")
    ap.add_argument("--val-start", default="2026-02-01")
    ap.add_argument("--val-end", default="2026-02-15")
    ap.add_argument("--sample", type=int, default=1_000_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mem", default="10GB")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    tr = (to_us(args.train_start), to_us(args.train_end))
    va = (to_us(args.val_start), to_us(args.val_end))
    con = duckdb.connect()
    con.execute(f"PRAGMA memory_limit='{args.mem}'")
    con.execute(f"PRAGMA threads={args.threads}")
    for sym in args.syms.split(","):
        run_symbol(con, sym, tr, va, args.sample, args.seed)


if __name__ == "__main__":
    main()
