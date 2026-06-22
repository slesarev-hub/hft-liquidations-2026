"""
Feature evaluation harness for the liquidation maker-filter task.

Experiment 1: weighted markout-IC of new microstructure features
              (signed taker flow / OFI, direction-relative microprice deviation,
               trade-size-vs-depth) with sub-period stability.
Experiment 2: linear factor raw_score -> Score(keep-rate) curve on TRAIN.

All features are strictly causal (use only data with timestamp < t_i).
Markout uses forward-fill mid at t_i + tau (timestamp <= t_i + tau).

Flow features: prefix-sum over the full trade stream + searchsorted (O(n)),
NOT a SQL RANGE window (which is O(n * window_rows) and far too slow here).
BBO / markout: DuckDB ASOF joins evaluated only at the sampled query points.
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent / "liquidation_task" / "data"
TRD = {"btc": str(BASE / "binance_trades/perp_btcusdt.parquet"),
       "eth": str(BASE / "binance_trades/perp_ethusdt.parquet")}
BBO = {"btc": str(BASE / "binance_booktickers/perp_btcusdt.parquet"),
       "eth": str(BASE / "binance_booktickers/perp_ethusdt.parquet")}

TAUS = (30, 120, 300)
US = 1_000_000

# Candidate features. prior = expected sign of corr(feature, pnl).
# "same-side" pressure features are expected NEGATIVELY related to maker pnl
# (more continuation in the taker direction => worse fill).
FEATURES = [
    ("flow_same_30s",   "signed taker flow same-side, 30s (OFI)",       -1),
    ("flow_same_5s",    "signed taker flow same-side, 5s (OFI)",        -1),
    ("flow_imb_30s",    "taker flow imbalance same-side, 30s",          -1),
    ("flow_imb_5s",     "taker flow imbalance same-side, 5s",           -1),
    ("mp_dev_same",     "microprice deviation same-side (bps)",         -1),
    ("imb_same",        "book imbalance same-side",                     -1),
    ("depth_ratio",     "trade amount / same-side top depth",           -1),
    ("spread_bps",      "BBO spread (bps), direction-agnostic",         -1),
    ("mid_vel_5s_same", "mid velocity 5s same-side (bps) [reference]",  -1),
]


def to_us(date_str: str) -> int:
    d = dt.datetime.fromisoformat(date_str).replace(tzinfo=dt.timezone.utc)
    return int(d.timestamp() * US)


def w_mean(x, w):
    m = np.isfinite(x) & np.isfinite(w)
    return np.sum(x[m] * w[m]) / np.sum(w[m]) if m.any() else np.nan


def weighted_ic(feat, pnl, w, kind="spearman"):
    m = np.isfinite(feat) & np.isfinite(pnl) & np.isfinite(w)
    f, p, ww = feat[m], pnl[m], w[m]
    if len(f) < 100:
        return np.nan
    if kind == "spearman":
        f = pd.Series(f).rank().to_numpy()
        p = pd.Series(p).rank().to_numpy()
    fm, pm = w_mean(f, ww), w_mean(p, ww)
    cov = np.sum(ww * (f - fm) * (p - pm))
    vf = np.sum(ww * (f - fm) ** 2)
    vp = np.sum(ww * (p - pm) ** 2)
    return cov / np.sqrt(vf * vp) if vf > 0 and vp > 0 else np.nan


def build_sample(con, sym, t0, t1, n_sample, seed):
    """Load full trade stream, compute causal flow via prefix-sum+searchsorted,
    then draw a random sample of query trades with computed flow features."""
    a = con.execute(f"""
        SELECT timestamp AS ts,
               CASE WHEN side='buy' THEN 1 ELSE -1 END AS s,
               price, amount
        FROM read_parquet('{TRD[sym]}')
        WHERE timestamp BETWEEN {t0} AND {t1}
        ORDER BY timestamp
    """).fetchnumpy()
    ts = np.asarray(a["ts"], dtype=np.int64)
    s = np.asarray(a["s"], dtype=np.int64)
    price = np.asarray(a["price"], dtype=np.float64)
    amount = np.asarray(a["amount"], dtype=np.float64)
    notional = price * amount
    w = np.minimum(notional, 100_000.0)

    # prefix sums (inclusive); flow over [t-W, t) = cum[hi-1] - cum[lo-1]
    cum_sw = np.concatenate([[0.0], np.cumsum(s * w)])   # cum_sw[i] = sum of first i rows
    cum_w = np.concatenate([[0.0], np.cumsum(w)])

    rng = np.random.default_rng(seed)
    start = int(np.searchsorted(ts, t0 + 30 * US, side="left"))  # warm-up for 30s flow
    pool = np.arange(start, len(ts))
    take = min(n_sample, len(pool))
    idx = np.sort(rng.choice(pool, size=take, replace=False))

    tq = ts[idx]
    hi = np.searchsorted(ts, tq, side="left")            # rows strictly before t
    lo30 = np.searchsorted(ts, tq - 30 * US, side="left")
    lo5 = np.searchsorted(ts, tq - 5 * US, side="left")

    sq = s[idx].astype(float)
    sf30 = cum_sw[hi] - cum_sw[lo30]
    tf30 = cum_w[hi] - cum_w[lo30]
    sf5 = cum_sw[hi] - cum_sw[lo5]
    tf5 = cum_w[hi] - cum_w[lo5]

    df = pd.DataFrame({
        "k": np.arange(take, dtype=np.int64),
        "ts": tq, "s": sq, "price": price[idx], "amount": amount[idx],
        "notional": notional[idx], "w": w[idx],
        "flow_same_30s": sq * sf30,
        "flow_same_5s": sq * sf5,
        "flow_imb_30s": sq * np.where(tf30 > 0, sf30 / tf30, np.nan),
        "flow_imb_5s": sq * np.where(tf5 > 0, sf5 / tf5, np.nan),
    })
    return df


def add_book_and_markout(con, sym, t0, t1, df):
    """ASOF-join BBO state at t (strict <), t-5s, and t+tau, only for sampled rows."""
    max_tau_us = max(TAUS) * US
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE bbo_slice AS
        SELECT timestamp, bid_price, bid_amount, ask_price, ask_amount
        FROM read_parquet('{BBO[sym]}')
        WHERE timestamp BETWEEN {t0 - 5*US} AND {t1 + max_tau_us}
    """)
    con.register("samp", df[["k", "ts"]])
    tau_joins = "\n        ".join(
        f"ASOF LEFT JOIN bbo_slice b{tau} ON samp.ts + {tau*US} >= b{tau}.timestamp"
        for tau in TAUS)
    tau_cols = ",\n           ".join(
        f"(b{tau}.bid_price + b{tau}.ask_price)/2.0 AS mid_{tau}" for tau in TAUS)
    out = con.execute(f"""
        SELECT samp.k,
               b0.bid_price, b0.bid_amount, b0.ask_price, b0.ask_amount,
               (bm5.bid_price + bm5.ask_price)/2.0 AS mid_m5,
               {tau_cols}
        FROM samp
        ASOF LEFT JOIN bbo_slice b0  ON samp.ts          >  b0.timestamp
        ASOF LEFT JOIN bbo_slice bm5 ON samp.ts - {5*US} >  bm5.timestamp
        {tau_joins}
        ORDER BY samp.k
    """).df()
    con.unregister("samp")
    m = df.merge(out, on="k", how="left")

    s = m["s"].to_numpy(float)
    bid, ask = m["bid_price"].to_numpy(float), m["ask_price"].to_numpy(float)
    bida, aska = m["bid_amount"].to_numpy(float), m["ask_amount"].to_numpy(float)
    mid0 = (bid + ask) / 2.0
    micro = (bid * aska + ask * bida) / (bida + aska)
    imb = (bida - aska) / (bida + aska)
    same_depth = np.where(s > 0, aska, bida)
    with np.errstate(divide="ignore", invalid="ignore"):
        m["mp_dev_same"] = s * (micro - mid0) / mid0 * 1e4
        m["imb_same"] = s * imb
        m["depth_ratio"] = m["amount"].to_numpy(float) / same_depth
        m["spread_bps"] = (ask - bid) / mid0 * 1e4
        m["mid_vel_5s_same"] = s * (mid0 - m["mid_m5"].to_numpy(float)) / mid0 * 1e4
        for tau in TAUS:
            mid_tau = m[f"mid_{tau}"].to_numpy(float)
            pnl = -s * (mid_tau - m["price"].to_numpy(float)) / m["price"].to_numpy(float) * 1e4 + 0.5
            pnl[~np.isfinite(mid_tau)] = np.nan        # edge trades excluded
            m[f"pnl_{tau}"] = pnl
    return m


def build_scored(con, sym, t0, t1, n_sample, seed):
    df = build_sample(con, sym, t0, t1, n_sample, seed)
    df = add_book_and_markout(con, sym, t0, t1, df)
    return df


def ic_table(df, t0, t1):
    """Exp1: weighted Spearman markout-IC with sub-period stability."""
    mid_ts = (t0 + t1) // 2
    half1 = df["ts"].to_numpy() < mid_ts
    rows = []
    for name, desc, prior in FEATURES:
        row = {"feature": name}
        for tau in TAUS:
            pnl = df[f"pnl_{tau}"].to_numpy(); w = df["w"].to_numpy()
            f = df[name].to_numpy().astype(float)
            row[f"IC{tau}"] = weighted_ic(f, pnl, w)
            i1 = weighted_ic(f[half1], pnl[half1], w[half1])
            i2 = weighted_ic(f[~half1], pnl[~half1], w[~half1])
            row[f"st{tau}"] = "ok" if np.isfinite(i1) and np.isfinite(i2) and np.sign(i1) == np.sign(i2) else "FLIP"
        rows.append(row)
    return pd.DataFrame(rows)


def fit_factor(df):
    """Per-feature z-norm params (mu,sd) and per-tau IC weights, fit on TRAIN."""
    zp = {}
    for name, _, _ in FEATURES:
        f = df[name].to_numpy().astype(float)
        zp[name] = (np.nanmean(f), np.nanstd(f))
    wt = {}
    for tau in TAUS:
        pnl = df[f"pnl_{tau}"].to_numpy(); w = df["w"].to_numpy()
        wt[tau] = {name: weighted_ic(df[name].to_numpy().astype(float), pnl, w)
                   for name, _, _ in FEATURES}
    return zp, wt


def factor_score(df, zp, weights_tau):
    score = np.zeros(len(df))
    for name, _, _ in FEATURES:
        mu, sd = zp[name]
        f = df[name].to_numpy().astype(float)
        z = np.nan_to_num((f - mu) / sd if sd > 0 else f * 0.0, nan=0.0)
        ic_k = weights_tau[name]
        if np.isfinite(ic_k):
            score += ic_k * z
    return score


def keep_curve(df, score, tau, num_days):
    pnl = df[f"pnl_{tau}"].to_numpy(); w = df["w"].to_numpy()
    valid = np.isfinite(pnl) & np.isfinite(w)
    pnl_all = w_mean(pnl[valid], w[valid])
    order = np.argsort(-score); order = order[valid[order]]
    ws, ps = w[order], pnl[order]
    cw, cwp = np.cumsum(ws), np.cumsum(ws * ps)
    frac = cw / cw[-1]; pnl_kept = cwp / cw
    return frac, pnl_kept - pnl_all, pnl_kept, cw / num_days, pnl_all


def report_curve(label, df, score, num_days):
    for tau in TAUS:
        frac, sc, pk, turn, pnl_all = keep_curve(df, score, tau, num_days)
        print(f"  [{label}] tau={tau:>3}  PnL_all={pnl_all:+.4f}")
        for kr in (0.2, 0.4, 0.61, 0.8, 1.0):    # 0.61 = the benchmark operating point
            i = min(int(np.searchsorted(frac, kr)), len(frac) - 1)
            tag = " <-bench" if kr == 0.61 else ""
            print(f"      keep={kr*100:4.0f}%  Score={sc[i]:+.4f}  PnL_kept={pk[i]:+.4f}  "
                  f"turnover=${turn[i]/1e6:,.1f}M/d{tag}")


def run_symbol(con, sym, tr, va, n_sample, seed, out_dir):
    t0, t1 = tr
    tr_days = (t1 - t0) / (US * 86400)
    print(f"\n{'='*72}\n{sym.upper()}   sample={n_sample:,}   train_days={tr_days:.1f}\n{'='*72}")
    train = build_scored(con, sym, t0, t1, n_sample, seed)
    print(f"train rows={len(train):,}  valid pnl_30={np.isfinite(train['pnl_30']).sum():,}")

    ic = ic_table(train, t0, t1)
    pd.set_option("display.float_format", lambda v: f"{v:+.4f}")
    pd.set_option("display.width", 220)
    print("\n--- Exp1: weighted Spearman markout-IC on TRAIN (sign auto-handled by factor) ---")
    print(ic.to_string(index=False))
    ic.to_csv(out_dir / f"ic_{sym}.csv", index=False)

    zp, wt = fit_factor(train)
    print("\n--- Exp2: Score(keep-rate), factor fit on TRAIN ---")
    report_curve("TRAIN(in-sample)", train, factor_score(train, zp, wt[120]), tr_days)

    if va is not None:
        v0, v1 = va
        va_days = (v1 - v0) / (US * 86400)
        val = build_scored(con, sym, v0, v1, n_sample, seed)
        print(f"\nval rows={len(val):,}  valid pnl_30={np.isfinite(val['pnl_30']).sum():,}")
        print("--- Exp2 FAIR: train-fit factor applied to VALIDATION (Feb) ---")
        report_curve("VAL(out-of-sample)", val, factor_score(val, zp, wt[120]), va_days)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--syms", default="btc,eth")
    ap.add_argument("--train-start", default="2025-12-01")
    ap.add_argument("--train-end", default="2025-12-22")    # 3-week train slice (memory-bounded)
    ap.add_argument("--val-start", default="2026-02-01")
    ap.add_argument("--val-end", default="2026-02-15")       # 2-week val slice; "" disables
    ap.add_argument("--sample", type=int, default=1_000_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mem", default="13GB")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    tr = (to_us(args.train_start), to_us(args.train_end))
    va = (to_us(args.val_start), to_us(args.val_end)) if args.val_start and args.val_end else None
    out_dir = Path(__file__).resolve().parent / "out"
    out_dir.mkdir(exist_ok=True)

    con = duckdb.connect()
    con.execute(f"PRAGMA memory_limit='{args.mem}'")
    con.execute(f"PRAGMA threads={args.threads}")

    for sym in args.syms.split(","):
        run_symbol(con, sym, tr, va, args.sample, args.seed, out_dir)


if __name__ == "__main__":
    main()
