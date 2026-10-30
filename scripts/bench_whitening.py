"""Benchmark the frozen-Gaussian-copula prototypes on the galactic-binary example.

Runs ``normalizing_flows_fit_auto`` over a candidate ladder that mixes the shipped
architectures with their ``whiten=True`` counterparts plus a zero-parameter
``flow_layers=0, whiten=True`` rung (the data's Gaussian copula, fitted by nothing).

Reported numbers are strictly held-out (``refit_best=False``), and every candidate in a
given seed sees the identical split, so the comparison is paired.

    .venv/bin/python scripts/bench_whitening.py [--seeds 3] [--max-epochs 200]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np
import pandas as pd

from coppuccino import normalizing_flows_fit_auto, sample_and_log_prob
from coppuccino.copula_flows import normalizing_flows_fit

REPO = pathlib.Path(__file__).resolve().parent.parent
DATA = REPO / "examples" / "data" / "LDC2A_00000007_posterior.feather"
PARAMS = [
    "Frequency", "Frequency Derivative", "Amplitude", "Ecliptic Longitude",
    "sinlat", "cosinc", "Initial Phase", "Polarization",
]
PHASE_IDX = PARAMS.index("Initial Phase")

CANDIDATES = [
    {"knots": 2, "flow_layers": 4},
    {"knots": 4, "flow_layers": 6},
    {"knots": 8, "flow_layers": 8},
    {"flow_layers": 0, "whiten": True},
    {"knots": 2, "flow_layers": 4, "whiten": True},
    {"knots": 4, "flow_layers": 6, "whiten": True},
    {"knots": 8, "flow_layers": 8, "whiten": True},
]


def load() -> np.ndarray:
    """Posterior samples with the periodic phase recentered (see the example notebook)."""
    x = np.array(pd.read_feather(DATA)[PARAMS].to_numpy(), dtype=float)
    x[:, PHASE_IDX] = (x[:, PHASE_IDX] + np.pi / 2) % (2 * np.pi)
    return x


def label(cand: dict) -> str:
    if cand.get("flow_layers") == 0:
        return "gaussian-copula (0 params)"
    tag = "  +whiten" if cand.get("whiten") else ""
    return f"knots={cand['knots']} layers={cand['flow_layers']}{tag}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--max-epochs", type=int, default=200)
    args = ap.parse_args()

    data = load()
    print(f"data {data.shape}  candidates {len(CANDIDATES)}  seeds {args.seeds}\n")

    runs = []
    for seed in range(args.seeds):
        rng_seed = 42 + 1000 * seed
        t0 = time.perf_counter()
        _, table = normalizing_flows_fit_auto(
            data,
            candidates=CANDIDATES,
            rng_seed=rng_seed,
            max_epochs=args.max_epochs,
            refit_best=False,        # reported scores are strictly held-out
            return_table=True,
            selection="smallest_comparable",
        )
        dt = time.perf_counter() - t0
        print(f"seed {rng_seed}  ({dt:.1f}s)")
        for row in table:
            print(f"  {label(row['config']):32s} logp {row['val_logp']:+9.4f} "
                  f"+- {row['sem']:.4f}  params {str(row['n_params']):>6s}  "
                  f"comparable {str(row['comparable']):5s}"
                  f"{'  <- selected' if row['selected'] else ''}")
        runs.append({"rng_seed": rng_seed, "seconds": dt, "table": [
            {k: v for k, v in row.items() if k != "marginal_details"} for row in table
        ]})
        print()

    # Wall-clock cost of a fit, and of the MCMC-relevant sample+density call.
    print("cost per configuration (single fit, seed 42)")
    costs = []
    for cand in CANDIDATES:
        t0 = time.perf_counter()
        flow = normalizing_flows_fit(data, rng_seed=42, max_epochs=args.max_epochs,
                                     show_progress=False, **cand)
        fit_s = time.perf_counter() - t0
        sample_and_log_prob(flow, 1024, rng_seed=7)          # warm the compile
        t0 = time.perf_counter()
        for _ in range(20):
            sample_and_log_prob(flow, 1024, rng_seed=7)
        prop_ms = (time.perf_counter() - t0) / 20 * 1e3
        print(f"  {label(cand):32s} fit {fit_s:7.1f}s   sample_and_log_prob(1024) {prop_ms:6.2f} ms")
        costs.append({"config": cand, "fit_seconds": fit_s, "proposal_ms": prop_ms})

    out = REPO / "scripts" / "bench_whitening_results.json"
    out.write_text(json.dumps({"runs": runs, "costs": costs}, indent=1, default=str))
    print(f"\nwrote {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
