"""Does the fleet's coupling flow beat a Gaussian copula, in the fleet's own pipeline?

The fleet maps each marginal onto [0, 1] with a unit-range RQS and models the copula
with a coupling flow on the cube. The Gaussian-copula alternative keeps those exact
same marginals, pushes u through ndtri, and models the copula as N(0, C) with C the
copula-space correlation of the training rows -- closed form, no training at all.

Holding the marginals fixed isolates the copula model. Both are scored by mean
held-out log-density in the original parameter space, so the numbers are directly
comparable to each other.
"""
from __future__ import annotations

import pathlib
import time

import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy.special import ndtri

from coppuccino.fleet import _build_fleet_knots, _u_and_logdet, normalizing_flows_fit_fleet

REPO = pathlib.Path(__file__).resolve().parent.parent
DATA = REPO / "examples" / "data" / "LDC2A_00000007_posterior.feather"
PARAMS = [
    "Frequency", "Frequency Derivative", "Amplitude", "Ecliptic Longitude",
    "sinlat", "cosinc", "Initial Phase", "Polarization",
]
PHASE_IDX = PARAMS.index("Initial Phase")
TWO_PI = 2 * np.pi


def load() -> np.ndarray:
    x = np.array(pd.read_feather(DATA)[PARAMS].to_numpy(), dtype=float)
    x[:, PHASE_IDX] = (x[:, PHASE_IDX] + np.pi / 2) % TWO_PI
    return x


def bounds(x: np.ndarray) -> np.ndarray:
    """Physical prior box where the parameter has one; padded data range otherwise."""
    lo, hi = x.min(0).copy(), x.max(0).copy()
    pad = 0.05 * (hi - lo) + 1e-12
    lo, hi = lo - pad, hi + pad
    for j, (a, b) in {3: (0.0, TWO_PI), 4: (-1.0, 1.0), 5: (-1.0, 1.0),
                      6: (0.0, TWO_PI), 7: (0.0, np.pi)}.items():
        lo[j], hi[j] = a, b
    return np.stack([lo, hi], axis=1)


def gaussian_copula_logprob(u_tr, u_val, ld_val, jitter=1e-10):
    """Held-out log-density under N(0, C) fitted in copula space (zero trainable params)."""
    z_tr, z_val = ndtri(np.clip(u_tr, 1e-15, 1 - 1e-15)), ndtri(np.clip(u_val, 1e-15, 1 - 1e-15))
    d = z_tr.shape[1]
    C = np.corrcoef(z_tr, rowvar=False) + jitter * np.eye(d)
    sign, logdet = np.linalg.slogdet(C)
    quad = np.einsum("ij,jk,ik->i", z_val, np.linalg.inv(C), z_val)
    log_gauss = -0.5 * (d * np.log(2 * np.pi) + logdet + quad)
    log_phi = np.sum(-0.5 * (z_val**2 + np.log(2 * np.pi)), axis=1)
    return log_gauss - log_phi + ld_val, C   # copula density x marginal jacobian


def main() -> None:
    x = load()
    pb = bounds(x)
    rng = np.random.default_rng(0)

    print(f"data {x.shape}\n")
    results = {}
    for seed in (0, 1, 2):
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(x))
        n_val = int(round(0.15 * len(x)))
        val, tr = x[perm[:n_val]], x[perm[n_val:]]

        # Marginals are built from the training rows only, and shared by both models.
        lo, hi = pb[None, :, 0], pb[None, :, 1]
        xk, yk, dk = _build_fleet_knots(tr[None], lo, hi, num_points=200)
        u_tr, _ = _u_and_logdet(jnp.asarray(tr), xk[0], yk[0], dk[0])
        u_val, ld_val = _u_and_logdet(jnp.asarray(val), xk[0], yk[0], dk[0])
        u_tr, u_val, ld_val = map(np.asarray, (u_tr, u_val, ld_val))

        gc_lp, C = gaussian_copula_logprob(u_tr, u_val, ld_val)

        t0 = time.perf_counter()
        fleet = normalizing_flows_fit_fleet(tr[None], prior_bounds=pb, max_epochs=400,
                                            rng_seed=seed, verbose=False)
        fit_s = time.perf_counter() - t0
        fl_lp = np.asarray(fleet.log_prob(val[None]))[0]

        off = C[np.triu_indices_from(C, 1)]
        print(f"seed {seed}:  fleet flow {fl_lp.mean():+9.3f}   "
              f"gaussian copula {gc_lp.mean():+9.3f}   "
              f"flow advantage {fl_lp.mean() - gc_lp.mean():+7.3f} nats"
              f"   (fit {fit_s:.1f}s, max|rho| {np.abs(off).max():.2f})")
        results.setdefault("flow", []).append(fl_lp.mean())
        results.setdefault("gc", []).append(gc_lp.mean())

    f, g = np.array(results["flow"]), np.array(results["gc"])
    print(f"\nmean over 3 splits: flow {f.mean():+.3f}   gaussian copula {g.mean():+.3f}"
          f"   advantage {f.mean() - g.mean():+.3f} nats")


if __name__ == "__main__":
    main()
