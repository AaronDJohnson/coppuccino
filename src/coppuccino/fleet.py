"""Fleet fitting: train one uniform-copula flow per source, all simultaneously.

Designed for the LISA galactic-binary use case: ~10^4 sources, each with its
own bounded posterior samples, whose proposal flows are refit periodically.
A single flow fit is a tiny network that would leave a GPU idle; the win is
that all sources' fits are the *same program with different data and
parameters*, which ``jax.vmap`` fuses into single kernels: one optimizer step
advances every source's flow at once. The same code runs on CPU (where vmap
still vectorizes) and becomes fast on a GPU simply by installing ``jax[cuda]``.

Architecture is FIXED across the fleet (vmap requires identical parameter
pytrees): the default ``knots=4, flow_layers=6`` is the measured middle
ground for galactic-binary posteriors -- near-optimal on simple (few-parameter,
unimodal) fits and mid-rung on hard (8-parameter, bimodal) ones. Both are
exposed as kwargs.

Pipeline per source (all boundary-safe, mirroring ``copula_space="uniform"``):

1. *Marginals*: empirical-quantile monotone RQS CDFs built on CPU (cheap,
   O(n log n) per source-dimension) and stored as stacked knot tensors
   ``(n_sources, dim, K)`` -- no per-source Python closures, so evaluation is
   vmappable end to end. Endpoints map to exactly 0/1 at the prior bounds
   (Hazen plotting positions; strictly positive wall density).
2. *Copula*: an RQS coupling flow on the unit cube over a uniform base, one
   parameter stack for the whole fleet.

Example
-------
>>> import numpy as np
>>> from coppuccino import normalizing_flows_fit_fleet
>>> chains = np.random.rand(64, 2000, 8)          # (n_sources, n_samples, dim)
>>> bounds = np.tile(np.array([0.0, 1.0]), (8, 1))
>>> fleet = normalizing_flows_fit_fleet(chains, prior_bounds=bounds,
...                                     max_epochs=100)      # doctest: +SKIP
>>> x = fleet.sample(512, rng_seed=0)             # (64, 512, 8)  # doctest: +SKIP
>>> lp = fleet.log_prob(x)                        # (64, 512)     # doctest: +SKIP
>>> x7 = fleet.sample(512, rng_seed=0, source_index=7)  # doctest: +SKIP
"""
import io
from typing import Optional

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
import cloudpickle

from coppuccino.bijections import _POS_FLOOR

__all__ = ["normalizing_flows_fit_fleet", "FleetFlow"]

_MIN_SAMPLES = 64


# ----------------------------------------------------------------------
# marginal knots (CPU construction; stacked arrays, no closures)
# ----------------------------------------------------------------------

def _marginal_knots(x: np.ndarray, lo: float, hi: float, num_points: int):
    """Empirical-quantile RQS knots for one source-dimension.

    Returns ``(xk, yk, dk)`` of fixed length ``num_points + 2``: interior knots
    at the data quantiles with Hazen ``(counts - 1/2)/n`` CDF values, plus the
    exact endpoints ``(lo, 0)`` and ``(hi, 1)``. ``dk`` are monotone PCHIP
    slopes (strictly positive), so the resulting rational-quadratic spline is
    an exact bijection ``[lo, hi] <-> [0, 1]`` with positive wall density.
    """
    from scipy.interpolate import PchipInterpolator

    n = len(x)
    span = max(hi - lo, _POS_FLOOR)
    # keep the quantile grid strictly inside the bounds: samples sitting
    # exactly on a bound (e.g. clipped data) would otherwise collide with the
    # endpoint knots after tie-bumping
    margin = 1e-9 * span
    xs = np.sort(np.clip(x.astype(float), lo + margin, hi - margin))
    xq = np.quantile(xs, np.linspace(0.0, 1.0, num_points))
    # strictly increasing interior x-knots (bump ties; preserves count)
    eps_x = 1e-12 * span
    for i in range(1, num_points):
        if xq[i] <= xq[i - 1]:
            xq[i] = xq[i - 1] + eps_x
    counts = np.searchsorted(xs, xq, side="right")
    yq = (counts - 0.5) / n
    yq = np.clip(yq, 1e-7, 1.0 - 1e-7)
    eps_y = 1e-9
    for i in range(1, num_points):
        if yq[i] <= yq[i - 1]:
            yq[i] = yq[i - 1] + eps_y

    xk = np.concatenate([[lo], xq, [hi]])
    yk = np.concatenate([[0.0], yq, [1.0]])
    # guard: quantile grid must sit strictly inside the bounds
    if xk[1] <= xk[0]:
        xk[1] = xk[0] + eps_x
    if xk[-2] >= xk[-1]:
        xk[-2] = xk[-1] - eps_x
    dk = np.maximum(PchipInterpolator(xk, yk).derivative()(xk), _POS_FLOOR)
    return xk, yk, dk


def _build_fleet_knots(chains: np.ndarray, lo: np.ndarray, hi: np.ndarray,
                       num_points: int):
    """Stacked marginal knots for the whole fleet: three (N, d, K) arrays."""
    n_src, _, dim = chains.shape
    K = num_points + 2
    xk = np.empty((n_src, dim, K))
    yk = np.empty((n_src, dim, K))
    dk = np.empty((n_src, dim, K))
    for s in range(n_src):
        for j in range(dim):
            xk[s, j], yk[s, j], dk[s, j] = _marginal_knots(
                chains[s, :, j], float(lo[s, j]), float(hi[s, j]), num_points)
    return jnp.asarray(xk), jnp.asarray(yk), jnp.asarray(dk)


# ----------------------------------------------------------------------
# vectorized rational-quadratic spline (Gregory-Delbourgo), knots as args
# ----------------------------------------------------------------------

def _rqs_fwd(x, xk, yk, dk):
    """CDF: x -> u. ``x`` any shape; ``xk/yk/dk`` 1-D knots."""
    k = jnp.clip(jnp.searchsorted(xk, x, side="right") - 1, 0, len(xk) - 2)
    x0, x1 = jnp.take(xk, k), jnp.take(xk, k + 1)
    y0, y1 = jnp.take(yk, k), jnp.take(yk, k + 1)
    d0, d1 = jnp.take(dk, k), jnp.take(dk, k + 1)
    w = x1 - x0
    s = (y1 - y0) / w
    xi = jnp.clip((x - x0) / w, 0.0, 1.0)
    num = (y1 - y0) * (s * xi**2 + d0 * xi * (1 - xi))
    den = s + (d1 + d0 - 2 * s) * xi * (1 - xi)
    return y0 + num / den


def _rqs_deriv(x, xk, yk, dk):
    """Density: dCDF/dx (strictly positive on [lo, hi])."""
    k = jnp.clip(jnp.searchsorted(xk, x, side="right") - 1, 0, len(xk) - 2)
    x0, x1 = jnp.take(xk, k), jnp.take(xk, k + 1)
    y0, y1 = jnp.take(yk, k), jnp.take(yk, k + 1)
    d0, d1 = jnp.take(dk, k), jnp.take(dk, k + 1)
    w = x1 - x0
    s = (y1 - y0) / w
    xi = jnp.clip((x - x0) / w, 0.0, 1.0)
    den = s + (d1 + d0 - 2 * s) * xi * (1 - xi)
    num = s**2 * (d1 * xi**2 + 2 * s * xi * (1 - xi) + d0 * (1 - xi) ** 2)
    return jnp.maximum(num / den**2, _POS_FLOOR)


def _rqs_inv(u, xk, yk, dk):
    """Quantile: u -> x (closed-form stable quadratic root)."""
    k = jnp.clip(jnp.searchsorted(yk, u, side="right") - 1, 0, len(yk) - 2)
    x0, x1 = jnp.take(xk, k), jnp.take(xk, k + 1)
    y0, y1 = jnp.take(yk, k), jnp.take(yk, k + 1)
    d0, d1 = jnp.take(dk, k), jnp.take(dk, k + 1)
    w = x1 - x0
    dy = y1 - y0
    s = dy / w
    alpha = u - y0
    coef = d1 + d0 - 2 * s
    a = dy * (s - d0) + alpha * coef
    b = dy * d0 - alpha * coef
    c = -s * alpha
    disc = jnp.maximum(b**2 - 4 * a * c, 0.0)
    xi = jnp.clip(2 * c / (-b - jnp.sqrt(disc)), 0.0, 1.0)
    return x0 + xi * w


# per-source marginal maps: x (b, d) with knots (d, K)
def _u_and_logdet(x, xk, yk, dk):
    per_dim = jax.vmap(lambda col, a, b_, c: (_rqs_fwd(col, a, b_, c),
                                              jnp.log(_rqs_deriv(col, a, b_, c))),
                       in_axes=(1, 0, 0, 0))
    u_t, ld_t = per_dim(x, xk, yk, dk)          # (d, b) each
    return u_t.T, jnp.sum(ld_t, axis=0)          # (b, d), (b,)


def _x_from_u(u, xk, yk, dk):
    per_dim = jax.vmap(lambda col, a, b_, c: _rqs_inv(col, a, b_, c),
                       in_axes=(1, 0, 0, 0))
    return per_dim(u, xk, yk, dk).T              # (b, d)


# ----------------------------------------------------------------------
# stacked copula flows
# ----------------------------------------------------------------------

def _template(dim: int, knots: int, flow_layers: int, rng_seed: int = 0):
    from coppuccino.copula_flows import _build_uniform_inner
    return _build_uniform_inner(("uniform", dim, knots, flow_layers),
                                rng_seed=rng_seed)


def _stack_params(dim, knots, flow_layers, n_src, rng_seed):
    """Loop-construct N template flows and stack their array leaves."""
    models = [_template(dim, knots, flow_layers, rng_seed=rng_seed + i)
              for i in range(n_src)]
    params_list = [eqx.partition(m, eqx.is_inexact_array)[0] for m in models]
    static = eqx.partition(models[0], eqx.is_inexact_array)[1]
    stacked = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *params_list)
    return stacked, static


# ----------------------------------------------------------------------
# FleetFlow
# ----------------------------------------------------------------------

class FleetFlow:
    """A fleet of independently fitted boundary-safe copula flows.

    Holds one stacked parameter pytree (leading axis = source) plus stacked
    marginal knot tensors. ``sample`` / ``log_prob`` operate on the whole
    fleet at once (vmapped) or on a single source via ``source_index``.
    Returns NumPy arrays, matching the coppuccino API convention.
    """

    def __init__(self, params, static, arch, xk, yk, dk, lo, hi):
        self.params = params            # stacked pytree, leading axis n_sources
        self.static = static            # shared template statics
        self.arch = arch                # ("fleet-uniform", dim, knots, layers, n_sources, template_seed)
        self.xk, self.yk, self.dk = xk, yk, dk   # (N, d, K)
        self.lo, self.hi = lo, hi                # (N, d)
        self.n_sources = int(xk.shape[0])
        self.dim = int(xk.shape[1])
        self._jit_cache = {}

    # ---- internals -----------------------------------------------------
    def _model(self, p):
        return eqx.combine(p, self.static)

    def _take(self, i):
        p = jax.tree_util.tree_map(lambda a: a[i], self.params)
        return p, self.xk[i], self.yk[i], self.dk[i]

    def _fn(self, name):
        """Build (and cache) the jitted batched evaluation functions."""
        if name in self._jit_cache:
            return self._jit_cache[name]
        static = self.static

        def logp_one(p, x, xk, yk, dk):
            u, ld = _u_and_logdet(x, xk, yk, dk)
            return eqx.combine(p, static).log_prob(u) + ld

        def sample_one(p, key, n, xk, yk, dk):
            u = eqx.combine(p, static).sample(key, (n,))
            return _x_from_u(u, xk, yk, dk)

        if name == "logp_all":
            f = eqx.filter_jit(jax.vmap(logp_one, in_axes=(0, 0, 0, 0, 0)))
        elif name == "logp_one":
            f = eqx.filter_jit(logp_one)
        elif name == "sample_all":
            f = eqx.filter_jit(jax.vmap(sample_one,
                                        in_axes=(0, 0, None, 0, 0, 0)))
        else:  # sample_one
            f = eqx.filter_jit(sample_one)
        self._jit_cache[name] = f
        return f

    # ---- public API ------------------------------------------------------
    def log_prob(self, x, source_index: Optional[int] = None) -> np.ndarray:
        """Log density. ``x``: (n_sources, b, dim) batched, or (b, dim) with
        ``source_index``. Returns (n_sources, b) or (b,)."""
        x = jnp.asarray(x)
        if source_index is None:
            return np.asarray(self._fn("logp_all")(self.params, x,
                                                   self.xk, self.yk, self.dk))
        p, xk, yk, dk = self._take(int(source_index))
        return np.asarray(self._fn("logp_one")(p, x, xk, yk, dk))

    def sample(self, n_samples: int, rng_seed: Optional[int] = None,
               source_index: Optional[int] = None) -> np.ndarray:
        """Draw samples: (n_sources, n_samples, dim), or (n_samples, dim) for
        one source. Fresh entropy when ``rng_seed`` is None."""
        from coppuccino.copula_flows import _resolve_seed
        key = jr.key(_resolve_seed(rng_seed))
        if source_index is None:
            keys = jr.split(key, self.n_sources)
            return np.asarray(self._fn("sample_all")(self.params, keys,
                                                     n_samples, self.xk,
                                                     self.yk, self.dk))
        p, xk, yk, dk = self._take(int(source_index))
        return np.asarray(self._fn("sample_one")(p, key, n_samples, xk, yk, dk))

    # ---- persistence -----------------------------------------------------
    def save(self, path) -> None:
        buf = io.BytesIO()
        eqx.tree_serialise_leaves(buf, self.params)
        with open(path, "wb") as f:
            cloudpickle.dump({
                "arch": self.arch,
                "params_bytes": buf.getvalue(),
                "xk": np.asarray(self.xk), "yk": np.asarray(self.yk),
                "dk": np.asarray(self.dk),
                "lo": np.asarray(self.lo), "hi": np.asarray(self.hi),
            }, f)

    @classmethod
    def load(cls, path) -> "FleetFlow":
        with open(path, "rb") as f:
            d = cloudpickle.load(f)
        # rebuild with the SAME template seed: the coupling layers' integer
        # permutation arrays live in the static partition and are
        # seed-dependent, so an arbitrary-seed skeleton would evaluate the
        # deserialized parameters under different permutations
        _, dim, knots, layers, n_src, tmpl_seed = d["arch"]
        skeleton, static = _stack_params(dim, knots, layers, n_src,
                                         rng_seed=tmpl_seed)
        params = eqx.tree_deserialise_leaves(io.BytesIO(d["params_bytes"]),
                                             skeleton)
        return cls(params, static, d["arch"],
                   jnp.asarray(d["xk"]), jnp.asarray(d["yk"]),
                   jnp.asarray(d["dk"]),
                   np.asarray(d["lo"]), np.asarray(d["hi"]))


# ----------------------------------------------------------------------
# fleet fitting
# ----------------------------------------------------------------------

def normalizing_flows_fit_fleet(chains: np.ndarray,
                                prior_bounds: np.ndarray,
                                knots: int = 4,
                                flow_layers: int = 6,
                                learning_rate: float = 1e-3,
                                max_epochs: int = 200,
                                patience: int = 30,
                                batch_size: int = 128,
                                val_prop: float = 0.1,
                                num_points: int = 200,
                                rng_seed: int = 999,
                                warm_start: Optional[FleetFlow] = None,
                                verbose: bool = False) -> FleetFlow:
    """Fit one boundary-safe copula flow per source, all sources at once.

    Every source shares a fixed architecture (``knots``, ``flow_layers`` --
    default 4/6, the measured middle ground for galactic-binary posteriors),
    which is what lets ``vmap`` fuse the whole fleet's training into single
    kernels: one Adam step advances every source simultaneously. On CPU this
    vectorizes; on a GPU (``jax[cuda]``) the same code trains ~10^4 sources
    in minutes.

    Parameters
    ----------
    chains : np.ndarray
        Posterior samples, shape ``(n_sources, n_samples, dim)``. Use a few
        thousand *decorrelated* (thinned) samples per source; all sources
        must supply the same ``n_samples`` (subsample upstream). NaNs are not
        allowed here (clean per source upstream).
    prior_bounds : np.ndarray
        ``(dim, 2)`` shared bounds or ``(n_sources, dim, 2)`` per source;
        each row ``[low, high]``. Required: the fleet models bounded
        (prior-box) support exactly.
    knots, flow_layers : int, optional
        Fleet-wide copula architecture. Defaults 4 and 6.
    learning_rate, max_epochs, patience, batch_size, val_prop : optional
        Training controls. ``patience`` acts on the fleet-mean validation
        loss; per-source best parameters are tracked and returned regardless.
    num_points : int, optional
        Marginal CDF quantile-grid size (clamped to ``n_samples // 3``).
    rng_seed : int, optional
        Seed for initialization, batching, and the validation split.
    warm_start : FleetFlow, optional
        Reuse a previous fleet's parameters as initialization (architecture
        and ``n_sources`` must match). For periodic refreshes this typically
        cuts the required epochs several-fold.
    verbose : bool, optional
        Print per-epoch fleet-mean validation loss.

    Returns
    -------
    FleetFlow
    """
    chains = np.asarray(chains, dtype=float)
    if chains.ndim != 3:
        raise ValueError(f"chains must be (n_sources, n_samples, dim); "
                         f"got shape {chains.shape}.")
    n_src, n_samp, dim = chains.shape
    if n_samp < _MIN_SAMPLES:
        raise ValueError(f"need at least {_MIN_SAMPLES} samples per source; "
                         f"got {n_samp}.")
    if np.isnan(chains).any():
        raise ValueError("chains contain NaNs; clean or subsample upstream.")

    pb = np.asarray(prior_bounds, dtype=float)
    if pb.shape == (dim, 2):
        pb = np.broadcast_to(pb, (n_src, dim, 2)).copy()
    if pb.shape != (n_src, dim, 2):
        raise ValueError(f"prior_bounds must be (dim, 2) or "
                         f"(n_sources, dim, 2); got {pb.shape}.")
    lo, hi = pb[..., 0], pb[..., 1]
    if (chains < lo[:, None, :]).any() or (chains > hi[:, None, :]).any():
        raise ValueError("chains contain samples outside prior_bounds.")

    # ---- marginals (CPU) + u-space training tensors ----
    num_points = max(20, min(num_points, n_samp // 3))
    xk, yk, dk = _build_fleet_knots(chains, lo, hi, num_points)
    u_all, _ = eqx.filter_jit(jax.vmap(_u_and_logdet))(
        jnp.asarray(chains), xk, yk, dk)                       # (N, n, d)

    # per-source val split (shared indices; sizes identical by construction)
    rng = np.random.default_rng(rng_seed)
    perm = rng.permutation(n_samp)
    n_val = max(1, int(round(n_samp * val_prop)))
    u_val = u_all[:, perm[:n_val]]
    u_tr = u_all[:, perm[n_val:]]
    n_tr = u_tr.shape[1]
    batch_size = min(batch_size, n_tr)
    steps = n_tr // batch_size

    # ---- stacked flows ----
    arch = ("fleet-uniform", dim, knots, flow_layers, n_src, rng_seed)
    if warm_start is not None:
        if warm_start.arch[:5] != arch[:5]:
            raise ValueError(f"warm_start arch {warm_start.arch[:5]} does not "
                             f"match requested {arch[:5]}.")
        # adopt the previous fleet's parameters AND static (its params were
        # trained under its own template's permutations)
        params, static, arch = warm_start.params, warm_start.static, warm_start.arch
    else:
        params, static = _stack_params(dim, knots, flow_layers, n_src, rng_seed)

    def nll(p, u):
        return -jnp.mean(eqx.combine(p, static).log_prob(u))

    opt = optax.adam(learning_rate)
    opt_state = opt.init(eqx.filter(params, eqx.is_inexact_array))

    @eqx.filter_jit
    def train_step(params, opt_state, u_batch):
        losses, grads = jax.vmap(jax.value_and_grad(nll))(params, u_batch)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = eqx.apply_updates(params, updates)
        return params, opt_state, losses

    @eqx.filter_jit
    def val_loss(params):
        return jax.vmap(nll)(params, u_val)                    # (N,)

    key = jr.key(rng_seed)
    best_val = np.asarray(val_loss(params))
    best_params = params
    fleet_best = float(np.mean(best_val))
    stall = 0
    for epoch in range(max_epochs):
        key, sub = jr.split(key)
        order = jr.permutation(sub, n_tr)
        for s in range(steps):
            idx = order[s * batch_size:(s + 1) * batch_size]
            params, opt_state, _ = train_step(params, opt_state,
                                              u_tr[:, idx])
        v = np.asarray(val_loss(params))                        # (N,)
        improved = v < best_val
        if improved.any():
            imp = jnp.asarray(improved)
            best_params = jax.tree_util.tree_map(
                lambda b, n: jnp.where(imp.reshape((-1,) + (1,) * (n.ndim - 1)),
                                       n, b),
                best_params, params)
            best_val = np.where(improved, v, best_val)
        if verbose:
            print(f"[fleet] epoch {epoch + 1}/{max_epochs}: "
                  f"mean val NLL {np.mean(v):.4f} "
                  f"(best {np.mean(best_val):.4f})", flush=True)
        if float(np.mean(best_val)) < fleet_best - 1e-12:
            fleet_best = float(np.mean(best_val))
            stall = 0
        else:
            stall += 1
            if stall >= patience:
                break

    return FleetFlow(best_params, static, arch, xk, yk, dk, lo, hi)
