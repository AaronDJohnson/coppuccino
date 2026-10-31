import jax
import jax.numpy as jnp
import numpy as np
import pytest

from coppuccino import bijections, fleet
from coppuccino.bijections import (
    _POS_FLOOR,
    _rqs_bulk,
    _rqs_knot_derivatives,
    _smooth_marginal_bijection,
    make_smooth_marginal,
)


class TestSmoothMarginalBoundaryAtoms:
    """Samples sitting exactly on a prior bound -- the documented recommended
    use of ``prior_bounds``, and what a clipping/reflecting MCMC kernel
    produces."""

    @staticmethod
    def _chain_with_upper_atom(n=1500, lo=-1.0, hi=1.0):
        rng = np.random.default_rng(0)
        return np.clip(rng.normal(0.9, 0.25, n), lo, hi), lo, hi

    @staticmethod
    def _chain_with_lower_atom(n=1500, lo=-1.0, hi=1.0):
        rng = np.random.default_rng(0)
        return np.clip(rng.normal(-0.9, 0.25, n), lo, hi), lo, hi

    @staticmethod
    def _log_q(z_of_theta, log_dtheta_dz):
        def log_q(theta):
            z = z_of_theta(jnp.asarray(theta))
            return float(-0.5 * z ** 2 - 0.5 * np.log(2 * np.pi)
                         - log_dtheta_dz(z))
        return log_q

    @pytest.mark.parametrize("side,max_excess", [("lower", 12.0), ("upper", 8.0)])
    def test_bound_row_reports_a_bounded_wall_density(self, side, max_excess):
        """A row on the bound must report a density of interior magnitude.

        The constructor insets a wall atom before building the knots, so the
        atom's mass becomes a resolvable shoulder and the tail below it
        carries a finite wall density. Both halves matter: without the
        finite-density tail the wall value runs off to +-infinity (the
        exponential wall approach reported -7.6e5 nats here), and without the
        inset the shoulder is only ``eps0`` = 1e-9 of the span wide, so the
        same mass is reported over a 1e9-times-narrower window.

        Measured excess of the at-bound log-density over the interior median,
        8 seeds (``default_rng(0..7)``) of ``clip(N(-+0.9, 0.25), -1, 1)``
        (33% of 1500 rows on the bound): 6.34 to 11.03 nats at the lower wall
        and -0.66 to 3.84 at the upper one. Reverting the inset to the raw
        ``eps0`` margin spreads the same atom over a window 1e9 times
        narrower, which pushes both far higher. The two walls differ because
        the Hazen tie count puts a tie block's plotting position at the top
        of the block, so a lower-wall atom lands in the tail (with its
        outermost knot dragged in to z = -0.4 or so, and the wall therefore
        reached at |z| = 1.82 to 1.96) and an upper-wall one in the outermost bulk bin
        (knot at the usual ndtri(0.5/n), wall at |z| = 4.90). The lower
        branch had no test at all before this one -- deleting it left the
        suite green.
        """
        x, lo, hi = (self._chain_with_lower_atom() if side == "lower"
                     else self._chain_with_upper_atom())
        bound = lo if side == "lower" else hi
        on_bound = x <= lo if side == "lower" else x >= hi
        assert on_bound.sum() > 0.2 * len(x)
        _, z_of_theta, log_dtheta_dz, _ = make_smooth_marginal(
            x, prior_low=lo, prior_high=hi)
        log_q = self._log_q(z_of_theta, log_dtheta_dz)

        z_bound = float(z_of_theta(jnp.array(bound)))
        assert abs(z_bound) < 10.0, f"z at the bound is {z_bound}"
        interior = float(np.median(x[~on_bound]))
        assert log_q(bound) - log_q(interior) < max_excess, (
            log_q(bound), log_q(interior))
        assert log_q(bound) - log_q(interior) > -5.0, (
            log_q(bound), log_q(interior))
        # and it is a genuine limit, not a clamp: the density converges as the
        # query approaches the bound instead of stepping at a window edge
        sign = 1.0 if side == "lower" else -1.0
        ladder = np.array([log_q(bound + sign * 10.0 ** -k)
                           for k in range(6, 13)])
        assert np.max(np.abs(np.diff(ladder))) < 1.0, ladder
        assert abs(ladder[-1] - log_q(bound)) < 0.1, ladder

    def test_bound_row_and_nan_free_log_det(self):
        """Both directions stay finite and mutually consistent at the bound."""
        x, lo, hi = self._chain_with_upper_atom()
        bij = _smooth_marginal_bijection(x, prior_low=lo, prior_high=hi)
        clamps = (float(bij.x_clamp_lo), float(bij.x_clamp_hi))

        # the tail saturates one _WALL_Z_MARGIN below its outermost knot, so
        # the last sliver of the support does share one z; probe outside it
        inside = hi - 10.0 * (hi - clamps[1])
        zs = []
        for xv in (hi, inside, hi + 1.0):
            z, log_det = bij.inverse_and_log_det(jnp.array(xv))
            assert bool(jnp.isfinite(z)) and bool(jnp.isfinite(log_det))
            assert abs(float(z)) < 10.0
            zs.append(float(z))
        # a value beyond the bound saturates at the wall, not extrapolated
        assert zs[2] == pytest.approx(zs[0], abs=1e-12)
        # and the saturated sliver is exactly the one the builder reports:
        # everything above ``x_clamp_hi`` shares the bound's z, everything
        # below it is resolved separately
        assert float(bij.inverse_and_log_det(jnp.array(clamps[1]))[0]) == \
            pytest.approx(zs[0], abs=1e-12)
        assert zs[1] < zs[0]

    def test_unclamped_forward_draws_are_never_absorbed(self):
        """``sample`` redraws exactly the rows that land on ``x_clamp_hi``, so
        every x the forward map emits *below* the clamp must still invert
        through the tail. Otherwise a kept draw would silently report the knot
        density instead of the density it was drawn from."""
        x, lo, hi = self._chain_with_upper_atom()
        theta_of_z, z_of_theta, _, (_, clamp_hi) = make_smooth_marginal(
            x, prior_low=lo, prior_high=hi)
        assert float(theta_of_z(jnp.array(1e6))) == clamp_hi

        zs = np.linspace(4.0, 2000.0, 400)
        xs = np.asarray(theta_of_z(jnp.asarray(zs)))
        kept = xs < clamp_hi
        assert kept.any()
        z_back = np.asarray(z_of_theta(jnp.asarray(xs[kept])))
        assert np.allclose(z_back, zs[kept], rtol=1e-6)

    def test_unbounded_and_atom_free_sides_are_untouched(self):
        """Absorption is tied to an atom actually being relocated.

        The map is built with its upper side bounded at 1 but carrying no
        wall atom, and its lower side unbounded.

        The bounded side keeps its analytic tail over the whole region the
        tail resolves, which is everything up to ``x_clamp_hi``. The saturated
        sliver beyond it is the ``_WALL_Z_MARGIN`` trade, not absorption: it
        is set by the outermost knot's z, not by any atom, and it is the same
        fraction of the knot-to-bound distance on a chain with no atom at all.
        Here it is 9.4e-5 wide against a 6.6e-2 knot-to-bound gap (0.14% of
        it), so probes an order of magnitude inside it are still resolved
        separately.

        The unbounded side has no saturation point at all: no clamp is
        reported, and theta keeps moving, invertibly, however deep z goes. A
        wall tail or an inset applied there by mistake would show up as a
        clamp and a plateau.
        """
        rng = np.random.default_rng(5)
        x = rng.normal(0.0, 0.3, 1500)
        assert x.max() < 1.0 - 1e-6  # no wall atom
        theta_of_z, z_of_theta, _, clamps = make_smooth_marginal(
            x, prior_low=None, prior_high=1.0)
        sliver = 1.0 - clamps[1]
        assert sliver < 0.01 * (1.0 - x.max())
        # strictly monotone right up to the saturation point
        probes = jnp.array([1.0 - 1e-2, 1.0 - 1e-3,
                            1.0 - 10.0 * sliver, 1.0 - sliver])
        zs = np.asarray(z_of_theta(probes))
        assert np.all(np.diff(zs) > 0)

        assert clamps[0] is None
        deep = np.array([-40.0, -20.0, -10.0, -5.0])
        theta = np.asarray(theta_of_z(jnp.asarray(deep)))
        assert np.all(np.isfinite(theta)) and np.all(np.diff(theta) > 0)
        np.testing.assert_allclose(np.asarray(z_of_theta(jnp.asarray(theta))),
                                   deep, rtol=1e-12)


class TestWallDepthTracksTheTrainingEdge:
    """A bounded tail stops one margin past its own outermost knot.

    That knot sits on the most extreme sample, at Hazen position
    ``0.5 / n``, so it is exactly the deepest ``|z|`` the copula flow
    downstream was ever trained on in this dimension; everything past it is
    extrapolation, and ``_WALL_Z_MARGIN`` is how much of it the marginal is
    willing to hand over. No constant can express that, because
    ``|ndtri(0.5 / n)|`` runs from 2.94 at n = 300 to 4.42 at n = 100000: a
    constant is either far past the edge on a small chain or inside it on a
    large one, and once it is inside, the half-gap guard takes over and the
    reported depth drifts with n regardless of the number named (the 8.0 this
    replaced sat 4.6 past the edge at n = 1500; a 4.0 would have been 0.6
    past it there and *inside* it by n = 6000).
    """

    @pytest.mark.parametrize("n", [300, 1500, 6000, 100000])
    def test_the_wall_z_is_one_margin_past_the_outermost_knot(self, n):
        from scipy.special import ndtri

        rng = np.random.default_rng(0)
        x = rng.uniform(0.0, 1.0, n)
        _, z_of_theta, _, clamps = make_smooth_marginal(
            x, prior_low=0.0, prior_high=1.0)
        edge = float(ndtri(0.5 / n))  # the outermost Hazen plotting position
        assert float(z_of_theta(jnp.array(0.0))) == pytest.approx(
            edge - bijections._WALL_Z_MARGIN, abs=1e-6)
        # and the wall resolution therefore *improves* with more data rather
        # than staying pinned: 2.8e-3 of the knot-to-bound gap at n = 300
        # down to 3.3e-4 at n = 100000. That trend needs no test of its own:
        # the wall shape has unit slope at the bound, so to first order the
        # fraction is Phi(wall z) / Phi(edge), and the assertion above pins
        # both z values exactly at every n.
        assert clamps[0] / float(x.min()) < 3e-3


class TestWallShapeJoinPin:
    """The wall shape's ``S'(1) = beta`` pin closes the bulk/tail join.

    :class:`bijections._MarginalTail` presents that pin as what makes the
    bounded tail's density continuous with the bulk where they meet. Nothing
    was checking it: replacing the fitted ``beta`` with 1.0 in ``_make_tail``
    -- deleting the pin, so the wall shape becomes the identity and the tail's
    join density is off by exactly that factor -- left every other test in
    this file passing.
    """

    LO, HI = 0.0, 1.0

    @staticmethod
    def _join_jumps(x, lo, hi):
        """``log dtheta/dz`` across each bulk/tail join, tail minus bulk."""
        _, z_of_theta, log_dtheta_dz, _ = make_smooth_marginal(
            x, prior_low=lo, prior_high=hi)
        out = []
        for theta in (float(np.min(x)), float(np.max(x))):
            z_join = float(z_of_theta(jnp.array(theta)))
            out.append(float(log_dtheta_dz(jnp.array(z_join - 1e-7)))
                       - float(log_dtheta_dz(jnp.array(z_join + 1e-7))))
        return out

    def test_the_density_is_continuous_across_both_wall_joins(self):
        rng = np.random.default_rng(0)
        x = rng.beta(3.0, 3.0, 1500)
        for jump in self._join_jumps(x, self.LO, self.HI):
            assert abs(jump) < 1e-3, jump

    def test_dropping_the_pin_opens_both_joins(self, monkeypatch):
        """The fixture has teeth: without the pin these joins step by nats.

        Beta(3, 3) puts the outermost knots where the density is falling
        steeply, so the join-side derivative the pin carries is far from 1 on
        both sides and the mutant cannot pass by luck.
        """
        real = bijections._make_tail

        def unpinned(*args, **kwargs):
            tail = real(*args, **kwargs)
            if not tail.wall:
                return tail
            return tail._replace(beta=1.0, shape_d=jnp.asarray([1.0, 1.0]))

        monkeypatch.setattr(bijections, "_make_tail", unpinned)
        rng = np.random.default_rng(0)
        x = rng.beta(3.0, 3.0, 1500)
        jumps = self._join_jumps(x, self.LO, self.HI)
        assert min(abs(j) for j in jumps) > 0.5, jumps


class TestSharedRQSKernel:
    """One rational-quadratic-spline kernel, used by every RQS call site.

    The marginal splines here and the fleet's stacked marginals used to carry
    separate transcriptions of the same Gregory-Delbourgo algebra with
    different out-of-range policies.
    """

    @staticmethod
    def _shared_knots(seed=0, n=12):
        """Knots both call sites can be evaluated on."""
        rng = np.random.default_rng(seed)
        xk = np.sort(rng.normal(size=n))
        yk = np.sort(rng.uniform(size=n))
        yk = (yk - yk[0]) / (yk[-1] - yk[0])
        return xk, yk

    def test_the_call_sites_hold_one_kernel(self):
        """A re-fork must be caught here, not by luck of a differing guard."""
        assert fleet._rqs_forward is bijections._rqs_forward
        assert fleet._rqs_inverse is bijections._rqs_inverse
        assert fleet._rqs_derivative is bijections._rqs_derivative
        assert fleet._POS_FLOOR is bijections._POS_FLOOR

    def test_bulk_and_fleet_paths_agree_bitwise(self):
        """Forward, inverse and log-derivative agree to exactly 0.0.

        Each module is driven through its own entry point, on shared knots,
        and including queries outside the knot range — where the copies used
        to disagree by construction.
        """
        xk, yk = self._shared_knots()
        cdf, inverse, deriv = _rqs_bulk(xk, yk)
        xk_j, yk_j = jnp.asarray(xk), jnp.asarray(yk)
        dk_j = jnp.asarray(_rqs_knot_derivatives(xk, yk))

        inside = jnp.asarray(np.linspace(xk[0], xk[-1], 257))
        outside = jnp.asarray(np.concatenate([xk[0] - np.geomspace(1e-6, 5.0, 8),
                                              xk[-1] + np.geomspace(1e-6, 5.0, 8)]))
        u = jnp.asarray(np.linspace(yk[0], yk[-1], 257))

        # fleet drives one source with one dimension: x is (b, d), knots (d, K)
        def fleet_forward(x):
            return fleet._u_and_logdet(x[:, None], xk_j[None], yk_j[None],
                                       dk_j[None])[0][:, 0]

        for x in (inside, outside):
            assert jnp.all(cdf(x) == fleet_forward(x))

        # the log-det is only comparable where the fleet reports one: outside
        # the prior box it deliberately returns -inf (a support test)
        log_deriv = jnp.log(jnp.maximum(deriv(inside), _POS_FLOOR))
        assert jnp.all(log_deriv == fleet._u_and_logdet(
            inside[:, None], xk_j[None], yk_j[None], dk_j[None])[1])

        assert jnp.all(inverse(u) == fleet._x_from_u(
            u[:, None], xk_j[None], yk_j[None], dk_j[None])[:, 0])

    def test_out_of_range_queries_saturate(self):
        """Extrapolating the rational quadratic turns it over: the CDF falls
        below zero and rises again, and the density goes negative. Queries
        outside the knot range are evaluated at the nearest end knot instead.
        """
        xk, yk = self._shared_knots(seed=1)
        cdf, _, deriv = _rqs_bulk(xk, yk)

        span = xk[-1] - xk[0]
        probe = jnp.asarray(np.concatenate([
            xk[0] - np.linspace(3.0 * span, 0.0, 200)[:-1],
            np.linspace(xk[0], xk[-1], 200),
            xk[-1] + np.linspace(0.0, 3.0 * span, 200)[1:],
        ]))
        values = np.asarray(cdf(probe))
        densities = np.asarray(deriv(probe))

        assert np.all(np.diff(values) >= 0.0)
        assert values.min() >= yk[0] and values.max() <= yk[-1]
        assert np.all(densities >= 0.0)
        # constant beyond each end knot, i.e. genuinely saturated
        assert cdf(jnp.asarray(xk[0] - 1e3)) == cdf(jnp.asarray(xk[0] - 1.0))
        assert cdf(jnp.asarray(xk[-1] + 1e3)) == cdf(jnp.asarray(xk[-1] + 1.0))
        assert deriv(jnp.asarray(xk[-1] + 1e3)) == deriv(jnp.asarray(xk[-1]))

    def test_knot_derivative_floors_are_parameters(self):
        """One knot-slope builder, with each caller's floor policy as an argument."""
        xk = np.array([0.0, 1.0, 2.0, 3.0])
        yk = np.array([0.0, 0.5, 0.5 + 1e-14, 1.0])  # near-flat middle bin

        assert np.all(_rqs_knot_derivatives(xk, yk) >= _POS_FLOOR)
        assert np.all(_rqs_knot_derivatives(xk, yk, floor=1e-8) >= 1e-8)

        # PCHIP's one-sided endpoint rule zeroes the first knot derivative
        # whenever the second bin is more than three times as steep
        steepening = np.array([0.0, 0.01, 1.0, 1.5])
        assert _rqs_knot_derivatives(xk, steepening)[0] == _POS_FLOOR
        floored = _rqs_knot_derivatives(xk, steepening, edge_slope_floor=0.05)
        assert floored[0] == 0.05 * (steepening[1] - steepening[0]) / (xk[1] - xk[0])


class TestFloat32Path:
    """The x64-disabled path had no coverage anywhere in tests/.

    ``coppuccino`` enables x64 on import, but that is a process-wide setting
    a caller can turn back off, and several constants in this module are
    float64 literals that underflow below it. Everything here runs inside
    ``jax.experimental.disable_x64()``.
    """

    def test_smooth_marginal_stays_a_finite_monotone_map_without_x64(self):
        """The bounded tails must not go NaN or non-monotone in float32."""
        rng = np.random.default_rng(1)
        x = np.clip(rng.beta(0.5, 3.0, 3000), 0.0, 1.0)
        with jax.experimental.disable_x64():
            theta_of_z, z_of_theta, log_dtheta_dz, clamps = (
                make_smooth_marginal(x, prior_low=0.0, prior_high=1.0))
            z = jnp.asarray(np.linspace(-7.0, 7.0, 1001, dtype=np.float32))
            theta = np.asarray(theta_of_z(z))
            log_det = np.asarray(log_dtheta_dz(z))
            back = np.asarray(z_of_theta(jnp.asarray(theta)))
        assert np.all(np.isfinite(theta)) and np.all(np.isfinite(log_det))
        assert np.all(np.diff(theta) >= 0.0)
        assert theta.min() > 0.0 and theta.max() < 1.0
        core = np.abs(np.asarray(z)) <= 3.0
        assert np.max(np.abs(back[core] - np.asarray(z)[core])) < 1e-3
        assert clamps[0] > 0.0 and clamps[1] < 1.0
