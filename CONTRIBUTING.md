# Contributing to coppuccino

Thanks for your interest in improving coppuccino! Bug reports, feature requests,
and pull requests are all welcome.

## Development setup

The project is managed with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/AaronDJohnson/coppuccino.git
cd coppuccino
uv sync            # create the environment and install coppuccino + dev deps
```

## Before opening a pull request

Please run the same checks CI runs:

```bash
uv run pytest              # full test suite
uv run ruff check .        # lint (pyflakes, bugbear, comprehensions, isort,
                           #       simplify, and McCabe complexity <= 15)
uv run mypy                # type check (src/coppuccino)
uv lock --check            # the committed lockfile must match pyproject.toml
```

- Add or update tests for any behavior you change. The numerical core
  (`src/coppuccino/bijections.py`) is delicate — prefer assertions that pin down
  *values* (round-trip exactness, monotonicity, density accuracy, Jacobian
  log-dets) over shape/`isfinite` smoke checks.
- `pytest` runs with `filterwarnings = ["error"]`, so a new warning fails the
  suite. If the library is *meant* to warn, assert it with `pytest.warns` at the
  call site rather than adding a global allowance.
- Keep public-API changes documented in the NumPy-style docstrings and, if the
  behavior is observable to users, in `CHANGELOG.md` under an `Unreleased`
  heading.
- Match the surrounding style by hand. `ruff format` is *not* run in CI and is
  not a gate: adopting it would rewrite most of the hand-laid numerical code in
  one pass. `[tool.ruff.format]` in `pyproject.toml` only fixes the settings for
  anyone who chooses to run it on a file they are already rewriting.
- One function (`make_empirical_cdf_spline`) carries a `# noqa: C901` exemption
  from the complexity ceiling, with a comment naming the reason. New code is
  expected to stay under it rather than add a second.

## Benchmarks

`scripts/bench_*.py` are one-off measurement drivers, not part of the package or
the test suite. They need the optional `bench` group:

```bash
uv run --group bench python scripts/bench_fleet_copula.py
```

## Reporting bugs

Open a GitHub issue with a minimal reproducer (the data shape, the call, and the
observed vs. expected behavior). For numerical issues, a small synthetic array
that triggers the problem is ideal.

## Releasing (maintainers)

1. Update the version in `pyproject.toml` and add a dated section to
   `CHANGELOG.md`.
2. Tag the release (`git tag vX.Y.Z && git push --tags`) and publish a GitHub
   Release — the `publish.yml` workflow builds and uploads to PyPI via Trusted
   Publishing.
