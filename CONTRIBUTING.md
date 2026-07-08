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
uv run ruff check .        # lint
uv run mypy                # type check (src/coppuccino)
```

- Add or update tests for any behavior you change. The numerical core
  (`src/coppuccino/bijections.py`) is delicate — prefer assertions that pin down
  *values* (round-trip exactness, monotonicity, density accuracy, Jacobian
  log-dets) over shape/`isfinite` smoke checks.
- Keep public-API changes documented in the NumPy-style docstrings and, if the
  behavior is observable to users, in `CHANGELOG.md` under an `Unreleased`
  heading.
- Match the surrounding style; `ruff` handles formatting nits.

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
