import os
import stat
import tempfile
from pathlib import Path

import cloudpickle
import numpy as np
import pytest
from flowjax.distributions import Transformed
from paramax import unwrap

import coppuccino.model_io as model_io
from coppuccino.copula_flows import log_prob, normalizing_flows_fit, sample
from coppuccino.model_io import (
    FLEET_KIND,
    FLOW_KIND,
    FORMAT_VERSION,
    _atomic_write_bytes,
    _extract_spline_data,
    load_flow,
    save_flow,
)


@pytest.fixture(scope="module")
def small_flow():
    """A cheap fitted flow shared by every test that only needs *a* flow.

    A fit costs about two seconds of tracing and compiling whatever its epoch
    count, so the file fits once here instead of once per test.
    """
    np.random.seed(42)
    chain = np.random.randn(200, 2)
    return normalizing_flows_fit(chain, max_epochs=5, patience=2,
                                 show_progress=False)


@pytest.fixture(scope="module")
def tail_alpha_flow():
    """A 2,400-row flow fitted with a non-default ``tail_alpha=0.5``.

    One fit serves two tests: the size law needs a second chain length, and
    the tail_alpha round trip needs a non-default tail_alpha. tail_alpha is
    one float per dimension whatever its value, so it does not move the
    payload size the size law measures.
    """
    chain = np.random.default_rng(0).standard_normal((2400, 2))
    return normalizing_flows_fit(chain, max_epochs=3, patience=2,
                                 tail_alpha=0.5, show_progress=False)


def _n_rows(flow):
    """Training rows a fitted flow retains (one column per marginal)."""
    return int(_extract_spline_data(flow)['samples'][0].size)


class TestPayloadSizeLaw:
    """Pin the file-size scaling that `save_flow`'s Notes quote figures for."""

    def test_size_is_one_float64_per_row_per_dim_plus_a_fixed_part(
            self, small_flow, tail_alpha_flow):
        """Growing the chain must cost exactly the retained training columns.

        The docstring's MB figures are only reproducible while this holds: if a
        future writer stores something else that scales with ``n`` (a knot grid
        per row, a cached z column) the quoted sizes stop following from the
        chain, and the advice to thin the chain stops being the whole story.
        """
        dim = 2
        sizes = {}
        with tempfile.TemporaryDirectory() as tmpdir:
            for flow in (small_flow, tail_alpha_flow):
                n = _n_rows(flow)
                path = os.path.join(tmpdir, f"flow_{n}.pkl")
                save_flow(flow, path)
                sizes[n] = os.path.getsize(path)

        small, large = sorted(sizes)
        assert (small, large) == (200, 2400)
        grew_by = sizes[large] - sizes[small]
        expected = 8 * dim * (large - small)
        # Slack absorbs pickle's variable-width length prefixes only: measured
        # at 36 bytes of drift across n=100..500,000, so anything of the order
        # of a second per-row array would blow straight through it.
        assert abs(grew_by - expected) < 128, (
            f"payload grew by {grew_by} bytes for {large - small} extra rows "
            f"x {dim} dims; one float64 each is {expected}")

        fixed = sizes[small] - 8 * dim * small
        assert 0 < fixed < 64_000, (
            f"fixed (chain-independent) part of the payload is {fixed} bytes")


class TestSaveLoadRoundTrip:
    """A reloaded flow is the flow that was saved."""

    def test_reload_reproduces_the_flow_and_can_be_saved_again(self, small_flow):
        """Draws and densities both survive a reload, and so does a re-save.

        ``sample`` exercises the forward marginal maps and their tail clamps,
        ``log_prob`` the inverse maps and their log-determinant, so each can
        break without the other. Saving the *reloaded* flow again is what a
        user who loads, inspects and re-publishes a model does; it has to give
        the same draws, not drift a little on every cycle.
        """
        original_samples = sample(small_flow, n_samples=100, rng_seed=999)
        test_data = np.random.default_rng(1).standard_normal((50, 2))
        original_log_probs = log_prob(small_flow, test_data)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "flow.pkl")
            save_flow(small_flow, path)
            loaded = load_flow(path)

            np.testing.assert_array_almost_equal(
                original_samples, sample(loaded, n_samples=100, rng_seed=999))
            # numerical precision may differ slightly on the inverse side
            np.testing.assert_array_almost_equal(
                original_log_probs, log_prob(loaded, test_data), decimal=5)

            resaved = Path(tmpdir) / "resaved.pkl"
            save_flow(loaded, resaved)
            np.testing.assert_array_almost_equal(
                original_samples,
                sample(load_flow(resaved), n_samples=100, rng_seed=999))

    def test_round_trip_preserves_tail_alpha(self, tail_alpha_flow):
        """A non-default tail_alpha survives save/load and the reloaded flow
        reproduces the original samples exactly."""
        assert _extract_spline_data(tail_alpha_flow)['tail_alpha'] == [0.5, 0.5]

        original_samples = sample(tail_alpha_flow, n_samples=100, rng_seed=999)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "flow.pkl"
            save_flow(tail_alpha_flow, path)
            loaded_flow = load_flow(path)
        assert [float(unwrap(b.tail_alpha))
                for b in loaded_flow.bijection.bijections] == [0.5, 0.5]
        loaded_samples = sample(loaded_flow, n_samples=100, rng_seed=999)
        np.testing.assert_array_almost_equal(original_samples, loaded_samples)


class TestOldFormatFilesAreRefused:
    """Copula-flow files below save format 2 are refused, naming the way out.

    load_flow rebuilds the marginals from the saved training columns with the
    current builder, so an older file would reload as a different density
    rather than fail. The real 1.0.0 file (no stamps, 'rqs' marginals) was
    checked by hand to unpickle and reach this refusal.
    """

    @staticmethod
    def _rewrite(path, edit):
        with open(path, "rb") as f:
            payload = cloudpickle.load(f)
        edit(payload)
        with open(path, "wb") as f:
            cloudpickle.dump(payload, f)

    def _assert_refused(self, path, version):
        with pytest.raises(ValueError, match=f"save format version {version}") as exc:
            load_flow(path)
        assert str(path) in str(exc.value)
        assert "coppuccino<2" in str(exc.value)

    def test_an_unstamped_file_is_refused_as_version_1(self, small_flow):
        """1.0.0 wrote neither stamp; a missing version reads as 1."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "flow.pkl"
            save_flow(small_flow, path)

            def strip_stamps(payload):
                for key in ("format_version", "kind", "writer_version"):
                    del payload[key]

            self._rewrite(path, strip_stamps)
            self._assert_refused(path, 1)

    def test_a_version_1_file_is_refused_even_with_smooth_marginals(self, small_flow):
        """Pre-release builds wrote version 1 with smooth marginals whose
        builder has since changed; they are refused too, not rebuilt."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "flow.pkl"
            save_flow(small_flow, path)
            self._rewrite(path, lambda p: p.update(format_version=1))
            self._assert_refused(path, 1)

    def test_a_version_2_file_is_refused(self, small_flow):
        """Version 2 came from pre-release builds of 2.0.0.

        Their marginal maps were compiled with the spline and tail constants
        baked into the program, and XLA rounds some operations differently
        that way (a division by a constant becomes a multiplication by its
        reciprocal), so near the walls and deep in the tails a reload moved
        by up to a few hundred float64 ulps. That is small, but a reload is
        meant to reproduce the fitted density exactly, so they are refused
        rather than rebuilt.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "flow.pkl"
            save_flow(small_flow, path)
            self._rewrite(path, lambda p: p.update(format_version=2))
            self._assert_refused(path, 2)

    def test_a_unit_cube_file_is_refused(self):
        """The pre-release unit-cube pipeline's files are version 1 too.

        They also lack 'inner_flow_like', so this pins that the version
        refusal comes before the missing-key check: the caller gets the way
        out (refit, or pin coppuccino<2), not a complaint about a key.
        """
        payload = {
            'spline_data': {'num_dims': 0},
            'inner_flow_bytes': b'',
            'inner_arch': ('uniform', 2, 4, 6),
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "flow.pkl"
            with open(path, 'wb') as f:
                cloudpickle.dump(payload, f)
            self._assert_refused(path, 1)


class _FullDisk:
    """A cloudpickle stand-in whose serialization fails with ENOSPC.

    ``dump`` writes half the payload before failing, which is what a writer
    that opens the destination directly leaves behind; ``dumps`` fails before
    producing any bytes at all. Injecting the same fault at both seams keeps
    the test independent of which call `save_flow` happens to use.
    """

    @staticmethod
    def dump(obj, f, *args, **kwargs):
        blob = cloudpickle.dumps(obj, *args, **kwargs)
        f.write(blob[: len(blob) // 2])
        raise OSError(28, "No space left on device")

    @staticmethod
    def dumps(obj, *args, **kwargs):
        raise OSError(28, "No space left on device")


class TestAtomicSave:
    """A failed save must not destroy the model already on disk."""

    def test_failed_save_leaves_previous_model_intact(self, small_flow, monkeypatch):
        """A save that fails leaves the old file untouched and no temp file
        behind, whether it fails while serializing or while writing.

        The serialization fault is the one the old writer died on: it opened
        the destination 'wb' first, so a half-written payload replaced a good
        model. The fsync fault lands later, once the temporary sibling exists
        and holds the whole payload, so only the cleanup in _atomic_write_bytes
        stands between it and a stray ``flow.pkl.<random>.tmp``, and only the
        rename-on-success stands between it and an overwritten destination.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "flow.pkl"
            save_flow(small_flow, path)
            # Mark the previous file so that it differs from the payload the
            # failing save would write; otherwise a writer that went straight
            # to the destination would leave identical bytes behind.
            payload = cloudpickle.loads(path.read_bytes())
            payload["previous"] = True
            path.write_bytes(cloudpickle.dumps(payload))
            before = path.read_bytes()

            with monkeypatch.context() as m:
                m.setattr(model_io, "cloudpickle", _FullDisk)
                with pytest.raises(OSError):
                    save_flow(small_flow, path)
            assert path.read_bytes() == before
            assert sorted(os.listdir(tmpdir)) == ["flow.pkl"]

            def disk_full(fd):
                raise OSError(28, "No space left on device")

            with monkeypatch.context() as m:
                m.setattr(model_io.os, "fsync", disk_full)
                with pytest.raises(OSError) as excinfo:
                    save_flow(small_flow, path)
            assert path.read_bytes() == before
            assert sorted(os.listdir(tmpdir)) == ["flow.pkl"]
            message = str(excinfo.value)
            assert str(path) in message
            assert ".tmp" not in message

            assert isinstance(load_flow(path), Transformed)

    def test_overwrite_preserves_file_permissions(self, small_flow):
        """Publishing through a temp file must not tighten the mode: mkstemp
        creates 0o600, so the destination's own permissions are reapplied."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "flow.pkl"
            save_flow(small_flow, path)
            os.chmod(path, 0o640)

            save_flow(small_flow, path)

            assert stat.S_IMODE(os.stat(path).st_mode) == 0o640

    def test_a_new_file_gets_the_umask_mode(self):
        """A file that did not exist yet is published with the mode a plain
        ``open(path, 'wb')`` would give it under the current umask.

        Without the chmod it would keep the owner-only 0o600 that mkstemp gives
        the temporary sibling, silently hiding a new model from the rest of a
        shared model directory. 0o027 is chosen so the expected 0o640 differs
        from both 0o600 and the common 0o644.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "payload.bin"
            previous = os.umask(0o027)
            try:
                _atomic_write_bytes(path, b"new")
            finally:
                os.umask(previous)

            assert stat.S_IMODE(os.stat(path).st_mode) == 0o640

    def test_atomic_write_bytes_replaces_content_wholesale(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "payload.bin"
            path.write_bytes(b"old contents, longer than the new ones")

            _atomic_write_bytes(path, b"new")

            assert path.read_bytes() == b"new"
            assert sorted(os.listdir(tmpdir)) == ["payload.bin"]

    def test_atomic_write_bytes_keeps_the_old_file_on_failure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "payload.bin"
            path.write_bytes(b"original")

            # A str is rejected by the underlying binary write, so the failure
            # lands after the temp file exists but before publication.
            with pytest.raises(TypeError):
                _atomic_write_bytes(path, "not bytes")

            assert path.read_bytes() == b"original"
            assert sorted(os.listdir(tmpdir)) == ["payload.bin"]

    def test_symlinked_destination_updates_the_target(self):
        """os.replace would swap out the link itself, where the direct
        open(path, 'wb') it replaced wrote *through* it -- so a published
        model would stop being visible at the shared store it points into."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = Path(tmpdir) / "store"
            store.mkdir()
            target = store / "model.bin"
            target.write_bytes(b"original")
            link = Path(tmpdir) / "latest.bin"
            link.symlink_to(target)

            _atomic_write_bytes(link, b"published")

            assert link.is_symlink()
            assert os.readlink(link) == str(target)
            assert target.read_bytes() == b"published"
            # the temp file belongs next to the target, not next to the link
            assert sorted(os.listdir(store)) == ["model.bin"]


class TestWriteErrorsNameTheDestination:
    """Every publication failure must quote the path the caller passed, never
    the ``<name>.<random>.tmp`` sibling it cannot find again."""

    def test_missing_parent_names_the_destination(self):
        """The parent is not created either: a missing directory is a clear,
        actionable error rather than a silently created directory tree."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nope" / "flow.pkl"
            with pytest.raises(FileNotFoundError) as excinfo:
                _atomic_write_bytes(path, b"payload")

            message = str(excinfo.value)
            assert str(path) in message
            assert ".tmp" not in message
            assert not path.parent.exists()

    def test_directory_destination_fails_before_writing_anything(self, monkeypatch):
        """A directory destination can never be published, so it is caught up
        front rather than after the whole payload has been written to a temp
        sibling that os.replace then fails on."""
        def _refuse(*args, **kwargs):
            raise AssertionError("mkstemp must not run for a directory destination")

        monkeypatch.setattr(model_io.tempfile, "mkstemp", _refuse)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "adir"
            path.mkdir()
            with pytest.raises(IsADirectoryError) as excinfo:
                _atomic_write_bytes(path, b"payload")

            message = str(excinfo.value)
            assert str(path) in message
            assert ".tmp" not in message
            assert sorted(os.listdir(tmpdir)) == ["adir"]

    @pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0,
                        reason="root ignores directory write permissions")
    def test_unwritable_directory_names_the_destination(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            locked = Path(tmpdir) / "locked"
            locked.mkdir()
            path = locked / "flow.pkl"
            path.write_bytes(b"original")
            os.chmod(locked, 0o500)
            try:
                with pytest.raises(PermissionError) as excinfo:
                    _atomic_write_bytes(path, b"payload")
            finally:
                os.chmod(locked, 0o700)

            message = str(excinfo.value)
            assert str(path) in message
            assert ".tmp" not in message
            assert path.read_bytes() == b"original"

    def test_empty_path_is_rejected_without_touching_the_cwd(self, monkeypatch):
        """Path('') is Path('.'), whose parent is a directory too, so the
        write used to land a temp file in the current directory and fail with
        EINVAL on a path the caller never wrote."""
        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.chdir(tmpdir)
            with pytest.raises(ValueError, match="not a file"):
                _atomic_write_bytes("", b"payload")
            assert sorted(os.listdir(".")) == []


class TestSaveFlowArgumentChecks:
    """save_flow validates `flow` instead of dying on an internal attribute."""

    # Every non-flow reaches the same isinstance check; an array (the
    # historical failure) and None are enough to pin the message.
    @pytest.mark.parametrize("bad", [np.zeros((10, 2)), None])
    def test_non_flow_raises_typeerror(self, bad):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "flow.pkl"
            with pytest.raises(TypeError) as excinfo:
                save_flow(bad, path)

            message = str(excinfo.value)
            assert "save_flow" in message
            assert type(bad).__name__ in message
            # the old failure was an AttributeError about .bijection
            assert "bijection" not in message
            assert sorted(os.listdir(tmpdir)) == []


class TestLoadFlowFailureMessages:
    """Unreadable files must fail by name, not by internal key."""

    def _write(self, tmpdir, payload_bytes):
        path = Path(tmpdir) / "flow.pkl"
        path.write_bytes(payload_bytes)
        return path

    def test_missing_payload_key_names_the_file_and_the_key(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # a flow-shaped payload with one key removed: used to raise a bare
            # KeyError: 'inner_flow_like'
            path = self._write(tmpdir, cloudpickle.dumps({
                "format_version": FORMAT_VERSION,
                "kind": FLOW_KIND,
                "spline_data": {"num_dims": 0},
                "inner_flow_bytes": b"",
            }))
            with pytest.raises(ValueError) as excinfo:
                load_flow(path)

            message = str(excinfo.value)
            assert str(path) in message
            assert "inner_flow_like" in message

    def test_truncated_file_names_the_file(self):
        good = cloudpickle.dumps({
            "spline_data": {"num_dims": 0},
            "inner_flow_bytes": b"",
            "inner_flow_like": None,
        })
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write(tmpdir, good[: len(good) // 2])
            with pytest.raises(ValueError) as excinfo:
                load_flow(path)
            assert str(path) in str(excinfo.value)

    def test_garbage_file_names_the_file(self):
        """A file that is not a pickle at all goes through the same guard as a
        truncated one."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write(tmpdir, b"not a pickle at all\n" * 4)
            with pytest.raises(ValueError) as excinfo:
                load_flow(path)
            assert str(path) in str(excinfo.value)

    def test_missing_file_still_raises_filenotfound(self):
        """The deserialization guard must not swallow a plain missing file."""
        with tempfile.TemporaryDirectory() as tmpdir, pytest.raises(FileNotFoundError):
            load_flow(Path(tmpdir) / "absent.pkl")


class TestFormatStamps:
    """Saved payloads carry a format version and a kind tag."""

    def test_saved_payload_carries_version_and_kind(self, small_flow):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "flow.pkl"
            save_flow(small_flow, path)
            with open(path, "rb") as f:
                payload = cloudpickle.load(f)

            assert payload["format_version"] == FORMAT_VERSION
            assert payload["kind"] == FLOW_KIND

    def test_saved_payload_records_the_writer_version(self, small_flow):
        """A pickled model outlives its session; the payload must say which
        install wrote it, without that becoming a compatibility gate."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "flow.pkl"
            save_flow(small_flow, path)
            with open(path, "rb") as f:
                payload = cloudpickle.load(f)

            assert payload["writer_version"] == model_io._writer_version()
            assert isinstance(payload["writer_version"], str)
            assert payload["writer_version"]

            # dropping it must not change what load_flow does: older files
            # have no such key
            del payload["writer_version"]
            with open(path, "wb") as f:
                cloudpickle.dump(payload, f)
            assert isinstance(load_flow(path), Transformed)

    def test_fleet_file_points_at_the_fleet_loader(self):
        """A fleet payload used to die on `KeyError: 'spline_data'`."""
        fleet_payload = {
            "arch": (2, 8, 6),
            "params_bytes": b"",
            "xk": np.zeros((1, 2, 3)),
            "yk": np.zeros((1, 2, 3)),
            "dk": np.zeros((1, 2, 3)),
            "lo": np.zeros((1, 2)),
            "hi": np.ones((1, 2)),
        }
        for stamped in (False, True):
            payload = dict(fleet_payload)
            if stamped:
                payload["format_version"] = FORMAT_VERSION
                payload["kind"] = FLEET_KIND
            with tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir) / "fleet.pkl"
                with open(path, "wb") as f:
                    cloudpickle.dump(payload, f)
                with pytest.raises(ValueError, match="FleetFlow.load"):
                    load_flow(path)

    def test_future_format_version_is_rejected(self, small_flow):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "flow.pkl"
            save_flow(small_flow, path)
            with open(path, "rb") as f:
                payload = cloudpickle.load(f)
            payload["format_version"] = FORMAT_VERSION + 1
            with open(path, "wb") as f:
                cloudpickle.dump(payload, f)

            with pytest.raises(ValueError, match="save format version"):
                load_flow(path)

    def test_unrecognised_payload_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "junk.pkl"
            with open(path, "wb") as f:
                cloudpickle.dump({"something": "else"}, f)
            with pytest.raises(ValueError, match="'unknown' model"):
                load_flow(path)

            with open(path, "wb") as f:
                cloudpickle.dump([1, 2, 3], f)
            with pytest.raises(ValueError, match="does not hold a coppuccino model"):
                load_flow(path)


class TestPickleTrustBoundary:
    """load_flow deserializes with cloudpickle, which executes code from the
    file; the docstring must say so."""

    def test_load_flow_documents_the_trust_boundary(self):
        doc = load_flow.__doc__
        assert ".. warning::" in doc
        lowered = doc.lower()
        for phrase in ("pickle", "arbitrary", "trust"):
            assert phrase in lowered
