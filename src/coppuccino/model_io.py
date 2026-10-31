"""
Model I/O utilities for saving and loading copula flows.

This module provides functions to serialize and deserialize copula flow models,
including the spline-based empirical CDF transforms that cannot be directly
serialized with standard equinox serialization.
"""

import contextlib
import io
import os
import stat
import tempfile
from importlib import metadata
from pathlib import Path
from typing import Union

import cloudpickle
import equinox as eqx
import numpy as np
from flowjax.bijections import Stack
from flowjax.distributions import Transformed
from paramax import non_trainable, unwrap

from coppuccino.bijections import SmoothMarginalToGaussian, make_smooth_marginal

__all__ = ["save_flow", "load_flow"]

# On-disk payload format. Purely additive fields do not bump this: they are
# read back through ``dict.get(key, <legacy default>)``, so an older writer's
# file still loads correctly. Bump only when an older coppuccino would
# silently MISREAD a newer file.
#
# A copula-flow file stores its training columns, and load_flow rebuilds the
# marginals from them with the current builder, so a change to that builder
# silently changes every reloaded flow (a file from one pre-release build,
# reloaded by a later one, moved its near-wall log_prob by 58 to 217 nats with
# no warning).
# Bump this whenever such a change lands, and raise _MIN_FLOW_FORMAT_VERSION
# with it.
FORMAT_VERSION = 2

# Oldest copula-flow format load_flow rebuilds. Version 2 is the first written
# by the smooth-only marginal builder; everything older came from coppuccino
# 1.0.0 (whose 'rqs' / 'pchip' marginals no longer exist) or from a
# pre-release build. Fleet files keep their own rule (see FleetFlow.load).
_MIN_FLOW_FORMAT_VERSION = 2

# ``kind`` tags. The two writers in this package store disjoint key sets, so a
# cross-kind load can never succeed; tagging lets it fail by name instead of on
# whichever internal dict key happens to be read first.
FLOW_KIND = 'copula-flow'
FLEET_KIND = 'fleet'

_LOADER_FOR_KIND = {
    FLOW_KIND: 'coppuccino.load_flow',
    FLEET_KIND: 'coppuccino.FleetFlow.load',
}


def _writer_version() -> str:
    """
    Version of the coppuccino that is writing a payload.

    Returns
    -------
    str
        The installed distribution version, or ``'unknown'`` when the package
        metadata is absent (running from a source tree that was never
        installed).

    Notes
    -----
    Provenance only: no loader branches on it. Compatibility is decided by
    ``format_version``, which says what the reader must understand; the writer
    version says which install produced the file, which is what a bug report
    about a specific ``.pkl`` needs.
    """
    try:
        return metadata.version("coppuccino")
    except metadata.PackageNotFoundError:
        return "unknown"


def _publication_mode(path: Path) -> int:
    """
    Permission bits a newly published file at ``path`` should carry.

    Parameters
    ----------
    path : Path
        Destination path, which may or may not exist yet.

    Returns
    -------
    int
        The existing file's mode, or the mode a plain ``open(path, 'wb')``
        would have produced.
    """
    try:
        return stat.S_IMODE(os.stat(path).st_mode)
    except OSError:
        # os.umask is the only way to read the umask, so set-and-restore.
        umask = os.umask(0o022)
        os.umask(umask)
        return 0o666 & ~umask


def _as_destination_error(exc: OSError, path: Path) -> OSError:
    """
    Rebuild an ``OSError`` so that it names the destination, not the temp file.

    Parameters
    ----------
    exc : OSError
        The error raised while writing or publishing the temporary sibling.
    path : Path
        Destination the caller asked for.

    Returns
    -------
    OSError
        An error of the same class carrying the same ``errno`` and the original
        text, prefixed with the destination.

    Notes
    -----
    Left alone, every filesystem error in the publication sequence quotes the
    ``<name>.<random>.tmp`` sibling, which the caller never named and -- once
    the cleanup below has run -- can no longer find, so a read-only directory
    reads as a failure on a path that does not exist.
    """
    detail = exc.strerror or str(exc)
    return type(exc)(exc.errno, f"cannot write {path}: {detail}")


def _atomic_write_bytes(path: Union[str, Path], data: bytes) -> None:
    """
    Write ``data`` to ``path`` so that the destination is never left partial.

    The bytes go to a temporary file alongside the destination and are moved
    into place with :func:`os.replace`, which is atomic within a filesystem.
    Opening the destination directly would truncate the previous file on entry,
    so an interrupt or a full disk part-way through would destroy a good model
    and leave an unloadable one behind.

    Parameters
    ----------
    path : str or Path
        Destination path. Its parent directory must already exist.
    data : bytes
        Complete file contents.

    Raises
    ------
    ValueError
        If ``path`` names no file at all (``''``, ``'.'``, ``'/'``).
    IsADirectoryError
        If ``path`` is an existing directory.
    OSError
        For any other filesystem failure. The message always names ``path``.

    Notes
    -----
    Intended for the writers in this package (``save_flow`` and
    ``FleetFlow.save``) so they share one publication policy.
    """
    requested = path
    path = Path(path)
    # os.replace swaps out a symlink itself, where the previous
    # open(path, 'wb') wrote *through* it. Resolve first so a destination that
    # is a link into a shared model store keeps updating the store.
    if path.is_symlink():
        path = Path(os.path.realpath(path))
    if not path.name:
        # Path('') is Path('.'), whose parent is also a directory, so without
        # this the temp file is created in the *current* directory and only
        # os.replace fails -- with EINVAL on a path the caller never wrote.
        raise ValueError(
            f"cannot write {requested!r}: it names the directory {path}, "
            "not a file.")
    if path.is_dir():
        # os.replace would report this against the temp sibling instead.
        raise IsADirectoryError(
            f"cannot write {path}: it is an existing directory.")
    if not path.parent.is_dir():
        # mkstemp would raise here too, but naming its random sibling instead
        # of the destination the caller passed is no help to anyone.
        raise FileNotFoundError(
            f"cannot write {path}: the parent directory {path.parent} does "
            "not exist.")
    mode = _publication_mode(path)
    try:
        fd, tmp_path = tempfile.mkstemp(
            prefix=path.name + '.', suffix='.tmp', dir=path.parent
        )
    except OSError as exc:
        # An unwritable parent directory fails here, before there is anything
        # to clean up.
        raise _as_destination_error(exc, path) from None
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(data)
            f.flush()
            # Without the fsync the rename can be durable before the contents
            # are, which after a crash yields an empty or short file.
            os.fsync(f.fileno())
        # mkstemp creates owner-only files; publish with the permissions the
        # destination already had, so atomicity does not silently tighten
        # access for a shared model directory.
        os.chmod(tmp_path, mode)
        os.replace(tmp_path, path)
    except BaseException as exc:
        # The destination still holds whatever was there before; only the
        # half-written sibling needs cleaning up.
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        if isinstance(exc, OSError):
            raise _as_destination_error(exc, path) from None
        raise


def _infer_legacy_kind(save_dict: dict) -> str:
    """
    Identify a payload written before the ``kind`` stamp existed.

    Parameters
    ----------
    save_dict : dict
        Deserialized payload.

    Returns
    -------
    str
        ``FLOW_KIND``, ``FLEET_KIND``, or ``'unknown'``.
    """
    if 'spline_data' in save_dict:
        return FLOW_KIND
    if 'arch' in save_dict and 'params_bytes' in save_dict:
        return FLEET_KIND
    return 'unknown'


def _check_payload(save_dict: object, path: Union[str, Path],
                   expected_kind: str) -> None:
    """
    Validate the format version and kind stamps of a deserialized payload.

    Parameters
    ----------
    save_dict : object
        Whatever the pickle contained; anything other than a dict is rejected.
    path : str or Path
        Source path, quoted in the error messages.
    expected_kind : str
        The kind the calling loader can rebuild, ``FLOW_KIND`` or
        ``FLEET_KIND``.

    Raises
    ------
    ValueError
        If the payload is not a dict, was written in a newer format version, or
        holds a different kind of model than the caller handles.

    Notes
    -----
    A missing ``format_version`` is read as version 1: every file written
    before the stamp existed lacks the key, so a strict check would reject them
    all. A missing ``kind`` is inferred from the payload's own keys for the
    same reason.
    """
    if not isinstance(save_dict, dict):
        raise ValueError(
            f"{path} does not hold a coppuccino model "
            f"(found a {type(save_dict).__name__})."
        )

    version = save_dict.get('format_version', 1)
    try:
        version = int(version)
    except (TypeError, ValueError):
        raise ValueError(
            f"{path} has a malformed 'format_version' ({version!r}); "
            "the file is not a readable coppuccino model."
        ) from None
    if version > FORMAT_VERSION:
        raise ValueError(
            f"{path} was written in save format version {version}, but this "
            f"coppuccino reads up to version {FORMAT_VERSION}. Upgrade "
            "coppuccino to load it."
        )

    kind = save_dict.get('kind') or _infer_legacy_kind(save_dict)
    if kind != expected_kind:
        loader = _LOADER_FOR_KIND.get(kind)
        hint = f" Load it with {loader}." if loader else ""
        raise ValueError(
            f"{path} holds a '{kind}' model, but "
            f"{_LOADER_FOR_KIND[expected_kind]} loads '{expected_kind}' "
            f"models.{hint}"
        )


def _extract_spline_data(flow: Transformed) -> dict:
    """
    Extract spline reconstruction data from a fitted copula flow.

    Parameters
    ----------
    flow : Transformed
        A fitted copula flow with smooth-tail marginal transforms.

    Returns
    -------
    dict
        Everything needed to rebuild the marginals, one list entry per
        dimension: 'samples', 'num_points', 'prior_low', 'prior_high',
        'tail_alpha' and 'marginal' (always 'smooth'), plus 'num_dims'.
    """
    # The flow structure is: Transformed(inner_flow, Stack([SmoothMarginalToGaussian, ...]))
    transform = flow.bijection

    spline_data: dict = {
        'samples': [],
        'num_points': [],
        'prior_low': [],
        'prior_high': [],
        'marginal': [],
        'tail_alpha': [],
        'num_dims': 0
    }

    if not isinstance(transform, Stack):
        raise ValueError(f"Expected Stack transform, got: {type(transform)}")
    spline_data['num_dims'] = len(transform.bijections)
    for bij in transform.bijections:
        if not isinstance(bij, SmoothMarginalToGaussian):
            raise ValueError(f"Unexpected bijection type: {type(bij)}")
        # Fully determined by (samples, num_points, prior bounds, tail_alpha).
        spline_data['marginal'].append('smooth')
        spline_data['samples'].append(np.array(unwrap(bij.samples)))
        spline_data['num_points'].append(int(unwrap(bij.num_points)))
        p_low = unwrap(bij.prior_low)
        p_high = unwrap(bij.prior_high)
        spline_data['prior_low'].append(float(p_low) if p_low is not None else None)
        spline_data['prior_high'].append(float(p_high) if p_high is not None else None)
        spline_data['tail_alpha'].append(float(unwrap(bij.tail_alpha)))

    return spline_data


def _reconstruct_empirical_transforms(spline_data: dict) -> Stack:
    """
    Reconstruct the marginal transforms from saved spline data.

    Parameters
    ----------
    spline_data : dict
        Dictionary containing spline reconstruction data, as written by
        :func:`_extract_spline_data`.

    Returns
    -------
    Stack
        Stacked marginal transforms for each parameter dimension.
    """
    empirical_transforms = []

    for i in range(spline_data['num_dims']):
        samples = spline_data['samples'][i]
        num_points = spline_data['num_points'][i]
        prior_low = spline_data['prior_low'][i]
        prior_high = spline_data['prior_high'][i]
        tail_alpha = spline_data['tail_alpha'][i]

        theta_of_z, z_of_theta, log_dtheta_dz, x_clamps = make_smooth_marginal(
            samples, num_points=num_points,
            prior_low=prior_low, prior_high=prior_high,
            tail_alpha=tail_alpha,
        )
        # mypy folds flowjax's AbstractVar shape/cond_shape into the
        # synthesised __init__; see the ClassVar note on the class.
        empirical_transforms.append(non_trainable(SmoothMarginalToGaussian(  # type: ignore[call-arg]
            samples, theta_of_z, z_of_theta, log_dtheta_dz,
            num_points=num_points,
            prior_low=prior_low, prior_high=prior_high,
            tail_alpha=tail_alpha,
            x_clamp_lo=x_clamps[0], x_clamp_hi=x_clamps[1],
        )))

    return Stack(empirical_transforms)


def save_flow(flow: Transformed, path: Union[str, Path]) -> None:
    """
    Save a copula flow model to disk.

    This function saves both the flow parameters and the spline data needed
    to reconstruct the empirical CDF transforms. The saved file can be loaded
    with `load_flow`.

    Parameters
    ----------
    flow : Transformed
        A fitted copula flow model from `normalizing_flows_fit`.
    path : str or Path
        Path to save the model. Recommended extension is '.pkl'. The parent
        directory must already exist; it is not created.

    Raises
    ------
    TypeError
        If `flow` is not a fitted copula flow. A `FleetFlow` writes its own
        payload format via `fleet.save(path)` and reads it back with
        `FleetFlow.load(path)`.

    Notes
    -----
    The file is published atomically: the payload is written to a temporary
    file in the same directory and then renamed over `path`. An interrupted or
    failed save therefore leaves any previously saved model at `path` intact
    rather than truncating it.

    File size scales with the training set. Each fitted marginal keeps the
    training column it was built from, because the marginal maps are rebuilt
    at load time by re-running the knot construction on those samples (see
    `_reconstruct_empirical_transforms`); nothing smaller reproduces them
    bit-for-bit. The payload is therefore one float64 per training row per
    dimension on top of a fixed part that does not grow with the chain.
    Measured with the default architecture on standard-normal chains
    (3 dimensions, `max_epochs=3`, one fit per size): 0.147 MB for 5,000 rows,
    1.227 MB for 50,000 and 12.027 MB for 500,000. Each is ``24 * n_rows``
    bytes plus a remainder of 26.6 kB that moved by 36 bytes in total across
    n = 100 to 500,000; of that remainder only 7.2 kB is the serialized flow
    leaves, the rest being the pickled tree skeleton.

    Load cost splits the same way, and the fixed part dominates until the
    chain is large. The wall times below are from one machine (arm64, JAX CPU
    backend) and only the shape of the scaling should be read across
    hardware. Warm (same process, 7 repeat loads per file, medians over
    3 independent runs of the sweep above) `load_flow` took 7-8 ms for both a
    100-row and a 5,000-row file -- the two are indistinguishable, being a
    floor of pickle and knot-construction work -- rising to 11-13 ms at 50,000
    rows and 65-72 ms at 500,000, i.e. roughly linear in rows above the floor.
    The *first* load in a fresh process pays a one-off cost on top of that,
    measured at 316-374 ms and the same at every size, so a cold load of
    anything under ~10^5 rows is essentially all overhead. Do not read the two
    small warm figures as a size effect; they are the floor.

    Sampling and density evaluation are unaffected -- the retained columns are
    never read after reconstruction. Thin the chain before fitting if the file
    size matters.

    Examples
    --------
    >>> import numpy as np
    >>> from coppuccino.copula_flows import normalizing_flows_fit
    >>> from coppuccino.model_io import save_flow, load_flow
    >>> # Fit a flow
    >>> data = np.random.randn(1000, 3)
    >>> flow = normalizing_flows_fit(data, max_epochs=50)
    >>> # Save the flow
    >>> save_flow(flow, "my_flow.pkl")
    >>> # Load it back
    >>> loaded_flow = load_flow("my_flow.pkl")
    """
    # Deferred so this module keeps ``coppuccino.bijections`` as its only
    # module-scope coppuccino dependency, which is what lets ``coppuccino.fleet``
    # import the payload helpers above at module scope without a cycle.
    from coppuccino.copula_flows import _reject_fleet_flow

    _reject_fleet_flow(flow, "save_flow", "fleet.save(path)")

    # Anything else that is not a fitted flow used to reach _extract_spline_data
    # and die on "'numpy.ndarray' object has no attribute 'bijection'", naming
    # an attribute the caller never asked about.
    if not isinstance(flow, Transformed):
        raise TypeError(
            f"save_flow() takes a fitted copula flow (a flowjax Transformed) "
            f"as returned by normalizing_flows_fit; flow is a "
            f"{type(flow).__name__}."
        )

    path = Path(path)

    # Extract spline data for reconstruction
    spline_data = _extract_spline_data(flow)

    # Get the inner flow (the actual normalizing flow without the empirical transform)
    inner_flow = flow.base_dist

    # Serialize the inner flow using equinox to a bytes buffer
    inner_flow_buffer = io.BytesIO()
    eqx.tree_serialise_leaves(inner_flow_buffer, inner_flow)
    inner_flow_bytes = inner_flow_buffer.getvalue()

    save_dict = {
        'format_version': FORMAT_VERSION,
        'kind': FLOW_KIND,
        # Provenance for the file, not a compatibility gate: nothing reads it
        # back. A pickled model outlives the session that made it, and without
        # this there is no way to tell which install wrote a given .pkl.
        'writer_version': _writer_version(),
        'spline_data': spline_data,
        'inner_flow_bytes': inner_flow_bytes,
    }

    save_dict['inner_flow_like'] = inner_flow  # Keep structure for deserialization

    # cloudpickle (a hard dependency) serializes the JAX/equinox objects.
    # Serializing to bytes first keeps the destination untouched until the
    # whole payload exists.
    _atomic_write_bytes(path, cloudpickle.dumps(save_dict))


def _read_payload(path, writer: str):
    """Unpickle a saved model file, naming ``path`` if it is not one.

    ``writer`` is the save method the file should have come from, quoted in the
    error. ``open`` is outside the guard: its own errors (missing file,
    directory, permissions) already name the path.
    """
    with open(path, 'rb') as f:
        try:
            return cloudpickle.load(f)
        except Exception as exc:
            # A truncated or non-pickle file surfaces as UnpicklingError,
            # EOFError or UnicodeDecodeError, none of which mention the file
            # being read -- unhelpful when the caller is walking a directory.
            raise ValueError(
                f"{path} could not be deserialized as a coppuccino model "
                f"({type(exc).__name__}: {exc}). The file is truncated or was "
                f"not written by {writer}."
            ) from exc


def load_flow(path: Union[str, Path]) -> Transformed:
    """
    Load a copula flow model from disk.

    This function loads a flow that was saved with `save_flow`, reconstructing
    both the flow parameters and the spline-based empirical CDF transforms.

    .. warning::
        The saved format is a Python pickle. Loading it executes arbitrary
        code contained in the file, so only load models from a source you
        trust. Never load a model received over a network or from an untrusted
        third party. This is the same caveat that applies to `pickle`, `joblib`
        and `torch.load`.

    Parameters
    ----------
    path : str or Path
        Path to the saved model file.

    Returns
    -------
    Transformed
        The loaded copula flow model, ready for sampling and density evaluation.

    Raises
    ------
    FileNotFoundError
        If `path` does not exist.
    ValueError
        If the file cannot be deserialized at all (truncated, or not a pickle),
        holds a different kind of coppuccino model (for example a fleet, saved
        by `FleetFlow.save`), was written in a newer save format than this
        install understands, was written by coppuccino 1.0.0 or a pre-release
        build (save format below 2: refit the chain, or pin coppuccino<2), or
        is missing keys `save_flow` always writes. Every message quotes
        `path`.

    Examples
    --------
    >>> from coppuccino.model_io import load_flow
    >>> from coppuccino.copula_flows import sample
    >>> # Load a previously saved flow
    >>> flow = load_flow("my_flow.pkl")
    >>> # Generate samples
    >>> samples = sample(flow, n_samples=100, rng_seed=42)
    """
    path = Path(path)
    save_dict = _read_payload(path, "save_flow")
    _check_payload(save_dict, path, FLOW_KIND)

    # _check_payload has already parsed the stamp, so this cast cannot fail.
    version = int(save_dict.get('format_version', 1))
    if version < _MIN_FLOW_FORMAT_VERSION:
        raise ValueError(
            f"{path} was written in save format version {version}, by "
            "coppuccino 1.0.0 or a pre-release build. Its marginals were built "
            "by code this coppuccino no longer has, so reloading it would not "
            "reproduce the fitted density. Refit the chain, or pin "
            "coppuccino<2 to read a file written by 1.0.0."
        )

    # Every save_flow payload since the first release carries all three, so a
    # gap means a hand-built or partially rewritten file rather than an old
    # one; without this the first read raised a bare KeyError naming neither
    # the file nor what it is missing.
    missing = [key for key in ('spline_data', 'inner_flow_bytes',
                               'inner_flow_like') if key not in save_dict]
    if missing:
        raise ValueError(
            f"{path} is missing the required payload key(s) "
            f"{', '.join(repr(key) for key in missing)}; it is not a complete "
            "copula-flow file written by save_flow."
        )

    spline_data = save_dict['spline_data']

    # Deserialize the inner flow
    inner_flow_bytes = save_dict['inner_flow_bytes']
    inner_flow_like = save_dict['inner_flow_like']

    inner_flow_buffer = io.BytesIO(inner_flow_bytes)
    inner_flow = eqx.tree_deserialise_leaves(inner_flow_buffer, inner_flow_like)

    # Reconstruct the empirical transforms with working splines
    transform = _reconstruct_empirical_transforms(spline_data)

    return Transformed(inner_flow, transform)
