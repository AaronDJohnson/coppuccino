"""coppuccino: fit distributions with copula normalizing flows in JAX.

.. note::
    Importing this package enables JAX 64-bit (float64) mode globally via
    ``jax.config.update('jax_enable_x64', True)``. The empirical-CDF transforms
    and log-determinant computations rely on float64 accuracy, but be aware that
    this is a process-wide setting: it also affects any other JAX code in the
    same session.
"""
import jax
jax.config.update('jax_enable_x64', True)

from coppuccino.copula_flows import (
    normalizing_flows_fit,
    sample,
    log_prob,
    sample_and_log_prob,
)
from coppuccino.model_io import save_flow, load_flow
from coppuccino.hdr import compute_injection_hdr, check_in_support

__all__ = [
    "normalizing_flows_fit",
    "sample",
    "log_prob",
    "sample_and_log_prob",
    "save_flow",
    "load_flow",
    "compute_injection_hdr",
    "check_in_support",
]
