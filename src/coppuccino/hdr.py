import numpy as np
from coppuccino.copula_flows import normalizing_flows_fit
from coppuccino.copula_flows import sample_and_log_prob, log_prob


def compute_injection_hdr(samples: np.ndarray, injection_params: np.ndarray, num_samples: int = 100_000, return_flow=False, **nf_kwargs):
    default_kwargs = {'knots':64, 'interval':16, 'patience':20, 'learning_rate':1e-4, 'max_epochs':400, 'nn_depth':16}
    kwargs = nf_kwargs if nf_kwargs else default_kwargs
    # fit NF to samples
    flow = normalizing_flows_fit(samples, **kwargs)  # TODO: document kwargs in docstring
    # sample from flow and compute log probability of those samples
    _, gen_log_probs = sample_and_log_prob(flow, n_samples=num_samples)
    injection_probs = log_prob(flow, injection_params)

    hdrs = []
    if injection_params.ndim == 0:
        raise ValueError("injection_params must be at least 1D")
    elif injection_params.ndim == 1:
        fraction = np.sum(gen_log_probs >= injection_probs) / num_samples
        hdrs.append(fraction)
    else:
        # import matplotlib.pyplot as plt
        for injection_prob in injection_probs:
            # plt.plot(gen_log_probs)
            # plt.axhline(injection_prob)
            # plt.show()
            fraction = np.sum(gen_log_probs >= injection_prob) / num_samples
            hdrs.append(fraction)
    if return_flow:
        return np.array(hdrs), flow

    return np.array(hdrs)
