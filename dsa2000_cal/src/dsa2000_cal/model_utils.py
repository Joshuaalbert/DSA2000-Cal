from functools import partial

import jax
import tensorflow_probability.substrates.jax as tfp

from dsa2000_fm.actors.average_utils import average_rule

tfpd = tfp.distributions


@partial(jax.jit, static_argnames=['Tm', 'Cm'])
def average_model(vis_model, Tm: int, Cm: int):
    """
    Averages model vis data.

    Args:
        vis_model: [D, T, B, C, ...]
        Tm: number of model times
        Cm: number of model channels

    Returns:
        [D, Tm, B, Cm, ...]
    """
    # average data to match model: [Ts, B, Cs[, 2, 2]] -> [Tm, B, Cm[, 2, 2]]
    if Tm is not None:
        time_average_rule = partial(
            average_rule,
            num_model_size=Tm,
            axis=1
        )
    else:
        time_average_rule = lambda x: x
    if Cm is not None:
        freq_average_rule = partial(
            average_rule,
            num_model_size=Cm,
            axis=3
        )
    else:
        freq_average_rule = lambda x: x
    vis_model_avg = time_average_rule(freq_average_rule(vis_model))
    return vis_model_avg
