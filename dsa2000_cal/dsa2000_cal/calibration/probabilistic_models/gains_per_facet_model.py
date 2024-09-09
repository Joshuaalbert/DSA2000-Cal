import dataclasses
from typing import Any

import astropy.time as at
import jax
import numpy as np
import tensorflow_probability.substrates.jax as tfp
from jax import numpy as jnp
from jaxns import Model
from jaxns.framework import context as ctx
from jaxns.framework import ops
from jaxns.internals.constraint_bijections import quick_unit, quick_unit_inverse

from dsa2000_cal.calibration.probabilistic_models.gain_prior_models import AbstractGainPriorModel
from dsa2000_cal.calibration.probabilistic_models.probabilistic_model import AbstractProbabilisticModel, \
    ProbabilisticModelInstance
from dsa2000_cal.common.jax_utils import promote_pytree
from dsa2000_cal.delay_models.far_field import VisibilityCoords
from dsa2000_cal.measurement_sets.measurement_set import VisibilityData, MeasurementSet
from dsa2000_cal.types import CalibrationSolutions
from dsa2000_cal.visibility_model.rime_model import RIMEModel

tfpd = tfp.distributions


@dataclasses.dataclass(eq=False)
class GainsPerFacet(AbstractProbabilisticModel):
    rime_model: RIMEModel
    gain_prior_model: AbstractGainPriorModel

    def create_model_instance(self,
                              freqs: jax.Array,
                              times: jax.Array,
                              vis_data: VisibilityData,
                              vis_coords: VisibilityCoords
                              ) -> ProbabilisticModelInstance:
        model_data = self.rime_model.get_model_data(
            times=times
        )  # [facets]

        vis = self.rime_model.predict_visibilities(
            model_data=model_data,
            visibility_coords=vis_coords
        )  # [num_cal, num_row, num_chan[, 2, 2]]

        # vis now contains the model visibilities for each calibrator
        def prior_model():
            gain_prior_model = self.gain_prior_model.build_prior_model(
                num_source=self.rime_model.num_facets,
                num_ant=self.rime_model.num_antennas,
                freqs=freqs,
                times=times
            )
            gains: jax.Array = yield from gain_prior_model()  # [num_source, num_ant, num_chan[, 2, 2]]
            visibilities = self.rime_model.apply_gains(
                gains=gains,
                vis=vis,
                visibility_coords=vis_coords
            )  # [num_row, num_chan[, 2, 2]]
            return visibilities, gains

        def log_likelihood(vis_model: jax.Array, gains: jax.Array):
            """
            Compute the log probability of the data given the gains.

            Args:
                vis_model: [num_rows, num_chan, 4]
                gains: [num_source, num_ant, num_chan, 2, 2]

            Returns:
                log_prob: scalar
            """

            vis_variance = 1. / vis_data.weights  # Should probably use measurement set SIGMA here
            vis_stddev = jnp.sqrt(vis_variance)
            obs_dist_real = tfpd.Normal(
                *promote_pytree('vis_real', (jnp.real(vis_data.vis), vis_stddev))
            )
            obs_dist_imag = tfpd.Normal(
                *promote_pytree('vis_imag', (jnp.imag(vis_data.vis), vis_stddev))
            )
            log_prob_real = obs_dist_real.log_prob(jnp.real(vis_model))
            log_prob_imag = obs_dist_imag.log_prob(jnp.imag(vis_model))  # [num_rows, num_chan, 4]
            log_prob = log_prob_real + log_prob_imag  # [num_rows, num_chan, 4]

            # Mask out flagged data or zero-weighted data.
            mask = jnp.logical_or(vis_data.weights == 0, vis_data.flags)
            log_prob = jnp.where(mask, 0., log_prob)
            return jnp.sum(log_prob)

        model = Model(
            prior_model=prior_model,
            log_likelihood=log_likelihood
        )

        def get_init_params():
            # Could use model.sample_W() for a random start
            return jax.tree.map(quick_unit_inverse, model._W_placeholder())

        def forward(params):
            W = jax.tree.map(quick_unit, params)

            def _forward():
                # Use jaxns.framework.ops to transform the params into the args for likelihood
                return ops.prepare_input(W=W, prior_model=prior_model)

            return ctx.transform(_forward).apply(params=model.params, rng=jax.random.PRNGKey(0)).fn_val

        def log_prob_joint(params):
            W = jax.tree.map(quick_unit, params)

            def _log_prob_joint():
                # Use jaxns.framework.ops to compute the log prob of joint
                log_prob_prior = ops.compute_log_prob_prior(
                    W=W,
                    prior_model=model.prior_model
                )
                log_prob_likelihood = ops.compute_log_likelihood(
                    W=W,
                    prior_model=model.prior_model,
                    log_likelihood=model.log_likelihood,
                    allow_nan=False
                )
                return log_prob_prior + log_prob_likelihood

            return ctx.transform(_log_prob_joint).apply(params=model.params, rng=jax.random.PRNGKey(0)).fn_val

        return ProbabilisticModelInstance(
            get_init_params_fn=get_init_params,
            forward_fn=forward,
            log_prob_joint_fn=log_prob_joint
        )

    def save_solution(self, solution: Any, file_name: str, times: at.Time, ms: MeasurementSet):
        # Save to file
        solution = CalibrationSolutions(
            gains=np.asarray(solution),
            times=times,
            freqs=ms.meta.freqs,
            antennas=ms.meta.antennas,
            antenna_labels=ms.meta.antenna_labels,
            pointings=ms.meta.pointings
        )
        with open(file_name, "w") as fp:
            fp.write(solution.json(indent=2))
