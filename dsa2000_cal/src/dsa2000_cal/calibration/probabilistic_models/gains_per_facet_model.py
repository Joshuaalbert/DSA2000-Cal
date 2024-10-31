import dataclasses
from typing import Any, List

import jax
import numpy as np
import tensorflow_probability.substrates.jax as tfp
from astropy import time as at, coordinates as ac, units as au
from jax import numpy as jnp
from jaxns import Model

from dsa2000_cal.calibration.probabilistic_models.gain_prior_models import AbstractGainPriorModel
from dsa2000_cal.calibration.probabilistic_models.probabilistic_model import AbstractProbabilisticModel, \
    ProbabilisticModelInstance
from dsa2000_cal.common.jax_utils import promote_pytree
from dsa2000_cal.common.serialise_utils import SerialisableBaseModel
from dsa2000_cal.delay_models.far_field import VisibilityCoords
from dsa2000_cal.measurement_sets.measurement_set import  VisibilityData, MeasurementSet
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

        # jax.debug.print(
        #     '{time_idx}\n{antenna1}\n{antenna2}',
        #     time_idx=vis_coords.time_idx,
        #     antenna1=vis_coords.antenna_1, antenna2=vis_coords.antenna_2
        # )

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

        U = model.sample_U(jax.random.PRNGKey(0))
        if len(U) > 0:
            raise ValueError("This model has non-parametrised variables. Please update.")

        def get_init_params():
            return model.params

        def forward(params):
            return model(params).prepare_input(U=U)

        def log_prob_joint(params):
            return model(params).log_prob_joint(U=U, allow_nan=False)

        return ProbabilisticModelInstance(
            get_init_params_fn=get_init_params,
            forward_fn=forward,
            log_prob_joint_fn=log_prob_joint
        )

    def save_solution(self, solution: Any, file_name: str, times: at.Time, ms: MeasurementSet):
        solution: jax.Array = solution
        solution = jax.tree.map(np.asarray, solution)
        # Save to file
        data = CalibrationSolutions(
            gains=solution * au.dimensionless_unscaled,
            times=times,
            freqs=ms.meta.freqs,
            antennas=ms.meta.antennas,
            antenna_labels=ms.meta.antenna_names,
            pointings=ms.meta.pointings
        )
        with open(file_name, "w") as fp:
            fp.write(data.json(indent=2))


class CalibrationSolutions(SerialisableBaseModel):
    """
    Calibration solutions, stored in a serialisable format.
    """
    times: at.Time  # [time]
    pointings: ac.ICRS | None  # [[ant]]
    antennas: ac.EarthLocation  # [ant]
    antenna_labels: List[str]  # [ant]
    freqs: au.Quantity  # [chan]
    gains: au.Quantity  # [facet, time, ant, chan[, 2, 2]]
