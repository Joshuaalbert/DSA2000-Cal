import dataclasses
from typing import Any, List

import jax
import numpy as np
import tensorflow_probability.substrates.jax as tfp
from astropy import time as at, coordinates as ac, units as au
from jax import numpy as jnp
from jaxns import Model

from dsa2000_cal.calibration.probabilistic_models.probabilistic_model import AbstractProbabilisticModel, \
    ProbabilisticModelInstance
from dsa2000_cal.calibration.probabilistic_models.rfi_prior_models import AbstractRFIPriorModel
from dsa2000_cal.common.interp_utils import InterpolatedArray
from dsa2000_cal.common.jax_utils import promote_pytree
from dsa2000_cal.common.serialise_utils import SerialisableBaseModel
from dsa2000_cal.delay_models.far_field import VisibilityCoords
from dsa2000_cal.measurement_sets import VisibilityData, MeasurementSet
from dsa2000_cal.visibility_model.source_models.rfi.parametric_rfi_emitter import ParametricDelayACF
from dsa2000_cal.visibility_model.source_models.rfi.rfi_emitter_source_model import RFIEmitterModelData, \
    RFIEmitterPredict

tfpd = tfp.distributions


@dataclasses.dataclass(eq=False)
class HorizonRFIModel(AbstractProbabilisticModel):
    rfi_prior_model: AbstractRFIPriorModel
    rfi_predict: RFIEmitterPredict

    def create_model_instance(self, freqs: jax.Array,
                              times: jax.Array,
                              vis_data: VisibilityData,
                              vis_coords: VisibilityCoords
                              ) -> ProbabilisticModelInstance:
        def prior_model():
            rfi_prior_model = self.rfi_prior_model.build_prior_model(
                freqs=freqs,
                times=times
            )
            rfi_emitter_model_data: RFIEmitterModelData = yield from rfi_prior_model()  # [num_source, num_ant, num_chan[, 2, 2]]

            visibilities = self.rfi_predict.predict(
                model_data=rfi_emitter_model_data,
                visibility_coords=vis_coords
            )  # [num_row, num_chan[, 2, 2]]
            return visibilities, rfi_emitter_model_data

        def log_likelihood(vis_model: jax.Array, rfi_emitter_model_data: RFIEmitterModelData):
            """
            Compute the log probability of the data given the gains.

            Args:
                vis_model: [num_rows, num_chan, 4]
                rfi_emitter_model_data: [num_source, num_ant, num_chan, 2, 2]

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
        solution: RFIEmitterModelData = solution
        solution = jax.tree.map(np.asarray, solution)
        data = RFIEmitterSolutions(
            times=times,
            pointings=ms.meta.pointings,
            freqs=ms.meta.freqs,
            position_enu=solution.position_enu * au.m,
            array_location=ms.meta.array_location,
            delay_acf=solution.delay_acf,
            antennas=ms.meta.antennas,
            antenna_labels=ms.meta.antenna_names,
            gains=solution.gains * au.dimensionless_unscaled
        )
        with open(file_name, "w") as fp:
            fp.write(data.json(indent=2))


class RFIEmitterSolutions(SerialisableBaseModel):
    """
    Calibration solutions, stored in a serialisable format.
    """

    times: at.Time  # [time]
    pointings: ac.ICRS | None  # [[ant]]
    freqs: au.Quantity  # [num_chans]
    position_enu: au.Quantity  # [E, 3]
    array_location: ac.EarthLocation
    delay_acf: InterpolatedArray | ParametricDelayACF # [E]

    antennas: ac.EarthLocation  # [ant]
    antenna_labels: List[str]  # [ant]
    gains: au.Quantity | None = None  # [[E,] time, ant, chan[, 2, 2]]
