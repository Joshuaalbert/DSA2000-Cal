import dataclasses
from abc import ABC, abstractmethod
from functools import partial
from typing import Tuple, Callable

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax as tfp
from jax import lax
from jaxns import Prior, Model, PriorModelType

from dsa2000_cal.common.jax_utils import promote_pytree, multi_vmap
from dsa2000_cal.measurement_sets.measurement_set import VisibilityData
from dsa2000_cal.uvw.far_field import VisibilityCoords
from dsa2000_cal.common.vec_utils import kron_product
from dsa2000_cal.visibility_model.rime_model import RIMEModel
from dsa2000_cal.visibility_model.source_models import flatten_coherencies

tfpd = tfp.distributions


class AbstractGainProbabilisticModel(ABC):

    @abstractmethod
    def get_init_params(self):
        """
        Get the initial parameters for the gains.

        Returns:
            params: [[num_chan,] ...] the initial parameters, possibly sharded over channel.
        """
        ...

    @abstractmethod
    def log_prob_joint(self, params: jax.Array) -> jax.Array:
        """
        Compute the joint log probability of the gains and the data.

        Args:
            params: [[num_chan,] ...]

        Returns:
            scalar
        """
        ...

    @abstractmethod
    def forward(self, params: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """
        Forward model for the gains.

        Args:
            params: [[num_chan,] ...]

        Returns:
            vis_model: [num_row, num_chan, 4] the model visibilities
            gains: [num_source, num_ant, num_chan, 2, 2]
        """
        ...


class AbstractGainPriorModel(ABC):
    @abstractmethod
    def prior_model(self, num_source: int, num_ant: int, freqs: jax.Array) -> PriorModelType:
        """
        Define the prior model for the gains.

        Args:
            num_source: the number of sources
            num_ant: the number of antennas
            freqs: [num_chan] the frequencies, should use for sharding

        Returns:
            gains: [num_source, num_ant, num_chan, 2, 2], using `freqs` to shard axis 2
        """
        ...


@dataclasses.dataclass(eq=False)
class ReplicatedGainProbabilisticModel(AbstractGainProbabilisticModel):
    rime_model: RIMEModel
    gain_prior_model: AbstractGainPriorModel
    freqs: jax.Array  # [num_chan] sharded over chan
    preapply_gains: jax.Array  # [num_source, num_ant, num_chan, 2, 2] sharded over chan
    vis_data: VisibilityData  # [num_row, num_chan, 2, 2] sharded over chan
    vis_coords: VisibilityCoords  # [num_row, 2]

    def _apply_gains(self, gains: jax.Array, vis: jax.Array, vis_coords: VisibilityCoords):
        """
        Apply the gains to the visibilities.

        Args:
            gains: [num_source, num_ant, num_chan, 2, 2]
            vis: [num_source, num_row, num_chan, 2, 2]
            vis_coords: the visibility coordinates

        Returns:
            vis_model: [num_row, num_chan, 4] the model visibilities
        """

        # V_ij = G_i * V_ij * G_j^H
        g1 = gains[:, vis_coords.antenna_1, ...]  # [num_source, num_rows, num_chans, 2, 2]
        g2 = gains[:, vis_coords.antenna_2, ...]  # [num_source, num_rows, num_chans, 2, 2]

        def body_fn(accumulated_vis, x):
            g1, g2, vis = x

            # g1, g2: [num_rows, num_chans, 2, 2]
            # vis: [num_rows, num_chans, 2, 2]
            @partial(multi_vmap, in_mapping="[r,c,2,2],[r,c,2,2],[r,c,2,2]", out_mapping="[r,c]", verbose=True)
            def transform(g1, g2, vis):
                return flatten_coherencies(kron_product(g1, vis, g2.conj().T))  # [4]

            accumulated_vis += transform(g1, g2, vis)  # [num_rows, num_chans, 4]
            return accumulated_vis, ()

        model_vis = jnp.zeros(np.shape(vis)[1:-2] + (4,), vis.dtype)  # [num_rows, num_chans, 4]

        accumulated_vis, _ = lax.scan(body_fn, model_vis, (g1, g2, vis))
        return accumulated_vis  # [num_rows, num_chans, 4]

    def __post_init__(self):
        self._get_init_params, self._forward, self._log_prob_joint = self._build_model()

    def _constrain_params(self, params: jax.Array) -> jax.Array:
        # TODO: Not the most efficiency parametrisation. Should JAXNS consider this?
        #  X=quantile(sigmoid(params)), because JAXNS likes to formulate the problem in the unit hypercube.
        return jax.nn.sigmoid(params)

    def get_init_params(self):
        # Unconstrained params
        return self._get_init_params()

    def forward(self, params: jax.Array) -> Tuple[jax.Array, jax.Array]:
        return self._forward(self._constrain_params(params))

    def log_prob_joint(self, params: jax.Array) -> jax.Array:
        return self._log_prob_joint(self._constrain_params(params))

    def _build_model(self) -> Tuple[Callable, Callable, Callable]:
        num_source, num_time, num_ant, num_chan, _, _ = np.shape(self.preapply_gains)

        # TODO: explore using checkpointing
        vis = self.rime_model.predict_facets_model_visibilities(
            freqs=self.freqs,
            apply_gains=self.preapply_gains,
            visibility_coords=self.vis_coords,
            flat_coherencies=False
        )  # [num_cal, num_row, num_chan, 2, 2]

        # jax.debug.inspect_array_sharding(vis, callback=print) # (1, 1, 'chan', 1, 1)

        # vis = jax.lax.with_sharding_constraint(vis, NamedSharding(mesh, P(None, None, 'chan')))'
        # TODO: https://jax.readthedocs.io/en/latest/notebooks/shard_map.html#fsdp-tp-with-shard-map-at-the-top-level

        # vis now contains the model visibilities for each calibrator

        def prior_model():
            gains: jax.Array = yield from self.gain_prior_model.prior_model(
                num_source=num_source,
                num_ant=num_ant,
                freqs=self.freqs
            )  # [num_source, num_ant, num_chan, 2, 2]
            return self._apply_gains(gains, vis, self.vis_coords), gains

        def log_likelihood(vis_model: jax.Array, gains: jax.Array):
            """
            Compute the log probability of the data given the gains.

            Args:
                vis_model: [num_rows, num_chan, 4]

            Returns:
                log_prob: scalar
            """

            vis_variance = 1. / self.vis_data.weights  # Should probably use measurement set SIGMA here
            vis_stddev = jnp.sqrt(vis_variance)
            obs_dist_real = tfpd.Normal(*promote_pytree('vis_real', (jnp.real(self.vis_data.vis), vis_stddev)))
            obs_dist_imag = tfpd.Normal(*promote_pytree('vis_imag', (jnp.imag(self.vis_data.vis), vis_stddev)))
            log_prob_real = obs_dist_real.log_prob(jnp.real(vis_model))
            log_prob_imag = obs_dist_imag.log_prob(jnp.imag(vis_model))  # [num_rows, num_chan, 4]
            log_prob = log_prob_real + log_prob_imag  # [num_rows, num_chan, 4]

            # Mask out flagged data or zero-weighted data.
            mask = jnp.logical_or(self.vis_data.weights == 0, self.vis_data.flags)
            log_prob = jnp.where(mask, 0., log_prob)
            return jnp.sum(log_prob)

        model = Model(
            prior_model=prior_model,
            log_likelihood=log_likelihood
        )

        def get_init_params():
            return jnp.zeros((model.U_ndims,))

        def forward(params):
            return model.prepare_input(params)

        def log_prob_joint(params):
            return model.log_prob_joint(params, allow_nan=False)

        return get_init_params, forward, log_prob_joint


@dataclasses.dataclass(eq=False)
class UnconstrainedGain(AbstractGainPriorModel):
    """
    A gain model with unconstrained complex Gaussian priors with zero mean.
    """
    gain_stddev: float = 2.

    def prior_model(self, num_source: int, num_ant: int, freqs: jax.Array) -> PriorModelType:
        loc = jnp.zeros((num_source, num_ant, 2, 2))
        scale = jnp.full((num_source, num_ant, 2, 2), self.gain_stddev)
        gains_real = yield Prior(
            tfpd.Normal(loc=loc + 1.,
                        scale=scale
                        ),
            name='gains_real'
        )
        gains_imag = yield Prior(
            tfpd.Normal(loc=loc,
                        scale=scale
                        ),
            name='gains_imag'
        )
        gains = gains_real + 1j * gains_imag  # [num_source, num_ant, 2, 2]
        gains = jax.vmap(lambda _freq: gains, out_axes=2)(freqs)  # [num_source, num_ant, num_chan, 2, 2]
        return gains


@dataclasses.dataclass(eq=False)
class DiagonalUnconstrainedGain(AbstractGainPriorModel):
    """
    A gain model with unconstrained complex Gaussian priors with zero mean, but only on the diagonal.
    """
    gain_stddev: float = 2.

    def prior_model(self, num_source: int, num_ant: int, freqs: jax.Array) -> PriorModelType:
        loc = jnp.zeros((num_source, num_ant, 2))
        scale = jnp.full((num_source, num_ant, 2), self.gain_stddev)
        gains_real = yield Prior(
            tfpd.Normal(
                loc=loc + 1.,
                scale=scale
            ),
            name='gains_real'
        )
        gains_imag = yield Prior(
            tfpd.Normal(
                loc=loc,
                scale=scale
            ),
            name='gains_imag'
        )
        diag_gains = gains_real + 1j * gains_imag  # [num_source, num_ant, 2]
        gains = jax.vmap(jax.vmap(jnp.diag))(diag_gains)  # [num_source, num_ant, 2, 2]
        gains = jax.vmap(lambda _freq: gains, out_axes=2)(freqs)  # [num_source, num_ant, num_chan, 2, 2]
        return gains
