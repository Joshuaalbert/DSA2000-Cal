from typing import Tuple

import jax
import numpy as np
import pytest
from astropy import units as au, time as at
from jax import numpy as jnp

from dsa2000_cal.assets.content_registry import fill_registries
from dsa2000_cal.assets.registries import array_registry, rfi_model_registry
from dsa2000_cal.common.mixed_precision_utils import mp_policy
from dsa2000_cal.delay_models.base_near_field_delay_engine import build_near_field_delay_engine
from dsa2000_cal.visibility_model.source_models.rfi.base_rfi_emitter_source_model import \
    RFIEmitterSourceModel, RFIEmitterModelData


def build_mock_rfi_model_data(is_gains: bool, full_stokes: bool, direction_dependent_gains: bool, chan: int,
                              num_time: int) -> Tuple[RFIEmitterModelData]:
    # Create a simple model
    fill_registries()
    # rfi_model = rfi_model_registry.get_instance(rfi_model_registry.get_match('mock_cell_tower'))
    rfi_model = rfi_model_registry.get_instance(rfi_model_registry.get_match('parametric_mock_cell_tower'))
    # Create a simple model
    freqs = np.linspace(55, 59, chan) * au.MHz
    rfi_model_params = rfi_model.make_source_params(freqs=freqs, full_stokes=full_stokes)
    source_model = RFIEmitterSourceModel(rfi_model_params)
    if full_stokes:
        assert source_model.is_full_stokes()
    else:
        assert not source_model.is_full_stokes()
    array = array_registry.get_instance(array_registry.get_match('dsa2000W_small'))
    time = at.Time('2021-01-01T00:00:00', scale='utc')

    antennas = array.get_antennas()
    delay_engine = build_near_field_delay_engine(
        antennas=antennas,
        start_time=time,
        end_time=time,
        ref_time=time
    )
    predict = RFIEmitterPredict(delay_engine=delay_engine)
    ant = len(antennas)
    if is_gains:
        if full_stokes:
            if direction_dependent_gains:
                gains_shape = (1, num_time, ant, len(freqs), 2, 2)
            else:
                gains_shape = (num_time, ant, len(freqs), 2, 2)
        else:
            if direction_dependent_gains:
                gains_shape = (1, num_time, ant, len(freqs))
            else:
                gains_shape = (num_time, ant, len(freqs),)
        gains = 1. + 1j * 1e-3 * jax.random.normal(jax.random.PRNGKey(42), gains_shape)
    else:
        gains = None
    model_data = source_model.get_model_data(gains=gains)
    return model_data, predict


def build_mock_visibility_coord(rows: int, ant: int, time: int) -> VisibilityCoords:
    uvw = 20e3 * jax.random.normal(jax.random.PRNGKey(42), (rows, 3))
    uvw = uvw.at[:, 2].mul(1e-3)
    time_obs = jnp.zeros((rows,))
    antenna1 = jax.random.randint(jax.random.PRNGKey(42), (rows,), 0, ant)
    antenna2 = jax.random.randint(jax.random.PRNGKey(43), (rows,), 0, ant)
    time_idx = jax.random.randint(jax.random.PRNGKey(44), (rows,), 0, time)

    visibility_coords = VisibilityCoords(
        uvw=mp_policy.cast_to_length(uvw),
        time_obs=mp_policy.cast_to_time(time_obs),
        antenna1=mp_policy.cast_to_index(antenna1),
        antenna2=mp_policy.cast_to_index(antenna2),
        time_idx=mp_policy.cast_to_index(time_idx)
    )
    return visibility_coords


@pytest.mark.parametrize("is_gains", [True, False])
@pytest.mark.parametrize("direction_dependent_gains", [True, False])
@pytest.mark.parametrize("full_stokes", [True, False])
def test_rfi_emitter_predict_shapes_correct(is_gains: bool, direction_dependent_gains: bool, full_stokes: bool):
    chan = 10
    time = 1
    ant = 24
    row = ant * (ant - 1) // 2

    model_data, predict = build_mock_rfi_model_data(is_gains, full_stokes, direction_dependent_gains, chan, time)
    visibility_coords = build_mock_visibility_coord(row, ant, time)

    vis = predict.predict(model_data=model_data, visibility_coords=visibility_coords)
    if full_stokes:
        assert np.shape(vis) == (row, chan, 2, 2)
    else:
        assert np.shape(vis) == (row, chan)
    assert np.all(np.isfinite(vis))
    print(vis)
