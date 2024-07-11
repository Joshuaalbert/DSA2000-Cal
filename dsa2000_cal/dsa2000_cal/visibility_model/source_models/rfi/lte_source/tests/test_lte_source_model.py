import jax
import numpy as np
import pytest
from astropy import units as au, time as at
from jax import numpy as jnp

from dsa2000_cal.assets.content_registry import fill_registries
from dsa2000_cal.assets.registries import rfi_model_registry, array_registry
from dsa2000_cal.uvw.far_field import VisibilityCoords
from dsa2000_cal.uvw.near_field import NearFieldDelayEngine
from dsa2000_cal.visibility_model.source_models.rfi.lte_source.lte_source_model import LTESourceModel, LTEPredict


@pytest.mark.parametrize("is_gains", [True, False])
@pytest.mark.parametrize("direction_dependent_gains", [True, False])
@pytest.mark.parametrize("full_stokes", [True, False])
def test_lte_predict(is_gains: bool, direction_dependent_gains: bool, full_stokes: bool):
    # Create a simple model
    fill_registries()
    rfi_model = rfi_model_registry.get_instance(rfi_model_registry.get_match('lte_cell_tower'))
    freqs = np.linspace(701, 705, 10) * au.MHz
    rfi_model_params = rfi_model.make_source_params(freqs=freqs, full_stokes=full_stokes)
    source_model = LTESourceModel(rfi_model_params)
    if full_stokes:
        assert source_model.is_full_stokes()
    else:
        assert not source_model.is_full_stokes()
    array = array_registry.get_instance(array_registry.get_match('dsa2000W'))
    time = at.Time('2021-01-01T00:00:00', scale='utc')

    antennas = array.get_antennas()
    delay_engine = NearFieldDelayEngine(
        antennas=antennas,
        start_time=time,
        end_time=time
    )
    predict = LTEPredict(delay_engine=delay_engine)
    ant = len(antennas)
    num_times = 1
    if is_gains:
        if full_stokes:
            if direction_dependent_gains:
                gains_shape = (1, num_times, ant, len(freqs), 2, 2)
            else:
                gains_shape = (num_times, ant, len(freqs), 2, 2)
        else:
            if direction_dependent_gains:
                gains_shape = (1, num_times, ant, len(freqs))
            else:
                gains_shape = (num_times, ant, len(freqs),)
        gains = 1j * 1e-3 * jax.random.normal(jax.random.PRNGKey(42), gains_shape)
    else:
        gains = None
    model_data = source_model.get_model_data(gains=gains)

    row = ant * (ant - 1) // 2

    antennas = 20e3 * jax.random.normal(jax.random.PRNGKey(42), (ant, 3))
    antenna_1 = jax.random.randint(jax.random.PRNGKey(42), (row,), 0, ant)
    antenna_2 = jax.random.randint(jax.random.PRNGKey(42), (row,), 0, ant)

    uvw = antennas[antenna_2] - antennas[antenna_1]
    uvw = uvw.at[:, 2].mul(1e-3)

    times = jnp.linspace(0, 1, num_times)
    time_idx = jax.random.randint(jax.random.PRNGKey(42), (row,), 0, num_times)
    time_obs = times[time_idx]
    visibility_coords = VisibilityCoords(
        uvw=uvw,
        time_obs=time_obs,
        antenna_1=antenna_1,
        antenna_2=antenna_2,
        time_idx=time_idx
    )

    vis = predict.predict(model_data=model_data, visibility_coords=visibility_coords)
    if full_stokes:
        assert np.shape(vis) == (row, len(freqs), 2, 2)
    else:
        assert np.shape(vis) == (row, len(freqs))
    assert np.all(np.isfinite(vis))
    print(vis)
