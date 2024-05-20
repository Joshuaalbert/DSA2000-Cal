import astropy.coordinates as ac
import astropy.time as at
import astropy.units as au
import matplotlib.pyplot as plt
import numpy as np
import pytest

from dsa2000_cal.assets.content_registry import fill_registries
from dsa2000_cal.assets.registries import source_model_registry
from dsa2000_cal.source_models.gaussian_stokes_I_source_model import GaussianSourceModel
from dsa2000_cal.source_models.point_stokes_I_source_model import PointSourceModel
from dsa2000_cal.source_models.wsclean_stokes_I_source_model import WSCleanSourceModel


def test_wsclean_sources():
    fill_registries()
    time = at.Time('2021-01-01T00:00:00', scale='utc')

    # -00:36:28.234,58.50.46.396
    source_file = source_model_registry.get_instance(
        source_model_registry.get_match('cyg_a')).get_wsclean_clean_component_file()
    # phase_tracking = ac.SkyCoord("-00h36m28.234s", "58d50m46.396s", frame='icrs')
    phase_tracking = ac.SkyCoord("-00h36m28.234s", "78d50m46.396s", frame='icrs')

    # -04:00:28.608,40.43.33.595
    # source_file = source_model_registry.get_instance(source_model_registry.get_match('cyg_a')).get_wsclean_source_file()
    # phase_tracking = ac.SkyCoord("-04h00m28.608s", "40d43m33.595s", frame='icrs')

    freqs = au.Quantity([65e6, 77e6], 'Hz')

    wsclean_sources = WSCleanSourceModel.from_wsclean_model(
        wsclean_clean_component_file=source_file,
        time=time,
        phase_tracking=phase_tracking,
        freqs=freqs
    )

    wsclean_sources.plot()


@pytest.mark.parametrize("num_point", [0, 1, 2])
@pytest.mark.parametrize("num_gaussian", [0, 1, 2])
def test_empty_sources(num_point: int, num_gaussian: int):
    freqs = [50e6, 80e6] * au.Hz

    point_sources = None
    if num_point > 0:
        point_sources = PointSourceModel(
            freqs=freqs,
            l0=np.zeros(num_point) * au.dimensionless_unscaled,
            m0=np.zeros(num_point) * au.dimensionless_unscaled,
            A=np.ones((num_point, len(freqs))) * au.Jy
        )

    gaussian_sources = None
    if num_gaussian > 0:
        gaussian_sources = GaussianSourceModel(
            freqs=freqs,
            l0=np.zeros(num_gaussian) * au.dimensionless_unscaled,
            m0=np.zeros(num_gaussian) * au.dimensionless_unscaled,
            A=np.ones((num_gaussian, len(freqs))) * au.Jy,
            major=np.ones(num_gaussian) * au.dimensionless_unscaled,
            minor=np.ones(num_gaussian) * au.dimensionless_unscaled,
            theta=np.zeros(num_gaussian) * au.deg
        )

    if point_sources is None and gaussian_sources is None:
        with pytest.raises(ValueError):
            WSCleanSourceModel(
                point_source_model=point_sources,
                gaussian_source_model=gaussian_sources,
                freqs=freqs
            )
    else:
        source_model = WSCleanSourceModel(
            point_source_model=point_sources,
            gaussian_source_model=gaussian_sources,
            freqs=freqs
        )

        print(source_model.flux_weighted_lmn())

        source_model.plot()


def test_wsclean_sources_projection_effect():
    fill_registries()
    time = at.Time('2021-01-01T00:00:00', scale='utc')
    freqs = au.Quantity([50e6, 80e6], 'Hz')

    # -00:36:28.234,58.50.46.396
    source_file = source_model_registry.get_instance(
        source_model_registry.get_match('cyg_a')).get_wsclean_clean_component_file()
    phase_tracking = ac.SkyCoord("-04h00m28.608s", "40d43m33.595s", frame='icrs')
    wsclean_sources = WSCleanSourceModel.from_wsclean_model(
        wsclean_clean_component_file=source_file,
        time=time,
        phase_tracking=phase_tracking,
        freqs=freqs,
        lmn_transform_params=True
    )
    lvec, mvec, flux_model_1 = wsclean_sources.get_flux_model()

    # phase_tracking = ac.SkyCoord("-00h36m28.234s", "78d50m46.396s", frame='icrs')

    wsclean_sources = WSCleanSourceModel.from_wsclean_model(
        wsclean_clean_component_file=source_file,
        time=time,
        phase_tracking=phase_tracking,
        freqs=freqs,
        lmn_transform_params=False
    )
    _, _, flux_model_2 = wsclean_sources.get_flux_model(lvec, mvec)

    flux_model_diff = flux_model_2 - flux_model_1

    fig, axs = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(6, 12), squeeze=False)

    im = axs[0][0].imshow(flux_model_1, origin='lower', extent=(lvec[0], lvec[-1], mvec[0], mvec[-1]))
    plt.colorbar(im, ax=axs[0][0])
    axs[0][0].set_title('With correction')
    im = axs[1][0].imshow(flux_model_2, origin='lower', extent=(lvec[0], lvec[-1], mvec[0], mvec[-1]))
    plt.colorbar(im, ax=axs[1][0])
    axs[1][0].set_title('Without correction')
    im = axs[2][0].imshow(flux_model_diff, origin='lower', extent=(lvec[0], lvec[-1], mvec[0], mvec[-1]))
    plt.colorbar(im, ax=axs[2][0])
    axs[2][0].set_title('Difference')

    fig.tight_layout()
    plt.show()
