import os

import numpy as np
import pylab as plt
import pytest
from astropy import time as at, coordinates as ac, units as au

from dsa2000_cal.antenna_model.utils import get_dish_model_beam_widths
from dsa2000_cal.assets.content_registry import fill_registries
from dsa2000_cal.assets.registries import array_registry, source_model_registry
from dsa2000_cal.assets.rfi.lte_rfi.lwa_cell_tower import LWACellTower
from dsa2000_cal.assets.source_models.cyg_a.source_model import CygASourceModel
from dsa2000_cal.assets.tests.test_source_models import get_lm_coords_image
from dsa2000_cal.visibility_model.source_models.celestial.fits_source_model import FITSSourceModel


@pytest.mark.parametrize('array_name', ['dsa2000W_small', 'dsa2000W', 'lwa'])
def test_array_layouts(array_name: str):
    fill_registries()
    array = array_registry.get_instance(array_registry.get_match(array_name))
    antennas = array.get_antennas()

    import pylab as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(antennas.itrs.cartesian.x.to('km'),
               antennas.itrs.cartesian.y.to('km'),
               antennas.itrs.cartesian.z.to('km'))
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')
    ax.set_title(array_name)
    plt.show()


@pytest.mark.parametrize('array_name', ['dsa2000W_small', 'dsa2000W', 'lwa'])
def test_dsa2000_antenna_beam(array_name: str):
    fill_registries()
    array = array_registry.get_instance(array_registry.get_match(array_name))
    antenna_model = array.get_antenna_model()

    # amplitude = antenna_model.get_amplitude()[..., 0, 0]
    # circular_mean = np.mean(amplitude, axis=1)
    # phi = antenna_model.get_phi()
    # theta = antenna_model.get_theta()
    # for i, freq in enumerate(antenna_model.get_freqs()):
    #     plt.plot(theta, amplitude[:, i], label=freq)
    # plt.legend()
    # plt.show()
    # for i, freq in enumerate(antenna_model.get_freqs()):
    #     for k, th in enumerate(theta):
    #         if circular_mean[k, i] < 0.01:
    #             break
    #     print(f"Freq: {i}, {freq}, theta: {k}, {th}")

    freqs, beam_widths = get_dish_model_beam_widths(
        antenna_model=antenna_model,
        threshold=0.5
    )
    plt.plot(freqs, beam_widths.to('deg').value)
    plt.ylabel('Beam Half Power Width (deg)')
    plt.xlabel('Frequency (Hz)')
    plt.show()


def test_lwa_cell_tower():
    lwa_cell_tower = LWACellTower(seed='abc')
    lwa_cell_tower.plot_acf()


def test_model():
    for file in CygASourceModel(seed='abc').get_wsclean_fits_files():
        assert os.path.isfile(file)


@pytest.mark.parametrize('source', ['cas_a', 'cyg_a', 'tau_a', 'vir_a'])
def test_orientations(source: str):
    fill_registries()
    time = at.Time('2021-01-01T00:00:00', scale='utc')

    # source_file = source_model_registry.get_instance(source_model_registry.get_match('cas_a')).get_wsclean_source_file()
    # -00:36:28.234,58.50.46.396
    # phase_tracking = ac.SkyCoord("-00h36m28.234s", "58d50m46.396s", frame='icrs')
    # phase_tracking = ac.SkyCoord("-00h36m28.234s", "78d50m46.396s", frame='icrs')

    wsclean_fits_files = source_model_registry.get_instance(
        source_model_registry.get_match(source)).get_wsclean_fits_files()
    # -04:00:28.608,40.43.33.595
    phase_tracking = ac.SkyCoord("-04h00m28.608s", "40d43m33.595s", frame='icrs')

    freqs = au.Quantity([65e6, 77e6], 'Hz')

    fits_sources = FITSSourceModel.from_wsclean_model(wsclean_fits_files=wsclean_fits_files,
                                                      phase_tracking=phase_tracking, freqs=freqs)
    assert isinstance(fits_sources, FITSSourceModel)

    # Visually verified against ds9, that RA increases over column, and DEC increases over rows.
    fits_sources.plot()
    image, lmn = get_lm_coords_image(wsclean_fits_files[0], time=time, phase_tracking=phase_tracking)
    print(lmn.shape)
    print(image.shape)  # [Nm, Nl]
    l = lmn[:, :, 0]
    m = lmn[:, :, 1]
    _, dl = np.gradient(l)
    dm, _ = np.gradient(m)
    dA = dl * dm
    # dl = np.diff(l, axis=1, prepend=l[:, 1] - l[:, 0])
    # dm = np.diff(m, axis=0, prepend=m[1, :] - m[0, :])
    # print(dl)
    import pylab as plt
    plt.imshow(dl, origin='lower',
               extent=(l.min(), l.max(), m.min(), m.max()))
    plt.colorbar()
    plt.xlabel('l')
    plt.ylabel('m')
    plt.title('dl(l,m)')
    plt.show()

    # print(dm)
    plt.imshow(dm, origin='lower',
               extent=(l.min(), l.max(), m.min(), m.max()))
    plt.colorbar()
    plt.xlabel('l')
    plt.ylabel('m')
    plt.title('dm(l,m)')
    plt.show()

    # print(dA)
    plt.imshow(dA, origin='lower',
               extent=(l.min(), l.max(), m.min(), m.max()))
    plt.colorbar()
    plt.xlabel('l')
    plt.ylabel('m')
    plt.title('dA(l,m)')
    plt.show()


def test_wsclean_component_files():
    fill_registries()
    # Create a sky model for calibration
    for source in ['cas_a', 'cyg_a', 'tau_a', 'vir_a']:
        source_model_asset = source_model_registry.get_instance(source_model_registry.get_match(source))
        time = at.Time('2021-01-01T00:00:00', format='isot', scale='utc')
        freqs = np.linspace(700e6, 2000e6, 1) * au.Hz
        phase_tracking = ac.ICRS(ra=ac.Angle('0h'), dec=ac.Angle('0d'))
        # source_model = WSCleanSourceModel.from_wsclean_model(
        #     wsclean_clean_component_file=source_model_asset.get_wsclean_clean_component_file(),
        #     time=at.Time('2021-01-01T00:00:00', format='isot', scale='utc'),
        #     freqs=np.linspace(700e6, 2000e6, 2) * au.Hz,
        #     phase_tracking=ac.ICRS(ra=ac.Angle('0h'), dec=ac.Angle('0d'))
        # )

        fits_model = FITSSourceModel.from_wsclean_model(source_model_asset.get_wsclean_fits_files(),
                                                        phase_tracking, freqs, ignore_out_of_bounds=True)
        fits_model.plot()
