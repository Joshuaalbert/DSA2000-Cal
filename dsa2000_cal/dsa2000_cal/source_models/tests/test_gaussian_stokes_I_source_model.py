import astropy.coordinates as ac
import astropy.time as at
import astropy.units as au
import numpy as np

from dsa2000_cal.assets.content_registry import fill_registries
from dsa2000_cal.assets.registries import source_model_registry
from dsa2000_cal.source_models.gaussian_stokes_I_source_model import GaussianSourceModel, ellipse_rotation, ellipse_eval


def test_gaussian_sources():
    fill_registries()
    time = at.Time('2021-01-01T00:00:00', scale='utc')

    # source_file = source_model_registry.get_instance(source_model_registry.get_match('cas_a')).get_wsclean_source_file()
    # -00:36:28.234,58.50.46.396
    # phase_tracking = ac.SkyCoord("-00h36m28.234s", "58d50m46.396s", frame='icrs')
    # phase_tracking = ac.SkyCoord("-00h36m28.234s", "78d50m46.396s", frame='icrs')

    source_file = source_model_registry.get_instance(
        source_model_registry.get_match('cyg_a')).get_wsclean_clean_component_file()
    # -04:00:28.608,40.43.33.595
    phase_tracking = ac.SkyCoord("-04h00m28.608s", "40d43m33.595s", frame='icrs')

    freqs = au.Quantity([50e6, 80e6], 'Hz')

    gaussian_source_model = GaussianSourceModel.from_wsclean_model(
        wsclean_clean_component_file=source_file,
        time=time,
        phase_tracking=phase_tracking,
        freqs=freqs,
        lmn_transform_params=True
    )

    assert isinstance(gaussian_source_model, GaussianSourceModel)
    assert gaussian_source_model.num_sources > 0

    gaussian_source_model.plot()


def test_ellipse_rotation():
    # (x,y) -> (x',y')
    up = np.asarray([0, 1])
    right = np.asarray([1, 0])
    left = np.asarray([-1, 0])
    down = np.asarray([0, -1])
    np.testing.assert_allclose(ellipse_rotation(np.pi / 2.) @ up, right, atol=1e-6)
    np.testing.assert_allclose(ellipse_rotation(np.pi) @ up, down, atol=1e-6)
    np.testing.assert_allclose(ellipse_rotation(-np.pi / 2.) @ up, left, atol=1e-6)


def test_ellipse_eval():
    b_major = 3.5
    b_minor = 2.5
    A = 4. * np.pi / (np.log(2) * b_major * b_minor)
    # For point on ellipse get 1/2
    pos_angle = 0.
    l = b_minor / 2.
    m = 0
    l0 = 0.
    m0 = 0.
    np.testing.assert_allclose(ellipse_eval(A, b_major, b_minor, pos_angle, l, m, l0, m0), 0.5, atol=1e-6)

    l = 0
    m = b_major / 2.
    l0 = 0.
    m0 = 0.
    np.testing.assert_allclose(ellipse_eval(A, b_major, b_minor, pos_angle, l, m, l0, m0), 0.5, atol=1e-6)

    pos_angle = np.pi / 2.
    l = 0
    m = b_minor / 2.
    l0 = 0.
    m0 = 0.
    np.testing.assert_allclose(ellipse_eval(A, b_major, b_minor, pos_angle, l, m, l0, m0), 0.5, atol=1e-6)

    pos_angle = np.pi / 2.
    l = b_major / 2.
    m = 0.
    l0 = 0.
    m0 = 0.
    np.testing.assert_allclose(ellipse_eval(A, b_major, b_minor, pos_angle, l, m, l0, m0), 0.5, atol=1e-6)
