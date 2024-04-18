import pytest
from astropy import units as au, coordinates as ac, time as at

from dsa2000_cal.common.astropy_utils import create_spherical_grid
from dsa2000_cal.forward_model.synthetic_sky_model import choose_dr, SyntheticSkyModelProducer
from dsa2000_cal.source_models.gaussian_stokes_I_source_model import GaussianSourceModel
from dsa2000_cal.source_models.point_stokes_I_source_model import PointSourceModel
from dsa2000_cal.source_models.wsclean_stokes_I_source_model import WSCleanSourceModel


@pytest.mark.parametrize('total_n, expected_n', [
    (1, 1),
    (2, 7),
    (7, 7),
    (18, 7),
    (19, 19)
])
def test_choose_dr(total_n, expected_n):
    field_of_view = 4 * au.deg
    dr = choose_dr(field_of_view=field_of_view, total_n=total_n)
    sources = create_spherical_grid(
        pointing=ac.ICRS(15 * au.deg, 0 * au.deg),
        angular_radius=0.5 * field_of_view,
        dr=dr,
        sky_rotation=40 * au.deg
    )
    import pylab as plt
    plt.plot(sources.ra, sources.dec, 'o')
    plt.show()
    assert len(sources) == expected_n


def test_create_sky_model():
    sky_model_producer = SyntheticSkyModelProducer(
        phase_tracking=ac.ICRS(15 * au.deg, 0 * au.deg),
        obs_time=at.Time('2021-01-01T00:00:00'),
        freqs=au.Quantity([0.7, 1.4, 2], unit=au.GHz),
        num_bright_sources=7,
        num_faint_sources=7,
        field_of_view=au.Quantity(2, au.deg),
        mean_major=au.Quantity(1, au.arcmin),
        mean_minor=au.Quantity(1, au.arcmin),
        seed=42
    )
    sky_model = sky_model_producer.create_sky_model()

    for source_params in sky_model.bright_sources:
        source = PointSourceModel.from_point_source_params(source_params)
        source.plot()
    for source_params in sky_model.faint_sources:
        source = GaussianSourceModel.from_gaussian_source_params(source_params)
        source.plot()

    wsclean_source_models = sky_model.to_wsclean_source_models()
    print(wsclean_source_models)

    # combine to plot
    points = [s.point_source_model for s in wsclean_source_models if s.point_source_model is not None]
    gaussians = [s.gaussian_source_model for s in wsclean_source_models if s.gaussian_source_model is not None]
    points = sum(points[1:], start=points[0])
    gaussians = sum(gaussians[1:], start=gaussians[0])
    sky_model = WSCleanSourceModel(point_source_model=points, gaussian_source_model=gaussians,
                                   freqs=points.freqs)
    sky_model.plot()
