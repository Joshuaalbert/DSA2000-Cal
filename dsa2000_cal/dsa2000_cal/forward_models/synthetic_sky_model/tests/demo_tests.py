from astropy import coordinates as ac, units as au

from dsa2000_cal.assets.content_registry import fill_registries
from dsa2000_cal.forward_models.synthetic_sky_model.synthetic_sky_model_producer import SyntheticSkyModelProducer


def test_create_sky_model():
    fill_registries()
    synthetic_sky_model_producer = SyntheticSkyModelProducer(
        phase_tracking=ac.ICRS(15 * au.deg, 0 * au.deg),
        freqs=au.Quantity([700], unit='MHz'),
        field_of_view=4 * au.deg,
        seed=42
    )
    bright_point_sources = synthetic_sky_model_producer.create_sources_outside_fov(num_bright_sources=100,
                                                                                   full_stokes=False)
    bright_point_sources.plot(save_file='bright_point_sources.png')
    assert bright_point_sources.num_sources == 100
    inner_point_sources = synthetic_sky_model_producer.create_sources_inside_fov(num_sources=100, full_stokes=False)
    inner_point_sources.plot(save_file='inner_point_sources.png')

    (bright_point_sources + inner_point_sources).plot(save_file='all_point_sources.png')

    assert inner_point_sources.num_sources == 37  # Should debug
    inner_diffuse_sources = synthetic_sky_model_producer.create_diffuse_sources_inside_fov(num_sources=100,
                                                                                           full_stokes=False)
    inner_diffuse_sources.plot(save_file='inner_diffuse_sources.png')
    assert inner_diffuse_sources.num_sources == 37  # Should debug
    rfi_emitter_sources = synthetic_sky_model_producer.create_rfi_emitter_sources(full_stokes=False)
    rfi_emitter_sources[0].plot(save_file='rfi_emitter_sources.png')
    assert len(rfi_emitter_sources) == 1
    a_team_sources = synthetic_sky_model_producer.create_a_team_sources(a_team_sources=['cas_a'])
    a_team_sources[0].plot(save_file='cas_a.png')
    assert len(a_team_sources) == 1
    illustris_sources = synthetic_sky_model_producer.create_illustris_sources()
    illustris_sources[0].plot(save_file='illustris.png')
    assert len(illustris_sources) == 1