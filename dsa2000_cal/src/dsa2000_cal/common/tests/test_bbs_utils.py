import astropy.units as au
from h5parm.utils import parse_coordinates_bbs

from dsa2000_cal.common.bbs_utils import BBSSkyModel


def test_bbs_sky_model_I_only(tmp_path):
    sky_model_bbs = "# (Name, Type, Ra, Dec, I) = format\n" \
                    "A, POINT, 00:00:00.123456, +37.07.47.12345, 1.0\n" \
                    "B, POINT, 00:00:00.123456, +37.37.47.12345, 1.0\n" \
                    "C, POINT, 00:00:00.123456, +38.07.47.12345, 1.0\n"
    sky_model_file = tmp_path / 'test_sky_model.txt'
    with open(sky_model_file, 'w') as f:
        f.write(sky_model_bbs)
    pointing_centre = parse_coordinates_bbs("00:00:00.0", "+37.07.47.0")
    bbs_sky_model = BBSSkyModel(sky_model_file, pointing_centre=pointing_centre,
                                chan0=800e6 * au.Hz,
                                chan_width=2e6 * au.Hz,
                                num_channels=5
                                )
    source_model = bbs_sky_model.get_source()
    assert source_model.corrs == ['I', 'Q', 'U', 'V']
    assert source_model.image.shape == (3, 5, 4)
    assert source_model.lm.shape == (3, 2)
    assert source_model.freqs.shape == (5,)


def test_bbs_sky_model_all_only(tmp_path):
    sky_model_bbs = "# (Name, Type, Ra, Dec, I=0, U=0, V=0) = format\n" \
                    "A, POINT, 00:00:00.123456, +37.07.47.12345, 1.0, , , \n" \
                    "B, POINT, 00:00:00.123456, +37.37.47.12345, 1.0, , , \n" \
                    "C, POINT, 00:00:10.123456, +37.37.47.12345, 1.0, , , "
    sky_model_file = tmp_path / 'test_sky_model.txt'
    with open(sky_model_file, 'w') as f:
        f.write(sky_model_bbs)
    pointing_centre = parse_coordinates_bbs("00:00:00.0", "+37.07.47.0")
    bbs_sky_model = BBSSkyModel(sky_model_file, pointing_centre=pointing_centre,
                                chan0=800e6 * au.Hz,
                                chan_width=2e6 * au.Hz,
                                num_channels=5
                                )
    source_model = bbs_sky_model.get_source()
    assert source_model.corrs == ['I', 'Q', 'U', 'V']
    assert source_model.image.shape == (3, 5, 4)
    assert source_model.lm.shape == (3, 2)
    print(source_model)
