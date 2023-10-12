import os

import astropy.units as au
from astropy.coordinates import SkyCoord
from h5parm.utils import parse_coordinates_bbs

from dsa2000_cal.bbs_sky_model import BBSSkyModel, create_sky_model


def test_bbs_sky_model_I_only():
    sky_model_bbs = "# (Name, Type, Ra, Dec, I) = format\n" \
                    "A, POINT, 00:00:00.123456, +37.07.47.12345, 1.0\n" \
                    "B, POINT, 00:00:00.123456, +37.37.47.12345, 1.0"
    sky_model_file = 'test_sky_model.txt'
    with open(sky_model_file, 'w') as f:
        f.write(sky_model_bbs)
    pointing_centre = parse_coordinates_bbs("00:00:00.0", "+37.07.47.0")
    bbs_sky_model = BBSSkyModel(sky_model_file, pointing_centre=pointing_centre,
                                chan0=800e6 * au.Hz,
                                chan_width=2e6 * au.Hz,
                                num_channels=2
                                )
    source_model = bbs_sky_model.get_source()
    assert len(source_model.corrs) == 4
    assert source_model.image.shape == (2, 2, 4)
    assert source_model.lm.shape == (2, 2)
    os.remove(sky_model_file)


def test_bbs_sky_model_all_only():
    sky_model_bbs = "# (Name, Type, Ra, Dec, I=0, U=0, V=0) = format\n" \
                    "A, POINT, 00:00:00.123456, +37.07.47.12345, 1.0, , , \n" \
                    "B, POINT, 00:00:00.123456, +37.37.47.12345, 1.0, , , "
    sky_model_file = 'test_sky_model.txt'
    with open(sky_model_file, 'w') as f:
        f.write(sky_model_bbs)
    pointing_centre = parse_coordinates_bbs("00:00:00.0", "+37.07.47.0")
    bbs_sky_model = BBSSkyModel(sky_model_file, pointing_centre=pointing_centre,
                                chan0=800e6 * au.Hz,
                                chan_width=2e6 * au.Hz,
                                num_channels=2
                                )
    source_model = bbs_sky_model.get_source()
    assert len(source_model.corrs) == 4
    assert source_model.image.shape == (2, 2, 4)
    assert source_model.lm.shape == (2, 2)
    print(source_model)
    os.remove(sky_model_file)


def test_file_creation():
    filename = "test_sky_model.txt"
    create_sky_model(filename, 5, 1.0, SkyCoord(ra=10 * au.degree, dec=10 * au.degree, frame='icrs'))
    assert os.path.exists(filename)
    os.remove(filename)


def test_correct_number_of_sources():
    filename = "test_sky_model.txt"
    create_sky_model(filename, 5, 1.0, SkyCoord(ra=10 * au.degree, dec=10 * au.degree, frame='icrs'))
    with open(filename, 'r') as f:
        lines = f.readlines()
    # Subtracting 1 for the header
    assert len(lines) - 1 == 5
    os.remove(filename)
