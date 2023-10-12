import os

import astropy.coordinates as ac
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from dsa2000_cal.assets.mocks.mock_data import MockData
from dsa2000_cal.faint_sky_model import repoint_fits, down_sample_fits


def test_repoint_fits():
    pointing_centre = ac.ICRS(ra=ac.Angle("19h59m28.356s"), dec=ac.Angle("+40d44m02.10s"))

    MockData()

    ds_output_file = "down_sampled_faint_sky_model.fits"

    down_sample_fits(
        fits_file=MockData().faint_sky_model(),
        output_file=ds_output_file,
        desired_ra_size=128,
        desired_dec_size=128
    )

    # Check the dimensions of the output file
    with fits.open(ds_output_file) as hdu:
        data = hdu[0].data
        assert data.shape[2:] == (128, 128)

    rp_output_file = "repointed_down_sampled_faint_sky_model.fits"
    repoint_fits(
        fits_file=ds_output_file,
        output_file=rp_output_file,
        pointing_centre=pointing_centre
    )

    # Check if the output file is created
    assert os.path.exists(rp_output_file)

    # Check the pointing center of the output file
    with fits.open(rp_output_file) as hdu:
        wcs = WCS(hdu[0].header)
        assert np.isclose(wcs.wcs.crval[0], pointing_centre.ra.deg, atol=1e-6)
        assert np.isclose(wcs.wcs.crval[1], pointing_centre.dec.deg, atol=1e-6)

    # Cleanup
    os.remove(ds_output_file)
    os.remove(rp_output_file)
