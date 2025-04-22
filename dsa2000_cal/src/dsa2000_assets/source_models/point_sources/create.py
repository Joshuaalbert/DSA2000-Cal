import astropy.coordinates as ac
import astropy.time as at
import astropy.units as au
import numpy as np

from dsa2000_common.common.fits_utils import ImageModel, save_image_to_fits


def main():
    pointing = ac.ICRS(ra=0.0 * au.deg, dec=0.0 * au.deg)
    ref_time = at.Time('2025-06-10T00:00:00', scale='utc')
    freqs = [1350] * au.MHz
    bandwidth = 1300 * au.MHz
    dl = (0.8 * au.arcsec).to('rad').value

    num_l = num_m = 512
    dirty = np.zeros((num_l, num_m), dtype=np.float32)
    dirty[num_l // 2, num_m // 2] = 500e-3
    dirty[num_l // 3, num_m // 3] = 500e-3

    beam_major = dl * au.rad
    beam_minor = dl * au.rad
    beam_pa = 0.0 * au.rad

    image_model = ImageModel(
        phase_center=pointing,
        obs_time=ref_time,
        dl=dl * au.rad,
        dm=dl * au.rad,
        freqs=freqs,
        bandwidth=bandwidth,
        coherencies=('I',),
        beam_major=beam_major,
        beam_minor=beam_minor,
        beam_pa=beam_pa,
        unit='JY/PIXEL',
        object_name=f'POINT_SOURCES',
        image=dirty[:, :, None, None] * au.Jy  # [num_l, num_m, 1, 1]
    )
    save_image_to_fits("point_sources.fits", image_model=image_model, overwrite=True)


if __name__ == '__main__':
    main()
