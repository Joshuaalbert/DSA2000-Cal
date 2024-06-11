import itertools

import numpy as np
import pytest
from ducc0.wgridder import dirty2vis


@pytest.mark.parametrize("center_offset", [0.0, 0.1, 0.2])
@pytest.mark.parametrize("negate_w", ['neg_w', 'pos_w'])
@pytest.mark.parametrize("convention", ['casa', 'fourier'])
@pytest.mark.parametrize("negate_wgridder_center_xy", ['neg_center', 'normal_center'])
def test_wrong_w(center_offset: float, negate_w: str, convention: str, negate_wgridder_center_xy: str):
    np.random.seed(42)
    N = 512
    num_ants = 100
    num_freqs = 1

    pixsize = 0.5 * np.pi / 180 / 3600.  # 1 arcsec ~ 4 pixels / beam, so we'll avoid aliasing
    l0 = center_offset
    m0 = center_offset
    dl = pixsize
    dm = pixsize
    dirty = np.zeros((N, N))

    dirty[N // 2, N // 2] = 1.
    dirty[N // 4, N // 4] = 1.

    def pixel_to_lmn(xi, yi):
        l = l0 + (-N / 2 + xi) * dl
        m = m0 + (-N / 2 + yi) * dm
        n = np.sqrt(1. - l ** 2 - m ** 2)
        return np.asarray([l, m, n])

    lmn1 = pixel_to_lmn(N // 2, N // 2)
    lmn2 = pixel_to_lmn(N // 4, N // 4)

    antenna_1, antenna_2 = np.asarray(list(itertools.combinations(range(num_ants), 2))).T
    antennas = 10e3 * np.random.normal(size=(num_ants, 3))
    antennas[:, 2] *= 0.001
    uvw = antennas[antenna_2] - antennas[antenna_1]

    freqs = np.linspace(700e6, 2000e6, num_freqs)

    vis = dirty2vis(
        uvw=uvw,
        freq=freqs,
        dirty=dirty,
        wgt=None,
        pixsize_x=dl,
        pixsize_y=dm,
        center_x=-l0 if negate_wgridder_center_xy == 'neg_center' else l0,
        center_y=-m0 if negate_wgridder_center_xy == 'neg_center' else m0,
        epsilon=1e-4,
        do_wgridding=True,
        flip_v=False,
        divide_by_n=True,
        nthreads=1,
        verbosity=0,
    )

    lmn = [lmn1, lmn2]
    pixel_fluxes = [1., 1.]
    vis_explicit = explicit_degridder(uvw, freqs, lmn, pixel_fluxes, negate_w, convention)

    np.testing.assert_allclose(vis.real, vis_explicit.real, atol=1e-4)
    np.testing.assert_allclose(vis.imag, vis_explicit.imag, atol=1e-4)


def explicit_degridder(uvw, freqs, lmn, pixel_fluxes, negate_w, convention):
    vis = np.zeros((len(uvw), len(freqs)), dtype=np.complex128)
    c = 299792458.  # m/s

    if convention == 'casa':
        uvw = -uvw

    for row, (u, v, w) in enumerate(uvw):
        if negate_w == 'neg_w':
            w = -w
        for col, freq in enumerate(freqs):
            for flux, (l, m, n) in zip(pixel_fluxes, lmn):
                wavelength = c / freq
                phase = -2j * np.pi * (u * l + v * m + w * (n - 1)) / wavelength
                vis[row, col] += flux * np.exp(phase) / n
    return vis
