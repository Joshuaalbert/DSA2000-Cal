import jax
import numpy as np
import pytest
from astropy import time as at, units as au, coordinates as ac
from jax import numpy as jnp
from matplotlib import pyplot as plt
from tomographic_kernel.frames import ENU

from dsa2000_cal.delay_models.far_field import FarFieldDelayEngine


@pytest.mark.parametrize('time', [at.Time("2024-01-01T00:00:00", scale='utc'),
                                  at.Time("2024-04-01T00:00:00", scale='utc'),
                                  at.Time("2024-07-01T00:00:00", scale='utc'),
                                  at.Time("2024-10-01T00:00:00", scale='utc')])
@pytest.mark.parametrize('baseline', [3 * au.km, 10 * au.km])
def test_aberated_plane_of_sky(time: at.Time, baseline: au.Quantity):
    # aberation happens when uvw coordinates are assumed to be consistent for all points in the sky, however
    # tau = (-?) c * delay = u l + v m + w sqrt(1 - l^2 - m^2) ==> w = tau(l=0, m=0)
    # d/dl tau = u + w l / sqrt(1 - l^2 - m^2) ==> u = d/dl tau(l=0, m=0)
    # d/dm tau = v + w m / sqrt(1 - l^2 - m^2) ==> v = d/dm tau(l=0, m=0)
    # only true for l=m=0.

    # Let us see the error in delay for the approximation tau(l,m) = u*l + v*m + w*sqrt(1 - l^2 - m^2)
    array_location = ac.EarthLocation.of_site('vla')
    antennas = ENU(
        east=[0, baseline.to('km').value] * au.km,
        north=[0, 0] * au.km,
        up=[0, 0] * au.km,
        location=array_location,
        obstime=time
    )
    antennas = antennas.transform_to(ac.ITRS(obstime=time)).earth_location

    phase_centre = ENU(east=0, north=0, up=1, location=array_location, obstime=time).transform_to(ac.ICRS())

    engine = FarFieldDelayEngine(
        antennas=antennas,
        phase_center=phase_centre,
        start_time=time,
        end_time=time,
        verbose=True
    )
    uvw = engine.compute_uvw_jax(
        times=engine.time_to_jnp(time[None]),
        antenna_1=jnp.asarray([0]),
        antenna_2=jnp.asarray([1])
    )
    uvw = uvw * au.m

    lvec = mvec = jnp.linspace(-1, 1, 100)
    M, L = jnp.meshgrid(lvec, mvec, indexing='ij')
    N = jnp.sqrt(1 - L ** 2 - M ** 2)
    tau_approx = uvw[0, 0] * L + uvw[0, 1] * M + uvw[0, 2] * N
    tau_approx = tau_approx.to('m').value

    tau_exact = jax.vmap(
        lambda l, m: engine.compute_delay_from_lm_jax(
            l=l, m=m,
            t1=engine.time_to_jnp(time),
            i1=jnp.asarray(0),
            i2=jnp.asarray(1))
    )(L.ravel(), M.ravel()).reshape(L.shape)

    tau_diff = tau_exact - tau_approx

    # Plot exact, approx, and difference
    fig, axs = plt.subplots(3, 1, figsize=(5, 15), sharex=True, sharey=True, squeeze=False)

    im = axs[0, 0].imshow(tau_exact,
                          origin='lower',
                          extent=(lvec.min(), lvec.max(), mvec.min(), mvec.max()),
                          interpolation='nearest',
                          cmap='PuOr'
                          )
    axs[0, 0].set_title('Exact delay')
    fig.colorbar(im, ax=axs[0, 0],
                 label='Light travel dist. (m)')

    im = axs[1, 0].imshow(tau_approx,
                          origin='lower',
                          extent=(lvec.min(), lvec.max(), mvec.min(), mvec.max()),
                          interpolation='nearest',
                          cmap='PuOr'
                          )
    axs[1, 0].set_title('Approximated delay')
    fig.colorbar(im, ax=axs[1, 0],
                 label='Light travel dist. (m)')

    im = axs[2, 0].imshow(tau_diff,
                          origin='lower',
                          extent=(lvec.min(), lvec.max(), mvec.min(), mvec.max()),
                          interpolation='nearest',
                          cmap='PuOr'
                          )
    axs[2, 0].set_title(f'Difference: {time}')
    fig.colorbar(im, ax=axs[2, 0],
                 label='Light travel dist. (m)')

    axs[0, 0].set_ylabel('m')
    axs[1, 0].set_ylabel('m')
    axs[2, 0].set_ylabel('m')
    axs[2, 0].set_xlabel('l')

    fig.tight_layout()
    plt.show()

    freq = 70e6 * au.Hz
    difference_deg = tau_diff * freq.to('Hz').value / 299792458.0 * 180 / np.pi

    # The difference in delay in radians
    fig, axs = plt.subplots(1, 1, figsize=(5, 5), sharex=True, sharey=True, squeeze=False)

    im = axs[0, 0].imshow(difference_deg,
                          origin='lower',
                          extent=(lvec.min(), lvec.max(), mvec.min(), mvec.max()),
                          interpolation='nearest',
                          cmap='PuOr'
                          )
    # structure time like "1 Jan, 2024"
    axs[0, 0].set_title(
        fr'$\Delta \tau(l,m)$ over {baseline.to("km")} | {freq.to("MHz")} | {time.to_datetime().strftime("%d %b, %Y")}')
    fig.colorbar(im, ax=axs[0, 0],
                 label='Phase difference (deg)')

    axs[0, 0].set_ylabel('m')
    axs[0, 0].set_xlabel('l')

    fig.tight_layout()
    fig.savefig(f'phase_error_{baseline.to("km").value:.0f}km_{freq.to("MHz").value:.0f}MHz_{time.to_datetime().strftime("%d_%b_%Y")}.png')
    plt.show()

    # The difference in delay in (m)
    fig, axs = plt.subplots(1, 1, figsize=(5, 5), sharex=True, sharey=True, squeeze=False)

    im = axs[0, 0].imshow(tau_diff,
                          origin='lower',
                          extent=(lvec.min(), lvec.max(), mvec.min(), mvec.max()),
                          interpolation='nearest',
                          cmap='PuOr'
                          )
    # structure time like "1 Jan, 2024"
    axs[0, 0].set_title(
        fr'$\Delta \tau(l,m)$ over {baseline.to("km")} | {time.to_datetime().strftime("%d %b, %Y")}')
    fig.colorbar(im, ax=axs[0, 0],
                 label='Delay error (m)')

    axs[0, 0].set_ylabel('m')
    axs[0, 0].set_xlabel('l')

    fig.tight_layout()
    fig.savefig(
        f'delay_error_{baseline.to("km").value:.0f}km_{time.to_datetime().strftime("%d_%b_%Y")}.png')
    plt.show()
