import itertools
import json

import jax
import numpy as np
import pytest
from astropy import time as at, units as au, coordinates as ac
from astropy.coordinates import offset_by
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
    fig.savefig(
        f'phase_error_{baseline.to("km").value:.0f}km_{freq.to("MHz").value:.0f}MHz_{time.to_datetime().strftime("%d_%b_%Y")}.png')
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


def prepare_standard_test():
    np.random.seed(42)

    obstime = at.Time("2024-01-01T00:00:00", scale='utc')
    array_location = ac.EarthLocation.of_site('vla')

    phase_centers = []

    for body in ['sun', 'jupiter', 'moon', 'neptune', 'mars']:
        body_position_bcrs, body_velocity_bcrs = ac.get_body_barycentric_posvel(
            body=body,
            time=obstime
        )  # [T, N]
        frame = ac.CartesianRepresentation(body_position_bcrs)
        source = ac.ICRS().realize_frame(frame)

        for displacement_deg in [0., 0.25, 0.5, 1., 2., 4., 8.]:
            # displace by 1 deg in some random direction
            new_ra, new_dec = offset_by(
                lon=source.ra,
                lat=source.dec,
                posang=np.random.uniform(0., 2 * np.pi) * au.rad,
                distance=displacement_deg * au.deg
            )
            phase_centers.append(ac.ICRS(ra=new_ra, dec=new_dec))

    antennas = []

    for b_east in [0, 3, 10, 100]:
        for b_north in [0, 3, 10, 100]:
            az = np.arctan2(b_east, b_north) * au.rad
            dist = np.sqrt(b_east ** 2 + b_north ** 2)
            antenna = ac.AltAz(
                az=az,
                alt=0 * au.deg,
                distance=dist * au.km,
                location=array_location,
                obstime=obstime
            )
            antennas.append(antenna.transform_to(ac.ITRS(obstime=obstime, location=array_location)))

    antennas = ac.concatenate(antennas).earth_location


    antenna_1, antenna_2 = jnp.asarray(list(itertools.combinations(range(len(antennas)), 2))).T

    return obstime, array_location, antennas, antenna_1, antenna_2, phase_centers


def test_standard_test_dsa2000():
    obstime, array_location, antennas, antenna_1, antenna_2, phase_centers = prepare_standard_test()

    # For each phase center produce the UVW coordinates ordered by the antenna1 and antenna2, and save to file

    results = {}

    for i, phase_center in enumerate(phase_centers):
        far_field_engine = FarFieldDelayEngine(
            antennas=antennas,
            phase_center=phase_center,
            start_time=obstime,
            end_time=obstime,
            verbose=True
        )
        times = jnp.repeat(far_field_engine.time_to_jnp(obstime), len(antenna_1))
        uvw = far_field_engine.compute_uvw_jax(
            times=times,
            antenna_1=antenna_1,
            antenna_2=antenna_2
        )  # [N, 3] in meters
        results[str(i)] = uvw.tolist()

    with open('dsa2000_results.json', 'w') as f:
        json.dump(results, f, indent=2)

def test_compare_to_calc():
    with open('dsa2000_results.json', 'r') as f:
        results_dsa = json.load(f)
    with open('pycalc_results_noatmo.json', 'r') as f:
        results_calc = json.load(f)

    obstime, array_location, antennas, antenna_1, antenna_2, phase_centers = prepare_standard_test()

    antennas_gcrs = antennas.get_gcrs(obstime)
    antennas_gcrs = antennas_gcrs.cartesian.xyz.T # [N, 3]

    baselines = antennas_gcrs[antenna_2, :] - antennas_gcrs[antenna_1, :]
    baselines_norm = np.linalg.norm(baselines, axis=1).to('m').value

    diffs = []
    alts = []
    baseline_norms = []
    for key, phase_center in enumerate(phase_centers):
        alt = phase_center.transform_to(ac.AltAz(obstime=obstime, location=array_location)).alt.to('deg')

        value_calc = np.asarray(results_calc[str(key)])
        value_dsa = np.asarray(results_dsa[str(key)])
        value_dsa_norm = np.linalg.norm(value_dsa, axis=1)
        value_calc_norm = np.linalg.norm(value_calc, axis=1)
        for i1, i2, b_norm, dsa_norm, calc_norm in zip(antenna_1, antenna_2, baselines_norm, value_dsa_norm, value_calc_norm):
            if np.abs(dsa_norm - calc_norm) > 10:
                print(f"alt: {alt}, phase center: {phase_center.ra} {phase_center.dec} i1: {i1} i2: {i2} Baseline: {b_norm}, DSA: {dsa_norm}, Calc: {calc_norm}")

        diff = (value_dsa - value_calc).flatten()
        mean_diff = np.mean(diff)
        std_diff = np.std(diff)
        print(f"alt: {alt}, phase center: {phase_center.ra} {phase_center.dec} Mean diff: {mean_diff}, Std diff: {std_diff}")

        b_norm = np.repeat(np.linalg.norm(baselines[antenna_2] - baselines[antenna_1], axis=1, keepdims=True), 3, axis=1).flatten().to('m').value
        baseline_norms.extend(b_norm)
        diffs.extend((diff).tolist())
        alts.extend([alt.deg] * np.size(diff))

    import pylab as plt
    plt.hist(diffs, bins='auto')
    plt.xlabel('Difference (m)')
    plt.ylabel('Counts')
    plt.title('Distribution of differences between DSA2000 and PyCalc11')
    plt.show()
    # Do a scatter plot of the differences with color by alt
    fig, axs = plt.subplots(1, 1, figsize=(5, 5), squeeze=True)
    sc=axs.scatter(baseline_norms, diffs, c=alts, s=1, cmap='hsv',vmin=-90, vmax=90)
    fig.colorbar(sc, ax=axs, label='Alt (deg)')
    axs.set_xlabel('Baseline (m)')
    axs.set_ylabel('Diff (m)')
    axs.set_title('Difference vs Baseline')
    fig.tight_layout()
    plt.show()