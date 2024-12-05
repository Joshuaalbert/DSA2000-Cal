import astropy.coordinates as ac
import astropy.time as at
import astropy.units as au
import jax.lax
import numpy as np
import pylab as plt
from jax import numpy as jnp

from dsa2000_cal.common.noise import calc_baseline_noise
from dsa2000_cal.common.quantity_utils import time_to_jnp, quantity_to_jnp
from dsa2000_fm.forward_models.streaming.distributed.calibrator import Calibration
from dsa2000_fm.forward_models.utils import ObservationSetup
from dsa2000_cal.visibility_model.source_models.celestial.base_point_source_model import build_point_source_model


def main():
    full_stokes = True
    D = 1
    T = 1
    setup = ObservationSetup.create_tracking_from_array(
        array_name='dsa2000W_small',
        ref_time=at.Time('2021-01-01T00:00:00', scale='utc'),
        num_timesteps=T,
        phase_center=ac.ICRS(ra=20 * au.deg, dec=30 * au.deg)
    )
    print(setup)
    num_ant = len(setup.antennas)
    F = len(setup.freqs)

    calibration = Calibration(
        full_stokes=full_stokes,
        num_ant=num_ant,
        num_background_source_models=0,
        verbose=True
    )
    visibility_coords = setup.far_field_delay_engine.compute_visibility_coords(
        freqs=quantity_to_jnp(setup.freqs),
        times=time_to_jnp(setup.obstimes, setup.ref_time),
        with_autocorr=True,
        convention='physical'
    )

    true_gains = 1 + 0.1j * jax.random.normal(jax.random.PRNGKey(2), (D, T, num_ant, F, 1, 1))
    true_gains *= jnp.exp(1j * jnp.pi / 2)

    source_models = []
    vis_model = []
    for d in range(D):
        ra = (20 + 1 * np.random.normal(size=(1,))) * au.deg
        dec = (30 + 1 * np.random.normal(size=(1,))) * au.deg
        point_source_model = build_point_source_model(
            model_freqs=setup.freqs,
            ra=ra,
            dec=dec,
            A=np.ones((1, F, 2, 2)) * au.Jy if full_stokes else np.ones((1, F)) * au.Jy
        )
        source_models.append(point_source_model)
        g1 = true_gains[d][:, visibility_coords.antenna1, :, :, :]  # [T, B, F, 1, 1]
        g2 = true_gains[d][:, visibility_coords.antenna2, :, :, :]  # [T, B, F, 1, 1]
        vis_model.append(
            point_source_model.predict(
                visibility_coords=visibility_coords,
                gain_model=None,
                near_field_delay_engine=setup.near_field_delay_engine,
                far_field_delay_engine=setup.far_field_delay_engine,
                geodesic_model=setup.geodesic_model
            ) * (g1 * jnp.conj(g2))
        )

    vis_model = jnp.stack(vis_model, axis=0)
    vis_data = jnp.sum(vis_model, axis=0)

    B = len(visibility_coords.antenna1)

    assert np.shape(vis_data) == (T, B, F, 2, 2)
    assert np.shape(vis_model) == (D, T, B, F, 2, 2)
    assert np.shape(true_gains) == (D, T, num_ant, F, 1, 1)

    noise_scale = calc_baseline_noise(
        system_equivalent_flux_density=5000,
        chan_width_hz=130e3 * 40,
        t_int_s=1.5 * 4
    )
    print(noise_scale)
    noise_scale /= np.sqrt(2.)

    vis_data += noise_scale * jax.lax.complex(
        jax.random.normal(jax.random.PRNGKey(0), vis_data.shape),
        jax.random.normal(jax.random.PRNGKey(1), vis_data.shape)
    )
    weights = jnp.full(vis_data.shape, 1. / noise_scale ** 2)
    flags = jnp.zeros(vis_data.shape, dtype=jnp.bool_)

    state = None
    gains, vis_data_residuals, state, diagnostics = calibration.step(
        vis_model=vis_model,
        vis_data=vis_data,
        weights=weights,
        flags=flags,
        freqs=visibility_coords.freqs,
        times=visibility_coords.times,
        antenna1=visibility_coords.antenna1,
        antenna2=visibility_coords.antenna2,
        state=state
    )
    fig, axs = plt.subplots(2, 1, figsize=(6, 10), sharex=True)
    mask = diagnostics.damping > 0.
    axs[0].plot(diagnostics.iteration[mask], diagnostics.F_norm[mask])
    axs[0].set_ylabel('F norm')
    axs[1].plot(diagnostics.iteration[mask], diagnostics.error[mask])
    axs[1].set_ylabel('error')
    axs[-1].set_xlabel('iteration')
    plt.show()

    # plot error in abo and phase space

    baseline_dist = np.linalg.norm(visibility_coords.uvw, axis=-1)  # [T, B]
    freq_idx = 0
    time_idx = 0
    for d in range(D):
        fig, axs = plt.subplots(2, 1, figsize=(6, 10), sharex=True)

        for p_idx, q_idx in [(0, 0), (0, 1), (1, 1)]:
            g1 = gains[d, :, freq_idx, p_idx, q_idx][visibility_coords.antenna1]
            g2 = gains[d, :, freq_idx, p_idx, q_idx][visibility_coords.antenna2]
            g12 = g1 * np.conj(g2)
            g1_true = true_gains[d, time_idx, :, freq_idx, 0, 0][visibility_coords.antenna1]
            g2_true = true_gains[d, time_idx, :, freq_idx, 0, 0][visibility_coords.antenna2]
            g12_true = g1_true * np.conj(g2_true)
            error_abs = np.abs(g12) - np.abs(g12_true)
            error_phase = np.angle(g12) - np.angle(g12_true)

            axs[0].plot(baseline_dist[time_idx], error_abs, 'o',
                        label=f"{p_idx}, {q_idx}")
            axs[1].plot(baseline_dist[time_idx], error_phase, 'o',
                        label=f"{p_idx}, {q_idx}")
        axs[0].legend()
        axs[1].legend()
        axs[0].set_ylabel('gain error abs')
        axs[1].set_ylabel('gain error phase')
        axs[-1].set_xlabel('baseline distance')

        fig.tight_layout()
        plt.show()


if __name__ == '__main__':
    main()
