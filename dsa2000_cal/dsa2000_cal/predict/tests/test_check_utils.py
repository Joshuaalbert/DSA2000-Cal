import jax.numpy as jnp
import numpy as np
import pytest

from dsa2000_cal.predict.check_utils import check_dft_predict_inputs, check_fft_predict_inputs


def test_check_predict_inputs():
    source = 5
    chan = 6
    time = 7
    ant = 8

    # Image has too many channels
    with pytest.raises(ValueError):
        check_dft_predict_inputs(
            freqs=np.random.rand(chan, 2),
            image=np.random.rand(source, chan + 1, 2, 2),
            gains=np.random.rand(source, time, ant, chan, 2, 2),
            lmn=np.random.rand(source, 3)
        )

    # Image has too many sources
    with pytest.raises(ValueError):
        check_dft_predict_inputs(
            freqs=np.random.rand(chan),
            image=np.random.rand(source + 1, chan, 2, 2),
            gains=np.random.rand(source, time, ant, chan, 2, 2),
            lmn=np.random.rand(source, 3)
        )

    # Gains has too many channels
    with pytest.raises(ValueError):
        check_dft_predict_inputs(
            freqs=np.random.rand(chan),
            image=np.random.rand(source, chan, 2, 2),
            gains=np.random.rand(source, time, ant, chan + 1, 2, 2),
            lmn=np.random.rand(source, 3)
        )

    # Gains has too many sources
    with pytest.raises(ValueError):
        check_dft_predict_inputs(
            freqs=np.random.rand(chan),
            image=np.random.rand(source, chan, 2, 2),
            gains=np.random.rand(source + 1, time, ant, chan, 2, 2),
            lmn=np.random.rand(source, 3)
        )

    # Image missing channel dimension
    with pytest.raises(ValueError):
        check_dft_predict_inputs(
            freqs=np.random.rand(chan),
            image=np.random.rand(source, 2, 2),
            gains=np.random.rand(source, time, ant, chan, 2, 2),
            lmn=np.random.rand(source, 3)
        )

    # Gains missing channel dimension
    with pytest.raises(ValueError):
        check_dft_predict_inputs(
            freqs=np.random.rand(chan),
            image=np.random.rand(source, chan, 2, 2),
            gains=np.random.rand(time, ant, 2, 2),
            lmn=np.random.rand(source, 3)
        )

    # Image flat coherencies
    with pytest.raises(ValueError):
        check_dft_predict_inputs(
            freqs=np.random.rand(chan),
            image=np.random.rand(source, chan, 4),
            gains=np.random.rand(source, time, ant, chan, 2, 2),
            lmn=np.random.rand(source, 3)
        )

    # Gains flat coherencies
    with pytest.raises(ValueError):
        check_dft_predict_inputs(
            freqs=np.random.rand(chan),
            image=np.random.rand(source, chan, 2, 2),
            gains=np.random.rand(source, time, ant, chan, 4),
            lmn=np.random.rand(source, 3)
        )

    # Correct inputs: DD gains
    assert check_dft_predict_inputs(
        freqs=np.random.rand(chan),
        image=np.random.rand(source, chan, 2, 2),
        gains=np.random.rand(source, time, ant, chan, 2, 2),
        lmn=np.random.rand(source, 3)
    )

    # Correct inputs: DI gains
    assert not check_dft_predict_inputs(
        freqs=np.random.rand(chan),
        image=np.random.rand(source, chan, 2, 2),
        gains=np.random.rand(time, ant, chan, 2, 2),
        lmn=np.random.rand(source, 3)
    )


@pytest.mark.parametrize("image_has_chan", [True, False])
@pytest.mark.parametrize("gains_have_chan", [True, False])
@pytest.mark.parametrize("stokes_I_image", [True, False])
@pytest.mark.parametrize("direction_dependent_gains", [True, False])
def test_check_fft_predict_inputs(direction_dependent_gains, image_has_chan, gains_have_chan, stokes_I_image):
    Nx = 100
    Ny = 100
    chan = 5
    time = 6
    ant = 7

    # Generate inputs
    if image_has_chan:
        if stokes_I_image:
            image = jnp.ones((chan, Nx, Ny))
            incorrect_image_1 = jnp.ones((chan + 1, Nx, Ny))
            l0 = m0 = dl = dm = jnp.zeros((chan,))
        else:
            image = jnp.ones((chan, Nx, Ny, 2, 2))
            incorrect_image_1 = jnp.ones((chan + 1, Nx, Ny, 2, 2))
            l0 = m0 = dl = dm = jnp.zeros((chan,))
    else:
        if stokes_I_image:
            image = jnp.ones((Nx, Ny))
            incorrect_image_1 = jnp.ones((Nx, Ny, 4))
            l0 = m0 = dl = dm = jnp.zeros(())
        else:
            image = jnp.ones((Nx, Ny, 2, 2))
            incorrect_image_1 = jnp.ones((Nx, Ny, 1))
            l0 = m0 = dl = dm = jnp.zeros(())

    if gains_have_chan:
        if direction_dependent_gains:
            gains = jnp.ones((Nx, Ny, time, ant, chan, 2, 2))
            incorrect_gains_1 = jnp.ones((Nx, Ny, time, ant, chan + 1, 2, 2))
        else:
            gains = jnp.ones((time, ant, chan, 2, 2))
            incorrect_gains_1 = jnp.ones((time, ant, chan + 1, 2, 2))

    else:
        if direction_dependent_gains:
            gains = jnp.ones((Nx, Ny, time, ant, 2, 2))
            incorrect_gains_1 = jnp.ones((Nx, Ny, time, ant, 1))
        else:
            gains = jnp.ones((time, ant, 2, 2))
            incorrect_gains_1 = jnp.ones((time, ant, 4))

    freqs = jnp.ones((chan,))

    _direction_dependent_gains, _image_has_chan, _gains_have_chan, _stokes_I_image = check_fft_predict_inputs(
        freqs=freqs,
        image=image,
        gains=gains,
        l0=l0,
        m0=m0,
        dl=dl,
        dm=dm
    )
    assert _direction_dependent_gains == direction_dependent_gains
    assert _image_has_chan == image_has_chan
    assert _gains_have_chan == gains_have_chan
    assert _stokes_I_image == stokes_I_image

    with pytest.raises(ValueError):
        check_fft_predict_inputs(
            freqs=freqs,
            image=incorrect_image_1,
            gains=gains,
            l0=l0,
            m0=m0,
            dl=dl,
            dm=dm
        )

    with pytest.raises(ValueError):
        check_fft_predict_inputs(
            freqs=freqs,
            image=image,
            gains=incorrect_gains_1,
            l0=l0,
            m0=m0,
            dl=dl,
            dm=dm
        )

    with pytest.raises(ValueError):
        check_fft_predict_inputs(
            freqs=freqs,
            image=incorrect_image_1,
            gains=incorrect_gains_1,
            l0=l0,
            m0=m0,
            dl=dl,
            dm=dm
        )
