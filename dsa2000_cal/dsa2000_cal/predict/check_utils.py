from typing import Tuple

import jax
import numpy as np


def check_dft_predict_inputs(freqs, image, gains, lmn) -> bool:
    """
    Check the inputs for predict.

    Args:
        freqs: [chan] frequencies in Hz.
        image: [source, chan, 2, 2] in [[xx, xy], [yx, yy]] format.
        gains: [[source,] time, ant, chan, 2, 2]
        lmn: [source, 3]

    Returns:
        direction_dependent_gains: bool
    """
    if len(np.shape(lmn)) != 2 or np.shape(lmn)[1] != 3:
        raise ValueError(f"Expected lmn to have shape [source, 3], got {np.shape(lmn)}")
    if len(np.shape(freqs)) != 1:
        raise ValueError(f"Expected freqs to have shape [chan], got {np.shape(freqs)}")

    num_sources = np.shape(lmn)[0]
    num_chan = np.shape(freqs)[0]
    if np.shape(image) != (num_sources, num_chan, 2, 2):
        raise ValueError(f"Expected image to have shape [source, chan, 2, 2], got {np.shape(image)}")

    direction_dependent_gains = len(np.shape(gains)) == 6
    if direction_dependent_gains:
        _, time, ant, _, _, _ = np.shape(gains)
        if np.shape(gains) != (num_sources, time, ant, num_chan, 2, 2):
            raise ValueError(
                f"Expected gains to have shape [source, time, ant, chan, 2, 2], got {np.shape(gains)}.")
    else:
        if len(np.shape(gains)) != 5:
            raise ValueError(
                f"If gains doesn't have a source dimension then it must be shaped (time, ant, chan, 2, 2), "
                f"got {np.shape(gains)}.")
        time, ant, _, _, _ = np.shape(gains)
        if np.shape(gains) != (time, ant, num_chan, 2, 2):
            raise ValueError(
                f"Expected gains to have shape [time, ant, chan, 2, 2], got {np.shape(gains)}.")

    return direction_dependent_gains


def check_fft_predict_inputs(freqs: jax.Array, image: jax.Array, gains: jax.Array, l0: jax.Array, m0: jax.Array,
                             dl: jax.Array, dm: jax.Array) -> Tuple[bool, bool, bool, bool]:
    """
    Check the inputs for predict.

    Args:
        freqs: [chan] frequencies in Hz.
        image: [[chan,] Nx, Ny [2, 2]]
        gains: [[Nx, Ny,] time, ant, [chan,] 2, 2]
        l0: [[chan,]]
        m0: [[chan,]]
        dl: [[chan,]]
        dm: [[chan,]]

    Returns:
        direction_dependent_gains: bool
        image_has_chan: bool
        gains_have_chan: bool
        stokes_I_image: bool
    """
    if len(np.shape(freqs)) != 1:
        raise ValueError(f"Expected freqs to have shape [chan], got {np.shape(freqs)}")
    num_chans = np.shape(freqs)[0]

    # Check image
    if len(np.shape(image)) == 2:  # should be [Nx, Ny]
        image_has_chan = False
        stokes_I_image = True
        Nx, Ny = np.shape(image)
        # Check l0, m0, dl, dm are scalars
        if np.shape(l0) != () or np.shape(m0) != () or np.shape(dl) != () or np.shape(dm) != ():
            raise ValueError("If image doesn't have a channel then l0, m0, dl, and dm must be scalars.")
    elif len(np.shape(image)) == 3:  # should be [chan, Nx, Ny]
        image_has_chan = True
        stokes_I_image = True
        if np.shape(image)[0] != num_chans:
            raise ValueError("If image has a channel then it must match freqs.")
        _, Nx, Ny = np.shape(image)
        # Check l0, m0, dl, dm are [chan]
        if (np.shape(l0) != (num_chans,) or np.shape(m0) != (num_chans,)
                or np.shape(dl) != (num_chans,) or np.shape(dm) != (num_chans,)):
            raise ValueError("If image has a channel then l0, m0, dl, and dm must be shaped (freqs,).")
    elif len(np.shape(image)) == 4:  # should be [Nx, Ny, 2, 2]
        image_has_chan = False
        stokes_I_image = False
        Nx, Ny, _, _ = np.shape(image)
        # Check l0, m0, dl, dm are scalars
        if np.shape(l0) != () or np.shape(m0) != () or np.shape(dl) != () or np.shape(dm) != ():
            raise ValueError("If image doesn't have a channel then l0, m0, dl, and dm must be scalars.")
        if np.shape(image) != (Nx, Ny, 2, 2):
            raise ValueError(f"Expected image to have shape [Nx, Ny, 2, 2], got {np.shape(image)}")
    elif len(np.shape(image)) == 5:  # should be [chan, Nx, Ny, 2, 2]
        image_has_chan = True
        stokes_I_image = False
        _, Nx, Ny, _, _ = np.shape(image)
        # Check l0, m0, dl, dm are [chan]
        if (np.shape(l0) != (num_chans,) or np.shape(m0) != (num_chans,)
                or np.shape(dl) != (num_chans,) or np.shape(dm) != (num_chans,)):
            raise ValueError("If image has a channel then l0, m0, dl, and dm must be shaped (freqs,).")
        if np.shape(image) != (num_chans, Nx, Ny, 2, 2):
            raise ValueError(f"Expected image to have shape [chan, Nx, Ny, 2, 2], got {np.shape(image)}")
    else:
        raise ValueError(f"Expected image to have shape [[chan,] Nx, Ny[, 2, 2]], got {np.shape(image)}")

    # Check gains
    if len(np.shape(gains)) == 4:  # should be [time, ant, 2, 2]
        gains_have_chan = False
        direction_dependent_gains = False
        time, ant, _, _ = np.shape(gains)
        if np.shape(gains) != (time, ant, 2, 2):
            raise ValueError(f"Expected gains to have shape [time, ant, 2, 2], got {np.shape(gains)}")
    elif len(np.shape(gains)) == 5:  # should be [time, ant, chan, 2, 2]
        gains_have_chan = True
        direction_dependent_gains = False
        time, ant, _, _, _ = np.shape(gains)
        if np.shape(gains) != (time, ant, num_chans, 2, 2):
            raise ValueError(f"Expected gains to have shape [time, ant, chan, 2, 2], got {np.shape(gains)}")
    elif len(np.shape(gains)) == 6:  # should be [Nx, Ny, time, ant, 2, 2]
        gains_have_chan = False
        direction_dependent_gains = True
        _, _, time, ant, _, _ = np.shape(gains)
        if np.shape(gains) != (Nx, Ny, time, ant, 2, 2):
            raise ValueError(f"Expected gains to have shape [Nx, Ny, time, ant, 2, 2], got {np.shape(gains)}")
    elif len(np.shape(gains)) == 7:  # should be [Nx, Ny, time, ant, chan, 2, 2]
        gains_have_chan = True
        direction_dependent_gains = True
        _, _, time, ant, _, _, _ = np.shape(gains)
        if np.shape(gains) != (Nx, Ny, time, ant, num_chans, 2, 2):
            raise ValueError(f"Expected gains to have shape [Nx, Ny, time, ant, chan, 2, 2], got {np.shape(gains)}")
    else:
        raise ValueError(f"Expected gains to have shape [[Nx, Ny,] time, ant, [chan,] 2, 2], got {np.shape(gains)}")

    return direction_dependent_gains, image_has_chan, gains_have_chan, stokes_I_image
