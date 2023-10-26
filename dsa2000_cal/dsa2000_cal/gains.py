import numpy as np
from h5parm import DataPack


def extract_scalar_gains(h5parm: str) -> np.ndarray:
    """
    Extract scalar gains from an h5parm file, with same gain on diagonal of Jones matrix.

    Args:
        h5parm: path to h5parm file

    Returns:
        gains: [time, ant, source, chan, 2, 2]
    """
    gains_list = []
    with DataPack(h5parm, readonly=True) as dp:
        if dp.axes_order != ['pol', 'dir', 'ant', 'freq', 'time']:
            raise ValueError(f"Expects axes order must be ['pol', 'dir', 'ant', 'freq', 'time'], got {dp.axes_order}")
        dp.current_solset = 'sol000'
        if 'phase000' in dp.soltabs:
            phase, axes = dp.phase
            Np, Nd, Na, Nf, Nt = phase.shape
            phase = np.reshape(np.transpose(phase, (4, 2, 1, 3, 0)),
                               (Nt, Na, Nd, Nf, Np))  # [time, ant, dir, freq, pol]
            gains = np.zeros((Nt, Na, Nd, Nf, 2, 2), dtype=np.complex64)
            gains[..., 0, 0] = np.exp(1j * phase[..., 0])
            if Np == 1:
                gains[..., 1, 1] = gains[..., 0, 0]
            else:
                gains[..., 1, 1] = np.exp(1j * phase[..., 1])
            gains_list.append(gains)
        # if amplitude is present, multiply by it
        if 'amplitude000' in dp.soltabs:
            amplitude, axes = dp.amplitude
            Np, Nd, Na, Nf, Nt = amplitude.shape
            amplitude = np.reshape(np.transpose(amplitude, (4, 2, 1, 3, 0)),
                                   (Nt, Na, Nd, Nf, Np))  # [time, ant, dir, freq, pol]
            gains = np.zeros((Nt, Na, Nd, Nf, 2, 2), dtype=np.complex64)
            gains[..., 0, 0] = amplitude[..., 0]
            if Np == 1:
                gains[..., 1, 1] = gains[..., 0, 0]
            else:
                gains[..., 1, 1] = amplitude[..., 1]
            gains_list.append(gains)
        else:
            print(f"Amplitude not present in h5parm.")
    if len(gains_list) == 0:
        raise ValueError("No gains found in h5parm.")
    output = gains_list[0]
    for gains in gains_list[1:]:
        output *= gains
    return output
