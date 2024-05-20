import re
from typing import NamedTuple

import numpy as np
from astropy import coordinates as ac, units as au
from h5parm.utils import parse_coordinates_bbs

from dsa2000_cal.common.astropy_utils import dimensionless


def parse_wsclean_source_line(line):
    # Regular expression to match the required fields
    pattern = r'([^,]+),([^,]+),([^,]+),([^,]+),([^,]+),\[(.*?)\],([^,]*),([^,]*),([^,]*),([^,]*),([^,]*)'
    matches = re.search(pattern, line)

    if matches:
        fields = matches.groups()
        name = fields[0]
        type_ = fields[1]
        ra = fields[2]
        dec = fields[3]
        stokes_I = float(fields[4]) if fields[4] else None
        spectral_index = list(map(float, filter(None, fields[5].split(','))))
        logarithmic_si = True if fields[6].lower() == 'true' else False if fields[6] else None
        reference_frequency = float(fields[7]) if fields[7] else None
        major_axis = float(fields[8]) if fields[8] else None
        minor_axis = float(fields[9]) if fields[9] else None
        orientation = float(fields[10]) if fields[10] else None

        return {
            'Name': name,
            'Type': type_,
            'Ra': ra,
            'Dec': dec,
            'I': stokes_I,
            'SpectralIndex': spectral_index,
            'LogarithmicSI': logarithmic_si,
            'ReferenceFrequency': reference_frequency,
            'MajorAxis': major_axis,
            'MinorAxis': minor_axis,
            'Orientation': orientation
        }
    else:
        return None


class WSCleanLine(NamedTuple):
    type_: str
    direction: ac.ICRS
    spectrum: au.Quantity
    major: au.Quantity | None
    minor: au.Quantity | None
    theta: au.Quantity | None


def parse_and_process_wsclean_source_line(line, freqs: au.Quantity) -> WSCleanLine | None:
    """
    Parse a WSClean source line and return a WSCleanLine object.

    Args:
        line: the line to parse
        freqs: the frequencies to use for the spectrum

    Returns:
        WSCleanLine object
    """
    result = parse_wsclean_source_line(line)
    if result is None:
        return None
    direction = parse_coordinates_bbs(result['Ra'], result['Dec'])

    if result['LogarithmicSI']:
        spectrum = wsclean_log_spectral_index_spectrum_fn(
            stokesI=result['I'] * au.Jy,
            ref_nu=result['ReferenceFrequency'] * au.Hz,
            nu=freqs,
            spectral_indices=result['SpectralIndex']
        )
    else:
        spectrum = wsclean_linear_spectral_index_spectrum_fn(
            stokesI=result['I'] * au.Jy,
            ref_nu=result['ReferenceFrequency'] * au.Hz,
            nu=freqs,
            spectral_indices=result['SpectralIndex'] * au.Jy
        )

    major = result['MajorAxis'] * au.arcsec if result['MajorAxis'] is not None else None
    minor = result['MinorAxis'] * au.arcsec if result['MinorAxis'] is not None else None
    theta = result['Orientation'] * au.deg if result['Orientation'] is not None else None

    return WSCleanLine(
        type_=result['Type'],
        direction=direction,
        spectrum=spectrum,
        major=major,
        minor=minor,
        theta=theta
    )


def wsclean_log_spectral_index_spectrum_fn(stokesI, ref_nu, nu, spectral_indices):
    # Wrong -> flux(nu) = exp ( log stokesI + term0 log(nu/refnu) + term1 log(nu/refnu)^2 + ... )
    # Correct -> flux(nu) = I0 * (v/v0) ^ (c0 + c1 * log10(v/v0) + c2 * log10(v/v0)^2 + …)
    exponents = np.arange(len(spectral_indices))  # [n]
    freq_ratio = dimensionless(nu / ref_nu)  # [num_freq]
    tmp = spectral_indices * np.log10(freq_ratio[:, None]) ** exponents  # [num_freq, n]
    scaling = freq_ratio ** np.sum(tmp, axis=-1)  # [num_freq]
    return stokesI * scaling


def wsclean_linear_spectral_index_spectrum_fn(stokesI: au.Quantity, ref_nu: au.Quantity, nu: au.Quantity,
                                              spectral_indices: np.ndarray):
    # flux(nu) = I0 + c0 * (v/v0 - 1) + c1 * (v /v0 - 1)^2 + c2 * (v/v0 -1)^3 + …
    exponents = np.arange(len(spectral_indices)) + 1  # [n]
    freq_ratio = dimensionless(nu / ref_nu)  # [num_freq]
    shift = spectral_indices * (freq_ratio[:, None] - 1.) ** exponents  # [num_freq, n]
    return stokesI + np.sum(shift, axis=-1)
