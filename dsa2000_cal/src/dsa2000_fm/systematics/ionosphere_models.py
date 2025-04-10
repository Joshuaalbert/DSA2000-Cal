from datetime import datetime
from typing import List, NamedTuple

import astropy.units as au
import jax
import numpy as np
import pylab as plt
import requests
from jax import numpy as jnp

from dsa2000_common.common.array_types import FloatArray
from dsa2000_common.common.logging import dsa_logger
from dsa2000_common.common.quantity_utils import quantity_to_np
from dsa2000_fm.systematics.ionosphere import IonosphereLayer, IonosphereMultiLayer


class IonosphereModel(NamedTuple):
    dt: datetime  # The datetime with timezone
    confidence_score: float  # CS metric
    f0E: float  # MHz (E-layer critical frequency)
    f0F1: float  # MHz (F1-layer critical frequency)
    f0F2: float  # MHz (F2-layer critical frequency)
    hmE: float  # km (E-layer height)
    hmF1: float  # km (F1-layer height)
    hmF2: float  # km (F2-layer height)
    yE: float  # km (E-layer half thickness)
    yF1: float  # km (F1-layer half thickness)
    yF2: float  # km (F2-layer half thickness)
    vtec: float  # TECU (Total electron content)


def plot_fetched_data(data: List[IonosphereModel], savefile: str = None):
    data = jax.tree.map(lambda *x: np.stack(x), *data)
    fig, axs = plt.subplots(3, 4, figsize=(15, 10))

    for f0, hm, y, col, name in [
        (data.f0E, data.hmE, data.yE, 0, 'E'),
        (data.f0F1, data.hmF1, data.yF1, 1, 'F1'),
        (data.f0F2, data.hmF2, data.yF2, 2, 'F2')
    ]:
        axs[0, col].hist(f0, bins='auto')
        # axs[0, col].set_title(f"{name} f0 Histogram")
        axs[0, col].set_xlabel(f"{name} f0 [MHz]")

        axs[1, col].hist(hm, bins='auto')
        # axs[1, col].set_title(f"{name} hm Histogram")
        axs[1, col].set_xlabel(f"{name} hm [km]")

        axs[2, col].hist(y, bins='auto')
        # axs[2, col].set_title(f"{name} y Histogram")
        axs[2, col].set_xlabel(f"{name} y [km]")

    axs[0, 3].hist(data.vtec, bins='auto')
    # axs[0, 3].set_title("VTEC Histogram")
    axs[0, 3].set_xlabel("VTEC [TECU]")

    # Turn off extra
    axs[1, 3].set_visible(False)
    axs[2, 3].set_visible(False)

    if savefile is not None:
        fig.savefig("ionosphere_histograms.png")
        plt.close(fig)
    else:
        plt.show()


def fetch_ionosphere_data(start: datetime, end: datetime, ursi_station: str) -> List[IonosphereModel]:
    """
    Downloads ionospheric data for the given station and datetime interval,
    and returns a list of IonosphereModel instances. Lines that have any missing
    required values are skipped.

    Parameters:
      start (datetime): Start time of the query interval.
      end (datetime): End time of the query interval.
      ursi_station (str): The URSI code for the station.

    Returns:
      List[IonosphereModel]: A list of parsed measurement data.
    """
    base_url = "https://lgdc.uml.edu/common/DIDBGetValues"

    # Format dates as "YYYY/MM/DD HH:MM:SS" which will be URL-encoded by requests
    params = {
        "ursiCode": ursi_station,
        "charName": "foF2,foF1,foE,hmF2,hmF1,hmE,yF2,yF1,yE,TEC",
        "DMUF": "3000",
        "fromDate": start.strftime("%Y/%m/%d %H:%M:%S"),
        "toDate": end.strftime("%Y/%m/%d %H:%M:%S")
    }

    response = requests.get(base_url, params=params)
    response.raise_for_status()  # Raise an error if the request failed

    # Retrieve the response text containing header lines starting with '#'
    text = response.text
    models: List[IonosphereModel] = []

    # Process each non-empty, non-comment line.
    # The header line describes the columns (after skipping the comment lines):
    # Time, CS, foF2, QD, foF1, QD, foE, QD, hmE, QD, yE, QD, hmF2, QD, hmF1, QD, yF2, QD, yF1, QD, TEC, QD
    # We map these fields to the desired NamedTuple as follows:
    #   - dt: index 0 (datetime string)
    #   - confidence_score: index 1
    #   - f0F2: index 2, but note required output order is f0E, f0F1, f0F2 so we re-map accordingly.
    #   - f0F1: index 4
    #   - f0E: index 6
    #   - hmE: index 8
    #   - yE: index 10
    #   - hmF2: index 12
    #   - hmF1: index 14
    #   - yF2: index 16
    #   - yF1: index 18
    #   - vtec: index 20
    #
    # Finally, we reassemble these into:
    # IonosphereModel(dt, CS, f0E, f0F1, f0F2, hmE, hmF1, hmF2, yE, yF1, yF2, TEC)

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        tokens = line.split()

        # Ensure we have at least enough tokens to cover all columns.
        if len(tokens) < 21:
            continue

        # Define the indices for the tokens we need:
        # Note: tokens at odd positions following each measurement are quality data, which we ignore.
        required_indices = [0, 1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
        try:
            # Test conversion for each field (except the datetime at token[0])
            for idx in required_indices:
                if idx == 0:
                    continue
                # Attempt to convert the token to a float to ensure it is valid
                float(tokens[idx])

            # Parse the datetime field.
            dt_str = tokens[0]
            # Convert the ISO timestamp ending with "Z" to one with proper timezone info.
            if dt_str.endswith("Z"):
                dt_str = dt_str[:-1] + "+00:00"
            dt_obj = datetime.fromisoformat(dt_str)

            # Extract the values based on their column positions.
            cs = float(tokens[1])
            f0F2 = float(tokens[2])
            f0F1 = float(tokens[4])
            f0E = float(tokens[6])
            hmE = float(tokens[8])
            yE = float(tokens[10])
            hmF2 = float(tokens[12])
            hmF1 = float(tokens[14])
            yF2 = float(tokens[16])
            yF1 = float(tokens[18])
            vtec = float(tokens[20])

            # Note: reordering to match the desired tuple order:
            # f0E, f0F1, f0F2, hmE, hmF1, hmF2, yE, yF1, yF2, vtec.
            model = IonosphereModel(
                dt=dt_obj,
                confidence_score=cs,
                f0E=f0E,
                f0F1=f0F1,
                f0F2=f0F2,
                hmE=hmE,
                hmF1=hmF1,
                hmF2=hmF2,
                yE=yE,
                yF1=yF1,
                yF2=yF2,
                vtec=vtec
            )
            models.append(model)
        except ValueError:
            # If any conversion fails (i.e. if there is a missing or non-numeric value),
            # skip this line.
            continue

    return models


def construct_canonical_ionosphere(x0_radius: FloatArray, turbulent: bool = True, dawn: bool = True,
                                   high_sun_spot: bool = True):
    """
    Construct a canonical ionosphere model.

    Args:
        x0_radius: see `compute_x0_radius`
        turbulent: If True then uses larger relative spatial variations
        dawn: If True then uses smaller length scales
        high_sun_spot: If True then uses higher mean electron densities

    Returns:
        An ionosphere model
    """
    _bottom_E = 90
    _width_E = 10
    _bottom_F = 250
    _width_F = 100
    _bottom_velocity_E = 0.2
    _bottom_velocity_F = 0.3
    _radial_velocity_E = 0.
    _radial_velocity_F = 0.
    if high_sun_spot:
        _fed_mu_E = 10.  # 10^11 e / m^3
        _fed_mu_F = 200.  # 2 * 10^12 e / m^3
    else:
        _fed_mu_E = 1.  # 10^10 e / m^3
        _fed_mu_F = 50.  # 5 * 10^11 e / m^3
    if turbulent:
        _sigma_factor = 0.5
    else:
        _sigma_factor = 0.25
    if dawn:
        _length_scale_E = 1.
        _length_scale_F = 5.
    else:
        _length_scale_E = 2.
        _length_scale_F = 10.

    layers = [
        IonosphereLayer(
            length_scale=_length_scale_E,
            longitude_pole=0.,
            latitude_pole=np.pi / 2.,
            bottom_velocity=_bottom_velocity_E,
            radial_velocity=_radial_velocity_E,
            x0_radius=x0_radius,
            bottom=_bottom_E,
            width=_width_E,
            fed_mu=_fed_mu_E,
            fed_sigma=_fed_mu_E * _sigma_factor
        ),
        IonosphereLayer(
            length_scale=_length_scale_F,
            longitude_pole=0.,
            latitude_pole=np.pi / 2.,
            bottom_velocity=_bottom_velocity_F,
            radial_velocity=_radial_velocity_F,
            x0_radius=x0_radius,
            bottom=_bottom_F,
            width=_width_F,
            fed_mu=_fed_mu_F,
            fed_sigma=_fed_mu_F * _sigma_factor
        )
    ]
    return IonosphereMultiLayer(layers)


def construct_ionosphere_model_from_didb_db(
        start: datetime, end: datetime, ursi_station: str,
        x0_radius: FloatArray,
        latitude: au.Quantity = 0. * au.deg,
        longitude_pole: au.Quantity = 0. * au.deg,
        latitude_pole: au.Quantity = 90 * au.deg,
        turbulent: bool = True
):
    data = fetch_ionosphere_data(start, end, ursi_station)
    data = list(filter(lambda d: d.confidence_score == 100, data))
    if len(data) == 0:
        raise ValueError(f"No high quality data found for {ursi_station} in the given interval.")
    # randomly select one
    data = data[np.random.randint(len(data))]
    dsa_logger.info(f"Using data from {data.dt} for {ursi_station}.")
    vE = quantity_to_np(
        2 * np.pi * (x0_radius + data.hmE - data.yE) * np.cos(quantity_to_np(latitude, 'rad')) * au.km / (
                86400 * au.s),
        'km/s'
    )
    vF1 = quantity_to_np(
        2 * np.pi * (x0_radius + data.hmF1 - data.yF1) * np.cos(quantity_to_np(latitude, 'rad')) * au.km / (
                86400 * au.s),
        'km/s'
    )
    vF2 = quantity_to_np(
        2 * np.pi * (x0_radius + data.hmF2 - data.yF2) * np.cos(quantity_to_np(latitude, 'rad')) * au.km / (
                86400 * au.s),
        'km/s'
    )

    return construct_ionosphere_model(
        x0_radius=x0_radius,
        f0E=data.f0E,
        f0F1=data.f0F1,
        f0F2=data.f0F2,
        hmE=data.hmE,
        hmF1=data.hmF1,
        hmF2=data.hmF2,
        yE=data.yE,
        yF1=data.yF1,
        yF2=data.yF2,
        vtec=data.vtec,
        vE=vE,
        vF1=vF1,
        vF2=vF2,
        rvE=0.,
        rvF1=0.,
        rvF2=0.,
        longitude_pole=quantity_to_np(longitude_pole, 'rad'),
        latitude_pole=quantity_to_np(latitude_pole, 'rad'),
        turbulent=turbulent
    )


def construct_ionosphere_model(x0_radius: FloatArray,
                               f0E, f0F1, f0F2,
                               hmE, hmF1, hmF2,
                               yE, yF1, yF2,
                               vtec,
                               vE, vF1, vF2,
                               rvE=0., rvF1=0., rvF2=0.,
                               longitude_pole=0., latitude_pole=jnp.pi / 2.,
                               turbulent: bool = True):
    """
    Construct a canonical ionosphere model.

    Args:
        x0_radius: see `compute_x0_radius` in km
        f0E: the critical frequency of E layer in MHz
        f0F1: the critical frequency of F1 layer in MHz
        f0F2: the critical frequency of F2 layer in MHz
        hmE: the height of E layer in km
        hmF1: the height of F1 layer in km
        hmF2: the height of F2 layer in km
        yE: the half-width of E layer in km
        yF1: the half-width of F1 layer in km
        yF2: the half-width of F2 layer in km
        vtec: the vertical total electron content in TECU = 10^16 e/m^2 = 1e3 * (1e10 e/m^3) * km
        turbulent: If True then assumes shorter spatial scales

    Returns:
        An ionosphere model
    """

    def critical_freq_to_mu(f0):
        return 1e-10 * (1e6 * f0 / 8.979) ** 2  # [e/m^3]

    _bottoms = [hmE - yE, hmF1 - yF1, hmF2 - yF2]
    _widths = [2 * yE, 2 * yF1, 2 * yF2]
    _mus_peak = [critical_freq_to_mu(f0) for f0 in [f0E, f0F1, f0F2]]
    mean_vtec = 1e-3 * sum([w * mu for w, mu in zip(_widths, _mus_peak)])  # [TECU]
    scale = vtec / mean_vtec
    _mus = [mu * scale for mu in _mus_peak]
    _bottom_velocity = [vE, vF1, vF2]  # [km/s]
    _radial_velocity = [rvE, rvF1, rvF2]  # [km/s]
    # (mu_peak - mu) / mu = (mu_peak - scale * mu_peak) / scale * mu_peak = (1 - scale) / scale
    _sigma_factor = abs(1 - scale) / scale
    dsa_logger.info(
        f"Scaling Mean VTEC: {mean_vtec} TECU -> Peak VTEC {vtec} TECU ==> Scale: {scale}, Sigma factor: {_sigma_factor}")

    if turbulent:
        _length_scales = [yE / 10, yF1 / 10, yF2 / 10]
    else:
        _length_scales = [yE / 5, yF1 / 5, yF2 / 5]

    layers = []
    for i in range(3):
        layers.append(
            IonosphereLayer(
                length_scale=_length_scales[i],
                longitude_pole=longitude_pole,
                latitude_pole=latitude_pole,
                bottom_velocity=_bottom_velocity[i],
                radial_velocity=_radial_velocity[i],
                x0_radius=x0_radius,
                bottom=_bottoms[i],
                width=_widths[i],
                fed_mu=_mus[i],
                fed_sigma=_mus[i] * _sigma_factor
            )
        )
    return IonosphereMultiLayer(layers)
