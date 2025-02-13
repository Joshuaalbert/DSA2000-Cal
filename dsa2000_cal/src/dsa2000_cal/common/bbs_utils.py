import re
from typing import Tuple, Dict, Optional, List

import numpy as np
from astropy import coordinates as ac
from astropy import units as au
from pydantic import Field

from dsa2000_cal.common.serialise_utils import SerialisableBaseModel
from dsa2000_cal.common.wsclean_util import parse_coordinates_bbs


class SourceModel(SerialisableBaseModel):
    image: np.ndarray = Field(
        description="Source model of shape [source, chan, 2, 2]",
    )
    lm: np.ndarray = Field(
        description="Source direction cosines of shape [source, 2]",
    )
    corrs: List[str] = Field(
        description="Correlations in the source model",
    )
    freqs: np.ndarray = Field(
        description="Frequencies of shape [chan]",
    )


def extract_columns(line: str) -> Tuple[Dict[str, int], Dict[str, Optional[str]]]:
    """
    Extract column names and their respective indices, as well as their default values if provided,
    from a formatted header line.

    Args:
    - line (str): The header line containing column names possibly with default values.

    Returns:
    - Tuple[Dict[str, int], Dict[str, Optional[str]]]: A tuple containing two dictionaries:
        1. A mapping from column names to their indices.
        2. A mapping from column names to their default values, or None if no default is provided.

    Example:
        line = "# (Name,Type,Ra,Dec,I, ReferenceFrequency='55.468e6', SpectralIndex) = format"
        name_to_index, defaults = extract_columns(line)
        print(name_to_index)
        # {'Name': 0, 'Type': 1, 'Ra': 2, 'Dec': 3, 'I': 4, 'ReferenceFrequency': 5, 'SpectralIndex': 6}
        print(defaults)
        # {'Name': None, 'Type': None, 'Ra': None, 'Dec': None, 'I': None, 'ReferenceFrequency': '55.468e6', 'SpectralIndex': None}
    """
    # Use regex to find text within parentheses
    match = re.search(r'\((.*?)\)', line)
    if not match:
        return {}, {}

    # Extract matched text and split by commas
    columns = match.group(1).split(", ")

    # Create maps for column indices and default values
    name_to_index: Dict[str, int] = {}
    defaults: Dict[str, Optional[str]] = {}

    for idx, col in enumerate(columns):
        if '=' in col:
            name, default = col.split('=')
            name_to_index[name.strip()] = idx
            defaults[name.strip()] = default.strip('\'"')  # remove quotes if present
        else:
            name_to_index[col] = idx
            defaults[col] = None

    return name_to_index, defaults


def parse_data_line(line: str, name_to_index: Dict[str, int], defaults: Dict[str, Optional[str]]) -> Dict[
    str, List[str]]:
    """
    Parse a data line based on the provided header-to-index mapping and defaults mapping.

    Args:
    - line (str): The data line to parse.
    - name_to_index (Dict[str, int]): A mapping from column names to their indices.
    - defaults (Dict[str, Optional[str]]): A mapping from column names to their default values.

    Returns:
    - Dict[str, List[str]]: A dictionary where keys are header labels, and values are lists of data values for each header.

    Example:
        name_to_index = {'Name': 0, 'Type': 1, 'Ra': 2, 'Dec': 3, 'I': 4, 'ReferenceFrequency': 5, 'SpectralIndex': 6}
        defaults = {'Name': None, 'Type': None, 'Ra': None, 'Dec': None, 'I': None, 'ReferenceFrequency': '55.468e6', 'SpectralIndex': None}
        line = "3C196, POINT, 08:13:36.062300, +48.13.02.24900, 153.0, , [-0.56, -0.05212]"
        parsed_data = parse_data_line(line, name_to_index, defaults)
        print(parsed_data)
    """
    data_values = line.split(", ")
    parsed_data: Dict[str, List[str]] = {}

    for name, index in name_to_index.items():
        # Use the data value if it exists and is not empty, otherwise use the default.
        value = data_values[index] if index < len(data_values) and data_values[index] else defaults.get(name)
        # We need to handle list values, like for SpectralIndex
        if value and value.startswith("[") and value.endswith("]"):
            value = value[1:-1].split(", ")  # convert string list to actual list
        parsed_data[name] = value

    return parsed_data


class BBSSkyModel:
    def __init__(self, bbs_sky_model: str, pointing_centre: ac.ICRS, chan0: au.Quantity, chan_width: au.Quantity,
                 num_channels: int):
        """
        Initialize the BBS sky model.

        Args:
            bbs_sky_model: Path to the BBS sky model file.
            pointing_centre: The pointing centre of the observation.
            chan0: The (central) frequency of the first channel.
            chan_width: The width of each channel.
            num_channels: The number of channels.
        """
        self.bbs_sky_model = bbs_sky_model
        self.pointing_centre = pointing_centre
        chan0 = chan0.to('Hz').value
        chan_width = chan_width.to('Hz').value
        self.channels = np.arange(chan0, chan0 + num_channels * chan_width, chan_width, dtype=np.float32)

    def get_source(self) -> SourceModel:
        name_to_index, defaults = None, None
        with open(self.bbs_sky_model, 'r') as f:
            for line in f:
                if "= format" in line:
                    name_to_index, defaults = extract_columns(line)
                    break

        # Create a dictionary to hold values for each column
        data_dict: Dict[str, List[Optional[str]]] = {name: [] for name in name_to_index}

        with open(self.bbs_sky_model, 'r') as f:
            # Go through the lines and collect data
            for line in f:
                if line.startswith("#") or not line.strip():  # skip comments and empty lines
                    continue
                values = line.split(", ")
                for name, index in name_to_index.items():
                    value = values[index] if index < len(values) else None
                    if not value:  # If value is missing
                        value = defaults.get(name)  # Use default if available, else None
                    data_dict[name].append(value)

        for key in ['Ra', 'Dec', 'I']:
            if key not in data_dict:
                raise ValueError(f"Missing required column {key} in BBS sky model.")

        # Now parse data_dict to create a SourceModel
        directions = list(
            map(
                lambda ra_str, dec_str: parse_coordinates_bbs(ra_str, dec_str),
                data_dict['Ra'], data_dict['Dec']
            )
        )
        if len(directions) > 1:
            directions = ac.concatenate(directions)
        else:
            directions = directions[0].reshape((1,))
        directions = directions.transform_to(ac.ICRS())

        # Compute the separation angles in RA and Dec
        delta_alpha = (directions.ra - self.pointing_centre.ra).to(au.radian).value

        # Compute the direction cosines using the tangent plane approximation
        l = np.cos(directions.dec.radian) * np.sin(delta_alpha)
        m = np.sin(directions.dec.radian) * np.cos(self.pointing_centre.dec.radian) - np.cos(
            directions.dec.radian) * np.sin(self.pointing_centre.dec.radian) * np.cos(delta_alpha)

        # [source, 2]
        direction_cosines = np.stack([l, m], axis=1)

        num_sources = len(data_dict['I'])

        def _get_stokes_param(param: str):
            if param not in data_dict:
                return np.zeros(num_sources, dtype=np.float32)
            return np.asarray(list(map(lambda x: float(x), data_dict[param])), dtype=np.float32)

        stokes_image = np.stack(
            [_get_stokes_param(param=param) for param in ['I', 'Q', 'U', 'V']],
            axis=1
        )  # [source, corr]

        image_corr = np.tile(stokes_image[:, None, :], [1, len(self.channels), 1])  # [source, chan, 4]
        if 'ReferenceFrequency' in data_dict and 'SpectralIndex' in data_dict:
            ## TODO: Add spectral model if necessary
            pass

        return SourceModel(
            image=image_corr,
            lm=direction_cosines,
            corrs=['I', 'Q', 'U', 'V'],
            freqs=self.channels
        )
