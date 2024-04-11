import glob
import os
from typing import List, Tuple

from astropy import units as au
from astropy.io import fits

from dsa2000_cal.assets.registries import source_model_registry
from dsa2000_cal.assets.source_models.source_model import AbstractWSCleanSourceModel


@source_model_registry(template='cyg_a')
class CygASourceModel(AbstractWSCleanSourceModel):

    def get_wsclean_source_file(self) -> str:
        return os.path.join(*self.content_path, 'Cyg-sources.txt')

    def get_wsclean_fits_files(self) -> List[Tuple[au.Quantity, str]]:
        fits_files = glob.glob(os.path.join(*self.content_path, 'fits_models', 'Cyg-*-model.fits'))
        result = []
        for fits_file in fits_files:
            # Get frequency from header, open with astropy
            with fits.open(fits_file) as hdul:
                header = hdul[0].header
                # Try to find freq
                if 'FREQ' in header:
                    frequency = header['FREQ'] * au.Hz
                elif 'RESTFRQ' in header:
                    frequency = header['RESTFRQ'] * au.Hz
                elif 'CRVAL3' in header:  # Assuming the frequency is in the third axis
                    frequency = header['CRVAL3'] * au.Hz
                else:
                    raise KeyError(f"Frequency information not found in FITS header:\n{repr(header)}.")
                result.append((frequency, fits_file))
        return result

