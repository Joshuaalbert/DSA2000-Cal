from abc import ABC, abstractmethod
from typing import Tuple, List

import astropy.units as au

from dsa2000_cal.assets.base_content import BaseContent


class AbstractWSCleanSourceModel(ABC, BaseContent):
    """
    Abstract array class.
    """

    def __init__(self, *args, **kwargs):
        BaseContent.__init__(self, *args, **kwargs)

    @abstractmethod
    def get_wsclean_source_file(self) -> str:
        """
        Get the wsclean source file.

        Returns:
            the wsclean source file
        """
        ...

    @abstractmethod
    def get_wsclean_fits_files(self) -> List[Tuple[au.Quantity, str]]:
        """
        Get the wsclean fits file.

        Returns:
            list of tuples of frequencies and fits files
        """
        ...
