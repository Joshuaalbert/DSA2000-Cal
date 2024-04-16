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
    def get_wsclean_clean_component_file(self) -> str:
        """
        Get the wsclean source file.

        Returns:
            the wsclean source file
        """
        ...

    @abstractmethod
    def get_wsclean_fits_files(self) -> List[str]:
        """
        Get the files for the wsclean model.

        Returns:
            the fits files
        """
        ...
