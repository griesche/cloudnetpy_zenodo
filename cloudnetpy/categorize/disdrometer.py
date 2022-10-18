"""Mwr module, containing the :class:`Mwr` class."""
import numpy as np

from cloudnetpy import utils
from cloudnetpy.datasource import DataSource

_MMH = 3600*1000

class Disdrometer(DataSource):
    """Disdrometer radiometer class, child of DataSource.

    Args:
         full_path: Disdrometer Level 0 netCDF file.

    """

    def __init__(self, full_path: str):
        super().__init__(full_path)
        rain_rate = self.dataset.variables["rainfall_rate"][:]*_MMH # convert to mm h-1
        self.append_data(rain_rate, "rain_rate")

    def rebin_to_grid(self, time_grid: np.ndarray) -> None:
        """Approximates lwp and its error in a grid using mean.

        Args:
            time_grid: 1D target time grid.

        """
        for array in self.data.values():
            array.rebin_data(self.time, time_grid)

