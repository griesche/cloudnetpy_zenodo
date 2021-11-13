"""Module with a class for Lufft chm15k ceilometer."""
from typing import Optional
import logging
import netCDF4
from cloudnetpy.instruments.nc_lidar import NcLidar
from cloudnetpy import utils
import numpy.ma as ma
from cloudnetpy.instruments import instruments


class LufftCeilo(NcLidar):
    """Class for Lufft chm15k ceilometer."""

    def __init__(self, file_name: str, site_meta: dict, expected_date: Optional[str] = None):
        super().__init__()
        self.file_name = file_name
        self.site_meta = site_meta
        self.expected_date = expected_date
        self.instrument = instruments.CHM15K

    def read_ceilometer_file(self, calibration_factor: Optional[float] = None) -> None:
        """Reads data and metadata from Jenoptik netCDF file."""
        self.dataset = netCDF4.Dataset(self.file_name)
        self._fetch_range(reference='upper')
        self._fetch_beta_raw(calibration_factor)
        self._fetch_time_and_date()
        self._fetch_zenith_angle('zenith')
        self.dataset.close()

    def _fetch_beta_raw(self, calibration_factor: Optional[float] = None) -> None:
        if calibration_factor is None:
            logging.warning('Using default calibration factor')
            calibration_factor = 3e-12
        beta_raw = self._getvar('beta_raw')
        old_version = self._get_old_software_version()
        if old_version is not None:
            logging.warning(f'Software version {old_version}. Assuming data not range corrected.')
            data_std = self._getvar('stddev')
            normalised_apd = self._get_nn()
            beta_raw *= utils.transpose(data_std / normalised_apd)
            beta_raw *= self.data['range'] ** 2
        beta_raw *= calibration_factor
        self.data['calibration_factor'] = calibration_factor
        self.data['beta_raw'] = beta_raw

    def _get_old_software_version(self):
        version = self.dataset.software_version
        if len(str(version)) > 4:
            return None
        return version

    def _get_nn(self):
        nn1 = self._getvar('nn1', 'NN1')
        median_nn1 = ma.median(nn1)
        # Parameters taken from the matlab code and should be verified
        if 120 < median_nn1 < 160:
            step_factor, reference, scale = 1.24, 140, 5
        elif 3200 < median_nn1 < 4000:
            step_factor, reference, scale = 1.035, 3685, 1
        else:
            logging.warning('Unable to compute normalized APD')
            return 1
        return step_factor ** (-(nn1 - reference) / scale)

    def _getvar(self, *args):
        for arg in args:
            if arg in self.dataset.variables:
                var = self.dataset.variables[arg]
                return var[0] if utils.isscalar(var) else var[:]
