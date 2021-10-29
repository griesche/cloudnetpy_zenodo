"""Module for reading / converting disdrometer data."""
import glob
from typing import Optional
import logging
import netCDF4
import numpy as np
from numpy.testing import assert_array_equal
from cloudnetpy.metadata import MetaData
from cloudnetpy import output
from cloudnetpy import utils
from cloudnetpy.instruments.ceilometer import Ceilometer, NoiseParam
from cloudnetpy.instruments.ceilo import save_ceilo
import numpy.ma as ma


def pollyxt2nc(input_folder: str,
               output_file: str,
               site_meta: dict,
               keep_uuid: Optional[bool] = False,
               uuid: Optional[str] = None,
               date: Optional[str] = None) -> str:
    """"

    Args:
        input_folder: Filename of pollyxt file.
        output_file: Output filename.
        site_meta: Dictionary containing information about the site. Required keys are `name` and
            `altitude`. If the zenith angle of the instrument is NOT 5 degrees, it should be
            provided like this: {'zenith_angle': 6}.
        keep_uuid: If True, keeps the UUID of the old file, if that exists. Default is False
            when new UUID is generated.
        uuid: Set specific UUID for the file.
        date: Expected date of the measurements as YYYY-MM-DD.

    Returns:
        UUID of the generated file.

    """
    snr_limit = 5
    polly = PollyXt(site_meta, date)
    polly.fetch_data(input_folder)
    polly.get_date_and_time(polly.epoch)
    polly.fetch_zenith_angle()
    polly.calc_screened_products(snr_limit)
    polly.mask_nan_values()
    polly.prepare_data(site_meta)
    polly.prepare_metadata()
    polly.data_to_cloudnet_arrays()
    attributes = output.add_time_attribute(ATTRIBUTES, polly.metadata['date'])
    output.update_attributes(polly.data, attributes)
    polly.add_snr_info('beta', snr_limit)
    return save_ceilo(polly, output_file, keep_uuid, uuid)


class PollyXt(Ceilometer):

    def __init__(self, site_meta: dict, expected_date: Optional[str] = None):
        super().__init__()
        self.metadata = site_meta
        self.expected_date = expected_date
        self.model = 'PollyXT Raman lidar'
        self.wavelength = 1064
        self.epoch = None

    def mask_nan_values(self):
        for key, array in self.data.items():
            if getattr(array, 'ndim', 0) > 0:
                array[np.isnan(array)] = ma.masked

    def calc_screened_products(self, snr_limit: float = 5.0):
        keys = ('beta', 'depolarisation')
        for key in keys:
            self.data[key] = ma.masked_where(self.data['snr'] < snr_limit, self.data[f'{key}_raw'])
        del self.data['snr']

    def fetch_zenith_angle(self) -> None:
        default = 5
        self.data['zenith_angle'] = float(self.metadata.get('zenith_angle', default))

    def fetch_data(self, input_folder: str) -> None:
        """Read input data."""
        bsc_files = [file for file in glob.glob(f'{input_folder}/*[0-9]_att*.nc')]
        depol_files = [file for file in glob.glob(f'{input_folder}/*[0-9]_vol*.nc')]
        bsc_files.sort()
        depol_files.sort()
        if not bsc_files:
            logging.info('No pollyxt files found')
            return
        if len(bsc_files) != len(depol_files):
            logging.info('Inconsistent number of pollyxt bsc / depol files')
            return
        self.data['range'] = _read_array_from_multiple_files(bsc_files, depol_files, 'height')
        calibration_factors = []
        bsc_key = 'attenuated_backscatter_1064nm'
        for ind, (bsc_file, depol_file) in enumerate(zip(bsc_files, depol_files)):
            nc_bsc = netCDF4.Dataset(bsc_file, 'r')
            nc_depol = netCDF4.Dataset(depol_file, 'r')
            self.epoch = utils.get_epoch(nc_bsc['time'].unit)
            try:
                time = np.array(_read_array_from_file_pair(nc_bsc, nc_depol, 'time'))
            except AssertionError:
                _close(nc_bsc, nc_depol)
                continue
            beta_raw = nc_bsc.variables[bsc_key][:]
            depol_raw = nc_depol.variables['volume_depolarization_ratio_532nm'][:]
            snr = nc_bsc.variables['SNR_1064nm'][:]
            for array, key in zip([beta_raw, depol_raw, time, snr], ['beta_raw',
                                                                     'depolarisation_raw',
                                                                     'time', 'snr']):
                self.data = utils.append_data(self.data, key, array)
            calibration_factor = nc_bsc.variables[bsc_key].Lidar_calibration_constant_used
            calibration_factor = np.repeat(calibration_factor, len(time))
            calibration_factors = np.concatenate([calibration_factors, calibration_factor])
            _close(nc_bsc, nc_depol)
        self.data['calibration_factor'] = calibration_factors


def _read_array_from_multiple_files(files1: list, files2: list, key) -> np.ndarray:
    array = np.array([])
    for ind, (file1, file2) in enumerate(zip(files1, files2)):
        nc1 = netCDF4.Dataset(file1, 'r')
        nc2 = netCDF4.Dataset(file2, 'r')
        array1 = _read_array_from_file_pair(nc1, nc2, key)
        if ind == 0:
            array = array1
        _close(nc1, nc2)
        assert_array_equal(array, array1)
    return np.array(array)


def _read_array_from_file_pair(nc_file1: netCDF4.Dataset,
                               nc_file2: netCDF4.Dataset,
                               key: str) -> np.ndarray:
    array1 = nc_file1.variables[key][:]
    array2 = nc_file2.variables[key][:]
    assert_array_equal(array1, array2)
    return array1


def _close(*args) -> None:
    for arg in args:
        arg.close()


ATTRIBUTES = {
    'depolarisation': MetaData(
        long_name='Lidar volume linear depolarisation ratio',
        units='1',
        comment='SNR-screened lidar volume linear depolarisation ratio at 532 nm.'
    ),
    'depolarisation_raw': MetaData(
        long_name='Lidar volume linear depolarisation ratio',
        units='1',
        comment='Non-screened lidar volume linear depolarisation ratio at 532 nm.'
    ),
    'calibration_factor': MetaData(
        long_name='Attenuated backscatter calibration factor',
        units='1',
        comment='Calibration factor applied.'
    ),
}
