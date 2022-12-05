"""Module for reading / converting pollyxt data."""
import logging
import glob
from typing import Optional, Union

import netCDF4
import numpy as np
from numpy import ma
from numpy.testing import assert_array_equal
import csv
from datetime import datetime

from cloudnetpy import output, utils
from cloudnetpy.instruments import instruments
from cloudnetpy.instruments.ceilometer import Ceilometer
from cloudnetpy.metadata import MetaData
from cloudnetpy.utils import Epoch

WAVELEGNTHS = [1064,532,355]

def pollyxt2nc(
    input_folder: str,
    output_file: str,
    site_meta: dict,
    uuid: Optional[str] = None,
    date: Optional[str] = None,
) -> str:
    """
    Converts PollyXT Raman lidar data into Cloudnet Level 1b netCDF file.

    Args:
        input_folder: Path to pollyxt netCDF files.
        output_file: Output filename.
        site_meta: Dictionary containing information about the site with keys:

            - `name`: Name of the site (mandatory)
            - `altitude`: Site altitude in [m] (mandatory).
            - `latitude` (optional).
            - `longitude` (optional).
            - `zenith_angle`: If not the default 5 degrees (optional).
            - `snr_limit`: If not the default 2 (optional).
        uuid: Set specific UUID for the file.
        date: Expected date of the measurements as YYYY-MM-DD.

    Returns:
        UUID of the generated file.

    Examples:
        >>> from cloudnetpy.instruments import pollyxt2nc
        >>> site_meta = {'name': 'Mindelo', 'altitude': 13, 'zenith_angle': 6, 'snr_limit': 3}
        >>> pollyxt2nc('/path/to/files/', 'pollyxt.nc', site_meta)

    """
    snr_limit = site_meta.get("snr_limit", 2)
    polly = PollyXt(site_meta, date)
    epoch = polly.fetch_data(input_folder)
    polly.get_date_and_time(epoch)
    polly.fetch_zenith_angle()
    polly.calc_screened_products(snr_limit)
    polly.mask_nan_values()
    polly.prepare_data()
    polly.data_to_cloudnet_arrays()
    attributes = output.add_time_attribute(ATTRIBUTES, polly.date)
    output.update_attributes(polly.data, attributes)
    polly.add_snr_info("beta", snr_limit)
    uuid = output.save_level1b(polly, output_file, uuid)
    return uuid


class PollyXt(Ceilometer):
    def __init__(self, site_meta: dict, expected_date: Union[str, None]):
        super().__init__()
        self.site_meta = site_meta
        self.expected_date = expected_date
        self.instrument = instruments.POLLYXT

    def mask_nan_values(self):
        for array in self.data.values():
            if getattr(array, "ndim", 0) > 0:
                array[np.isnan(array)] = ma.masked

    def calc_screened_products(self, snr_limit: float = 5.0):
        for wavelength in WAVELEGNTHS:
            beta_key = "beta_%i" %wavelength
            snr_key = "snr_%i" %wavelength
            snr_key_nr = "snr_%i_nr" %wavelength
            if f"{beta_key}_raw" in self.data.keys(): 
                self.data[beta_key] = ma.masked_where(self.data[snr_key] < snr_limit, self.data[f"{beta_key}_raw"])
            beta_nr_key = "beta_%i_nr" %wavelength
            if f"{beta_nr_key}_raw" in self.data.keys():
                self.data[beta_nr_key] = ma.masked_where(self.data[snr_key_nr] < snr_limit, self.data[f"{beta_nr_key}_raw"])
            depol_key = "depolarisation_%i" %wavelength
            if f"{depol_key}_raw" in self.data.keys(): 
                self.data[depol_key] = ma.masked_where(self.data[snr_key] < snr_limit, self.data[f"{depol_key}_raw"])
                self.data[depol_key][self.data[depol_key] > 1] = ma.masked
                self.data[depol_key][self.data[depol_key] < 0] = ma.masked

    def fetch_zenith_angle(self) -> None:
        default = 5
        self.data["zenith_angle"] = float(self.metadata.get("zenith_angle", default))

    def fetch_data(self, input_folder: str) -> Epoch:
        """Read input data."""
        bsc_files = glob.glob(f"{input_folder}/*[0-9]_att*.nc")
        depol_files = glob.glob(f"{input_folder}/*[0-9]_vol*.nc")
        if not bsc_files:
            raise RuntimeError("No pollyxt files found")
        if len(bsc_files) != len(depol_files):
            raise RuntimeError("Inconsistent number of pollyxt bsc / depol files")
        bsc_nr_files = glob.glob(f'{input_folder}/*[0-9]_NR_att*.nc')
        if len(bsc_nr_files) > 0:
            if len(bsc_files) != len(bsc_nr_files):
                raise RuntimeError("Inconsistent number of pollyxt bsc / pollyxt bsc nr files")
            bsc_nr_files.sort()
            self.data["range"] = _read_array_from_three_filelists(bsc_files, bsc_nr_files, depol_files, "height")
            epoch = _include_nr_data(self, input_folder)
            return epoch
        bsc_files.sort()
        depol_files.sort()
        calibration_factors: np.ndarray = np.array([])
        self.data["range"] = _read_array_from_multiple_files(bsc_files, depol_files, "height")
        for (bsc_file, depol_file) in zip(bsc_files, depol_files):
            nc_bsc = netCDF4.Dataset(bsc_file, "r")
            nc_depol = netCDF4.Dataset(depol_file, "r")
            epoch = utils.get_epoch(nc_bsc["time"].unit)
            try:
                time = np.array(_read_array_from_file_pair(nc_bsc, nc_depol, "time"))
            except AssertionError:
                _close(nc_bsc, nc_depol)
                continue
            for idx_wavelength,wavelength in enumerate(WAVELEGNTHS):
                bsc_key = "attenuated_backscatter_%inm" %wavelength
                if bsc_key in nc_bsc.variables:
                    bsc_out_key = "beta_%i_raw" %wavelength
                    self.data = utils.append_data(self.data, bsc_out_key, nc_bsc.variables[bsc_key][:])
                    cal_fac_key = "calibration_factor_%i" %wavelength
                    calibration_factor = nc_bsc.variables[bsc_key].Lidar_calibration_constant_used
                    calibration_factor = np.repeat(calibration_factor, len(time))
                    self.data = utils.append_data(self.data, cal_fac_key, calibration_factor)
                snr_key = "SNR_%inm" %wavelength
                if snr_key in nc_bsc.variables:
                    snr_out_key = "snr_%i" %wavelength
                    self.data = utils.append_data(self.data, snr_out_key, nc_bsc.variables[snr_key][:])
                depol_key = "volume_depolarization_ratio_%inm" %wavelength
                if depol_key in nc_depol.variables:
                    depol_out_key = "depolarisation_%i_raw" %wavelength
                    self.data = utils.append_data(self.data, depol_out_key, nc_depol.variables[depol_key][:])
            self.data = utils.append_data(self.data, "time", time)
            _close(nc_bsc, nc_depol)
        return epoch

def _include_nr_data(self, input_folder: str) -> Epoch:
    bsc_files = glob.glob(f"{input_folder}/*[0-9]_att*.nc")
    bsc_nr_files = glob.glob(f"{input_folder}/*[0-9]_NR_att*.nc")
    depol_files = glob.glob(f"{input_folder}/*[0-9]_vol*.nc")
    bsc_files.sort()
    bsc_nr_files.sort()
    depol_files.sort()
    calibration_factors: np.ndarray = np.array([])
    for (bsc_file, bsc_nr_file, depol_file) in zip(bsc_files, bsc_nr_files, depol_files):
        nc_bsc = netCDF4.Dataset(bsc_file, "r")
        nc_bsc_nr = netCDF4.Dataset(bsc_nr_file, "r")
        nc_depol = netCDF4.Dataset(depol_file, "r")
        epoch = utils.get_epoch(nc_bsc["time"].unit)
        try:
            time = np.array(_read_array_from_file_triplet(nc_bsc, nc_bsc_nr,  nc_depol, "time"))
        except AssertionError:
            _close(nc_bsc, nc_bsc_nr, nc_depol)
            continue
        cal_facts = {"calibration_factor_1064":[],"calibration_factor_532":[],"calibration_factor_355":[]}
        for idx_wavelength,wavelength in enumerate(WAVELEGNTHS):
            bsc_key = "attenuated_backscatter_%inm" %wavelength
            if bsc_key in nc_bsc.variables:
                bsc_out_key = "beta_%i_raw" %wavelength
                self.data = utils.append_data(self.data, bsc_out_key, nc_bsc.variables[bsc_key][:])
                cal_fac_key = "calibration_factor_%i" %wavelength
                calibration_factor = nc_bsc.variables[bsc_key].Lidar_calibration_constant_used
                calibration_factor = np.repeat(calibration_factor, len(time))
                self.data = utils.append_data(self.data, cal_fac_key, calibration_factor)
            if bsc_key in nc_bsc_nr.variables:
                bsc_out_key = "beta_%i_nr_raw" %wavelength
                self.data = utils.append_data(self.data, bsc_out_key, nc_bsc_nr.variables[bsc_key][:])
                cal_fac_key = "calibration_factor_%i_nr" %wavelength
                calibration_factor = nc_bsc_nr.variables[bsc_key].Lidar_calibration_constant_used
                calibration_factor = np.repeat(calibration_factor, len(time))
                self.data = utils.append_data(self.data, cal_fac_key, calibration_factor)
            snr_key = "SNR_%inm" %wavelength
            if snr_key in nc_bsc.variables:
                snr_out_key = "snr_%i" %wavelength
                self.data = utils.append_data(self.data, snr_out_key, nc_bsc.variables[snr_key][:])
            if snr_key in nc_bsc_nr.variables:
                snr_out_key = "snr_%i_nr" %wavelength
                self.data = utils.append_data(self.data, snr_out_key, nc_bsc_nr.variables[snr_key][:])
            depol_key = "volume_depolarization_ratio_%inm" %wavelength
            if depol_key in nc_depol.variables:
                depol_out_key = "depolarisation_%i_raw" %wavelength
                self.data = utils.append_data(self.data, depol_out_key, nc_depol.variables[depol_key][:])
        self.data = utils.append_data(self.data, "time", time)
        _close(nc_bsc, nc_bsc_nr, nc_depol)
    return epoch

def _read_array_from_three_filelists(files1: list, files2: list, files3: list, key) -> np.ndarray:
    array: np.ndarray = np.array([])
    for ind, (file1, file2, file3) in enumerate(zip(files1, files2, files3)):
        nc1 = netCDF4.Dataset(file1, "r")
        nc2 = netCDF4.Dataset(file2, "r")
        nc3 = netCDF4.Dataset(file3, "r")
        array1 = _read_array_from_file_triplet(nc1, nc2, nc3, key)
        if ind == 0:
            array = array1
        _close(nc1, nc2, nc3)
        assert_array_equal(array, array1)
    return np.array(array)

def _read_array_from_file_triplet(
    nc_file1: netCDF4.Dataset, nc_file2: netCDF4.Dataset, nc_file3: netCDF4.Dataset,key: str
) -> np.ndarray:
    array1 = nc_file1.variables[key][:]
    array2 = nc_file2.variables[key][:]
    array3 = nc_file3.variables[key][:]
    assert_array_equal(array1, array2)
    assert_array_equal(array1, array3)
    return array1
####

def _read_array_from_multiple_files(files1: list, files2: list, key) -> np.ndarray:
    array: np.ndarray = np.array([])
    for ind, (file1, file2) in enumerate(zip(files1, files2)):
        nc1 = netCDF4.Dataset(file1, "r")
        nc2 = netCDF4.Dataset(file2, "r")
        array1 = _read_array_from_file_pair(nc1, nc2, key)
        if ind == 0:
            array = array1
        _close(nc1, nc2)
        assert_array_equal(array, array1)
    return np.array(array)

def _read_array_from_file_pair(
    nc_file1: netCDF4.Dataset, nc_file2: netCDF4.Dataset, key: str
) -> np.ndarray:
    array1 = nc_file1.variables[key][:]
    array2 = nc_file2.variables[key][:]
    assert_array_equal(array1, array2)
    return array1

def _close(*args) -> None:
    for arg in args:
        arg.close()


ATTRIBUTES = {
    "beta_1064": MetaData(
        long_name="Attenuated backscatter coefficient",
        units="sr-1 m-1",
        comment="SNR-screened attenuated backscatter coefficient at 1064 nm. SNR threshold applied: 2.",
    ),
    "beta_532": MetaData(
        long_name="Attenuated backscatter coefficient",
        units="sr-1 m-1",
        comment="SNR-screened attenuated backscatter coefficient at 532 nm. SNR threshold applied: 2.",
    ),
    "beta_355": MetaData(
        long_name="Attenuated backscatter coefficient",
        units="sr-1 m-1",
        comment="SNR-screened attenuated backscatter coefficient at 355 nm. SNR threshold applied: 2.",
    ),
    "beta_1064_raw": MetaData(
        long_name="Attenuated backscatter coefficient",
        units="sr-1 m-1",
        comment="Non-screened attenuated backscatter coefficient at 1064 nm.",
    ),
    "beta_532_raw": MetaData(
        long_name="Attenuated backscatter coefficient",
        units="sr-1 m-1",
        comment="Non-screened attenuated backscatter coefficient at 532 nm.",
    ),
    "beta_355_raw": MetaData(
        long_name="Attenuated backscatter coefficient",
        units="sr-1 m-1",
        comment="Non-screened attenuated backscatter coefficient at 355 nm.",
    ),
    "depolarisation_1064": MetaData(
        long_name="Lidar volume linear depolarisation ratio",
        units="1",
        comment="SNR-screened lidar volume linear depolarisation ratio at 1064 nm.",
    ),
    "depolarisation_532": MetaData(
        long_name="Lidar volume linear depolarisation ratio",
        units="1",
        comment="SNR-screened lidar volume linear depolarisation ratio at 532 nm.",
    ),
    "depolarisation_355": MetaData(
        long_name="Lidar volume linear depolarisation ratio",
        units="1",
        comment="SNR-screened lidar volume linear depolarisation ratio at 355 nm.",
    ),
    "depolarisation_1064_raw": MetaData(
        long_name="Lidar volume linear depolarisation ratio",
        units="1",
        comment="Non-screened lidar volume linear depolarisation ratio at 1064 nm.",
    ),
    "depolarisation_532_raw": MetaData(
        long_name="Lidar volume linear depolarisation ratio",
        units="1",
        comment="Non-screened lidar volume linear depolarisation ratio at 532 nm.",
    ),
    "depolarisation_355_raw": MetaData(
        long_name="Lidar volume linear depolarisation ratio",
        units="1",
        comment="Non-screened lidar volume linear depolarisation ratio at 355 nm.",
    ),
    "snr_1064": MetaData(
        long_name="Signal-to-Noise Ratio (1064 nm)",
        units=" ",
        comment="SNR of respective channel calculated according to Heese et al., 2010, ACP: \n Ceilometer lidar comparison: backscatter coefficient retrieval and signal-to-noise ratio determination.",
    ),
    "snr_532": MetaData(
        long_name="Signal-to-Noise Ratio (532 nm)",
        units=" ",
        comment="SNR of respective channel calculated according to Heese et al., 2010, ACP: \n Ceilometer lidar comparison: backscatter coefficient retrieval and signal-to-noise ratio determination.",
    ),
    "snr_355": MetaData(
        long_name="Signal-to-Noise Ratio (355 nm)",
        units=" ",
        comment="SNR of respective channel calculated according to Heese et al., 2010, ACP: \n Ceilometer lidar comparison: backscatter coefficient retrieval and signal-to-noise ratio determination.",
    ),
    "snr_532_nr": MetaData(
        long_name="Signal-to-Noise Ratio (532 nm near range)",
        units=" ",
        comment="SNR of respective channel calculated according to Heese et al., 2010, ACP: \n Ceilometer lidar comparison: backscatter coefficient retrieval and signal-to-noise ratio determination.",
    ),
    "snr_355_nr": MetaData(
        long_name="Signal-to-Noise Ratio (355 nm near range)",
        units=" ",
        comment="SNR of respective channel calculated according to Heese et al., 2010, ACP: \n Ceilometer lidar comparison: backscatter coefficient retrieval and signal-to-noise ratio determination.",
    ),
    "calibration_factor_1064": MetaData(
        long_name="Attenuated backscatter at 1064 calibration factor",
        units="1",
        comment="Calibration factor applied.",
    ),
    "calibration_factor_532": MetaData(
        long_name="Attenuated backscatter at 532 calibration factor",
        units="1",
        comment="Calibration factor applied.",
    ),
    "calibration_factor_355": MetaData(
        long_name="Attenuated backscatter at 355 calibration factor",
        units="1",
        comment="Calibration factor applied.",
    ),
    "calibration_factor_532_nr": MetaData(
        long_name="Attenuated backscatter at 532 near range calibration factor",
        units="1",
        comment="Calibration factor applied.",
    ),
    "calibration_factor_355_nr": MetaData(
        long_name="Attenuated backscatter at 355 near range calibration factor",
        units="1",
        comment="Calibration factor applied.",
    ),
}
