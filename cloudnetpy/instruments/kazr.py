"""Module for reading raw cloud radar data."""
import logging
import os
from tempfile import NamedTemporaryFile
from typing import List, Optional
import glob

import numpy as np
from numpy import ma

from datetime import datetime,timedelta

from cloudnetpy import concat_lib, output, utils
from cloudnetpy.exceptions import ValidTimeStampError
from cloudnetpy.instruments import general
from cloudnetpy.instruments.instruments import KAZR
from cloudnetpy.instruments.nc_radar import NcRadar
from cloudnetpy.metadata import MetaData


def kazr2nc(
    raw_kazr: str,
    output_file: str,
    site_meta: dict,
    uuid: Optional[str] = None,
    date: Optional[str] = None,
) -> str:
    """Converts ARM KAZR cloud radar data into Cloudnet Level 1b netCDF file.

    This function converts raw KAZR file(s) into a much smaller file that
    contains only the relevant data and can be used in further processing
    steps.

    Args:
        raw_kazr: Filename of a daily KAZR .nc file. Can be also a folder containing several
            non-concatenated .nc files from one day.
        output_file: Output filename.
        site_meta: Dictionary containing information about the site. Required key value pair
            is `name`.
        uuid: Set specific UUID for the file.
        date: Expected date as YYYY-MM-DD of all profiles in the file.

    Returns:
        UUID of the generated file.

    Raises:
        ValidTimeStampError: No valid timestamps found.

    Examples:
          >>> from cloudnetpy.instruments import kazr2nc
          >>> site_meta = {'name': 'MOSAiC'}
          >>> kazr2nc('raw_radar.nc', 'radar.nc', site_meta)
          >>> kazr2nc('/one/day/of/kazr/nc/files/', 'radar.nc', site_meta)

    """
    keymap = {
        "reflectivity": "Zh",
        "mean_doppler_velocity": "v",
        "spectral_width": "width",
        #"linear_depolarization_ratio": "ldr",   # no real ldr saved in kazr files, linear correlation to Zh -> likely only crosstalk
        "signal_to_noise_ratio_copolar_h": "SNR",
        "elevation": "elevation",
        "azimuth": "azimuth_angle",
        #"aziv": "azimuth_velocity",            # not available in kazr file
        #"fft_len": "nfft",                     # ||
        #"num_spectral_averages": "nave",       # ||
        #"pulse_repetition_frequency": "prf",   # ||
        "prt": "prt",
        "nyquist_velocity": "nyquist_velocity",
    }

    calibration_offset = 1 # kazr_ge = 1 , kazr_md = 6

    if os.path.isdir(raw_kazr):
        temp_file = NamedTemporaryFile()  # pylint: disable=R1732
        nc_filename = temp_file.name
        valid_filenames = utils.get_sorted_filenames(raw_kazr, ".nc")
        valid_filenames = general.get_files_with_common_range(valid_filenames)
        variables = list(keymap.keys())
        concat_lib.concatenate_files(valid_filenames, nc_filename, variables=variables)
        # gives error for kazr variable 'sweep_mode' (len(sweep_mode)=1 but dimensions(sweep,mode=(1,22))
    else:
        nc_filename = raw_kazr

    kazr = Kazr(nc_filename, site_meta)
    kazr.init_data(keymap)
    if kazr.init_time[0] == '23':
        kazr.time_offset = kazr._get_time_offset()
        kazr.date = kazr._init_kazr_date()
    if date is not None:
        kazr.screen_by_date(date)
        kazr.date = date.split("-")
    kazr.sort_timestamps()
    kazr.remove_duplicate_timestamps()
    #general.linear_to_db(kazr, ("Zh", "ldr", "SNR")) # is db already
    kazr.screen_by_snr()
    kazr.mask_invalid_data()
    kazr.add_time_and_range()
    kazr.remove_lowest_two_rangegates()
    kazr.calibrate_Z(calibration_offset)
    general.add_site_geolocation(kazr)
    general.add_radar_specific_variables(kazr)
    valid_indices = kazr.add_solar_angles()
    general.screen_time_indices(kazr, valid_indices)
    general.add_height(kazr)
    kazr.close()
    attributes = output.add_time_attribute(ATTRIBUTES, kazr.date)
    output.update_attributes(kazr.data, attributes)
    uuid = output.save_level1b(kazr, output_file, uuid)
    return uuid


class Kazr(NcRadar):
    """Class for KAZR raw radar data. Child of NcRadar().

    Args:
        full_path: Filename of a daily KAZR .nc NetCDF file.
        site_meta: Site properties in a dictionary. Required keys are: `name`.

    """

    epoch = (1970, 1, 1)
    def __init__(self, full_path: str, site_meta: dict):
        super().__init__(full_path, site_meta)
        self.time_offset = 0
        self.date = self._init_kazr_date()
        self.init_time = self._init_kazr_time()
        self.instrument = KAZR

    def screen_by_date(self, expected_date: str) -> None:
        """Screens incorrect time stamps."""
        time_stamps = self.getvar("time")+time_offset
        valid_indices = []
        for ind, timestamp in enumerate(time_stamps):
            date = "-".join(utils.seconds2date(timestamp, self.epoch)[:3])
            if date == expected_date:
                valid_indices.append(ind)
        if not valid_indices:
            raise ValidTimeStampError
        general.screen_time_indices(self, valid_indices)

    def sort_timestamps(self):
        """Sorts data by timestamps."""
        ind = self.time.argsort()
        self._screen_by_ind(ind)

    def calibrate_Z(self,cal_offset: int):
        self.data["Zh"].data = self.data["Zh"].data.__radd__(cal_offset)

    def remove_duplicate_timestamps(self):
        """Removes duplicate timestamps."""
        _, ind = np.unique(self.time, return_index=True)
        self._screen_by_ind(ind)

    def _screen_by_ind(self, ind: np.ndarray):
        n_time = len(self.time)
        for array in self.data.values():
            if array.data.ndim == 1 and array.data.shape[0] == n_time:
                array.data = array.data[ind]
            if array.data.ndim == 2 and array.data.shape[0] == n_time:
                array.data = array.data[ind, :]
        self.time = self.time[ind]

    def screen_by_snr(self, snr_limit: float = -17) -> None:
        """Screens by SNR."""
        ind = np.where(self.data["SNR"][:] < snr_limit)
        for cloudnet_array in self.data.values():
            if cloudnet_array.data.ndim == 2:
                cloudnet_array.mask_indices(ind)

    def remove_lowest_two_rangegates(self) -> None:
        """remove data from lowest tw range gates -> no reliable data."""
        hidx_min = 2
        keys = ["Zh", "v", "width", "SNR", "range"]
        for key in keys:
            if self.data[key].data.ndim == 1:
                self.data[key].data = self.data[key].data[hidx_min:]
            else:
                self.data[key].data = self.data[key].data[:,hidx_min:]

    def mask_invalid_data(self) -> None:
        """Makes sure Z and v masks are also in other 2d variables."""
        z_mask = self.data["Zh"][:].mask
        v_mask = self.data["v"][:].mask
        for cloudnet_array in self.data.values():
            if cloudnet_array.data.ndim == 2:
                cloudnet_array.mask_indices(z_mask)
                cloudnet_array.mask_indices(v_mask)

    def add_solar_angles(self) -> list:
        """Adds solar zenith and azimuth angles and returns valid time indices."""
        elevation = self.data["elevation"].data
        if "azimuth_velocity" in self.data.keys():
            azimuth_vel = self.data["azimuth_velocity"].data
        else:
            azimuth_vel = np.diff(self.data["azimuth_angle"].data)/np.diff(self.data["time"].data)
            azimuth_vel = np.append(azimuth_vel,np.nanmax(azimuth_vel))
            self.data["azimuth_velocity"] = azimuth_vel
        zenith = 90 - elevation
        is_stable_zenith = np.isclose(zenith, ma.median(zenith), atol=0.1)
        is_stable_azimuth = np.isclose(azimuth_vel, 0, atol=1e-6)
        is_stable_profile = is_stable_zenith & is_stable_azimuth
        n_removed = len(is_stable_profile) - np.count_nonzero(is_stable_profile)
        if n_removed >= len(zenith) - 1:
            raise ValidTimeStampError("No profiles with valid zenith / azimuth angles")
        if n_removed > 0:
            logging.warning(f"Filtering {n_removed} profiles due to varying zenith / azimuth angle")
        self.append_data(zenith, "zenith_angle")
        for key in ("elevation", "azimuth_velocity"):
            del self.data[key]
        return list(is_stable_profile)

    def _init_kazr_date(self) -> List[str]:
        time_stamps = self.getvar("time")+self.time_offset
        return utils.seconds2date(time_stamps[0], self.epoch)[:3]

    def _init_kazr_time(self) -> List[str]:
        time_stamps = self.getvar("time")+self.time_offset
        return utils.seconds2date(time_stamps[0], self.epoch)[3:]

    def _get_time_offset(self):
        time_stamps = self.getvar("time")
        dt_init_kazr = utils.seconds2date(time_stamps[0], self.epoch)
        timedelta_kazr = (datetime(int(dt_init_kazr[0]),int(dt_init_kazr[1]),int(dt_init_kazr[2]))+timedelta(days=1)) - datetime(int(dt_init_kazr[0]),int(dt_init_kazr[1]),int(dt_init_kazr[2]),int(dt_init_kazr[3]),int(dt_init_kazr[4]),int(dt_init_kazr[5]))
        time_offset = timedelta_kazr.seconds
        return time_offset


ATTRIBUTES = {
    "SNR": MetaData(
        long_name="Signal-to-noise ratio",
        units="dB",
    ),
    "nfft": MetaData(
        long_name="Number of FFT points",
        units="1",
    ),
    "nave": MetaData(
        long_name="Number of spectral averages (not accounting for overlapping FFTs)",
        units="1",
    ),
    "rg0": MetaData(long_name="Number of lowest range gates", units="1"),
    "prf": MetaData(
        long_name="Pulse Repetition Frequency",
        units="Hz",
    ),
    "prt": MetaData(
        long_name="Pulse Repetition Time",
        units="s",
    ),
}

