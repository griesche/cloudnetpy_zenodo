"""Module for concatenating netCDF files."""
from typing import Optional, Set, Union

import netCDF4
import numpy as np

from cloudnetpy.exceptions import InconsistentDataError


def update_nc(old_file: str, new_file: str) -> int:
    """Appends data to existing netCDF file.

    Args:
        old_file: Filename of a existing netCDF file.
        new_file: Filename of a new file whose data will be appended to the end.

    Returns:
        1 = success, 0 = failed to add new data.

    Notes:
        Requires 'time' variable with unlimited dimension.

    """
    try:
        with netCDF4.Dataset(old_file, "a") as nc_old, netCDF4.Dataset(new_file) as nc_new:
            valid_ind = _find_valid_time_indices(nc_old, nc_new)
            if len(valid_ind) > 0:
                _update_fields(nc_old, nc_new, valid_ind)
                return 1
            return 0
    except OSError:
        return 0


def concatenate_files(
    filenames: list,
    output_file: str,
    concat_dimension: str = "time",
    variables: Optional[list] = None,
    new_attributes: Optional[dict] = None,
    ignore: Optional[list] = None,
) -> None:
    """Concatenate netCDF files in one dimension.

    Args:
        filenames: List of files to be concatenated.
        output_file: Output file name.
        concat_dimension: Dimension name for concatenation. Default is 'time'.
        variables: List of variables with the 'concat_dimension' to be concatenated.
            Default is None when all variables with 'concat_dimension' will be saved.
        new_attributes: Optional new global attributes as {'attribute_name': value}.
        ignore: List of variables to be ignored.

    Notes:
        Arrays without 'concat_dimension', scalars, and global attributes will be taken from
        the first file. Groups, possibly present in a NETCDF4 formatted file, are ignored.

    """
    with Concat(filenames, output_file, concat_dimension) as concat:
        concat.get_common_variables()
        concat.create_global_attributes(new_attributes)
        concat.concat_data(variables, ignore)


class Concat:
    common_variables: Set[str]

    def __init__(self, filenames: list, output_file: str, concat_dimension: str = "time"):
        self.filenames = sorted(filenames)
        self.concat_dimension = concat_dimension
        self.first_filename = self.filenames[0]
        self.first_file = netCDF4.Dataset(self.first_filename)
        self.concatenated_file = self._init_output_file(output_file)
        self.common_variables = set()

    def create_global_attributes(self, new_attributes: Union[dict, None]) -> None:
        """Copies global attributes from one of the source files."""
        _copy_attributes(self.first_file, self.concatenated_file)
        if new_attributes is not None:
            for key, value in new_attributes.items():
                setattr(self.concatenated_file, key, value)

    def get_common_variables(self):
        """Finds variables which should have the same values in all files."""
        for key, value in self.first_file.variables.items():
            if self.concat_dimension not in value.dimensions:
                self.common_variables.add(key)

    def close(self):
        """Closes open files."""
        self.first_file.close()
        self.concatenated_file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def concat_data(self, variables: Optional[list] = None, ignore: Optional[list] = None):
        """Concatenates data arrays."""
        self._write_initial_data(variables, ignore)
        if len(self.filenames) > 1:
            for filename in self.filenames[1:]:
                self._append_data(filename)

    def _write_initial_data(self, variables: Union[list, None], ignore: Union[list, None]) -> None:
        for key in self.first_file.variables.keys():
            if (
                variables is not None
                and key not in variables
                and key not in self.common_variables
                and key != self.concat_dimension
            ):
                continue
            if ignore and key in ignore:
                continue

            self.first_file[key].set_auto_scale(False)
            array = self.first_file[key][:]
            if 'scale_factor' in self.first_file[key].ncattrs():
                array = array * self.first_file[key].scale_factor + self.first_file[key].add_offset
            if key == 'time':
                array = array + self.first_file["base_time"][:]
            dimensions = self.first_file[key].dimensions
            fill_value = getattr(self.first_file[key], "_FillValue", None)
            var = self.concatenated_file.createVariable(
                key,
                array.dtype,
                dimensions,
                zlib=True,
                complevel=3,
                shuffle=False,
                fill_value=fill_value,
            )
            var.set_auto_scale(False)
            if key == 'sweep_mode': # sweep mode in kazr file has scond dimension ('string_length_22')
                array = 'vertical pointing'
            var[:] = array
            _copy_attributes(self.first_file[key], var)

    def _append_data(self, filename: str) -> None:
        with netCDF4.Dataset(filename) as file:
            file.set_auto_scale(False)
            ind0 = len(self.concatenated_file.variables[self.concat_dimension])
            ind1 = ind0 + len(file.variables[self.concat_dimension])
            for key in self.concatenated_file.variables.keys():
                array = file[key][:]
                if key in self.common_variables:
                    if not np.array_equal(self.first_file[key][:], array):
                        raise InconsistentDataError(
                            f"Inconsistent values in variable '{key}' between "
                            f"files '{self.first_filename}' and '{filename}'"
                        )
                    continue
                if 'scale_factor' in file[key].ncattrs():
                    array = array * file[key].scale_factor + file[key].add_offset
                if key == 'time':
                    array = array + file["base_time"][:]
                if array.ndim == 0:
                    continue
                if array.ndim == 1:
                    self.concatenated_file.variables[key][ind0:ind1] = array
                else:
                    self.concatenated_file.variables[key][ind0:ind1, :] = array

    def _init_output_file(self, output_file: str) -> netCDF4.Dataset:
        data_model = "NETCDF4" if self.first_file.data_model == "NETCDF4" else "NETCDF4_CLASSIC"
        nc = netCDF4.Dataset(output_file, "w", format=data_model)
        for dim in self.first_file.dimensions.keys():
            dim_len = None if dim == self.concat_dimension else self.first_file.dimensions[dim].size
            nc.createDimension(dim, dim_len)
        return nc


def _copy_attributes(source: netCDF4.Dataset, target: netCDF4.Dataset) -> None:
    for attr in source.ncattrs():
        if attr == 'scale_factor' or attr == 'add_offset':
            continue
        if attr != "_FillValue":
            value = getattr(source, attr)
            setattr(target, attr, value)


def _find_valid_time_indices(nc_old: netCDF4.Dataset, nc_new: netCDF4.Dataset):
    return np.where(nc_new.variables["time"][:] > nc_old.variables["time"][-1])[0]


def _update_fields(nc_old: netCDF4.Dataset, nc_new: netCDF4.Dataset, valid_ind: list):
    ind0 = len(nc_old.variables["time"])
    idx = [ind0 + x for x in valid_ind]
    concat_dimension = nc_old.variables["time"].dimensions[0]
    for field in nc_new.variables:
        if field not in nc_old.variables:
            continue
        dimensions = nc_new.variables[field].dimensions
        if concat_dimension in dimensions:
            concat_ind = dimensions.index(concat_dimension)
            if len(dimensions) == 1:
                nc_old.variables[field][idx] = nc_new.variables[field][valid_ind]
            elif len(dimensions) == 2 and concat_ind == 0:
                nc_old.variables[field][idx, :] = nc_new.variables[field][valid_ind, :]
            elif len(dimensions) == 2 and concat_ind == 1:
                nc_old.variables[field][:, idx] = nc_new.variables[field][:, valid_ind]


def truncate_netcdf_file(filename: str, output_file: str, n_profiles: int):
    """Truncates netcdf file in 'time' dimension taking only n_profiles.
    Useful for creating small files for tests.
    """
    with netCDF4.Dataset(filename, "r") as nc, netCDF4.Dataset(
        output_file, "w", format=nc.data_model
    ) as nc_new:
        for dim in nc.dimensions.keys():
            dim_len = None if dim == "time" else nc.dimensions[dim].size
            nc_new.createDimension(dim, dim_len)
        for attr in nc.ncattrs():
            value = getattr(nc, attr)
            setattr(nc_new, attr, value)
        for key in nc.variables:
            array = nc.variables[key][:]
            dimensions = nc.variables[key].dimensions
            fill_value = getattr(nc.variables[key], "_FillValue", None)
            var = nc_new.createVariable(
                key, array.dtype, dimensions, zlib=True, fill_value=fill_value
            )
            if dimensions and "time" in dimensions[0]:
                if array.ndim == 1:
                    var[:] = array[:n_profiles]
                if array.ndim == 2:
                    var[:] = array[:n_profiles, :]
            else:
                var[:] = array
            for attr in nc.variables[key].ncattrs():
                if attr != "_FillValue":
                    value = getattr(nc.variables[key], attr)
                    setattr(var, attr, value)
