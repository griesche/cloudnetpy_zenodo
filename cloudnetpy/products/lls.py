"""Module for creating Cloudnet droplet effective radius using the Frisch et al. 2002 method."""
from collections import namedtuple
from typing import Optional

import numpy as np
from numpy import ma

from cloudnetpy import output, utils
from cloudnetpy.categorize import atmos
from cloudnetpy.datasource import DataSource
from cloudnetpy.metadata import MetaData
from cloudnetpy.products.product_tools import get_lls


def generate_lls(pollyxt_file: str, categorize_file: str, output_file: str, SNR_threshold: Optional[int] = 40, uuid: Optional[str] = None) -> str:
    """Generates Cloudnet low-level stratus product PollyXT data.

    This function calculates low-level stratus mask based on the PollyXT signal-to-noise ratio 
    of signals detected with the 532-nf near-field channel as described in Grieche et al. 2020.
    The results are written in a netCDF file.

    Args:
        pollyxt_file: PollyXT file name.
        categorize_file: Categorize file name.
        output_file: Output file name.
        SNR_threshold: threshold to detect LLS, deault 40
        uuid: Set specific UUID for the file.

    Returns:
        UUID of the generated file.

    Examples:
        >>> from cloudnetpy.products import generate_lls
        >>> generate_lls('pollyxt.nc', 'categorize.nc', 'lls.nc')

    References:
        Griesche, H. J., Seifert, P., Ansmann, A., Baars, H., Barrientos Velasco,
        C., Bühl, J., Engelmann, R., Radenz, M., Zhenping, Y., and Macke, A. (2020):
        Application of the shipborne remote sensing supersite OCEANET for
        profiling of Arctic aerosols and clouds during Polarstern cruise PS106,
        Atmos. Meas. Tech., 13, 5335–5358.
        from https://doi.org/10.5194/amt-13-5335-2020,

        Heese B., Flentje, H., Ansmann, A., and Frey, S. (2010):
        Ceilometer lidar comparison: backscatter coefficient retrieval 
        and signal-to-noise ratio determination,
        Atmos. Meas. Tech., 3, 1763-1770.
        from https://doi.org/10.5194/amt-3-1763-2010.

    """

    lls_source = LlsSource(pollyxt_file)
    lls_flag,lls_bounds = get_lls(pollyxt_file, categorize_file, SNR_threshold)
    lls_source.append_data(lls_flag, "lls_flag")
    lls_source.append_data(lls_bounds[:,0], "lls_lower_boundary")
    lls_source.append_data(lls_bounds[:,1], "lls_upper_boundary")
    date = lls_source.get_date()
    attributes = output.add_time_attribute(LLS_ATTRIBUTES, date)
    attributes = _add_lls_comment(attributes, SNR_threshold)
    output.update_attributes(lls_source.data, attributes)
    uuid = output.save_product_file("lls", lls_source, output_file, uuid)
    lls_source.close()
    return uuid


class LlsSource(DataSource):
    """Data container for low-level stratus calculations."""

    def __init__(self, pollyxt_file: str):
        super().__init__(pollyxt_file)

def _add_lls_comment(attributes: dict, SNR_threshold: int) -> dict:
    attributes["lls_flag"] = attributes["lls_flag"]._replace(
        comment=
        "This variable contains low level stratus occurrence information using the SNR\n"
        "of the PollyXT 532 nm near range channel. The presence of low-level stratus was\n"
        "identified in case the SNR (as defined in Heese et al. (2010)) exeeded a threshold\n"
        f"of {SNR_threshold}, according to Griesche et al., 2020. The LLS-flag indicates if in any height\n"
        "below the lowest Cloudnet range gate low-level stratus cloud was detected (=1) or not (=0)."
    )
    return attributes


COMMENTS = {
    "lls_lower_boundary": (
        "Lowest altitude matching the criteria for low-level stratus detection."
    ),
    "lls_upper_boundary": (
        "Highest (below the lowest Cloudnet range gate) altitude matching the criteria for low-level stratus detection."
    ),
}

LLS_ATTRIBUTES = {
    "lls_flag": MetaData(
        long_name="Low-level stratus cloud flag",
        units="",
    ),
    "lls_lower_boundary": MetaData(
        long_name="Lower LLS boundary",
        units="m",
        comment=COMMENTS["lls_upper_boundary"],
    ),
    "lls_upper_boundary": MetaData(
        long_name="Lower LLS boundary",
        units="m",
        comment=COMMENTS["lls_lower_boundary"],
    ),
}
