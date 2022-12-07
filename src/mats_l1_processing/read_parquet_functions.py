# -*- coding: utf-8 -*-
"""
Created on Mon Dec 05 09:57:37 2022

@author: skymandr

Functions used to read in MATS images and data from Parquet files.
Parquet files can either be local or on a remote server, such as Amazon S3.
"""

from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List, Optional

import numpy as np
import pyarrow as pa  # type: ignore
import pyarrow.dataset as ds  # type: ignore
import pyarrow.parquet as pq  # type: ignore
from pandas import Timestamp  # type: ignore
from PIL import Image


# Map string names to all channels
channel_num_to_str: Dict[int, str] = {
    1: "IR1",
    4: "IR2",
    3: "IR3",
    2: "IR4",
    5: "UV1",
    6: "UV2",
    7: "NADIR",
}

CCDItem = Dict[str, Any]


def rename_CCDitem_attributes(item: CCDItem) -> None:
    """Renaming of attributes to work with calibration code The names in the code
    are based on the old rac extract file (prior to May 2020).
    The names used in the parquet files are mostly the same as used in CSVs
    prior to November 2022. Exceptions are translated here.


    Args:
        item: Item from Parquet store for which to translate attributes.
    
    Returns:
        Nothing. Operation is performed in place.
    """

    if "EXP Nanoseconds" not in item:
        item["EXP Nanoseconds"] = item.pop("EXPNanoseconds")

    if "EXP Date" not in item:
        item["EXP Date"] = item.pop("EXPDate")

    if "WDW Mode" not in item:
        item["WDW Mode"] = item.pop("WDWMode")

    if "WDW InputDataWindow" not in item:
        item["WDW InputDataWindow"] = item.pop("WDWInputDataWindow")

    if "NCBIN CCDColumns" not in item:
        item["NCBIN CCDColumns"] = item.pop("NCBINCCDColumns")

    if "NCBIN FPGAColumns" not in item:
        item["NCBIN FPGAColumns"] = item.pop("NCBINFPGAColumns")

    if "GAIN Mode" not in item:
        item["GAIN Mode"] = item.pop("GAINMode")

    if "GAIN Timing" not in item:
        item["GAIN Timing"] = item.pop("GAINTiming")

    if "GAIN Truncation" not in item:
        item["GAIN Truncation"] = item.pop("GAINTruncation")

    if "BC" not in item:
        item["BC"] = item.pop("BadColumns")
    
    if "Image File Name" not in item:
        item["Image File Name"] = item.pop("ImageName")

    if "IMAGE" not in item:
        item["IMAGE"] = np.float64(Image.open(BytesIO(item.pop["ImageData"])))


def add_CCDItem_attributes(item: CCDItem):
    """Add some attributes to CCD Item that we need. The names in the code
    are based on the old rac extract file (prior to May 2020).

    Args:
        item: Item from Parquet store to which to add attributes.
    
    Returns:
        Nothing. Operation is performed in place.
    """

    item["channel"] = channel_num_to_str[item["CCDSEL"]]
    item["flipped"] = False

    # CCDitem["id"] should not be needed in operational retrieval. Keeping it
    # because protocol reading / CodeCalibrationReport needs it.  LM220908
    item["id"] = f"{item['EXP Nanoseconds']}_{item['CCDSEL']}"

    # Add temperature info fom OBC, the temperature info from the rac files are
    # better since they are based on the thermistors on the UV channels
    ADC_temp_in_mV = int(item["TEMP"]) / 32768 * 2048
    ADC_temp_in_degreeC = 1.0 / 0.85 * ADC_temp_in_mV - 296
    item["temperature_ADC"] = ADC_temp_in_degreeC

    # TODO: Replace with something like `add_temperature_info` when HTR data is
    # available, see: https://github.com/innosat-mats/level1a/issues/5
    item["temperature"] = ADC_temp_in_degreeC
    item["temperature_HTR"] = ADC_temp_in_degreeC


def add_and_rename_CCDitem_attributes(items: List[CCDItem]):
    """Applies renaming of CCD attributes as well as adding a few that are
    needed. Note that items with errors are removed, so the list may be shorter
    after processing.

    Args:
        items: List of items from Parquet store to be converted to CCCD items.
    
    Returns:
        Nothing. Operation is performed in place.
    """

    for ind, item in enumerate(items):
        try:
            rename_CCDitem_attributes(item)
            add_CCDItem_attributes(item)
        except Exception as err:
            print(
                f"Warning: could not prepare image {item.get('ImageName', 'unknown')} for calibration. Skipping. (Error: {err})"  # noqa: E501
            )
            items.pop(ind)


def read_CCDitems_interval(
    start: datetime,
    stop: datetime,
    path: str,
    filesystem: Optional[pa.fs.FileSystem] = None,
) -> List[CCDItem]:
    """Reads the CCD data and metadata from the specified path or S3 bucket
    between the specified times.

    Args:
        from:       Read CCD data from this timestamp (inclusive).
        to:         Read CCD data up to this timestamp.
        path:       Path to dataset. May be a directory or a bucket, depending
                    on filesystem.
        filesystem: Filesystem type. If not specified will assume that path
                    points to an ordinary directory disk .

    Returns:
        List of dictionary representations of the dataframes
    """

    CCDitems = ds.dataset(
        path,
        filesystem=filesystem,
    ).to_table(filter=(
        (ds.field("EXPDate") >= Timestamp(start))
        & (ds.field("EXPDate") <= Timestamp(stop))
    )).to_pandas().reset_index().to_dict("records")

    add_and_rename_CCDitem_attributes(CCDitems)

    return CCDitems


def read_CCDitems(
    path: str,
    filesystem: Optional[pa.fs.FileSystem] = None,
) -> List[CCDItem]:
    """Reads the CCD data and metadata from a singel file at the specified path.

    Args:
        path:       Path to parquet file containing the CCD data. May point to
                    object on disk or in a bucket, depending on filesystem.
        filesystem: Filesystem type. If not specified will assume that path
                    points to an ordinary directory on disk.

    Returns:
        List of dictionary representations of the dataframes
    """

    CCDitems = pq.read_table(
        path,
        filesystem=filesystem,
    ).to_pandas().reset_index().to_dict("records")

    add_and_rename_CCDitem_attributes(CCDitems)

    return CCDitems
