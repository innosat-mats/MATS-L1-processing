# -*- coding: utf-8 -*-
"""
Created on Mon Dec 05 09:57:37 2022

@author: skymandr

Functions used to read in MATS images and data from Parquet files.
Parquet files can either be local or on a remote server, such as Amazon S3.
"""

import logging
from datetime import datetime, timezone
from io import BytesIO
from typing import cast, Any, Dict, List, Optional, SupportsFloat, Tuple, Union

import numpy as np
import pyarrow as pa  # type: ignore
import pyarrow.dataset as ds  # type: ignore
import pyarrow.parquet as pq  # type: ignore
from pandas import DataFrame, Timestamp  # type: ignore
from PIL import Image


# Map all channels to string names
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


def rename_ccd_item_attributes(ccd_data: DataFrame) -> None:
    """Renaming of attributes to work with calibration code.
    The names in the code are based on the old rac extract file (prior to May
    2020).  The names used in the parquet files are mostly the same as used in
    CSVs prior to November 2022. Exceptions are translated here.

    Args:
        ccd_data (DataFrame):   CCD data for which translate attributes.

    Returns:
        None:   Operation is performed in place.
    """

    ccd_data.rename(
        columns={
            "EXPNanoseconds": "EXP Nanoseconds",
            "EXPDate": "EXP Date",
            "WDWMode": "WDW Mode",
            "WDWInputDataWindow": "WDW InputDataWindow",
            "NCBINCCDColumns": "NCBIN CCDColumns",
            "NCBINFPGAColumns": "NCBIN FPGAColumns",
            "GAINMode": "GAIN Mode",
            "GAINTiming": "GAIN Timing",
            "GAINTruncation": "GAIN Truncation",
            "BadColumns": "BC",
            "ImageName": "ImageFileName",
            "OriginFile": "File",
        },
        inplace=True,
    )


def add_ccd_item_attributes(ccd_data: DataFrame) -> None:
    """Add some attributes to CCD data that we need.
    Note that this function assumes the data has up to date names for columns,
    not the names used in the old rac extract file (prior to May 2020).
    Conversion to the old standard can be performed using
    `rename_ccd_item_attributes`, but that has to be done _after_ applying this
    function.

    Args:
        ccd_data (DataFrame):   CCD data to which to add attributes.

    Returns:
        None:   Operation is performed in place.
    """

    images: List[Optional[Any]] = []
    for ind, image_data in enumerate(ccd_data["ImageData"]):
        try:
            images.append(np.float64(cast(
                SupportsFloat,
                Image.open(BytesIO(image_data)),
            )))
        except Exception as err:
            image_name = ccd_data["ImageName"][ind]
            logging.warning(
                f"could not prepare image {image_name} for calibration. "
                f"Setting image data to `None`. (Error: {err})"
            )
            images.append(None)
    ccd_data["IMAGE"] = images

    ccd_data["channel"] = [channel_num_to_str[c] for c in ccd_data["CCDSEL"]]
    ccd_data["flipped"] = False

    # CCDitem["id"] should not be needed in operational retrieval. Keeping it
    # because protocol reading / CodeCalibrationReport needs it.  LM220908
    ccd_data["id"] = f"{ccd_data['EXPNanoseconds']}_{ccd_data['CCDSEL']}"

    # Add temperature info fom OBC, the temperature info from the rac files are
    # better since they are based on the thermistors on the UV channels
    ADC_temp_in_mV = ccd_data["TEMP"] / 32768 * 2048
    ADC_temp_in_degreeC = 1.0 / 0.85 * ADC_temp_in_mV - 296
    ccd_data["temperature_ADC"] = ADC_temp_in_degreeC

    # This needs to be updated when a better temperature estimate has been
    # designed. For now a de facto implementation of
    # get_temperature.add_temperature_info()
    ccd_data["temperature"] = ccd_data["HTR8A"]
    ccd_data["temperature_HTR"] = ccd_data["HTR8A"]


def remove_faulty_rows(
    ccd_data: DataFrame,
    remove_empty: bool = True,
    remove_errors: bool = True,
    remove_warnings: bool = False,
) -> DataFrame:
    """Remove rows of CCD data where image could not be parsed or other errors
    or warnings occurred.

    Args:
        ccd_data (DataFrame):   CCD data which to filter.
        remove_empty (bool):    Remove rows lacking image data. (Default: True)
        remove_errors (bool):   Remove rows with errors. (Default: True)
        remove_warnings (bool): Remove rows with warnings. (Default: False)

    Returns:
        DataFrame:   The filtered CCD data.
    """

    if remove_empty:
        ccd_data = ccd_data[ccd_data.IMAGE != None]  # noqa: E711

    if remove_errors:
        ccd_data = ccd_data[ccd_data.Errors != None]  # noqa: E711

    if remove_warnings:
        ccd_data = ccd_data[ccd_data.Warnings != None]  # noqa: E711

    return ccd_data


def dataframe_to_ccd_items(
    ccd_data: DataFrame,
    remove_empty: bool = True,
    remove_errors: bool = True,
    remove_warnings: bool = False,
) -> List[CCDItem]:
    """Returns a list of CCD Items converted from the input DataFrame

    Args:
        ccd_data (DataFrame):   The CCD data to convert.
        remove_empty (bool):    Remove rows lacking image data. (Default: True)
        remove_errors (bool):   Remove rows with errors. (Default: True)
        remove_warnings (bool): Remove rows with warnings. (Default: False)

    Returns:
        List[Dict[str, Any]]:   List of valid CCD items. List may be shorter
                                than than input, depending on applied filters.
    """

    data = ccd_data.copy()
    add_ccd_item_attributes(data)
    rename_ccd_item_attributes(data)

    return remove_faulty_rows(
        data,
        remove_empty,
        remove_errors,
        remove_warnings,
    ).to_dict("records")


def read_ccd_data_in_interval(
    start: datetime,
    stop: datetime,
    path: str,
    filesystem: Optional[pa.fs.FileSystem] = None,
    filter: dict = None,
    metadata: bool = False,
) -> Union[DataFrame, Tuple[DataFrame, pq.FileMetaData]]:
    """Reads the CCD data and metadata from the specified path or S3 bucket
    between the specified times. Optionally read file metadata.

    Args:
        start (datetime):           Read CCD data from this time (inclusive).
        stop (datetime):            Read CCD data up to this time (inclusive).
        path (str):                 Path to dataset. May be a directory or a
                                    bucket, depending on filesystem.
        filesystem (FileSystem):    Optional. File system to read. If not
                                    specified will assume that path points to
                                    an ordinary directory disk. (Default: None)
        filter (Optional[dict]):    Extra filters of the form:
                                    `{fieldname1: [min, max], ...}`
        metadata (bool):            If True, return Parquet file metadata along
                                    with data frame. (Default: False)

    Returns:
        DataFrame:      The CCD data.
        FileMetaData:   File metadata (optional).
    """

    if start.tzinfo is None:
        start.replace(tzinfo=timezone.utc)
    if stop.tzinfo is None:
        stop.replace(tzinfo=timezone.utc)

    filterlist = (
        (ds.field("EXPDate") >= Timestamp(start))
        & (ds.field("EXPDate") <= Timestamp(stop))
    )
    if filter != None:
        for variable in filter.keys():
            filterlist = filterlist & ((ds.field(variable) >= filter[variable][0]) &
                                       (ds.field(variable) <= filter[variable][1]))

    table = ds.dataset(
        path,
        filesystem=filesystem,
    ).to_table(filter=filterlist)
    dataframe = table.to_pandas().reset_index().set_index('TMHeaderTime')
    if metadata:
        return dataframe, table.schema.metadata
    return dataframe


def read_ccd_items_in_interval(
    start: datetime,
    stop: datetime,
    path: str,
    filesystem: Optional[pa.fs.FileSystem] = None,
) -> List[CCDItem]:
    """Reads the CCD data and metadata from the specified path or S3 bucket
    between the specified times.
    Some column names are translated and some convenience data is added before
    data is converted and and returned.

    Args:
        start (datetime):           Read CCD data from this time (inclusive).
        stop (datetime):            Read CCD data up to this time (inclusive).
        path (str):                 Path to dataset. May be a directory or a
                                    bucket, depending on filesystem.
        filesystem (FileSystem):    Optional. File system to read. If not
                                    specified will assume that path points to
                                    an ordinary directory disk. (Default: None)

    Returns:
        list[dict[str, Any]]:   List of dictionary representations of the CCD data.
    """

    return dataframe_to_ccd_items(
        read_ccd_data_in_interval(start, stop, path, filesystem)
    )


def read_ccd_data(
    path: str,
    filesystem: Optional[pa.fs.FileSystem] = None,
    metadata: bool = False,
) -> Union[DataFrame, Tuple[DataFrame, pq.FileMetaData]]:
    """Reads the CCD data and metadata from a singel file at the specified path.
    Optionally read file metadata.

    Args:
        path (str):                 Path to dataset. May be a directory or a
                                    bucket, depending on filesystem.
        filesystem (FileSystem):    Optional. File system to read. If not
                                    specified will assume that path points to
                                    an ordinary directory disk. (Default: None)
        metadata (bool):            If True, return Parquet file metadata along
                                    with data frame. (Default: False)

    Returns:
        DataFrame:      The CCD data.
        FileMetaData:   File metadata (optional).
    """

    table = pq.read_table(
        path,
        filesystem=filesystem,
    )
    dataframe = table.to_pandas().reset_index()
    if metadata:
        return dataframe, table.schema.metadata
    return dataframe


def read_ccd_items(
    path: str,
    filesystem: Optional[pa.fs.FileSystem] = None,
) -> List[CCDItem]:
    """Reads the CCD data and metadata from a singel file at the specified path.
    Some column names are translated and some convenience data is added before
    data is converted and and returned.

    Args:
        path (str):                 Path to dataset. May be a directory or a
                                    bucket, depending on filesystem.
        filesystem (FileSystem):    Optional. File system to read. If not
                                    specified will assume that path points to
                                    an ordinary directory disk. (Default: None)

    Return:
        list[dict[str, Any]]:   List of dictionary representations of the CCD data.
    """

    return dataframe_to_ccd_items(
        read_ccd_data(path, filesystem)
    )
