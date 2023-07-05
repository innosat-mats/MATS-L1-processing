import json
import os
from http import HTTPStatus
from traceback import format_tb
from typing import Any, Dict, Tuple

import pyarrow as pa  # type: ignore
import pyarrow.parquet as pq  # type: ignore
from pandas import (  # type: ignore
    DataFrame,
    concat,
)

from mats_l1_processing.instrument import Instrument  # type: ignore
from mats_l1_processing.L1_calibrate import L1_calibrate  # type: ignore
from mats_l1_processing.photometer import calibrate_pm  # type: ignore
from mats_l1_processing import read_parquet_functions as rpf  # type: ignore


BotoClient = Any
S3Client = BotoClient
SSMClient = BotoClient
Event = Dict[str, Any]
Context = Any


class InvalidMessage(Exception):
    pass


class UnknownDataSource(Exception):
    pass


class Level1BException(Exception):
    pass


def get_env_or_raise(variable_name: str) -> str:
    if (var := os.environ.get(variable_name)) is None:
        raise EnvironmentError(
            f"{variable_name} is a required environment variable"
        )
    return var


def parse_event_message(event: Event) -> Tuple[str, str]:
    try:
        message: Dict[str, Any] = json.loads(event["Records"][0]["body"])
        bucket = message["Records"][0]["s3"]["bucket"]["name"]
        key = message["Records"][0]["s3"]["object"]["key"]
    except (KeyError, TypeError):
        raise InvalidMessage
    return bucket, key


def handle_ccd_data(ccd_data: DataFrame, instrument: Instrument) -> DataFrame:
    ccd_items = rpf.dataframe_to_ccd_items(
        ccd_data,
        remove_empty=False,
        remove_errors=False,
        remove_warnings=False,
    )

    for ccd in ccd_items:
        if ccd["IMAGE"] is None:
            image_calibrated = None
            errors = None
        else:
            (
                _,
                _,
                _,
                _,
                _,
                image_calibrated,
                errors,
            ) = L1_calibrate(ccd, instrument, force_table=False)
        ccd["ImageCalibrated"] = image_calibrated
        ccd["CalibrationErrors"] = errors

    calibrated = DataFrame.from_records(
        ccd_items,
        columns=[
            "ImageCalibrated",
            "CalibrationErrors",
            "qprime",
        ],
    )
    l1b_data = concat([
        ccd_data,
        calibrated,
    ], axis=1).set_index("EXPDate").sort_index()
    l1b_data.drop(["ImageData", "Errors", "Warnings"], axis=1, inplace=True)
    l1b_data = l1b_data[l1b_data.ImageCalibrated != None]  # noqa: E711
    l1b_data["ImageCalibrated"] = [
        ic.tolist() for ic in l1b_data["ImageCalibrated"]
    ]
    l1b_data["CalibrationErrors"] = [
        ce.tolist() for ce in l1b_data["CalibrationErrors"]
    ]

    return l1b_data


def handle_pm_data(pm_data: DataFrame, instrument: Instrument) -> DataFrame:
    l1b_data = calibrate_pm(
        pm_data,
        instrument,
    ).set_index("PMTime").sort_index()
    l1b_data.drop(["Errors", "Warnings"], axis=1, inplace=True)
    return l1b_data


def lambda_handler(event: Event, context: Context):
    try:
        output_bucket = get_env_or_raise("L1B_BUCKET")
        code_version = get_env_or_raise("L1B_VERSION")
        data_source = get_env_or_raise("L1A_DATA_SOURCE")
        region = os.environ.get('AWS_REGION', "eu-north-1")
        s3 = pa.fs.S3FileSystem(region=region)

        try:
            bucket, object = parse_event_message(event)
            object_path = f"{bucket}/{object}"
        except InvalidMessage:
            return {
                'statusCode': HTTPStatus.NO_CONTENT,
                'headers': {'Content-Type': 'application/json'},
                'body': json.dumps({
                    'message': 'Failed to parse event, nothing to do.'
                })
            }

        if not object.endswith(".parquet"):
            return {
                'statusCode': HTTPStatus.NO_CONTENT,
                'headers': {'Content-Type': 'application/json'},
                'body': json.dumps({
                    'message': f'{object} is not a parquet file, nothing to do.'
                })
            }
    except Exception as err:
        tb = '|'.join(format_tb(err.__traceback__)).replace('\n', ';')
        msg = f"Failed to initialize handler: {err} ({type(err)}; {tb})"
        raise Level1BException(msg) from err

    try:
        data, metadata = rpf.read_ccd_data(
            object_path,
            filesystem=s3,
            metadata=True,
        )
    except Exception as err:
        msg = f"Failed to get {object_path}: {err}"
        raise Level1BException(msg) from err

    try:
        instrument = Instrument("/calibration_data/calibration_data.toml")
        if data_source.upper() == "CCD":
            l1b_data = handle_ccd_data(data, instrument)
        elif data_source.upper() == "PM":
            l1b_data = handle_ccd_data(data, instrument)
        else:
            raise UnknownDataSource(f"Unknown data source {data_source}")
    except Exception as err:
        msg = f"Failed to process {object_path}: {err}"
        raise Level1BException(msg) from err

    try:
        metadata.update({
            "L1BCode": code_version,
            "DataLevel": "L1B",
            "L1BDataBucket": output_bucket,
            "L1BDataPath": object_path,
        })
        if "CODE" in metadata.keys():
            metadata["RACCode"] = metadata.pop("CODE")
        elif b"CODE" in metadata.keys():
            metadata["RACCode"] = metadata.pop(b"CODE")
        if "pandas" in metadata.keys():
            del metadata["pandas"]
        elif b"pandas" in metadata.keys():
            del metadata[b"pandas"]

        ccd_items = rpf.dataframe_to_ccd_items(
            ccd_data,
            remove_empty=False,
            remove_errors=False,
            remove_warnings=False,
            legacy=False,
        )

        for ccd in ccd_items:
            if ccd["IMAGE"] is None:
                image_calibrated = None
                errors = None
            else:
                (
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                    image_calibrated,
                    errors,
                ) = L1_calibrate(ccd, instrument, force_table=False)
            ccd["ImageCalibrated"] = image_calibrated
            ccd["CalibrationErrors"] = errors
    except Exception as err:
        tb = '|'.join(format_tb(err.__traceback__)).replace('\n', ';')
        msg = f"Failed to process {object_path}: {err} ({type(err)}; {tb})"
        raise Level1BException(msg) from err

    try:
        calibrated = DataFrame.from_records(
            ccd_items,
            columns=[
                "ImageCalibrated",
                "CalibrationErrors",
                "qprime",
            ],
        )
        l1b_data = concat([
            ccd_data,
            calibrated,
        ], axis=1).set_index("EXPDate").sort_index()
        l1b_data.drop(["ImageData", "Errors", "Warnings"], axis=1, inplace=True)
        l1b_data = l1b_data[l1b_data.ImageCalibrated != None]  # noqa: E711
        l1b_data["ImageCalibrated"] = [
            ic.tolist() for ic in l1b_data["ImageCalibrated"]
        ]
        l1b_data["CalibrationErrors"] = [
            ce.tolist() for ce in l1b_data["CalibrationErrors"]
        ]
    except Exception as err:
        tb = '|'.join(format_tb(err.__traceback__)).replace('\n', ';')
        msg = f"Failed to prepare {object_path} for storage: {err} ({type(err)}; {tb})"  # noqa: E501
        raise Level1BException(msg) from err

    try:
        for key, val in metadata.items():
            l1b_data[
                key if isinstance(key, str) else key.decode()
            ] = val if isinstance(val, str) else val.decode()
        out_table = pa.Table.from_pandas(l1b_data)
        out_table = out_table.replace_schema_metadata({
            **metadata,
        })
        pq.write_table(
            out_table,
            f"{output_bucket}/{object}",
            filesystem=s3,
            version='2.6',
        )
    except Exception as err:
        tb = '|'.join(format_tb(err.__traceback__)).replace('\n', ';')
        msg = f"Failed to store {object_path}: {err} ({type(err)}; {tb})"
        raise Level1BException(msg) from err
