import json
import os
from http import HTTPStatus
from typing import Any, Dict, Tuple

import pyarrow as pa  # type: ignore
import pyarrow.parquet as pq  # type: ignore
from pandas import (  # type: ignore
    DataFrame,
    concat,
)

from mats_l1_processing import read_parquet_functions as rpf  # type: ignore
from mats_l1_processing.instrument import Instrument  # type: ignore
from mats_l1_processing.L1_calibrate import L1_calibrate  # type: ignore


BotoClient = Any
S3Client = BotoClient
SSMClient = BotoClient
Event = Dict[str, Any]
Context = Any


class InvalidMessage(Exception):
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


def lambda_handler(event: Event, context: Context):
    try:
        output_bucket = get_env_or_raise("L1B_BUCKET")
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
        raise Level1BException("Failed to initialize handler") from err

    try:
        instrument = Instrument("/calibration_data/calibration_data.toml")

        ccd_data = rpf.read_ccd_data(object_path, filesystem=s3)
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
                    _,
                    errors,
                ) = L1_calibrate(ccd, instrument)
            ccd["ImageCalibrated"] = image_calibrated
            ccd["CalibrationErrors"] = errors
    except Exception as err:
        msg = f"Failed to process {object_path}"
        raise Level1BException(msg) from err

    try:
        calibrated = DataFrame.from_records(
            ccd_items,
            columns=["EXP Date", "ImageCalibrated", "CalibrationErrors"],
        )
        l1b_data = concat([
            ccd_data,
            calibrated,
        ], axis=1).set_index("EXPDate").sort_index()
        l1b_data.drop(["ImageData", "Errors", "Warnings"], axis=1, inplace=True)
        l1b_data = l1b_data[l1b_data.ImageCalibrated != None]  # noqa: E711

        out_table = pa.Table.from_pandas(l1b_data)
        pq.write_table(
            out_table,
            f"{output_bucket}/{object_path}",
            filesystem=s3,
            version='2.6',
        )
    except Exception as err:
        msg = f"Failed to store {object_path}"
        raise Level1BException(msg) from err
