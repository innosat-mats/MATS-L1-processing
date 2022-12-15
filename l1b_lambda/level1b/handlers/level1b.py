import json
import os
import subprocess
from functools import lru_cache
from http import HTTPStatus
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Any, Dict, List, Tuple

import boto3
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


def get_rclone_config_path(
    ssm_client: SSMClient,
    rclone_config_ssm_name: str
) -> str:
    rclone_config = ssm_client.get_parameter(
        Name=rclone_config_ssm_name, WithDecryption=True
    )["Parameter"]["Value"]

    f = NamedTemporaryFile(buffering=0, delete=False)
    f.write(rclone_config.encode())

    return f.name


def format_rclone_command(
    config_path: str,
    source: str,
    destination: str,
) -> List[str]:
    return [
        "rclone",
        "--config",
        config_path,
        "copy",
        source,
        destination,
        "--size-only",
    ]


@lru_cache(maxsize=None)
def get_instrument(
    instrument_dir: str,
    instrument_bucket: str,
    rclone_config_path: str,
):
    subprocess.call(format_rclone_command(
        rclone_config_path,
        f"S3:{instrument_bucket}",
        instrument_dir,
    ))

    return Instrument(
        f"{instrument_dir}/calibration_data/calibration_data.toml",
    )


def lambda_handler(event: Event, context: Context):
    try:
        output_bucket = get_env_or_raise("L1B_BUCKET")
        instrument_bucket = get_env_or_raise("INSTRUMENT_BUCKET")
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
        with TemporaryDirectory(
            "_instrument",
            "/tmp",
        ) as instrument_path:
            rclone_config_path = get_rclone_config_path(
                boto3.client("ssm"),
                get_env_or_raise("RCLONE_CONFIG_SSM_NAME")
            )
            instrument = get_instrument(
                instrument_path,
                instrument_bucket,
                rclone_config_path,
            )

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
