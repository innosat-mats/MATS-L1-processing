#!/usr/bin/env python3

from pathlib import Path
from shutil import copytree

from aws_cdk import App

from stacks.level1b_stack import Level1BStack

app = App()

copytree(
    Path("..") / "src" / "mats_l1_processing",
    Path(".") / "level1b" / "mats_l1_processing",
    dirs_exist_ok=True,
)

Level1BStack(
    app,
    "Level1BStack",
    input_bucket_name="ops-payload-level1a-v0.1",
    output_bucket_name="ops-payload-level1b-v0.1",
    instrument_bucket_name="instrument-data",
    rclone_arn="arn:aws:lambda:eu-north-1:671150066425:layer:rclone-amd64:1",
    config_ssm_name="/rclone/l0-fetcher",
)

app.synth()
