#!/usr/bin/env python3

from pathlib import Path
from shutil import copyfile

from aws_cdk import App

from stacks.level1b_stack import Level1BStack

app = App()

copyfile(
    Path("..") / "dist" / "mats_l1_processing-0.0.0-py2.py3-none-any.whl",
    Path(".") / "mats_l1_processing-0.0.0-py2.py3-none-any.whl",
)

Level1BStack(
    app,
    "Level1BStack",
    input_bucket_name="ops-payload-level1a-v0.3",
    output_bucket_name="ops-payload-level1b-v0.2",
)

app.synth()
