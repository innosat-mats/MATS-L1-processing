#!/usr/bin/env python3
import os
from pathlib import Path
from shutil import copyfile
from typing import Optional

import git
from git import TagReference
from aws_cdk import App

from stacks.level1b_stack import Level1BStack

app = App()
repo = git.Repo("..")

copyfile(
    Path("..") / "dist" / "mats_l1_processing-0.0.0-py2.py3-none-any.whl",
    Path(".") / "mats_l1_processing-0.0.0-py2.py3-none-any.whl",
)

development = bool(os.environ.get("MATS_DEVELOPMENT", False))
if development:
    input_bucket_name_ccd = "dev-payload-level1a"
    output_bucket_name_ccd = "dev-payload-level1b"
    input_bucket_name_pm = "dev-payload-level1a-pm"
    output_bucket_name_ccd = "dev-payload-level1b-pm"
else:
    input_bucket_name_ccd = "ops-payload-level1a-v0.9"
    output_bucket_name_ccd = "ops-payload-level1b-v0.9"
    input_bucket_name_pm = "ops-payload-level1a-pm-v0.9"
    output_bucket_name_pm = "ops-payload-level1b-pm-v0.9"

try:
    tag: Optional[TagReference] = repo.tags[-1]
except IndexError:
    tag = None

Level1BStack(
    app,
    f"Level1BStackCCD{'Dev' if development else ''}",
    input_bucket_name=input_bucket_name_ccd,
    output_bucket_name=output_bucket_name_ccd,
    data_source="CCD",
    code_version=f"{tag} ({repo.head.commit})",
    development=development,
    memory_size=6144,
    storage_size=2048,
)

Level1BStack(
    app,
    f"Level1BStackPM{'Dev' if development else ''}",
    input_bucket_name=input_bucket_name_pm,
    output_bucket_name=output_bucket_name_pm,
    data_source="PM",
    code_version=f"{tag} ({repo.head.commit})",
    development=development,
    memory_size=512,
    storage_size=512,
)

app.synth()
