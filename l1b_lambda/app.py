#!/usr/bin/env python3
import os
import subprocess
from pathlib import Path
from shutil import copyfile
from typing import Optional

from aws_cdk import App

from stacks.level1b_stack import Level1BStack

app = App()


def _git_output(args: list[str]) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=Path(".."),
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() or None
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


tag = _git_output(["describe", "--tags", "--abbrev=0"])
commit = _git_output(["rev-parse", "HEAD"]) or "unknown"
code_version = f"{tag or 'no-tag'} ({commit})"

dist_wheels = sorted((Path("..") / "dist").glob("mats_l1_processing-*.whl"))
if not dist_wheels:
    raise FileNotFoundError("No built mats_l1_processing wheel found in ../dist")

copyfile(
    dist_wheels[-1],
    Path(".") / "mats_l1_processing-0.0.0-py2.py3-none-any.whl",
)

development = bool(os.environ.get("MATS_DEVELOPMENT", False))
if development:
    input_bucket_name_ccd = "dev-payload-level1a"
    output_bucket_name_ccd = "dev-payload-level1b"
    input_bucket_name_pm = "dev-payload-level1a-pm"
    output_bucket_name_pm = "dev-payload-level1b-pm"
else:
    input_bucket_name_ccd = "ops-payload-level1a-v1.0"
    output_bucket_name_ccd = "ops-payload-level1b-v1.0.2"
    input_bucket_name_pm = "ops-payload-level1a-pm-v1.0"
    output_bucket_name_pm = "ops-payload-level1b-pm-v1.0.2"

Level1BStack(
    app,
    f"Level1BStackCCD{'Dev' if development else ''}",
    input_bucket_name=input_bucket_name_ccd,
    output_bucket_name=output_bucket_name_ccd,
    data_source="CCD",
    code_version=code_version,
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
    code_version=code_version,
    development=development,
    memory_size=512,
    storage_size=512,
)

app.synth()
