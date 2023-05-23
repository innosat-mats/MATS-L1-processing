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
    output_bucket_name = "dev-payload-level1b"
else:
    output_bucket_name = "ops-payload-level1b-v0.4"

try:
    tag: Optional[TagReference] = repo.tags[-1]
except IndexError:
    tag = None

Level1BStack(
    app,
    "Level1BStack",
    input_bucket_name="ops-payload-level1a-v0.5",
    output_bucket_name=output_bucket_name,
    code_version=f"{tag} ({repo.head.commit})",
    development=development,
)

app.synth()
