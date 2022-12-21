import json
import os
from pathlib import Path
from unittest.mock import patch

import botocore
import pytest
from botocore.stub import Stubber

from level1b.handlers.level1b import (
    get_env_or_raise,
    parse_event_message,
)


@patch.dict(os.environ, {"DEFINITELY": "set"})
def test_get_env_or_raise():
    assert get_env_or_raise("DEFINITELY") == "set"


def test_get_env_or_raise_raises():
    with pytest.raises(
        EnvironmentError,
        match="DEFINITELYNOT is a required environment variable"
    ):
        get_env_or_raise("DEFINITELYNOT")


def test_parse_event_message():
    msg = {
        "Records": [{
            "body": json.dumps({
                "Records": [{
                    "s3": {
                        "bucket": {"name": "bucket-name"},
                        "object": {"key": "object-key"}
                    }
                }]
            }),
        }],
    }
    assert parse_event_message(msg) == ("bucket-name", "object-key")
