import json
import os
from pathlib import Path
from unittest.mock import patch

import botocore
import pytest
from botocore.stub import Stubber

from level1b.handlers.level1b import (
    format_rclone_command,
    get_env_or_raise,
    get_rclone_config_path,
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


def test_rclone_config_path():
    ssm_parameter = "param"

    ssm_client = botocore.session.get_session().create_client(
        "ssm",
        region_name="eu-north-1"
    )
    stubber = Stubber(ssm_client)
    stubber.add_response(
        "get_parameter",
        {"Parameter": {"Value": "config"}},
        expected_params={"Name": ssm_parameter, "WithDecryption": True}
    )
    stubber.activate()

    name = get_rclone_config_path(ssm_client, ssm_parameter)

    path = Path(name)
    assert path.exists()
    assert path.read_text() == "config"
    path.unlink()


def test_format_rclone_command():
    assert format_rclone_command("config", "from_path", "to_path") == [
        "rclone",
        "--config", "config",
        "copy", "from_path", "to_path",
        "--size-only"
    ]
