#!/usr/bin/env python3
"""Enqueue synthetic S3 event messages to the Level1B SQS queue.

This script sends SQS messages with the same payload shape used by S3 event
notifications, so Level1B processing can be triggered as if new Level1A files
arrived in the input bucket.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime, timedelta
from itertools import islice
from typing import Iterable, Iterator, Sequence

from botocore.exceptions import NoCredentialsError  # type: ignore


def parse_iso_datetime(value: str, *, end_of_day: bool) -> datetime:
    """Parse YYYY-MM-DD or ISO datetime and return a UTC datetime."""
    try:
        if len(value) == 10:
            dt_value = datetime.strptime(value, "%Y-%m-%d")
            if end_of_day:
                dt_value = dt_value + timedelta(days=1)
        else:
            dt_value = datetime.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid date/time '{value}'. Use YYYY-MM-DD or ISO-8601."
        ) from exc

    if dt_value.tzinfo is None:
        dt_value = dt_value.replace(tzinfo=UTC)
    else:
        dt_value = dt_value.astimezone(UTC)
    return dt_value


def iter_level1a_objects(
    s3_client,
    *,
    bucket: str,
    prefix: str,
    start: datetime,
    end: datetime,
    parquet_only: bool,
) -> Iterator[str]:
    """Yield object keys in [start, end) based on LastModified."""
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if parquet_only and not key.endswith(".parquet"):
                continue
            last_modified = obj["LastModified"].astimezone(UTC)
            if start <= last_modified < end:
                yield key


def iter_hourly_starts(start: datetime, end: datetime) -> Iterator[datetime]:
    """Yield hourly timestamps in [start, end)."""
    current = start.replace(minute=0, second=0, microsecond=0)
    while current < end:
        yield current
        current += timedelta(hours=1)


def normalize_prefix(prefix: str) -> str:
    if not prefix:
        return ""
    return prefix if prefix.endswith("/") else prefix + "/"


def iter_level1a_objects_by_partition(
    s3_client,
    *,
    bucket: str,
    prefix: str,
    start: datetime,
    end: datetime,
    parquet_only: bool,
) -> Iterator[str]:
    """Yield object keys by scanning hourly YYYY/MM/DD/HH partitions."""
    base_prefix = normalize_prefix(prefix)
    seen_keys: set[str] = set()

    for hour in iter_hourly_starts(start, end):
        partition_prefixes = {
            f"{base_prefix}{hour.strftime('%Y/%m/%d/%H')}/",
            f"{base_prefix}{hour.year}/{hour.month}/{hour.day}/{hour.hour}/",
        }
        paginator = s3_client.get_paginator("list_objects_v2")
        for hour_prefix in partition_prefixes:
            for page in paginator.paginate(Bucket=bucket, Prefix=hour_prefix):
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    if parquet_only and not key.endswith(".parquet"):
                        continue
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)
                    yield key


def chunked(items: Iterable[str], size: int) -> Iterator[list[str]]:
    """Yield fixed-size chunks from an iterable."""
    iterator = iter(items)
    while True:
        chunk = list(islice(iterator, size))
        if not chunk:
            return
        yield chunk


def build_s3_event_body(bucket: str, key: str) -> str:
    """Build the S3 event-shaped JSON body expected by the Level1B handler."""
    payload = {
        "Records": [
            {
                "s3": {
                    "bucket": {"name": bucket},
                    "object": {"key": key},
                }
            }
        ]
    }
    return json.dumps(payload)


def send_messages(
    sqs_client,
    *,
    queue_url: str,
    bucket: str,
    keys: Sequence[str],
) -> tuple[int, int]:
    """Send messages in batches of 10 and return (sent, failed)."""
    sent = 0
    failed = 0

    for key_batch in chunked(keys, 10):
        entries = [
            {
                "Id": str(i),
                "MessageBody": build_s3_event_body(bucket, key),
            }
            for i, key in enumerate(key_batch)
        ]
        result = sqs_client.send_message_batch(
            QueueUrl=queue_url,
            Entries=entries,
        )
        sent += len(result.get("Successful", []))
        failed += len(result.get("Failed", []))

        for failure in result.get("Failed", []):
            print(
                "Failed to enqueue message:",
                failure.get("Message", "unknown error"),
            )

    return sent, failed


def resolve_defaults(data_source: str, development: bool) -> tuple[str, str]:
    """Return default (bucket_name, queue_name)."""
    source = data_source.upper()
    if source not in {"CCD", "PM"}:
        raise ValueError("data_source must be CCD or PM")

    if development:
        bucket = "dev-payload-level1a" if source == "CCD" else "dev-payload-level1a-pm"
        queue = f"Level1BQueue{source}Dev"
    else:
        bucket = "ops-payload-level1a-v1.0" if source == "CCD" else "ops-payload-level1a-pm-v1.0"
        queue = f"Level1BQueue{source}"

    return bucket, queue


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Enqueue synthetic S3 event notifications to the Level1B SQS queue "
            "for all matching Level1A files in a date interval."
        )
    )
    parser.add_argument("--start", required=True, help="Start date/time (inclusive).")
    parser.add_argument("--end", required=True, help="End date/time (inclusive for dates).")
    parser.add_argument(
        "--data-source",
        choices=["CCD", "PM", "ccd", "pm"],
        default="CCD",
        help="Level1A/Level1B data source track.",
    )
    parser.add_argument(
        "--development",
        action="store_true",
        help="Use development queue and bucket defaults.",
    )
    parser.add_argument(
        "--bucket",
        help="Override input Level1A bucket name.",
    )
    parser.add_argument(
        "--queue-name",
        help="Override Level1B queue name.",
    )
    parser.add_argument(
        "--prefix",
        default="",
        help=(
            "Optional base object key prefix before YYYY/MM/DD/HH partitions, "
            "for example 'CCD'."
        ),
    )
    parser.add_argument(
        "--region",
        default="eu-north-1",
        help="AWS region for S3 and SQS clients.",
    )
    parser.add_argument(
        "--profile",
        help="Optional AWS profile name from ~/.aws/config and credentials.",
    )
    parser.add_argument(
        "--max-messages",
        type=int,
        default=0,
        help="Optional cap on number of messages to send (0 means no cap).",
    )
    parser.add_argument(
        "--include-non-parquet",
        action="store_true",
        help="Include non-.parquet objects too.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List matched keys and counts without sending messages.",
    )
    parser.add_argument(
        "--match-mode",
        choices=["partition", "last-modified"],
        default="partition",
        help=(
            "How to select objects in the interval. 'partition' scans "
            "YYYY/MM/DD/HH key paths; 'last-modified' filters by S3 metadata."
        ),
    )

    args = parser.parse_args()

    try:
        import boto3  # type: ignore
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "boto3 is required. Run with 'uv run --with boto3 "
            "scripts/enqueue_level1b_sqs.py ...' or install boto3 in your env."
        ) from exc

    start = parse_iso_datetime(args.start, end_of_day=False)
    end = parse_iso_datetime(args.end, end_of_day=True)
    if end <= start:
        raise SystemExit("--end must be after --start")

    default_bucket, default_queue = resolve_defaults(
        args.data_source,
        args.development,
    )
    bucket = args.bucket or default_bucket
    queue_name = args.queue_name or default_queue

    session_kwargs: dict[str, str] = {"region_name": args.region}
    if args.profile:
        session_kwargs["profile_name"] = args.profile
    session = boto3.session.Session(**session_kwargs)

    try:
        s3_client = session.client("s3")
        sqs_client = session.client("sqs")

        queue_url = sqs_client.get_queue_url(QueueName=queue_name)["QueueUrl"]

        if args.match_mode == "partition":
            matched_keys = list(
                iter_level1a_objects_by_partition(
                    s3_client,
                    bucket=bucket,
                    prefix=args.prefix,
                    start=start,
                    end=end,
                    parquet_only=not args.include_non_parquet,
                )
            )
        else:
            matched_keys = list(
                iter_level1a_objects(
                    s3_client,
                    bucket=bucket,
                    prefix=args.prefix,
                    start=start,
                    end=end,
                    parquet_only=not args.include_non_parquet,
                )
            )
    except NoCredentialsError as exc:
        raise SystemExit(
            "No AWS credentials found. Configure credentials first, e.g.\n"
            "  aws sso login --profile <profile>\n"
            "or\n"
            "  export AWS_PROFILE=<profile>\n"
            "Then rerun with optional --profile <profile>."
        ) from exc

    if args.max_messages > 0:
        matched_keys = matched_keys[: args.max_messages]

    print(f"Bucket: {bucket}")
    print(f"Queue: {queue_name}")
    print(f"Queue URL: {queue_url}")
    print(f"Interval (UTC): [{start.isoformat()} .. {end.isoformat()})")
    print(f"Match mode: {args.match_mode}")
    print(f"Matched objects: {len(matched_keys)}")

    if not matched_keys:
        return 0

    if args.dry_run:
        preview = min(len(matched_keys), 20)
        for key in matched_keys[:preview]:
            print(f"DRY-RUN key: {key}")
        if len(matched_keys) > preview:
            print(f"... and {len(matched_keys) - preview} more")
        return 0

    sent, failed = send_messages(
        sqs_client,
        queue_url=queue_url,
        bucket=bucket,
        keys=matched_keys,
    )
    print(f"Enqueued successfully: {sent}")
    print(f"Failed: {failed}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
