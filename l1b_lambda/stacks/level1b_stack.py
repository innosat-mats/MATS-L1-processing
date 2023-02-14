from aws_cdk import Duration, Stack, RemovalPolicy
from aws_cdk.aws_lambda import (
    Architecture, DockerImageFunction, DockerImageCode,
)
from aws_cdk.aws_lambda_event_sources import SqsEventSource
from aws_cdk.aws_s3 import Bucket
from aws_cdk.aws_s3_notifications import SqsDestination
from aws_cdk.aws_sqs import DeadLetterQueue, Queue
from constructs import Construct


class Level1BStack(Stack):
    def __init__(
        self,
        scope: Construct,
        id: str,
        input_bucket_name: str,
        output_bucket_name: str,
        lambda_timeout: Duration = Duration.seconds(900),
        queue_retention_period: Duration = Duration.days(14),
        code_version: str = "",
        **kwargs
    ) -> None:
        super().__init__(scope, id, **kwargs)

        input_bucket = Bucket.from_bucket_name(
            self,
            "Level1ABucket",
            input_bucket_name,
        )

        output_bucket = Bucket.from_bucket_name(
            self,
            "Level1BBucket",
            output_bucket_name,
        )

        level1b_lambda = DockerImageFunction(
            self,
            "Level1BLambda",
            code=DockerImageCode.from_image_asset("."),
            timeout=lambda_timeout,
            architecture=Architecture.X86_64,
            memory_size=4096,
            environment={
                "L1B_BUCKET": output_bucket.bucket_name,
                "L1B_VERSION": code_version,
            },
        )

        event_queue = Queue(
            self,
            "Level1AQueue",
            retention_period=queue_retention_period,
            visibility_timeout=lambda_timeout,
            removal_policy=RemovalPolicy.RETAIN,
            dead_letter_queue=DeadLetterQueue(
                max_receive_count=1,
                queue=Queue(
                    self,
                    "FailedCalibrationQueue",
                    retention_period=queue_retention_period,
                ),
            ),
        )

        input_bucket.add_object_created_notification(
            SqsDestination(event_queue),
        )

        level1b_lambda.add_event_source(SqsEventSource(
            event_queue,
            batch_size=1,
        ))

        input_bucket.grant_read(level1b_lambda)
        output_bucket.grant_put(level1b_lambda)
