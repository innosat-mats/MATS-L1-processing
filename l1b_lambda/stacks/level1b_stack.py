from aws_cdk import Duration, Size, Stack, RemovalPolicy
from aws_cdk.aws_lambda import Architecture, LayerVersion, Runtime
from aws_cdk.aws_lambda_event_sources import SqsEventSource
from aws_cdk.aws_lambda_python_alpha import PythonFunction
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
        instrument_bucket_name: str,
        rclone_arn: str,
        config_ssm_name: str,
        lambda_timeout: Duration = Duration.seconds(900),
        queue_retention_period: Duration = Duration.days(14),
        queue_visibility_timeout: Duration = Duration.hours(12),
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

        instrument_bucket = Bucket.from_bucket_name(
            self,
            "InstrumentBucket",
            instrument_bucket_name,
        )

        rclone_layer = LayerVersion.from_layer_version_arn(
            self,
            "RCloneLayer",
            rclone_arn,
        )

        level1b_lambda = PythonFunction(
            self,
            "Level1BLambda",
            entry="level1b",
            handler="lambda_handler",
            index="handlers/level1b.py",
            timeout=lambda_timeout,
            architecture=Architecture.X86_64,
            runtime=Runtime.PYTHON_3_9,
            memory_size=1024,
            ephemeral_storage_size=Size.mebibytes(1024),
            environment={
                "L1B_BUCKET": output_bucket.bucket_name,
                "INSTRUMENT_BUCKET": instrument_bucket.bucket_name,
                "RCLONE_CONFIG_SSM_NAME": config_ssm_name,
            },
            layers=[rclone_layer],
        )

        event_queue = Queue(
            self,
            "Level1AQueue",
            retention_period=queue_retention_period,
            visibility_timeout=queue_visibility_timeout,
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
        instrument_bucket.grant_read(level1b_lambda)
