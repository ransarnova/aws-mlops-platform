import os

from sagemaker.core import image_uris
from sagemaker.core.helper.session_helper import Session
from sagemaker.train import ModelTrainer
from sagemaker.train.configs import (
    Compute,
    InputData,
    OutputDataConfig,
    SourceCode,
    StoppingCondition,
)


REGION = "ap-south-1"

role_arn = os.environ["SAGEMAKER_ROLE_ARN"]
bucket_name = os.environ["MLOPS_BUCKET"]


# ---------------------------------------------------------
# 1. Create SageMaker session
# ---------------------------------------------------------

session = Session()

print(f"AWS Region: {session.boto_region_name}")


# ---------------------------------------------------------
# 2. Resolve AWS-managed Scikit-learn training image
# ---------------------------------------------------------

training_image = image_uris.retrieve(
    framework="sklearn",
    region=REGION,
    version="1.4-2",
    py_version="py3",
    instance_type="ml.m5.large",
    image_scope="training",
)

print(f"Training image: {training_image}")


# ---------------------------------------------------------
# 3. Define training source code
# ---------------------------------------------------------

source_code = SourceCode(
    source_dir="sagemaker",
    entry_script="train.py",
)


# ---------------------------------------------------------
# 4. Define compute
# ---------------------------------------------------------

compute = Compute(
    instance_type="ml.m5.large",
    instance_count=1,
    volume_size_in_gb=10,
)


# ---------------------------------------------------------
# 5. Safety limit
# ---------------------------------------------------------

stopping_condition = StoppingCondition(
    max_runtime_in_seconds=600,
)


# ---------------------------------------------------------
# 6. Define training input
# ---------------------------------------------------------

training_input = InputData(
    channel_name="training",
    data_source=f"s3://{bucket_name}/data/training/",
)


# ---------------------------------------------------------
# 7. Define model output location
# ---------------------------------------------------------

output_config = OutputDataConfig(
    s3_output_path=f"s3://{bucket_name}/artifacts/",
)


# ---------------------------------------------------------
# 8. Configure trainer
# ---------------------------------------------------------

trainer = ModelTrainer(
    training_image=training_image,
    source_code=source_code,
    compute=compute,
    stopping_condition=stopping_condition,
    output_data_config=output_config,
    role=role_arn,
    sagemaker_session=session,
    base_job_name="loan-default-training",
    hyperparameters={
        "test-size": 0.2,
        "random-state": 42,
    },
)


# ---------------------------------------------------------
# 9. Launch SageMaker training job
# ---------------------------------------------------------

training_job = trainer.train(
    input_data_config=[training_input],
    wait=True,
    logs=True,
)


# ---------------------------------------------------------
# 10. Print results
# ---------------------------------------------------------

print("\nTraining completed.")

print(f"Training job: {training_job.name}")

print(
    "Model artifact:",
    training_job.model_artifacts.s3_model_artifacts,
)
