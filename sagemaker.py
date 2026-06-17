import boto3
from sagemaker.huggingface import HuggingFace
from sagemaker import Session

role = "arn:aws:iam::042724764672:role/SageMakerExecutionRole"

session = Session()

estimator = HuggingFace(

    entry_point="train.py",

    source_dir="src/training-model",

    role=role,

    instance_count=1,

    instance_type="ml.g5.xlarge",

    transformers_version="4.39",

    pytorch_version="2.1",

    py_version="py310",

    output_path="s3://my-model-bucket/models/",

)

estimator.fit(

    {

        "training":"s3://shivamrtc-mlop/datasets/real/"

    },

    wait=True

)

print("Training Completed")

print("Model Location:")

print(estimator.model_data)

training_job_name = estimator.latest_training_job.name

print(training_job_name)

sm = boto3.client("sagemaker")

sm.delete_training_job(

    TrainingJobName=training_job_name

)

print("Training Job Deleted")