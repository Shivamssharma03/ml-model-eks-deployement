GitHub Actions

      │

      ▼

python sagemaker_train.py

      │

      ▼

HuggingFaceEstimator

      │

      ├── Zip src/training-model/

      ├── Upload source

      ├── Create SageMaker Training Job

      ├── Create GPU instance

      ├── Install dependencies

      ├── Run train.py

      ├── Save model

      ├── Create model.tar.gz

      ├── Upload model.tar.gz to S3

      ├── Stop GPU instance

      └── Return control

      │

      ▼

delete_training_job()

      │

      ▼

Training Job metadata deleted