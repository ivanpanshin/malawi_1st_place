#!/usr/bin/env

nohup mlflow server --backend-store-uri="mlflow_logs/" --default-artifact-root="mlflow_artifact_store/" --port=7053 --host 0.0.0.0 &