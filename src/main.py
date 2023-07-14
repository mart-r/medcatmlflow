import mlflow.server

if __name__ == "__main__":
    mlflow.server._run_server(
        host="0.0.0.0",
        file_store_path="./mlruns",
        registry_store_uri="sqlite:///mlflow.db",
        default_artifact_root="./mlruns",
        serve_artifacts=True,
        artifacts_only=False,
        artifacts_destination="./artifacts",
        port=5000,
    )
