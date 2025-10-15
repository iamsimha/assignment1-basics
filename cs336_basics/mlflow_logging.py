import mlflow
import os
from typing import Optional, Any, Mapping
from omegaconf import DictConfig, OmegaConf

def _flatten(d: Mapping[str, Any], parent_key: str = "", sep: str = ".") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, Mapping):
            out.update(_flatten(v, key, sep))
        else:
            # cast to plain types for MLflow
            if isinstance(v, (list, tuple, set)):
                v = json.dumps(list(v))
            out[key] = v
    return out

def _setup_mlflow(cfg: DictConfig):
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise ValueError("set MLFLOW_TRACKING_URI environment variable is required")
    mlflow.set_tracking_uri(tracking_uri)

    exp_name = getattr(cfg, "experiment_name", "transformer")
    mlflow.set_experiment(exp_name)

    run_name = getattr(cfg, "run_name", None)
    mlflow.start_run(run_name=run_name)

    # Log full config once (as dict artifact and flattened params)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    mlflow.log_dict(cfg_dict, artifact_file="config/config.json")
    mlflow.log_params(_flatten(cfg_dict))

def _teardown_mlflow():
    # End run if still active
    if mlflow.active_run() is not None:
        mlflow.end_run()