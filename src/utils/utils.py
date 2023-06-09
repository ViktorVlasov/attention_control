import mlflow
import os
from yacs.config import CfgNode


def setup_mlflow(config: CfgNode) -> str:
    mlflow.set_tracking_uri(config.MLFLOW.TRACKING_URI)
    experiment = mlflow.set_experiment(config.MLFLOW.EXPERIMENT)
    mlflow_run = mlflow.start_run(experiment_id=experiment.experiment_id,
                                  run_name=config.MLFLOW.RUN_NAME)
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = config.MLFLOW.S3_ENDPOINT_URL

    return mlflow_run.info.run_id


def log_config(config: CfgNode, config_path: str):
    params = {
        'DATASET.RAW_IMG_DIR': _get_rel_raw_dir(config.DATASET.RAW_IMG_DIR),
        'DATASET.TRAIN_SIZE': config.DATASET.TRAIN_SIZE,
        'DATASET.VAL_SIZE': config.DATASET.VAL_SIZE,
        'DATASET.TEST_SIZE': config.DATASET.TEST_SIZE,

        'DATAMODULE.IMG_SIZE': config.DATAMODULE.IMG_SIZE,
        'DATAMODULE.BATCH_SIZE': config.DATAMODULE.BATCH_SIZE,
        'DATAMODULE.NUM_WORKERS': config.DATAMODULE.NUM_WORKERS,
        'DATAMODULE.SHUFFLE_TRAIN': config.DATAMODULE.SHUFFLE_TRAIN,
        'DATAMODULE.SHUFFLE_VAL': config.DATAMODULE.SHUFFLE_VAL,
        'DATAMODULE.SHUFFLE_TEST': config.DATAMODULE.SHUFFLE_TEST,

        'LEARNING.BASE_MODEL' : config.LEARNING.BASE_MODEL,
        'LEARNING.ETA': config.LEARNING.ETA,
        'LEARNING.MAX_EPOCHS': config.LEARNING.MAX_EPOCHS,
        'LEARNING.DEVICE': config.LEARNING.DEVICE,
        'LEARNING.SEED': config.LEARNING.SEED
    }
    mlflow.log_params(params)
    mlflow.log_artifact(config_path)


def _get_rel_raw_dir(abs_path: str):
    """Делает из абсолютного пути RAW_IMG_DIR относительный
    Например,
    Было: '/Users/user/Desktop/projects/attention_contol/data/raw/mini_dataset'
    Стало: 'data/raw/mini_dataset'

    Args:
        abs_path (str): _description_

    Returns:
        _type_: _description_
    """
    substring = 'data/'
    parts = abs_path.split(substring)
    return substring + parts[1]
