from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import MLFlowLogger

from src.models.eff_net import EffNetClassifier
from src.configs.base_config import combine_config
from src.data.get_datamodule import get_datamodule
from src.utils.utils import setup_mlflow, log_config
from yacs.config import CfgNode
from typing import List, Optional
import pytorch_lightning as pl
import torch
import argparse
import mlflow


def train(config: CfgNode,
          mlflow_run_id: Optional[str] = None):
    """Функция для обучения модели.

    Args:
        config (CfgNode): Конфигурационный файл.
    """
    # фиксируем сид
    pl.seed_everything(cfg.LEARNING.SEED)
    # torch.set_default_dtype(torch.float32)
    # np.float64 = np.float32

    # инициализация модели и модуля с данными
    data_module = get_datamodule(config=config)
    model = EffNetClassifier(eta=config.LEARNING.ETA,
                             model=config.LEARNING.BASE_MODEL,
                             device=config.LEARNING.DEVICE)

    # список коллбэков (отслеживание lr, ранние остановки, сохранения весов)
    callbacks = get_callbacks(config)

    # mlflow логгер
    mlf_logger = MLFlowLogger(experiment_name=config.MLFLOW.EXPERIMENT,
                              tracking_uri=config.MLFLOW.TRACKING_URI,
                              log_model=config.MLFLOW.LOG_CHECKPOINTS,
                              run_id=mlflow_run_id)

    # обучение и сохранение модели
    trainer = Trainer(
        max_epochs=config.LEARNING.MAX_EPOCHS,
        accelerator=config.LEARNING.DEVICE,
        callbacks=callbacks,
        logger=mlf_logger
    )

    if config.LEARNING.CHECKPOINT_PATH:
        trainer.fit(model,
                    data_module,
                    ckpt_path=config.LEARNING.CHECKPOINT_PATH)
    else:
        trainer.fit(model,
                    data_module,
                    ckpt_path=None)

    torch.save(model, config.LOGGING.SAVING_MODEL_PATH)

    if config.MLFLOW.LOG_MODEL:
        mlflow.pytorch.log_model(model, "model")


def get_callbacks(config: CfgNode) -> List[object]:
    """Возвращает список коллбэков для обучения модели.
    На данный момент используются ранние остановки, отслеживание learning rate,
    чекпоинты модели.

    Args:
        config (CfgNode): Конфигурационный объект.

    Returns:
        List[object]: Список коллбэков.
    """

    # список для отслеживания lr, ранних остановок, сохранения весов
    callbacks = [
        ModelCheckpoint(
            dirpath=config.CHECKPOINT.CKPT_PATH,
            filename=config.CHECKPOINT.FILENAME,
            save_top_k=config.CHECKPOINT.SAVE_TOP_K,
            monitor=config.CHECKPOINT.CKPT_MONITOR,
            mode=config.CHECKPOINT.CKPT_MODE
        ),
        LearningRateMonitor(
            logging_interval=config.LOGGING.LOGGING_INTERVAL
        ),
        EarlyStopping(
            monitor=config.ES.MONITOR,
            min_delta=config.ES.MIN_DELTA,
            patience=config.ES.PATIENCE,
            verbose=config.ES.VERBOSE,
            mode=config.ES.MODE
        )
    ]

    return callbacks


def get_args_parser() -> argparse.Namespace:
    """Функция для парсинга аргументов, указанных при вызове скрипта

    Returns:
        argparse.Namespace: объект, который содержит аргументы в виде атрибутов
    """
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("-config_path",
                        default='src/configs/default_config.yaml',
                        type=str,
                        help="Путь к конфигурационному файлу")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args_parser()
    cfg_path = args.config_path
    cfg = combine_config(cfg_path)

    run_id = setup_mlflow(cfg)
    log_config(cfg, cfg_path)

    train(cfg, mlflow_run_id=run_id)
