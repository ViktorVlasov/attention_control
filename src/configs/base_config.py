"""
Базовый конфигурационный файл.
Предлагается использовать YACS config (https://github.com/rbgirshick/yacs) для
поддержки базового конфигурационного файла. Затем для каждого конкретного
эксперимента можно создать конфигурационный файл, который будет переопределять
необходимые параметры базового конфигурационного файла.


Пример использования:
    1. Создать конфигурационный файл для конкретного эксперимента
    (например, configs/experiment_1.yaml)
    2. В конфигурационном файле переопределить необходимые параметры
    базового конфигурационного файла
    3. В модуле, где необходимо использовать конфигурационные параметры,
    импортировать функцию combine_config
    4. Вызвать функцию combine_config, передав в качестве аргумента путь
    к конфигурационному файлу конкретного эксперимента
    5. Полученный объект yacs CfgNode можно использовать для доступа
    к конфигурационным параметрам
"""

import os.path as osp
from typing import Union, Optional
from datetime import datetime
from yacs.config import CfgNode as CN


_C = CN()

# Root directory of project
_C.ROOT = CN()
_C.ROOT.PATH = ''  # Путь к корневой директории проекта в Google Colab

# Dataset
_C.DATASET = CN()

# Raw image dir, interim (after split) image dir
_C.DATASET.RAW_IMG_DIR = 'data/raw/sfd_with_labels'
_C.DATASET.INTERIM_IMG_DIR = 'data/interim/dataset_split'
_C.DATASET.TRAIN_DIR = 'train'
_C.DATASET.VAL_DIR = 'val'
_C.DATASET.TEST_DIR = 'test'

# Proportion for split_dataset()
_C.DATASET.TRAIN_SIZE = 0.6
_C.DATASET.TEST_SIZE = 0.2
_C.DATASET.VAL_SIZE = 0.2

# datamodule params
_C.DATAMODULE = CN()
_C.DATAMODULE.IMG_SIZE = 224
_C.DATAMODULE.BATCH_SIZE = 1
_C.DATAMODULE.NUM_WORKERS = 2
_C.DATAMODULE.SHUFFLE_TRAIN = True
_C.DATAMODULE.SHUFFLE_VAL = False
_C.DATAMODULE.SHUFFLE_TEST = False

# Learning
_C.LEARNING = CN()
_C.LEARNING.BASE_MODEL = 'efficientnet-b0'
_C.LEARNING.CHECKPOINT_PATH: Optional[str] = None  # 'models/efficientnet-b0.ckpt'
_C.LEARNING.ETA = 3e-2
_C.LEARNING.MAX_EPOCHS = 10
_C.LEARNING.DEVICE = 'cpu'
_C.LEARNING.SEED = 42

# MLFLOW

_C.MLFLOW = CN()
_C.MLFLOW.TRACKING_URI = ''
_C.MLFLOW.S3_ENDPOINT_URL = ''
_C.MLFLOW.EXPERIMENT = ''
_C.MLFLOW.LOG_CHECKPOINTS = True
_C.MLFLOW.LOG_MODEL = True
_C.MLFLOW.RUN_NAME = f'{datetime.now().strftime("%yy_%mm_%dd_%Hh_%Mm_%Ss")}'

# Local logging
_C.LOGGING = CN()
_C.LOGGING.LOGS_FOLDER = 'logs'
_C.LOGGING.MODEL_NAME = f'{_C.LEARNING.BASE_MODEL}.pt'
_C.LOGGING.LOGGING_INTERVAL = 'step'

# Checkpoint
_C.CHECKPOINT = CN()
_C.CHECKPOINT.CKPT_FOLDER = 'checkpoints'
_C.CHECKPOINT.FILENAME = '{epoch}_{valid_acc:.2f}_{valid_loss:.2f}'
_C.CHECKPOINT.SAVE_TOP_K = 2
_C.CHECKPOINT.CKPT_MONITOR = 'valid_loss'
_C.CHECKPOINT.CKPT_MODE = 'min'

# Early stopping
_C.ES = CN()
_C.ES.MONITOR = 'valid_loss'
_C.ES.MIN_DELTA = 2e-4
_C.ES.PATIENCE = 10
_C.ES.VERBOSE = False,
_C.ES.MODE = 'min'

# test
_C.TEST = CN()
_C.TEST.SAVE = True


def get_cfg_defaults():
    """Возвращает yacs CfgNode объект со значениями по умолчанию"""
    return _C.clone()


def combine_config(cfg_path: Union[str, None] = None):
    """
    Объединяет базовый конфигурационный файл с
    конфигурационным файлом конкретного эксперимента
    Args:
         cfg_path (str): file in .yaml or .yml format with
         config parameters or None to use Base config
    Returns:
        yacs CfgNode object
    """
    base_config = get_cfg_defaults()
    if cfg_path is not None:
        if osp.exists(cfg_path):
            base_config.merge_from_file(cfg_path)
        else:
            raise FileNotFoundError(f'File {cfg_path} does not exists')

    # Join paths
    base_config.DATASET.RAW_IMG_DIR = osp.join(
        base_config.ROOT.PATH,
        base_config.DATASET.RAW_IMG_DIR
    )
    base_config.DATASET.INTERIM_IMG_DIR = osp.join(
        base_config.ROOT.PATH,
        base_config.DATASET.INTERIM_IMG_DIR
    )
    base_config.DATASET.TRAIN_DIR = osp.join(
        base_config.DATASET.INTERIM_IMG_DIR,
        base_config.DATASET.TRAIN_DIR
    )
    base_config.DATASET.VAL_DIR = osp.join(
        base_config.DATASET.INTERIM_IMG_DIR,
        base_config.DATASET.VAL_DIR
    )
    base_config.DATASET.TEST_DIR = osp.join(
        base_config.DATASET.INTERIM_IMG_DIR,
        base_config.DATASET.TEST_DIR
    )
    # путь к чекпоинту модели, с которого хотим продолжить обучение
    if base_config.LEARNING.CHECKPOINT_PATH:
        base_config.LEARNING.CHECKPOINT_PATH = osp.join(
            base_config.ROOT.PATH,
            base_config.LEARNING.CHECKPOINT_PATH
        )

    # локальное логгирование аналогично mlflow
    # путь к эксперименту
    base_config.LOGGING.EXPERIMENT_PATH = osp.join(
        base_config.ROOT.PATH,
        base_config.LOGGING.LOGS_FOLDER,
        base_config.MLFLOW.EXPERIMENT
    )
    # путь к запуску внутри эксперимента
    base_config.LOGGING.RUN_PATH = osp.join(
        base_config.LOGGING.EXPERIMENT_PATH,
        base_config.MLFLOW.RUN_NAME
    )
    # путь к сохраненной модели внутри запуска
    base_config.LOGGING.SAVING_MODEL_PATH = osp.join(
        base_config.LOGGING.RUN_PATH,
        base_config.LOGGING.MODEL_NAME
    )
    # путь к чекпоинтам модели внутри запуска
    base_config.CHECKPOINT.CKPT_PATH = osp.join(
        base_config.LOGGING.RUN_PATH,
        base_config.CHECKPOINT.CKPT_FOLDER
    )

    return base_config
