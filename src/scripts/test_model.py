import argparse
import torch
import pandas as pd
import mlflow
import os

from pathlib import Path
from tqdm.autonotebook import tqdm
from sklearn.metrics import classification_report
from mlflow import log_artifacts

from src.data.data_loader import DistractedDriverDataset
from src.data.augmentations import DataAugmentations
from src.configs.base_config import combine_config
from src.configs.augmentations_config import get_torchvision_transforms, \
    train_augmentations, \
    val_augmentations, \
    test_augmentations


def get_data_module(batch_size: int, cfg):
    """Функция, собирающая data_module из собственного конфига,

    Args:
        batch_size (int): размер батча
        cfg (type): конфиг

    Returns:
        DistractedDriverDataset: объект data_module
    """

    base_transforms = get_torchvision_transforms(image_size=cfg.DATASET.IMG_SIZE)

    augmentations = DataAugmentations(base_transforms=base_transforms,
                                      train_augmentations=train_augmentations,
                                      val_augmentations=val_augmentations,
                                      test_augmentations=test_augmentations)

    data_module = DistractedDriverDataset(image_size=cfg.DATASET.IMG_SIZE,
                                          batch_size=batch_size,
                                          train_dir=cfg.DATASET.TRAIN_DIR,
                                          val_dir=cfg.DATASET.VAL_DIR,
                                          test_dir=cfg.DATASET.TEST_DIR,
                                          num_workers=cfg.DATASET.NUM_WORKERS,
                                          train_shuffle=cfg.DATASET.SHUFFLE_TRAIN,
                                          val_shuffle=cfg.DATASET.SHUFFLE_VAL,
                                          test_shuffle=cfg.DATASET.SHUFFLE_TEST,
                                          augmentations=augmentations)

    return data_module


def test(model,
         dataloader,
         cfg=None):
    """Функция для тестирования и валидации результатов модели

    Args:
        model: Используемая модель
        dataloader: test_dataloader или val_dataloader
        name_experiment (str): Название эксперимента
        save (bool): Сохранять результаты. По умолчанию: 'True'
                     Результаты тестирования/валидации будут сохранены
                     по пути attention_control/results/{name_experiment}.csv

    Returns:
        pandas.DataFrame:
                результат classification_report (precision, recall, f1-score, accuracy)

    """
    predictions = []
    targets = []
    device = cfg.LEARNING.DEVICE
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for X, y in tqdm(dataloader):
            outputs = torch.argmax(model(X.to(torch.float32).to(device)), 1)
            predictions.append(outputs.detach())
            targets.append(y.detach())

    predictions = (torch.cat(predictions)).cpu().numpy()
    targets = (torch.cat(targets)).cpu().numpy()
    classes = dataloader.dataset.classes

    report = classification_report(targets, predictions,
                                   target_names=classes, output_dict=True)

    mlflow.set_tracking_uri(cfg.MLFLOW.TRACKING_URI)
    mlflow.set_experiment(cfg.LEARNING.EXPERIMENT)
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = cfg.MLFLOW.S3_ENDPOINT_URL

    df_report = pd.DataFrame(report).transpose()
    if cfg.TEST.SAVE:
        save_path = Path(__file__).parent / '../../results'
        if not save_path.exists():
            Path.mkdir(save_path, parents=True)
        output_file = cfg.LEARNING.EXPERIMENT + '.csv'
        df_report.to_csv(save_path / output_file)
        log_artifacts(save_path)

    return df_report


def get_args_parser():
    """Функция для парсинга аргументов, указанных при вызове скрипта

    Returns:
        argparse.Namespace: объект, который содержит аргументы в виде атрибутов
    """
    parser = argparse.ArgumentParser(description="Test model")
    parser.add_argument('--model',
                        default='models/efficientnet-b0.pt',
                        type=str,
                        help='Model name in models dir')
    parser.add_argument("--config",
                        default='default_config.yaml',
                        type=str)

    return parser.parse_args()


def get_init_params():
    """Функция для инициализации параметров функции test из аргументов,
    указанных при вызове скрипта.

    Returns:
        tuple: кортеж из параметров (model, dataloader, model_name, device)
    """
    args = get_args_parser()

    model_path = Path(__file__).parent / Path('../../') / args.model
    path_to_yaml = Path(__file__).parent / '../configs' / args.config
    cfg = combine_config(path_to_yaml)

    model = torch.load(model_path, map_location=cfg.LEARNING.DEVICE)
    batch_size = cfg.DATASET.BATCH_SIZE

    dm = get_data_module(batch_size, cfg)
    dm.setup()
    dataloader = dm.test_dataloader()

    return model, dataloader, cfg


if __name__ == "__main__":
    model, dataloader, cfg = get_init_params()
    test(model, dataloader, cfg)
