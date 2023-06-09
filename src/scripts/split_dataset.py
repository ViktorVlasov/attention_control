import argparse
from typing import Union
from pathlib import Path
from src.configs.base_config import combine_config
from src.data.data_splitter import (
    Proportion,
    DistractedDriverReader,
    DistractedDriverSplitter
    )


def create_parser():
    """Функция для парсинга аргументов, указанных при вызове скрипта

    Returns:
        argparse.Namespace: объект, который содержит аргументы в виде атрибутов
    """
    parser = argparse.ArgumentParser(description='Parameters to split dataset')
    parser.add_argument("-config_path",
                        default='src/configs/default_config.yaml',
                        type=str)
    args = parser.parse_args()
    return args


def split_dataset(dataset_path: Union[Path, str],
                  output_path: Union[Path, str],
                  train_size: float = 0.6,
                  val_size: float = 0.2,
                  test_size: float = 0.2,
                  seed: int = 42) -> None:
    """Разделяет набор данных на обучающую, валидационную и тестовую выборки.
    Использует библиотеку splitfolders.

    Args:
        dataset_path (Path): Путь к директории с набором данных
        output_path (Path): Путь к директории для вывода разделенных выборок
        train_size (float, optional): Пропорция обучающей выборки. Defaults to 0.6.
        val_size (float, optional): Пропорция валидационной выборки. Defaults to 0.2.
        test_size (float, optional): Пропорция тестовой выборки. Defaults to 0.2.
        seed (int, optional): Число для ГПСЧ в splitfolders
    """

    proportion = Proportion(train_size, val_size, test_size)
    reader = DistractedDriverReader(dataset_path)
    splitter = DistractedDriverSplitter(data_reader=reader,
                                        output_path=output_path,
                                        seed=seed)
    splitter.split_data(proportion)


if __name__ == '__main__':
    args = create_parser()
    cfg = combine_config(args.config_path)

    split_dataset(dataset_path=cfg.DATASET.RAW_IMG_DIR,
                  output_path=cfg.DATASET.INTERIM_IMG_DIR,
                  train_size=cfg.DATASET.TRAIN_SIZE,
                  test_size=cfg.DATASET.TEST_SIZE,
                  val_size=cfg.DATASET.VAL_SIZE,
                  seed=cfg.LEARNING.SEED)
