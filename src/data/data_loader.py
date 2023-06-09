"""
Модуль содержит классы для загрузки данных.
Предполагается что мы будем работать с PyTorch Lighting, поэтому класы должны
наследоваться от pytorch_lightning.LightningDataModule и в них должны быть реализованы
загрузчики данных для обучения, валидации и тестирования моделей.
"""
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from src.data.augmentations import DataAugmentations


class DistractedDriverDataset(pl.LightningDataModule):
    def __init__(self,
                 image_size: int,
                 batch_size: int,
                 train_dir: str,
                 val_dir: str,
                 test_dir: str,
                 num_workers: int,
                 train_shuffle: bool,
                 val_shuffle: bool,
                 test_shuffle: bool,
                 augmentations: DataAugmentations):
        """
        Класс загрузки данных для обучения, валидации и тестирования моделей

        Args:
            image_size: размер изображения
            batch_size: размер батча
            train_dir: путь к тренировочному набору данных
            val_dir: путь к валидационному набору данных
            test_dir: путь к тестовому набору данных
            num_workers: количество потоков для загрузки данных
            train_shuffle: перемешивать ли тренировочный набор данных
            val_shuffle: перемешивать ли валидационный набор данных
            test_shuffle: перемешивать ли тестовый набор данных
            augmentations: Класс с дополнительными преобразованиями изображений для
            каждой стадии (train, val, test). Изменение аугментаций происходит за счет
            обновления конфигурационного файла `src/configs/augmentations_config.py`

        Usage example:
            >>> from src.configs.base_config import combine_config
            >>> from src.data.augmentations import DataAugmentations
            >>> from src.data.data_loader import DistractedDriverDataset
            >>> from src.configs.augmentations_config import get_base_transforms, \
                                                             train_augmentations, \
                                                             val_augmentations, \
                                                             test_augmentations

            # Load config file for certain experiment and combine it with base config
            >>> cfg = combine_config(cfg_path='configs/experiment_1.yaml')
            # Get base transforms
            >>> base_transforms = get_base_transforms(image_size=cfg.DATASET.IMG_SIZE)
            # Define augmentations class with base transforms
            # and augmentations for each stage
            >>> augmentations = DataAugmentations(
                    base_transforms=base_transforms, \
                    train_augmentations=train_augmentations, \
                    val_augmentations=val_augmentations, \
                    test_augmentations=test_augmentations \
                    )
            # Define data module. Get params from config file and augmentations class
            >>> data_module = DistractedDriverDataset(
                    image_size=cfg.DATASET.IMG_SIZE, \
                    batch_size=cfg.DATASET.BATCH_SIZE, \
                    train_dir=cfg.DATASET.TRAIN_DIR, \
                    val_dir=cfg.DATASET.VAL_DIR, \
                    test_dir=cfg.DATASET.TEST_DIR,\
                    num_workers=cfg.DATASET.NUM_WORKERS, \
                    train_shuffle=cfg.DATASET.SHUFFLE_TRAIN, \
                    val_shuffle=cfg.DATASET.SHUFFLE_VAL, \
                    test_shuffle=cfg.DATASET.SHUFFLE_TEST, \
                    augmentations=augmentations)
        """
        super().__init__()
        self.image_size = image_size
        self.batch_size = batch_size
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.num_workers = num_workers
        self.train_shuffle = train_shuffle
        self.val_shuffle = val_shuffle
        self.test_shuffle = test_shuffle
        self.augmentations = augmentations
        self.test_set = None
        self.train_set = None
        self.val_set = None

    def setup(self, stage=None):
        self.train_set = ImageFolder(self.train_dir,
                                     transform=self.augmentations.train_transforms)
        self.val_set = ImageFolder(self.val_dir,
                                   transform=self.augmentations.val_transforms)
        self.test_set = ImageFolder(self.test_dir,
                                    transform=self.augmentations.test_transforms)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size,
                          shuffle=self.train_shuffle, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size,
                          shuffle=self.val_shuffle, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size,
                          shuffle=self.test_shuffle, num_workers=self.num_workers)
