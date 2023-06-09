"""
Модуль предназначен для разбивки данных на обучающий, валидационный и тестовый наборы.

Пример использования:
$ python data_splitter.py --dataset_path /path/to/dataset --output_path /path/to/output
"""

import splitfolders
from typing import List, Union
from pathlib import Path
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class Proportion:
    """Contains train, val, test splitting proportion"""
    train: float = 0.85
    validation: float = 0.1
    test: float = 0.05
    assert train + validation + test == 1


class DataReader(ABC):
    """Read data"""
    def __init__(self, root_folder: Path):
        self.root_folder = root_folder

    @abstractmethod
    def get_full_data_list(self) -> List[Path]:
        """Returns list, contains paths to all images"""
        pass


class DistractedDriverReader(DataReader):
    image_extensions = ('.png', '.jpg', '.jpeg', '.JPG')

    def __init__(self,
                 root_folder: Union[Path, str],
                 images_folder: Union[Path, str] = Path('')):

        super().__init__(Path(root_folder))
        self.images_folder = Path(images_folder)

    def get_full_data_list(self) -> List[Path]:
        """Returns list, contains paths to all images"""
        image_folder = self.get_subfolder_path(self.root_folder, self.images_folder)
        full_data = [p.resolve() for p in image_folder.glob("**/*")
                     if p.suffix in self.image_extensions]
        return full_data

    def _check_path(self, path: Path) -> None:
        if not isinstance(path, Path):
            raise TypeError(f'`path` must be pathlib.Path, not {type(path)}')

        if not path.is_dir():
            raise FileNotFoundError(f'The directory at path {path} does not exists')

    def get_subfolder_path(self, root_dir: Path, internal_dir: Path) -> Path:
        """Get path to images dir inside root dataset path"""
        internal_path = root_dir.joinpath(internal_dir)
        self._check_path(internal_path)
        return internal_path


class Splitter(ABC):
    """Define the structure of train/val/test splitter"""
    def __init__(self,
                 data_reader: DistractedDriverReader = DistractedDriverReader):
        self.data_reader = data_reader

    def read_data(self, data_reader: DataReader) -> List[Path]:
        """Returns list, contains paths to all images"""
        return data_reader.get_full_data_list()

    @abstractmethod
    def split_data(self, proportion) -> List:
        """Splits data in given proportion and
        return list of samples (train, val, test)"""
        pass


class DistractedDriverSplitter(Splitter):
    def __init__(self,
                 data_reader: DistractedDriverReader,
                 output_path: Path,
                 seed=42,
                 group_prefix=None,
                 move=False):
        super().__init__(data_reader)
        self.output_path = output_path
        self.seed = seed
        self.group_prefix = group_prefix
        self.move = move

    def split_data(self, proportion: Proportion) -> None:
        splitfolders.ratio(
            self.data_reader.get_subfolder_path(self.data_reader.root_folder,
                                                self.data_reader.images_folder),
            output=self.output_path,
            seed=self.seed,
            ratio=(proportion.train, proportion.validation, proportion.test),
            group_prefix=self.group_prefix,
            move=self.move
        )
