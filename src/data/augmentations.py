import copy
from typing import Union

import albumentations as A
from torchvision import transforms


class DataAugmentations:
    def __init__(self,
                 base_transforms: A.Compose,
                 train_augmentations: Union[dict, None],
                 val_augmentations: Union[dict, None],
                 test_augmentations: Union[dict, None]):
        """
        Класс для определения преобразований для тренировочного,
        валидационного и тестового наборов данных

        Args:
            base_transforms: базовые преобразования для изображений
            (например, Resize, Normalize, ToTensorV2)
            train_augmentations: дополнительные преобразования
            для тренировочного набора данных
            val_augmentations: дополнительные преобразования
            для валидационного набора данных
            test_augmentations: дополнительные преобразования
            для тестового набора данных

        Usage example:
            >>> from src.configs.base_config import combine_config
            >>> from src.configs.augmentations_config import get_base_transforms, \
                                                             train_augmentations, \
                                                             val_augmentations, \
                                                             test_augmentations

            >>> cfg = combine_config()

            >>> base_transforms = get_base_transforms(image_size=cfg.DATASET.IMG_SIZE)
            >>> data_augmentations = DataAugmentations(base_transforms=base_transforms,\
                                                       train_augmentations=train_augmentations,\
                                                       val_augmentations=val_augmentations,\
                                                       test_augmentations=test_augmentations)
            >>> print(data_augmentations.train_transforms)
            Compose([
                    Resize(always_apply=False, p=1,
                           height=244, width=244,
                           interpolation=1),
                    Compose([
                            HorizontalFlip(always_apply=False, p=0.5),
                            ],
                            p=1.0, bbox_params=None, keypoint_params=None,
                            additional_targets={}),
                    Normalize(always_apply=False, p=1.0, mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
                    ToTensorV2(always_apply=True, p=1.0, transpose_mask=False),
                    ],
                    p=1.0, bbox_params=None,
                    keypoint_params=None, additional_targets={})
        """
        self.base_transforms = base_transforms
        self.train_augmentations = train_augmentations
        self.val_augmentations = val_augmentations
        self.test_augmentations = test_augmentations

        self.train_transforms = self._get_final_transforms(
            additional_augmentations=self.train_augmentations
        )
        self.val_transforms = self._get_final_transforms(
            additional_augmentations=self.val_augmentations
        )
        self.test_transforms = self._get_final_transforms(
            additional_augmentations=self.test_augmentations
        )

    def _get_final_transforms(self, additional_augmentations: Union[dict, None]):
        final_transform = copy.deepcopy(self.base_transforms)

        if type(final_transform) is transforms.Compose:
            space_transforms = transforms
        elif type(final_transform) is A.Compose:
            space_transforms = A

        if additional_augmentations:
            add_aug = additional_augmentations.items()
            final_transform.transforms.insert(
                1,
                *[getattr(space_transforms, key)(**params) for key, params in add_aug]
            )
        return final_transform
