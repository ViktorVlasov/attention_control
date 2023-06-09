"""
Конфигурационный файл для определения аугментаций

Базовые преобразования для изображений определяются в get_base_transforms
Дополнительные преобразования для тренировочного, валидационного и тестового наборов
данных определяются в словарях train_augmentations, val_augmentations,
test_augmentations, где ключ - название аугментации в библиотеке ALBUMENTATIONS,
значение - параметры аугментации
"""

from albumentations.pytorch import ToTensorV2
from albumentations import Compose, Resize, Normalize
from torchvision import transforms


def get_base_transforms(image_size: int) -> Compose:
    """Определяет базовые преобразования для изображений"""
    base_transform = Compose([
        Resize(image_size, image_size),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    return base_transform


def get_torchvision_transforms(image_size: int) -> transforms.Compose:
    """
    Определяет базовые преобразования для изображений.
    Преобразования из библиотеки torchvision.
    """
    base_transform = transforms.Compose([transforms.Resize((image_size, image_size)),
                                         transforms.ToTensor(),
                                         transforms.Normalize(
                                             mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
                                         ])
    return base_transform


train_augmentations = {
    'RandomHorizontalFlip': {'p': 0.5},
}

val_augmentations = None
test_augmentations = None
