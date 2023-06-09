from src.configs.augmentations_config import get_torchvision_transforms, \
    train_augmentations, \
    val_augmentations, \
    test_augmentations
from src.data.data_loader import DistractedDriverDataset
from src.data.augmentations import DataAugmentations


def get_datamodule(config: callable = None):
    base_transforms = get_torchvision_transforms(
        image_size=config.DATAMODULE.IMG_SIZE
    )

    augmentations = DataAugmentations(
        base_transforms=base_transforms,
        train_augmentations=train_augmentations,
        val_augmentations=val_augmentations,
        test_augmentations=test_augmentations
    )
    # создаем DataModule-экземпляр
    data_module = DistractedDriverDataset(
        image_size=config.DATAMODULE.IMG_SIZE,
        batch_size=config.DATAMODULE.BATCH_SIZE,
        train_dir=config.DATASET.TRAIN_DIR,
        val_dir=config.DATASET.VAL_DIR,
        test_dir=config.DATASET.TEST_DIR,
        num_workers=config.DATAMODULE.NUM_WORKERS,
        train_shuffle=config.DATAMODULE.SHUFFLE_TRAIN,
        val_shuffle=config.DATAMODULE.SHUFFLE_VAL,
        test_shuffle=config.DATAMODULE.SHUFFLE_TEST,
        augmentations=augmentations
    )

    return data_module
