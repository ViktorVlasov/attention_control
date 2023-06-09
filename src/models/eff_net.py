# загружаем библиотеки
from efficientnet_pytorch import EfficientNet
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics import Accuracy


class EffNetClassifier(pl.LightningModule):
    def __init__(self,
                 model: str = "efficientnet-b0",
                 in_channels: int = 3,
                 out_classes: int = 10,
                 eta: int = 3e-4,
                 device: str = 'cpu',
                 **kwargs) -> None:
        """Бейзлайн модель EfficientNet, предобученная
         и с базовой архитектурой. Класс отнаследован
         от pl.LightningModule

        Args:
            model (str, optional): Версия модели, которая будет использоваться.
            По умолчанию "efficientnet-b0".
            in_channels (int, optional): Количество каналов на входе модели.
            По умолчанию 3.
            out_classes (int, optional): Количество каналов на выходе.
            По умолчанию 10.
            eta (int, optional): Параметр learning rate для оптимизатора.
            По умолчанию 3e-4.
        """
        super().__init__()
        self.model = EfficientNet.from_pretrained(model,
                                                  num_classes=out_classes,
                                                  in_channels=in_channels)
        self.out_classes = out_classes
        self.criterion = nn.CrossEntropyLoss()
        self.hparams.eta = eta
        self.model_device = device
        self.metrics = {
            "accuracy": Accuracy(task="multiclass",
                                 num_classes=out_classes
                                 ).to(self.model_device)
        }

        self.preds_stage = {"train": {"loss": [], "accuracy": []},
                            "valid": {"loss": [], "accuracy": []},
                            "test": {"loss": [], "accuracy": []}}

    def forward(self, x: torch.Tensor = None) -> torch.Tensor:
        """Функция для forward pass модели

        Args:
            x (torch.Tensor, required): Тензор с изображениями
            для forward pass. По умолчанию None.

        Returns:
            torch.Tensor: Возвращает тензор с результатом прохода модели.
        """
        return self.model(x)

    def shared_step(self,
                    sample=None,
                    stage: str = None) -> torch.Tensor:
        """Общий шаг для всех этапов - обучения,
        валидации и тестирования модели.

        Args:
            sample (required): Тензор с изображениями. По умолчанию None.
            stage (str, required): Этап, может принимать значения из списка:
            ['train', 'valid', 'test']. По умолчанию None.

        Returns:
            torch.Tensor: Возвращает тензор лоссов после прохода модели.
        """
        x, y = sample
        logits = self.forward(x.to(torch.float32))
        preds = torch.argmax(logits, 1)
        loss = self.criterion(logits, y.to(torch.int64))
        # собираем в атрибут
        self.preds_stage[stage]['loss'].append(loss.detach().cpu())
        self.preds_stage[stage]['accuracy'].append(self.metrics["accuracy"]
                                                   (preds, y).detach().cpu())
        return loss

    def shared_epoch_end(self,
                         stage: str = None) -> None:
        """Функция расчета лоссов и точности в конце эпохи
        и логгирования этих значений

        Args:
            stage (str, required): Этап, может принимать значения из списка:
            ['train', 'valid', 'test']. По умолчанию None.
        """
        # считаем по стадиям
        loss = self.preds_stage[stage]['loss']
        loss = torch.stack(loss)
        loss = np.mean([x.item() for x in loss])

        acc = self.preds_stage[stage]['accuracy']
        acc = torch.stack(acc)
        acc = np.mean([x.item() for x in acc])

        if self.model_device == 'mps':
            loss = loss.astype(np.float32)
            acc = acc.astype(np.float32)

        metrics = {
            f"{stage}_loss": loss,
            f"{stage}_acc": acc
        }

        self.log_dict(metrics, prog_bar=True)
        # чистим
        self.preds_stage[stage]['loss'].clear()
        self.preds_stage[stage]['accuracy'].clear()

    # создаем оптимизатор, шедулер
    def configure_optimizers(self) -> dict:
        """Функция создания оптимизатора и шедулера

        Returns:
            dict: Возвращает словарь вида
            {"optimizer": optimizer, "lr_scheduler": scheduler_dict},
            который содержит в себе оптимизатор и шедулер
        """
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.eta
        )

        scheduler_dict = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                patience=5
            ),
            "interval": "epoch",
            "monitor": "valid_loss"
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}

    # шаг обучения
    def training_step(self,
                      batch=None,
                      batch_idx=None) -> torch.Tensor:
        """Функция обучения модели

        Args:
            batch_idx (_type_): Идентификатор батча
            batch (_type_, optional): Тензор, массив с изображениями.
            По умолчанию None.

        Returns:
            torch.Tensor: Возвращает тензор лоссов после прохода модели.
        """
        return self.shared_step(batch, "train")

    # шаг подсчета метрик обучения
    def on_training_epoch_end(self) -> None:
        """Функция расчета лоссов и точности на конце эпохи"""
        return self.shared_epoch_end("train")

    # шаг валидации
    def validation_step(self,
                        batch=None,
                        batch_idx=None) -> torch.Tensor:
        """Функция валидации модели

        Args:
            batch_idx (_type_): Идентификатор батча
            batch (_type_, optional): Тензор, массив с изображениями.
            По умолчанию None.

        Returns:
            torch.Tensor: Возвращает тензор лоссов после прохода модели.
        """
        return self.shared_step(batch, "valid")

    # шаг подсчета метрик валидации
    def on_validation_epoch_end(self) -> None:
        """Функция расчета лоссов и точности на конце эпохи"""
        return self.shared_epoch_end("valid")

    # шаг тестирования
    def test_step(self,
                  batch=None,
                  batch_idx=None) -> torch.Tensor:
        """Функция тестирования модели

        Args:
            batch_idx (_type_): Идентификатор батча
            batch (_type_, optional): Тензор, массив с изображениями.
            По умолчанию None.

        Returns:
            torch.Tensor: Возвращает тензор лоссов после прохода модели.
        """
        return self.shared_step(batch, "test")

    # шаг подсчета метрик тестирования
    def on_test_epoch_end(self) -> None:
        """Функция расчета лоссов и точности на конце эпохи"""
        return self.shared_epoch_end("test")
