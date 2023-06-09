## Загрузка датасетов
В проекте используется два набора данных:
1. [AUC v2](https://abouelnaga.io/distracted-driver/)
2. [Kaggle State Farm](https://www.kaggle.com/competitions/state-farm-distracted-driver-detection/data)

Все датасеты хранятся в S3 хранилище. Доступ и версионирование осуществляется посредством dvc.
Для загрузки датасета в .dvc/config должны быть указаны параметры удаленного хранилища (s3 бакета).

Пример загрузки определенного датасета: 
```
dvc pull auc.zip.dvc
```

## Описание файлов:

- auc.zip  
Датасет auc.  
Состоит из двух частей: v1_cam1_no_split и v2_cam1_cam2_split_by_driver
  
- sfd_split_811.zip  
Часть датасета kaggle state farm. Размеченные данные,  
разделенные на train, test, val в пропорции (0.8, 0.1, 0.1).
  
- sfd_with_labels.zip  
Часть датасета kaggle state farm. Размеченные данные.  
  
- sfd_without_labels.zip  
Часть датасета kaggle state farm. Неразмеченные данные.