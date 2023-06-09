import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

from src.configs.base_config import combine_config
from src.configs.augmentations_config import get_base_transforms


def inference(args: argparse.Namespace, idx2label: list[str]):
    """Функция, выполняющая предсказание модели на одном изображении.

    Для предсказания используется:
    - модель из src/models, название модели указывается в аргументе --model
    - изображение находящееся по пути из аргумента --input_path

    Результат модели сохраняется по пути из аргумента --output_path

    Args:
        args (argparse.Namespace): Аргументы, указанные при запуске скрипта
        idx_to_classes (list[str]): Лист с названиями классов
    """
    image_size = cfg.DATASET.IMG_SIZE
    model_path = Path(__file__).parent / Path('../../') / args.model
    image_path = Path(__file__).parent / Path('../../') / args.input_path

    device = torch.device(args.device)
    model = torch.load(model_path, map_location=device)
    target_image = Image.open(image_path)

    transform_test = get_base_transforms(image_size)
    input = transform_test(image=np.array(target_image))['image']

    model.eval()
    with torch.inference_mode():
        target_image_pred = model(input.unsqueeze(0).to(device)).cpu()
    target_image_pred_prob = torch.softmax(target_image_pred, dim=1).max().item()
    target_image_pred_label = torch.argmax(target_image_pred, dim=1).item()

    result = {
        'target_image': image_path.name,
        'predict': idx2label[target_image_pred_label],
        'probability': np.round(target_image_pred_prob, decimals=3)
    }

    if args.save_json:
        filename_json = image_path.stem + '.json'
        output_json = Path(args.output_path) / filename_json
        with open(output_json, "w") as outfile:
            json.dump(result, outfile)

    if args.save_plot:
        output_jpg = Path(args.output_path) / image_path.name
        plt.title(f'predict: {result["predict"]}, \
                  probability: {result["probability"]}')
        plt.axis('off')
        plt.imshow(target_image)
        plt.savefig(output_jpg, bbox_inches='tight')

    print(result)


def get_args_parser():
    """Функция для парсинга аргументов, указанных при вызове скрипта

    Returns:
        argparse.Namespace: объект, который содержит аргументы в виде атрибутов
    """
    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument('--model',
                        default='logs/23_04_16_10_50/efficientnet-b0.pt',
                        type=str,
                        help='Model name in models dir')
    parser.add_argument('--input_path',
                        default='data/raw/test.jpg',
                        type=str,
                        help='Path to image Default: data/raw/test.jpg')
    parser.add_argument('--output_path',
                        default='.',
                        type=str,
                        help='Path to folder for output')
    parser.add_argument('--device',
                        default="cpu",
                        type=str,
                        help="Device (Use cuda or cpu Default: cpu)")
    parser.add_argument('--save_json', action='store_true')
    parser.add_argument('--save_plot', action='store_true')

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args_parser()
    cfg = combine_config()
    inference(args, cfg.DATASET.CLASSES)
