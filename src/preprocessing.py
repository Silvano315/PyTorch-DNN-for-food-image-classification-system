import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torchvision
from typing import Tuple

class Transforms:
    def __init__(self, transforms: A.Compose):
        self.transforms = transforms

    def __call__(self, img, *args, **kwargs):
        return self.transforms(image=np.array(img))['image']

def create_preprocessing_pipeline(
    img_size: Tuple[int, int] = (224, 224),
    mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    std: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    augment: bool = False
) -> Transforms:
    """
    Create a preprocessing pipeline for image classification tasks.

    Args:
        img_size (Tuple[int, int]): Target image size (height, width). Default is (224, 224).
        mean (Tuple[float, float, float]): Mean values for normalization. Default is (0.5, 0.5, 0.5).
        std (Tuple[float, float, float]): Standard deviation values for normalization. Default is (0.5, 0.5, 0.5).
        augment (bool): Whether to apply data augmentation. Default is False.

    Returns:
        Transforms: A Transforms object wrapping the Albumentations composition.

    Example:
        transform = create_preprocessing_pipeline(augment=True)
        trainset = torchvision.datasets.ImageFolder(root='dataset/train', transform=transform)
    """
    transforms = [
        A.Resize(height=img_size[0], width=img_size[1]),
        A.Normalize(mean=mean, std=std),
    ]

    if augment:
        transforms.extend([
            A.RandomRotate90(),
            A.Flip(),
            A.Transpose(),
            A.OneOf([
                A.GaussNoise(),
                A.GaussianBlur(blur_limit=3),
            ], p=0.5),
            A.OneOf([
                A.OpticalDistortion(distort_limit=1.0),
                A.GridDistortion(num_steps=5, distort_limit=1.),
                A.ElasticTransform(alpha=3),
            ], p=0.5),
            A.OneOf([
                A.CLAHE(clip_limit=4.0),
                A.Sharpen(),
                A.Emboss(),
                A.RandomBrightnessContrast(),
            ], p=0.5),
            A.HueSaturationValue(p=0.5),
        ])

    transforms.append(ToTensorV2())

    return Transforms(A.Compose(transforms))


def create_datasets(data_dir: str, augment_train: bool = False):
    """
    Create train, validation, and test datasets.

    Args:
        data_dir (str): Root directory containing 'train', 'val', and 'test' subdirectories.
        augment_train (bool): Whether to apply data augmentation to the training set. Default is False.

    Returns:
        tuple: (trainset, valset, testset)

    Example:
        trainset, valset, testset = create_datasets('path/to/dataset', augment_train=True)
    """
    train_transform = create_preprocessing_pipeline(augment=augment_train)
    val_transform = create_preprocessing_pipeline(augment=False)

    trainset = torchvision.datasets.ImageFolder(root=f'{data_dir}/train', transform=train_transform)
    valset = torchvision.datasets.ImageFolder(root=f'{data_dir}/val', transform=val_transform)
    testset = torchvision.datasets.ImageFolder(root=f'{data_dir}/test', transform=val_transform)

    return trainset, valset, testset