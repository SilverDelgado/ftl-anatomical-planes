"""Dataset para imágenes MRI FLAIR del proyecto 6."""
from pathlib import Path
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

# Estadísticas ImageNet (el modelo espera imágenes normalizadas así)
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]


def get_transform(train: bool, img_size: int = 224) -> T.Compose:
    steps = [T.Resize((img_size, img_size))]
    if train:
        steps.append(T.RandomHorizontalFlip())
    steps += [
        T.ToTensor(),
        T.Lambda(lambda x: x.repeat(3, 1, 1)),  # escala de grises → 3 canales
        T.Normalize(_MEAN, _STD),
    ]
    return T.Compose(steps)


class MSDataset(Dataset):
    """Carga slices MRI desde un CSV del proyecto 6.

    Columnas relevantes del CSV: image_path (relativa a data_root), label_id (0/1).
    """

    def __init__(self, csv_path: Path, data_root: Path, train: bool = True, img_size: int = 224):
        self.df        = pd.read_csv(csv_path)
        self.data_root = Path(data_root)
        self.transform = get_transform(train, img_size)

    # permite filtrar por hospital en el test set global
    @classmethod
    def from_df(cls, df: pd.DataFrame, data_root: Path, train: bool = False, img_size: int = 224):
        obj = cls.__new__(cls)
        obj.df        = df.reset_index(drop=True)
        obj.data_root = Path(data_root)
        obj.transform = get_transform(train, img_size)
        return obj

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row     = self.df.iloc[idx]
        img_path = self.data_root / row["image_path"]
        img     = Image.open(img_path).convert("L")
        label   = torch.tensor(float(row["label_id"]), dtype=torch.float32)
        return self.transform(img), label


def load_client_datasets(csvs, data_root, img_size=224):
    """Devuelve lista de MSDataset, uno por hospital."""
    return [MSDataset(csv, data_root, train=True, img_size=img_size) for csv in csvs]


def load_test_datasets(global_test_csv, data_root, img_size=224):
    """Devuelve (global_ds, [hospital_1_ds, ..., hospital_4_ds])."""
    df_global  = pd.read_csv(global_test_csv)
    global_ds  = MSDataset.from_df(df_global, data_root, train=False, img_size=img_size)
    client_ds  = [
        MSDataset.from_df(df_global[df_global["hospital_id"] == k], data_root, train=False, img_size=img_size)
        for k in range(1, 5)
    ]
    return global_ds, client_ds
