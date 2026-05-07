from pathlib import Path
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

# Estadísticas ImageNet, donde se preentrenó Resnet18, entonces normalizamos dataset porque el modelo espera esta distriubcion de entrada
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]


def get_transform(train: bool, img_size: int = 224) -> T.Compose:
    steps = [T.Resize((img_size, img_size))] #resize a 224x224
    if train:
        steps.append(T.RandomHorizontalFlip())
    steps += [
        T.ToTensor(), #(1,124,124)
        T.Lambda(lambda x: x.repeat(3, 1, 1)), # escala de grises a 3 canales.. ResNet18 se entrenó para RGB, por eso pasamos a 3 canales (3,124,124)
        T.Normalize(_MEAN, _STD),
    ]
    return T.Compose(steps)


# image_path, label_id (0/1) es lo que importa
class MSDataset(Dataset):

    def __init__(self, csv_path: Path, data_root: Path, train: bool = True, img_size: int = 224):
        self.df = pd.read_csv(csv_path)
        self.data_root = Path(data_root)
        self.transform = get_transform(train, img_size)

    # para filtrar por hospital en el test set global
    @classmethod
    def from_df(cls, df: pd.DataFrame, data_root: Path, train: bool = False, img_size: int = 224):
        obj = cls.__new__(cls)
        obj.df = df.reset_index(drop=True)
        obj.data_root = Path(data_root)
        obj.transform = get_transform(train, img_size)
        return obj

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx] #coge fila idx del df
        img_path = self.data_root / row["image_path"]
        img = Image.open(img_path).convert("L") #abre imagen en scala de grises
        label = torch.tensor(float(row["label_id"]), dtype=torch.float32) #label 0 o 1
        return self.transform(img), label #(imagen transformada, label)


def get_dataloader(dataset: Dataset, batch_size: int, shuffle: bool = True, num_workers: int = 0):
    from torch.utils.data import DataLoader
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )

def load_client_datasets(csvs, data_root, img_size=224):
    datasets = []
    for csv in csvs:
        datasets.append(MSDataset(csv, data_root, train=True, img_size=img_size))
    return datasets

def load_test_datasets(global_test_csv, data_root, img_size=224):
    # devuelve (global_ds, [h1_ds, h2_ds, h3_ds, h4_ds]) para eval
    df_global = pd.read_csv(global_test_csv)
    global_ds = MSDataset.from_df(df_global, data_root, train=False, img_size=img_size)
    client_ds = []
    for k in range(1, 5):
        client_ds.append(MSDataset.from_df(df_global[df_global["hospital_id"] == k], data_root, train=False, img_size=img_size))
    return global_ds, client_ds
