from pathlib import Path
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset as TorchDataset, DataLoader
import torchvision.transforms as T

_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]


def get_transform(train, img_size=224):
    steps = [T.Resize((img_size, img_size))]
    if train:
        steps.append(T.RandomHorizontalFlip())
    steps += [
        T.ToTensor(),
        T.Lambda(lambda x: x.repeat(3, 1, 1)),
        T.Normalize(_MEAN, _STD),
    ]
    return T.Compose(steps)


class Dataset(TorchDataset):
    def __init__(self, csv_path, data_root, train=True, img_size=224):
        self.df = pd.read_csv(csv_path)
        self.data_root = Path(data_root)
        self.transform = get_transform(train, img_size)

    @classmethod
    def from_df(cls, df, data_root, train=False, img_size=224):
        obj = cls.__new__(cls)
        obj.df = df.reset_index(drop=True)
        obj.data_root = Path(data_root)
        obj.transform = get_transform(train, img_size)
        return obj

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(self.data_root / row["image_path"]).convert("L")
        label = torch.tensor(float(row["label_id"]), dtype=torch.float32)
        return self.transform(img), label


def get_dataloader(dataset, batch_size, shuffle=True, num_workers=0):
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
        datasets.append(Dataset(csv, data_root, train=True, img_size=img_size))
    return datasets


def load_test_datasets(global_test_csv, data_root, img_size=224):
    df_global = pd.read_csv(global_test_csv)
    global_ds = Dataset.from_df(df_global, data_root, train=False, img_size=img_size)
    client_ds = []
    for k in range(1, 5):
        df_k = df_global[df_global["hospital_id"] == k]
        client_ds.append(Dataset.from_df(df_k, data_root, train=False, img_size=img_size))
    return global_ds, client_ds
