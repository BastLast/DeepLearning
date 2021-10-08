import numpy as np
from torch.utils.data import Dataset

class LoadImages(Dataset):
    def __init__(self, transformed_dataset_path: str, original_dataset_path: str) -> None:
        super(LoadImages, self).__init__()

        self.transformed_dataset_path = transformed_dataset_path
        self.original_dataset_path = original_dataset_path

        e1 = np.load(original_dataset_path)

        self.original_dataset = np.reshape(e1, (-1, 3, 96, 96))

        e2 = np.load(transformed_dataset_path)
        self.transformed_dataset = np.reshape(e2, (-1, 3, 96, 96))

        self.size = len(self.original_dataset)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return (self.transformed_dataset[idx], self.original_dataset[idx])
