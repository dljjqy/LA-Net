import torch
import pytorch_lightning as pl
import numpy as np
from torch.utils.data import Dataset, DataLoader

class LADataset(Dataset):
    def __init__(self, pathB, pathF):
        super().__init__()
        self.B = np.load(pathB)
        self.F = np.load(pathF)

    def __len__(self):
        return self.B.shape[0]

    def __getitem__(self, idx):
        data = self.F[idx, :]
        b = self.B[idx, :]
        f = torch.from_numpy(data[-1:, ...]).to(torch.float32)
        data = torch.from_numpy(data).to(torch.float32)
        b = torch.from_numpy(b).to(torch.float32)
        return  data, b, f

class LADataModule(pl.LightningDataModule):
    
    def __init__(self, data_path, batch_size, input_mode='F'):
        super().__init__()
        self.trainF = f'{data_path}{input_mode}.npy'
        self.valF = f'{data_path}Val{input_mode}.npy'

        self.trainB = f'{data_path}B.npy'
        self.valB = f'{data_path}ValB.npy'
        self.batch_size = batch_size
    def setup(self, stage):    
        if stage == 'fit' or stage is None:
            self.train_dataset = LADataset(self.trainB, self.trainF)
            self.val_dataset = LADataset(self.valB, self.valF)

        if stage == 'test':
            pass

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=6)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1, shuffle=False, num_workers=6)
    
    def test_dataloader(self):
        pass  