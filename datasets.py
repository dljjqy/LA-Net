from re import A
import torch
import pytorch_lightning as pl
import numpy as np
from torch.utils.data import Dataset, DataLoader

class LADataset(Dataset):
    def __init__(self, pathB, pathF, a, n, numerical_method='fd'):
        super().__init__()
        self.B = np.load(pathB)
        self.F = np.load(pathF)
        if numerical_method == 'fd':
            x = np.linspace(-a, a, n)
            y = np.linspace(-a, a, n)
            self.xx, self.yy = np.meshgrid(x, y)
        elif numerical_method == 'fv':
            h = 2*a / n
            x = np.linspace(-a + h/2, a - h/2, n)
            y = np.linspace(-a + h/2, a - h/2, n)
            self.xx, self.yy = np.meshgrid(x, y)

    def __len__(self):
        return self.B.shape[0]

    def __getitem__(self, idx):
        f = self.F[idx, :]
        b = self.B[idx, :]
        data = np.stack([self.xx, self.yy, f], axis=0)
        
        f = torch.from_numpy(f).to(torch.float32)
        b = torch.from_numpy(b).to(torch.float32)
        data = torch.from_numpy(data).to(torch.float32)
        return  data, b, f[None, ...]

class LADataModule(pl.LightningDataModule):
    
    def __init__(self, data_path, batch_size, a, n, numerical_method='fd'):
        super().__init__()
        self.a ,self.n, self.numerical_method = a, n, numerical_method
        self.trainF = f'{data_path}{numerical_method}_F.npy'
        self.valF = f'{data_path}{numerical_method}_ValF.npy'

        self.trainB = f'{data_path}{numerical_method}_B.npy'
        self.valB = f'{data_path}{numerical_method}_ValB.npy'
        self.batch_size = batch_size

    def setup(self, stage):    
        if stage == 'fit' or stage is None:
            self.train_dataset = LADataset(self.trainB, self.trainF, self.a, self.n, self.numerical_method)
            self.val_dataset = LADataset(self.valB, self.valF, self.a, self.n, self.numerical_method)
        if stage == 'test':
            pass

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=6)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1, shuffle=False, num_workers=6)
    
    def test_dataloader(self):
        pass  