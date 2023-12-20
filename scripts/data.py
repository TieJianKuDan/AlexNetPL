from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torchvision.datasets import MNIST
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torchvision import transforms

class MinistLDM(LightningDataModule):

    def __init__(self, val_ratio, batch_size, workers) -> None:
        super(MinistLDM, self).__init__()
        self.val_ratio = val_ratio
        self.batch_size = batch_size
        self.workers = workers
    
    def setup(self, stage: str) -> None:
        train_set = MNIST(
            "datasets", 
            train=True, 
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor()]
        ))
        self.train_set, self.val_set = random_split(
            train_set, 
            [1 - self.val_ratio, self.val_ratio]
        )
        self.test_set = MNIST(
            "datasets", 
            train=False, 
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor()]
        ))
    
    @property
    def sample_num(self):
        return len(self.train_set)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_set, 
            self.batch_size, 
            True,
            num_workers=self.workers,
            persistent_workers=True
        )
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.val_set, 
            self.batch_size, 
            False,
            num_workers=self.workers,
            persistent_workers=True
        )
    
    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.test_set, 
            self.batch_size, 
            False,
            num_workers=self.workers,
            persistent_workers=True
        )