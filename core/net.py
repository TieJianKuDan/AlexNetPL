from pytorch_lightning import LightningModule
import torch
from torch import nn
from torch.optim.adam import Adam
from torch.optim.sgd import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR


class AlexNet(nn.Module):

    def __init__(self, config):
        super(AlexNet, self).__init__()
        self._config = config
        self.features = nn.Sequential(
            nn.Conv2d(
                self._config.in_channels, 
                64, kernel_size=3, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, self._config.num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def warmup_lambda(warmup_steps, min_lr_ratio=0.1):
    def ret_lambda(epoch):
        if epoch <= warmup_steps:
            return min_lr_ratio + (1.0 - min_lr_ratio) * epoch / warmup_steps
        else:
            return 1.0
    return ret_lambda

class AlexNetPL(LightningModule):
    
    def __init__(self, config, sample_num) -> None:
        '''
        config = Namespace({
            "optim": {
            ...
            },
            
        })
        '''
        super(AlexNetPL, self).__init__()
        self.model = AlexNet(config.model)
        self.config = config
        self.max_steps = \
            config.optim.max_epochs * sample_num / config.optim.batch_size
        self.loss = nn.CrossEntropyLoss()
        self.save_hyperparameters()

    def forward(self, batch) -> torch.Tensor:
        return self.model(batch)

    def configure_optimizers(self):
        optim_config = self.config.optim
        # Adam is bad for AlexNet, Why?
        # optimizer = Adam(
        #     self.model.parameters(), 
        #     lr=optim_config.lr,
        #     betas=optim_config.betas
        # )
        optimizer = SGD(
            self.model.parameters(),
            lr=optim_config.lr,
            momentum=optim_config.momentum
        )
        warmup_scheduler = LambdaLR(
            optimizer=optimizer, 
            lr_lambda=warmup_lambda(optim_config.warmup_steps)
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer=optimizer,
            T_max=self.max_steps - optim_config.warmup_steps,
            eta_min=optim_config.min_lr_ratio * optim_config.lr
        )
        lr_scheduler = SequentialLR(
            optimizer=optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[optim_config.warmup_steps]
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler
        }

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_h = self(x)
        l = self.loss(y_h, y)
        self.log("train_loss", l, prog_bar=True)
        return l

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_h = self(x)
        l = self.loss(y_h, y)
        self.log("val_loss", l, prog_bar=True)
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_h = self(x)
        l = self.loss(y_h, y)
        self.log("test_loss", l, prog_bar=True)
        pred = torch.argmax(y_h, 1)
        accuracy = (pred == y).sum() / len(y)
        self.log("accuracy", accuracy, prog_bar=True)