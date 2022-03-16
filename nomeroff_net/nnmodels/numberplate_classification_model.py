"""
Numberplate Classification Model
python3 ./nomeroff_net/nnmodels/numberplate_classification_model.py
"""
import torch
from typing import Any
from pytorch_lightning import LightningModule


class ClassificationNet(LightningModule):
    def __init__(self):
        super(ClassificationNet, self).__init__()

    def forward(self, *args, **kwargs) -> Any:
        pass

    def training_step(self, batch, batch_idx):
        loss, acc = self.step(batch)
        self.log(f'Batch {batch_idx} train_loss', loss)
        self.log(f'Batch {batch_idx} accuracy', acc)
        return {
            'loss': loss,
            'acc': acc,
        }

    def validation_step(self, batch, batch_idx):
        loss, acc = self.step(batch)
        self.log('val_loss', loss)
        self.log(f'val_accuracy', acc)
        return {
            'val_loss': loss,
            'val_acc': acc,
        }

    def test_step(self, batch, batch_idx):
        loss, acc = self.step(batch)
        self.log('test_loss', loss)
        self.log(f'test_accuracy', acc)
        return {
            'test_loss': loss,
            'test_acc': acc,
        }


if __name__ == "__main__":
    classification_net = ClassificationNet()
    x = torch.rand((1, 64, 295))
    y = classification_net(x)
