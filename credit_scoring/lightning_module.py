import pytorch_lightning as pl
import torch
from torch import nn
from torchmetrics.classification import BinaryAUROC


class CreditRiskLitModule(pl.LightningModule):
    def __init__(self, model, lr):
        super().__init__()
        self.model = model
        self.lr = lr

        self.loss_fn = nn.BCEWithLogitsLoss()
        self.auroc = BinaryAUROC()

    def forward(self, features):
        return self.model(features)

    def training_step(self, batch, batch_idx):
        features, targets = batch
        logits = self(features).squeeze(1)
        loss = self.loss_fn(logits, targets)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        features, targets = batch
        logits = self(features).squeeze(1)
        loss = self.loss_fn(logits, targets)
        probs = torch.sigmoid(logits)
        auc = self.auroc(probs, targets.int())
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/roc_auc", auc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        features, targets = batch
        logits = self(features).squeeze(1)
        loss = self.loss_fn(logits, targets)
        probs = torch.sigmoid(logits)
        auc = self.auroc(probs, targets.int())
        self.log("test/loss", loss, prog_bar=True)
        self.log("test/roc_auc", auc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
