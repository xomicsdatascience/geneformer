import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR


class GeneformerForSequenceClassification(pl.LightningModule):
    def __init__(self, pretrained_model, number_of_classes=3, dropout_rate=0.1, learning_rate=2e-5):
        super().__init__()
        self.number_of_classes = number_of_classes
        self.learning_rate = learning_rate
        self.pretrained_model = pretrained_model
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(pretrained_model.embedding_dimension, number_of_classes)

    def forward(self, x, mask=None):
        encoder_output = self.pretrained_model(x, mask)
        pooled_output = encoder_output[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

    def training_step(self, batch, batch_idx):
        input_tensor, labels = batch
        input_tensor, labels = input_tensor.to(self.device), labels.to(self.device)
        logits = self(input_tensor)
        loss = F.cross_entropy(logits, labels)
        self.log('train_loss', loss, on_step=True)
        accuracy = (logits.argmax(dim=-1) == labels).float().mean()
        self.log('train_acc', accuracy, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_tensor, labels = batch
        input_tensor, labels = input_tensor.to(self.device), labels.to(self.device)
        logits = self(input_tensor)
        loss = F.cross_entropy(logits, labels)
        self.log('val_loss', loss)
        accuracy = (logits.argmax(dim=-1) == labels).float().mean()
        self.log('val_acc', accuracy)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=10000
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }