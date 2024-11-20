import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR
from torch.nn import functional as F
import pytorch_lightning as pl
from copy import deepcopy
import math
from attention_smithy.components import Encoder, Decoder, EncoderLayer, DecoderLayer
from transformers import get_linear_schedule_with_warmup
from geneformer.loss import MaskedLoss

class Geneformer(pl.LightningModule):
    def __init__(self,
                 vocab_size: int,
                 embedding_dimension: int,
                 self_attention,
                 feedforward_network,
                 numeric_embedding_facade,
                 dropout,
                 num_layers,
                 padding_token,
                 ):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.numeric_embedding_facade = numeric_embedding_facade
        self.token_embedding = nn.Embedding(vocab_size, embedding_dimension)
        encoder_layer = EncoderLayer(embedding_dimension, self_attention, feedforward_network, dropout)
        self.encoder = Encoder(encoder_layer, number_of_layers=num_layers)
        self.loss_method = MaskedLoss(embedding_dimension, vocab_size, padding_token)

    def forward(self, src_tensor, src_padding_mask):
        src_embedding = self.token_embedding(src_tensor) * math.sqrt(self.embedding_dimension)
        position_embedding = self.numeric_embedding_facade.calculate_sinusoidal_and_learned_tokenizations(src_embedding)
        event_encoded = self.encoder(src=src_embedding + position_embedding, src_padding_mask=src_padding_mask, numeric_embedding_facade=self.numeric_embedding_facade)
        return event_encoded

    def training_step(self, batch, batch_idx):
        masked_tensor, padding_mask, original_masked_value_tensor = batch
        logits = self(masked_tensor, padding_mask)
        loss = self.loss_method(logits, original_masked_value_tensor)
        self.log("train_loss", loss, prog_bar=False, batch_size=logits.shape[0])
        return loss

    def validation_step(self, batch, batch_idx):
        masked_tensor, padding_mask, original_masked_value_tensor = batch
        logits = self(masked_tensor, padding_mask)
        loss = self.loss_method(logits, original_masked_value_tensor)
        self.log("val_loss", loss, prog_bar=False, batch_size=logits.shape[0])
        return loss

    def configure_optimizers(self):
        optimizer = Adam(params=self.parameters(), lr=1e-3, weight_decay=0.001)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=10000,
                                                    num_training_steps=self.trainer.max_steps)
        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]

