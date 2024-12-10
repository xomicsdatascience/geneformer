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
                 learning_rate,
                 weight_decay,
                 num_warmup_steps,
                 num_masked_tokens_as_global:int = 10,
                 ):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.numeric_embedding_facade = numeric_embedding_facade
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_warmup_steps = num_warmup_steps
        self.token_embedding = nn.Embedding(vocab_size, embedding_dimension)
        encoder_layer = EncoderLayer(embedding_dimension, self_attention, feedforward_network, dropout)
        self.encoder = Encoder(encoder_layer, number_of_layers=num_layers)
        self.loss_method = MaskedLoss(embedding_dimension, vocab_size, padding_token)
        self.padding_token = padding_token
        self.num_masked_tokens_as_global = num_masked_tokens_as_global

    def forward(self, src_tensor, src_padding_mask, **kwargs):
        src_embedding = self.token_embedding(src_tensor) * math.sqrt(self.embedding_dimension)
        sequence_length = src_embedding.shape[1]
        position_embedding = self.numeric_embedding_facade.calculate_sinusoidal_and_learned_tokenizations(src_embedding, sequence_length=sequence_length)
        event_encoded = self.encoder(src=src_embedding + position_embedding, src_padding_mask=src_padding_mask, numeric_embedding_facade=self.numeric_embedding_facade, **kwargs)
        return event_encoded

    def training_step(self, batch, batch_idx):
        masked_tensor, padding_mask, original_masked_value_tensor = batch
        global_tokens = (original_masked_value_tensor != self.padding_token).int()
        logits = self(masked_tensor, padding_mask, global_tokens_query=global_tokens, global_tokens_kv=global_tokens)
        loss = self.loss_method(logits, original_masked_value_tensor)
        self.log("train_loss", loss, prog_bar=False, batch_size=logits.shape[0])
        return loss

    def validation_step(self, batch, batch_idx):
        masked_tensor, padding_mask, original_masked_value_tensor = batch
        global_tokens = (original_masked_value_tensor != self.padding_token).int()
        for i in range(global_tokens.shape[0]):
            global_tokens[i] = flip_to_zeros(global_tokens[i], self.num_masked_tokens_as_global)
        logits = self(masked_tensor, padding_mask, global_tokens_query=global_tokens, global_tokens_kv=global_tokens)
        loss = self.loss_method(logits, original_masked_value_tensor)
        self.log("val_loss", loss, prog_bar=False, batch_size=logits.shape[0])
        return loss

    def configure_optimizers(self):
        optimizer = Adam(params=self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.num_warmup_steps,
                                                    num_training_steps=self.trainer.max_steps)
        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]

def flip_to_zeros(tensor, num_masked_tokens_as_global):
    # Get the indices of the 1's in the tensor
    ones_indices = torch.nonzero(tensor).flatten()

    # If there are less than 10 1's, return the original tensor
    if len(ones_indices) < 10:
        return tensor

    # Randomly select 10 indices from the 1's
    selected_indices = torch.randperm(len(ones_indices))[num_masked_tokens_as_global:]

    # Flip the selected 1's to 0's
    tensor[ones_indices[selected_indices]] = 0
    #print(tensor)
    return tensor
