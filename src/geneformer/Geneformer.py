import torch
from torch import nn
from torch.optim import Adam
from torch.nn import functional as F
import pytorch_lightning as pl
import math
from attention_smithy.components import Encoder, EncoderLayer, MultiheadAttention, FeedForwardNetwork
from attention_smithy.numeric_embeddings import (
    SinusoidalPositionEmbedding, LearnedPositionEmbedding,
    RotaryPositionEmbedding, ALiBiPositionEmbedding,
    NumericEmbeddingManager, NoAddEmbedding, PassthroughEmbedding
)
from attention_smithy.attention import StandardAttentionMethod
from transformers import get_linear_schedule_with_warmup
from geneformer.loss import MaskedLoss


class Geneformer(pl.LightningModule):
    def __init__(self,
                 vocab_size: int,
                 padding_token: int,
                 **kwargs):
        """
        Initialize the Geneformer model with required parameters and optional kwargs.

        Required Args:
            vocab_size (int): Size of vocabulary
            padding_token (int): Padding token ID for vocabulary

        Optional Args (kwargs):
            embedding_dimension (int): Dimension of embeddings (default: 256)
            number_of_heads (int): Number of attention heads (default: 4)
            dropout (float): Dropout rate (default: 0.2)
            activation (str): Activation function (default: 'relu')
            feedforward_dimension (int): Dimension of feedforward layer (default: 512)
            num_layers (int): Number of encoder layers (default: 6)
            learning_rate (float): Learning rate (default: 1e-3)
            weight_decay (float): Weight decay (default: 0.001)
            num_warmup_steps (int): Warmup steps for scheduler (default: 10000)
            use_sinusoidal (bool): Use sinusoidal position embedding (default: True)
            use_learned (bool): Use learned position embedding (default: False)
            use_rotary (bool): Use rotary position embedding (default: False)
            use_alibi (bool): Use ALiBi position embedding (default: False)
        """
        super().__init__()

        self.config = {
            'embedding_dimension': 256,
            'number_of_heads': 4,
            'dropout': 0.2,
            'activation': 'relu',
            'feedforward_dimension': 512,
            'num_layers': 6,
            'learning_rate': 1e-3,
            'weight_decay': 0.001,
            'num_warmup_steps': 10000,
            'use_sinusoidal': True,
            'use_learned': False,
            'use_rotary': False,
            'use_alibi': False,
        }

        self.config.update(kwargs)
        self.save_hyperparameters()

        self.embedding_dimension = self.config['embedding_dimension']
        self.learning_rate = self.config['learning_rate']
        self.weight_decay = self.config['weight_decay']
        self.num_warmup_steps = self.config['num_warmup_steps']

        self.token_embedding = nn.Embedding(vocab_size, self.embedding_dimension)
        self.numeric_embedding_manager = self._create_embedding_manager()

        self_attention = MultiheadAttention(
            embedding_dimension=self.embedding_dimension,
            number_of_heads=self.config['number_of_heads'],
            attention_method=StandardAttentionMethod(self.config['dropout'])
        )

        feedforward_network = FeedForwardNetwork(
            self.embedding_dimension,
            self.config['feedforward_dimension'],
            self.config['activation'],
            self.config['dropout']
        )

        encoder_layer = EncoderLayer(
            self.embedding_dimension,
            self_attention,
            feedforward_network,
            self.config['dropout']
        )

        self.encoder = Encoder(encoder_layer, number_of_layers=self.config['num_layers'])
        self.loss_method = MaskedLoss(self.embedding_dimension, vocab_size, padding_token)

    def _create_embedding_manager(self):
        sinusoidal_position = (
            SinusoidalPositionEmbedding(self.embedding_dimension)
            if self.config['use_sinusoidal'] else NoAddEmbedding()
        )

        learned_position = (
            LearnedPositionEmbedding(max_sequence_length=3_000, embedding_dimension=self.embedding_dimension)
            if self.config['use_learned'] else NoAddEmbedding()
        )

        rotary_position = (
            RotaryPositionEmbedding(self.embedding_dimension // self.config['number_of_heads'])
            if self.config['use_rotary'] else PassthroughEmbedding()
        )

        alibi_position = (
            ALiBiPositionEmbedding(self.config['number_of_heads'])
            if self.config['use_alibi'] else NoAddEmbedding()
        )

        return NumericEmbeddingManager(
            sinusoidal_position=sinusoidal_position,
            learned_position=learned_position,
            rotary_position=rotary_position,
            alibi_position=alibi_position
        )

    def forward(self, src_tensor, src_padding_mask):
        src_embedding = self.token_embedding(src_tensor) * math.sqrt(self.embedding_dimension)
        position_embedding = self.numeric_embedding_manager.calculate_sinusoidal_and_learned_tokenizations(
            src_embedding)
        event_encoded = self.encoder(
            src=src_embedding + position_embedding,
            src_padding_mask=src_padding_mask,
            numeric_embedding_manager=self.numeric_embedding_manager
        )
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
        optimizer = Adam(params=self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=self.trainer.max_steps
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]