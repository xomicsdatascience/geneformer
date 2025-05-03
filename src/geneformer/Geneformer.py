import torch
from torch import nn
from torch.optim import Adam
from torch.nn import functional as F
import pytorch_lightning as pl
import math
from attention_smithy.components import Encoder, EncoderLayer, MultiheadAttention, FeedForwardNetwork, PerceiverEncoder, PerceiverEncoderLayer
from attention_smithy.numeric_embeddings import (
    SinusoidalPositionEmbedding, LearnedPositionEmbedding,
    RotaryPositionEmbedding, ALiBiPositionEmbedding,
    NumericEmbeddingManager
)
from attention_smithy.attention import StandardAttentionMethod, LongformerAttentionMethod, LinformerAttentionMethod
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
            'attention_method': 'longformer',
            'perceiver_latent_encoder_num_layers': 3,
            'perceiver_latent_length': 512,
            'longformer_local_attention_window_width': 128,
            'linformer_projected_k': 128,
            'maximum_sequence_length': 2048,
        }

        self.config.update(kwargs)
        self.save_hyperparameters()

        self.embedding_dimension = self.config['embedding_dimension']
        self.learning_rate = self.config['learning_rate']
        self.weight_decay = self.config['weight_decay']
        self.num_warmup_steps = self.config['num_warmup_steps']

        self.token_embedding = nn.Embedding(vocab_size, self.embedding_dimension)
        self.numeric_embedding_manager = self._create_embedding_manager()

        self._create_encoder()
        self.loss_method = MaskedLoss(self.embedding_dimension, vocab_size, padding_token)

    def forward(self, src_tensor, src_padding_mask):
        src_embedding = self.token_embedding(src_tensor) * math.sqrt(self.embedding_dimension)
        position_embedding = self.numeric_embedding_manager.create_positional_or_custom_embedding(
            token_embedding=src_embedding
        )
        batch_size, seq_len = src_tensor.shape
        global_attention_mask = torch.zeros(batch_size, seq_len, dtype=torch.int)
        global_attention_mask[:, 0] = 1

        event_encoded = self.encoder(
            src=src_embedding + position_embedding,
            src_padding_mask=src_padding_mask,
            numeric_embedding_manager=self.numeric_embedding_manager,
            global_attention_mask=global_attention_mask,
        )
        if self.config['attention_method'] == 'perceiver':
            event_encoded = self.decoder(src_embedding, event_encoded, src_padding_mask=None, numeric_embedding_manager=self.numeric_embedding_manager)
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

    def _create_encoder(self):
        if self.config['attention_method'] == 'perceiver':
            self._create_perceiver_encoder()
        else:
            if self.config['attention_method'] == 'longformer':
                attention_method = LongformerAttentionMethod(
                    attention_window=self.config['longformer_local_attention_window_width'],
                    dropout=self.config['dropout'])
            elif self.config['attention_method'] == 'linformer':
                attention_method = LinformerAttentionMethod(embedding_dim=self.config['embedding_dimension'],
                                                            sequence_length=self.config['maximum_sequence_length'],
                                                            k=self.config['linformer_projected_k'],
                                                            dropout=self.config['dropout'])
            self.encoder = self._create_attention_specified_encoder(attention_method)

    def _create_attention_specified_encoder(self, attention_method):
        self_attention = MultiheadAttention(
            embedding_dimension=self.embedding_dimension,
            number_of_heads=self.config['number_of_heads'],
            attention_method=attention_method,
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
        encoder = Encoder(encoder_layer, number_of_layers=self.config['num_layers'])
        return encoder

    def _create_perceiver_encoder(self):
        attention_method = StandardAttentionMethod(self.config['dropout'])
        self_attention = MultiheadAttention(
            embedding_dimension=self.embedding_dimension,
            number_of_heads=self.config['number_of_heads'],
            attention_method=attention_method,
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
        encoder = Encoder(encoder_layer, number_of_layers=self.config['perceiver_latent_encoder_num_layers'])
        perceiver_layer = PerceiverEncoderLayer(
            self.embedding_dimension,
            self_attention,
            feedforward_network,
            encoder,
            self.config['dropout'],
        )
        self.encoder = PerceiverEncoder(
            self.embedding_dimension,
            latent_length=self.config['perceiver_latent_length'],
            perceiver_encoder_layer=perceiver_layer,
            number_of_layers=self.config['num_layers'],
        )

        class IdentityModule(nn.Module):
            def forward(self, x, **kwargs):
                return x

        self.decoder = PerceiverEncoderLayer(
            self.embedding_dimension,
            self_attention,
            feedforward_network,
            IdentityModule(),
            self.config['dropout'],
        )

    def _create_embedding_manager(self):
        embedding_strategies = []
        if self.config['use_sinusoidal']:
            embedding_strategies.append(SinusoidalPositionEmbedding(self.config['embedding_dimension']))

        if self.config['use_learned']:
            embedding_strategies.append(LearnedPositionEmbedding(max_sequence_length=3_000, embedding_dimension=self.config['embedding_dimension']))

        if self.config['use_rotary']:
            embedding_strategies.append(RotaryPositionEmbedding(self.config['embedding_dimension'] // self.config['number_of_heads']))

        if self.config['use_alibi']:
            embedding_strategies.append(ALiBiPositionEmbedding(self.config['number_of_heads']))

        return NumericEmbeddingManager(embedding_strategies)
