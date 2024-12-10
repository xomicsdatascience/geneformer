"""
import logging
import sys
import io
from datetime import datetime

class StreamToLogger(io.TextIOBase):
    def __init__(self, logger, level=logging.INFO):
        self.logger = logger
        self.level = level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.level, line.rstrip())

# Configure the logging module
logging.basicConfig(filename=f'outputs/logs/model_script_output_{datetime.now().strftime("%Y%m%d-%H%M%S-%f")}.log', level=logging.INFO)

# Redirect stdout to the logger
stdout_logger = logging.getLogger('STDOUT')
sys.stdout = StreamToLogger(stdout_logger, logging.INFO)

# Redirect stderr to the logger
stderr_logger = logging.getLogger('STDERR')
sys.stderr = StreamToLogger(stderr_logger, logging.ERROR)

print(' '.join(sys.argv))
#"""

import argparse
import logging
import os
import sys
import time
import warnings
from IPython.utils import io
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from geneformer import Geneformer
from geneformer.data import GeneformerDataModule
from attention_smithy.numeric_embeddings import SinusoidalPositionEmbedding, LearnedPositionEmbedding, RotaryPositionEmbedding, ALiBiPositionEmbedding, NumericEmbeddingFacade, NoAddEmbedding, PassthroughEmbedding
from attention_smithy.components import MultiheadAttention, FeedForwardNetwork
from attention_smithy.attention import BigBirdAttentionMethod, StandardAttentionMethod
from attention_smithy.utils import seed_everything
from datasets import load_from_disk

warnings.filterwarnings("ignore")  # Disable data logger warnings
logging.getLogger("pytorch_lightning").setLevel(logging.INFO)  # Disable GPU/TPU prints

class TensorBoardLoggingModelCheckpoint(ModelCheckpoint):
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        super().on_save_checkpoint(trainer, pl_module, checkpoint)
        if self.monitor in trainer.callback_metrics:
            metric_value = trainer.callback_metrics[self.monitor]
            trainer.logger.experiment.add_scalar(
                f"checkpoint/{self.monitor}", metric_value, trainer.global_step
            )

def parse_args():
    parser = argparse.ArgumentParser(description="generformer-nas")
    parser.add_argument("--log_path", type=str, required=True, help="dir to place tensorboard logs from all trials")
    parser.add_argument('--sinusoidal_position', action='store_true', default=False)
    parser.add_argument('--rotary_position', action='store_true', default=False)
    parser.add_argument('--alibi_position', action='store_true', default=False)
    parser.add_argument('--learned_position', action='store_true', default=False)
    parser.add_argument("--embedding_dimension", type=int, required=True)
    parser.add_argument("--feedforward_dimension", type=int, required=True)
    parser.add_argument("--number_of_heads", type=int, required=True)
    parser.add_argument("--number_of_layers", type=int, required=True)
    parser.add_argument("--number_of_warmup_steps", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--weight_decay", type=float, required=True)
    parser.add_argument("--learning_rate", type=float, required=True)
    parser.add_argument("--dropout", type=float, required=True, help="dropout rate")
    parser.add_argument("--activation", type=str, required=True, help="activation function for all layers but the last one")
    return parser.parse_args()


def run_training_job(parsed_args, random_state=0):
    seed_everything(random_state)
    logger = TensorBoardLogger(
        "tb_logs",
        name=f"geneformer",
    )

    class ValidateAtCheckpoint(pl.Callback):
        def __init__(self, train_step_cutoff):
            self.train_step_cutoff = train_step_cutoff
            self.val_loss = -1

        def on_train_batch_end(self, trainer, pl_module, outputs):
            if trainer.global_step in self.train_step_cutoff:
                with torch.no_grad():
                    val_loss_accumulated = 0
                    batch_count = 0
                    for batch in trainer.val_dataloaders:
                        batch_count += 1
                        val_loss_accumulated += pl_module.validation_step(tuple([x.to(pl_module.device) for x in batch]), batch_idx)
                    self.val_loss = val_loss / batch_count
                trainer.should_stop = True
                trainer.train_dataloader.sampler.set_epoch(1_000_000)

    validation_checkpoint_callback = ValidateAtCheckpoint(train_step_cutoff=3_000)

    trainer = pl.Trainer(
        max_epochs=30,
        logger=logger,
        callbacks=[
            validation_checkpoint_callback,
        ],
        log_every_n_steps=200,
    )

    dataset = load_from_disk('../../data/')

    masking_token = 1
    padding_token = 0
    block_size = 1

    data_module = GeneformerDataModule(dataset=dataset, batch_size=parsed_args.batch_size, bb_block_size=block_size, test_val_size=0.01, padding_token=padding_token, masking_token=masking_token)

    sinusoidal_position_embedding = SinusoidalPositionEmbedding(parsed_args.embedding_dimension) if parsed_args.sinusoidal_position else NoAddEmbedding()
    learned_position_embedding = LearnedPositionEmbedding(max_sequence_length=3_000, embedding_dimension=parsed_args.embedding_dimension) if parsed_args.learned_position else NoAddEmbedding()
    rotary_position_embedding = RotaryPositionEmbedding(parsed_args.embedding_dimension // parsed_args.number_of_heads) if parsed_args.rotary_position else PassthroughEmbedding()
    alibi_position_embedding = ALiBiPositionEmbedding(parsed_args.number_of_heads) if parsed_args.alibi_position else NoAddEmbedding()

    numeric_embedding_facade = NumericEmbeddingFacade(sinusoidal_position=sinusoidal_position_embedding, learned_position=learned_position_embedding)
    #self_attention = MultiheadAttention(embedding_dimension= parsed_args.embedding_dimension, number_of_heads= parsed_args.number_of_heads, attention_method= StandardAttentionMethod(parsed_args.dropout))

    big_bird_attention_method = BigBirdAttentionMethod(block_size_query=block_size, block_size_kv=block_size, local_window_extension_length=1, num_random_blocks=3, dropout=parsed_args.dropout)
    self_attention = MultiheadAttention(embedding_dimension= parsed_args.embedding_dimension, number_of_heads= parsed_args.number_of_heads, attention_method= big_bird_attention_method)
    feedforward_network = FeedForwardNetwork(parsed_args.embedding_dimension, parsed_args.feedforward_dimension, parsed_args.activation, parsed_args.dropout)
    model = Geneformer(
        vocab_size=25425,
        self_attention=self_attention,
        feedforward_network=feedforward_network,
        numeric_embedding_facade=numeric_embedding_facade,
        embedding_dimension=parsed_args.embedding_dimension,
        dropout=parsed_args.dropout,
        num_layers=parsed_args.number_of_layers,
        padding_token=padding_token,
        learning_rate=parsed_args.learning_rate,
        weight_decay=parsed_args.weight_decay,
        num_warmup_steps=parsed_args.number_of_warmup_steps,
    )
    torch.set_printoptions(threshold=10000)

    trainer.fit(model, data_module)
    val_loss = validation_checkpoint_callback.val_loss
    return val_loss


if __name__ == "__main__":
    parsed_args = parse_args()
    loss = run_training_job(parsed_args)
    logger = TensorBoardLogger(parsed_args.log_path)
    logger.log_metrics({"val_loss": loss})
    logger.save()
    print(f'BEST VAL SCORE: {loss}')

