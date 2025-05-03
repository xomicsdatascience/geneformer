#"""
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
from attention_smithy.utils import seed_everything
from datasets import load_from_disk
from datetime import timedelta

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
    parser = argparse.ArgumentParser(description="machine-translation")

    # Required arguments
    parser.add_argument("--log_path", type=str, required=True, help="Dir to place logs from all trials")

    # Positional encoding options (boolean flags)
    parser.add_argument('--sinusoidal_position', action='store_true', help='Use sinusoidal positional encoding')
    parser.add_argument('--learned_position', action='store_true', help='Use learned positional encoding')
    parser.add_argument('--rotary_position', action='store_true', help='Use rotary positional encoding')
    parser.add_argument('--alibi_position', action='store_true', help='Use ALiBi positional encoding')

    # Model architecture
    parser.add_argument('--embedding_dimension', type=int, default=256, help='Embedding dimension (default: 256)')
    parser.add_argument('--number_of_heads', type=int, default=4, help='Number of attention heads (default: 4)')
    parser.add_argument('--dropout', type=float, required=True, help='Dropout rate (default: 0.2)')
    parser.add_argument('--activation', type=str, required=True, help='Activation function (default: relu)')
    parser.add_argument('--feedforward_dimension', type=int, default=512, help='Feedforward dimension (default: 512)')
    parser.add_argument('--number_of_layers', type=int, default=6, help='Number of transformer layers (default: 6)')

    # Training and optimization
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Initial learning rate (default: 1e-3)')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='Weight decay for optimizer (default: 0.001)')
    parser.add_argument('--scheduler_warmup_steps', type=int, default=10000, help='Number of warmup steps for LR scheduler (default: 10000)')

    parser.add_argument('--attention_method', type=str, required=True, help='Type of attention used (options are: standard, longformer, linformer, perceiver)')
    parser.add_argument('--perceiver_latent_encoder_num_layers', type=int, default=3, help='Each perceiver layer has a latent encoder. This parameter determines the number of layers in that encoder.')
    parser.add_argument('--perceiver_latent_length', type=int, default=512, help='The "sequence length" of the latent space.')
    parser.add_argument('--longformer_local_attention_window_width', type=int, default=128, help='The number to either side of a given token that attends to a token in local attention.')
    parser.add_argument('--linformer_projected_k', type=int, default=128, help='Linformer projection k.')
    parser.add_argument('--maximum_sequence_length', type=int, default=2048, help='Linformer requires a prior knowledge of the expected sequence length. This, all sequences are set to the maximum (2048).')

    return parser.parse_args()

def get_config_from_args():
    args = parse_args()

    config = {
        'log_path':args.log_path,
        'embedding_dimension': args.embedding_dimension,
        'number_of_heads': args.number_of_heads,
        'dropout': args.dropout,
        'activation': args.activation,
        'feedforward_dimension': args.feedforward_dimension,
        'num_layers': args.number_of_layers,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'num_warmup_steps': args.scheduler_warmup_steps,
        'use_sinusoidal': args.sinusoidal_position,
        'use_learned': args.learned_position,
        'use_rotary': args.rotary_position,
        'use_alibi': args.alibi_position,
        'attention_method': args.attention_method,
        'perceiver_latent_encoder_num_layers': args.perceiver_latent_encoder_num_layers,
        'perceiver_latent_length': args.perceiver_latent_length,
        'longformer_local_attention_window_width': args.longformer_local_attention_window_width,
        'linformer_projected_k': args.linformer_projected_k,
        'maximum_sequence_length': args.maximum_sequence_length,
    }

    return config

def run_training_job(configs, random_state=0):
    seed_everything(random_state)
    torch.set_float32_matmul_precision('medium')
    logger = TensorBoardLogger(
        "tb_logs",
        name=f"geneformer",
    )

    class StopAfterBatches(pl.Callback):
        def __init__(self, max_batches):
            super().__init__()
            self.max_batches = max_batches

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            if trainer.global_step >= self.max_batches:
                trainer.should_stop = True
                trainer.limit_val_batches = 0

    class ValidateAtCheckpoints(pl.Callback):
        def __init__(self, checkpoints):
            self.checkpoints = checkpoints
            self.start_time = time.time()
            self.last_checkpoint_time = self.start_time
            self.best_val_loss = float('inf')

        def format_time(self, seconds):
            return str(timedelta(seconds=int(seconds)))

        def on_train_batch_end(self, trainer, pl_module, outputs, train_batch, batch_idx, **kwargs):
            if batch_idx in self.checkpoints:
                current_time = time.time()
                elapsed_time_from_start = current_time - self.start_time
                elapsed_time_from_last_checkpoint = current_time - self.last_checkpoint_time

                print(f"Time elapsed from start: {self.format_time(elapsed_time_from_start)}")
                print(f"Time elapsed from last checkpoint: {self.format_time(elapsed_time_from_last_checkpoint)}")

                validation_start_time = time.time()
                val_losses = []
                with torch.no_grad():
                    for batch in trainer.val_dataloaders:
                        batch = tuple(x.to(pl_module.device) for x in batch)
                        val_loss = pl_module.validation_step(batch, batch_idx)
                        val_losses.append(val_loss)
                validation_end_time = time.time()
                validation_time = validation_end_time - validation_start_time

                avg_val_loss = sum(val_losses) / len(val_losses)
                if avg_val_loss < self.best_val_loss:
                    self.best_val_loss = avg_val_loss
                    print(f"New best avg val_loss: {self.best_val_loss:.4f}")

                print(f"Time spent on validation: {self.format_time(validation_time)}")

                self.last_checkpoint_time = validation_end_time

    validation_checkpoint_callback = ValidateAtCheckpoints(list(range(0, 856020, 600))[6:])

    trainer = pl.Trainer(
        max_epochs=1,
        logger=logger,
        callbacks=[
            validation_checkpoint_callback,
            StopAfterBatches(max_batches=6001)
        ],
        log_every_n_steps=200,
    )

    dataset = load_from_disk('../data/')

    masking_token = 1
    padding_token = 0

    data_module = GeneformerDataModule(dataset=dataset, batch_size=32, num_batches_per_megabatch=10, padding_token=padding_token, masking_token=masking_token)
    model = Geneformer(
        vocab_size=25500,
        padding_token=padding_token,
        **configs
    )

    trainer.fit(model, data_module)
    val_loss = validation_checkpoint_callback.best_val_loss
    return val_loss


if __name__ == "__main__":
    config = get_config_from_args()
    loss = run_training_job(config)
    logger = TensorBoardLogger(config['log_path'])
    logger.log_metrics({"val_loss": loss})
    logger.save()
    print(f'BEST VAL SCORE: {loss}')

