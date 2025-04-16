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
    parser.add_argument('--sinusoidal_position', action='store_true', required=True, help='Use sinusoidal positional encoding')
    parser.add_argument('--learned_position', action='store_true', required=True, help='Use learned positional encoding')
    parser.add_argument('--rotary_position', action='store_true', required=True, help='Use rotary positional encoding')
    parser.add_argument('--alibi_position', action='store_true', required=True, help='Use ALiBi positional encoding')

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

    return parser.parse_args()

def get_config_from_args():
    args = parse_args()

    config = {
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
    }

    return config

def run_training_job(configs, random_state=0):
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

    validation_checkpoint_callback = ValidateAtCheckpoint(train_step_cutoff=12_000)

    trainer = pl.Trainer(
        max_epochs=30,
        logger=logger,
        callbacks=[
            validation_checkpoint_callback,
        ],
        log_every_n_steps=200,
    )

    dataset = load_from_disk('../data/')

    masking_token = 1
    padding_token = 0

    data_module = GeneformerDataModule(dataset=dataset, batch_size=64, num_batches_per_megabatch=10, test_val_size=0.01, padding_token=padding_token, masking_token=masking_token)
    model = Geneformer(
        vocab_size=25425,
        padding_token=padding_token,
        **configs
    )

    trainer.fit(model, data_module)
    val_loss = validation_checkpoint_callback.val_loss
    return val_loss


if __name__ == "__main__":
    config = get_config_from_args()
    loss = run_training_job(config)
    logger = TensorBoardLogger(config['log_path'])
    logger.log_metrics({"val_loss": loss})
    logger.save()
    print(f'BEST VAL SCORE: {loss}')

