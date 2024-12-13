import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from geneformer import Geneformer
from geneformer.data import GeneformerDataModule
from attention_smithy.numeric_embeddings import SinusoidalPositionEmbedding, NumericEmbeddingFacade
from attention_smithy.components import MultiheadAttention, FeedForwardNetwork
from attention_smithy.attention import StandardAttentionMethod
from attention_smithy.utils import seed_everything
from datasets import load_from_disk
import time
from datetime import timedelta

class TensorBoardLoggingModelCheckpoint(ModelCheckpoint):
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        super().on_save_checkpoint(trainer, pl_module, checkpoint)
        if self.monitor in trainer.callback_metrics:
            metric_value = trainer.callback_metrics[self.monitor]
            trainer.logger.experiment.add_scalar(
                f"checkpoint/{self.monitor}", metric_value, trainer.global_step
            )

def train_model(
        embed_dim=256,
        num_heads=4,
        dim_feedforward=512,
        dropout=0.2,
        num_layers=6,
        random_seed=0,
        batch_size=128,
):
    seed_everything(random_seed)

    logger = WandbLogger(project='geneformer')

    train_loss_checkpoint_callback = ModelCheckpoint(
        dirpath=f"checkpoints/",
        every_n_train_steps=50,
        filename="train-loss-{epoch:02d}-{step:08d}",
        save_last=True,
    )

    val_loss_checkpoint_callback = TensorBoardLoggingModelCheckpoint(
        monitor="val_loss",
        dirpath=f"checkpoints/",
        filename="best-val-loss-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        mode="min",
    )

    class ValidateAtCheckpoints(pl.Callback):
        def __init__(self, checkpoints):
            self.checkpoints = checkpoints
            self.start_time = time.time()
            self.last_checkpoint_time = self.start_time

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
                with torch.no_grad():
                    for batch in trainer.val_dataloaders:
                        pl_module.validation_step(tuple([x.to(pl_module.device) for x in batch]), batch_idx)
                validation_end_time = time.time()
                validation_time = validation_end_time - validation_start_time

                print(f"Time spent on validation: {self.format_time(validation_time)}")

                self.last_checkpoint_time = validation_end_time
    trainer = pl.Trainer(
        max_epochs=1,
        logger=logger,
        callbacks=[
            train_loss_checkpoint_callback,
            val_loss_checkpoint_callback,
            ValidateAtCheckpoints(list(range(0, 847880, 1000))[1:]),
        ],
        log_every_n_steps=200,

    )

    dataset = load_from_disk('/common/meyerjlab/caleb__geneformer_files/genecorpus_30M_2048.dataset/')

    masking_token = 1
    padding_token = 0

    data_module = GeneformerDataModule(dataset=dataset, batch_size=batch_size, num_batches_per_megabatch=10, padding_token=padding_token, masking_token=masking_token)

    sinusoidal_position_embedding = SinusoidalPositionEmbedding(embed_dim)
    numeric_embedding_facade = NumericEmbeddingFacade(sinusoidal_position=sinusoidal_position_embedding)
    self_attention = MultiheadAttention(embedding_dimension = embed_dim, number_of_heads = num_heads, attention_method = StandardAttentionMethod(dropout))
    feedforward_network = FeedForwardNetwork(embed_dim, dim_feedforward, 'relu', dropout)

    model = Geneformer(
        vocab_size=25425,
        self_attention=self_attention,
        feedforward_network=feedforward_network,
        numeric_embedding_facade=numeric_embedding_facade,
        embedding_dimension=embed_dim,
        dropout=dropout,
        num_layers=num_layers,
        padding_token=padding_token,
        learning_rate=1e-3,
        weight_decay=0.001,
        num_warmup_steps=10000,
    )

    trainer.fit(model, data_module)
    torch.save(model, 'model.pth')


if __name__ == "__main__":
    print("start")
    train_model()
