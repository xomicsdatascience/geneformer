import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from geneformer import Geneformer
from geneformer.fine_tuned_model import GeneformerForSequenceClassification, GeneformerDataModuleForSequenceClassification
from attention_smithy.numeric_embeddings import SinusoidalPositionEmbedding, NumericEmbeddingManager
from attention_smithy.components import MultiheadAttention, FeedForwardNetwork
from attention_smithy.attention import StandardAttentionMethod
from attention_smithy.utils import seed_everything
from datasets import load_from_disk
import time
from datetime import timedelta

torch.set_float32_matmul_precision('medium')

def fine_tune_model(
        embed_dim=256,
        num_heads=4,
        dim_feedforward=512,
        dropout=0.2,
        num_layers=6,
        random_seed=0,
        batch_size=32,
):
    seed_everything(random_seed)

    logger = WandbLogger(project='geneformer-fine-tune')

    train_loss_checkpoint_callback = ModelCheckpoint(
        dirpath=f"checkpoints/",
        every_n_train_steps=50,
        filename="fine-tune-train-loss-{epoch:02d}-{step:08d}",
        save_last=True,
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
                        pl_module.validation_step(batch, batch_idx)
                validation_end_time = time.time()
                validation_time = validation_end_time - validation_start_time

                print(f"Time spent on validation: {self.format_time(validation_time)}")

                self.last_checkpoint_time = validation_end_time
    trainer = pl.Trainer(
        max_epochs=20,
        logger=logger,
        callbacks=[
            train_loss_checkpoint_callback,
            ValidateAtCheckpoints(list(range(0, 18100, 500)) + [20, 40, 60, 80, 100]),
        ],
        log_every_n_steps=20,

    )


    dataset = load_from_disk('data/example_input_files/cell_classification/disease_classification/human_dcm_hcm_nf.dataset/')

    masking_token = 1
    padding_token = 0

    cell_types = ['ActivatedFibroblast', 'Adipocyte', 'Cardiomyocyte1', 'Cardiomyocyte2', 'Cardiomyocyte3', 'Endocardial', 'Endothelial1', 'Endothelial2', 'Endothelial3', 'Epicardial', 'Fibroblast1', 'Fibroblast2', 'LymphaticEndothelial', 'Lymphocyte', 'Macrophage', 'MastCell', 'Neuronal', 'Pericyte1', 'Pericyte2', 'ProliferatingMacrophage', 'VSMC']
    cell_tokenizer = {cell_type:idx for idx, cell_type in enumerate(cell_types)}

    data_module = GeneformerDataModuleForSequenceClassification(dataset=dataset, batch_size=batch_size, num_batches_per_megabatch=10, padding_token=padding_token, masking_token=masking_token, output_tokenizer=cell_tokenizer)
    sinusoidal_position_embedding = SinusoidalPositionEmbedding(embed_dim)
    numeric_embedding_manager = NumericEmbeddingManager(sinusoidal_position=sinusoidal_position_embedding)
    self_attention = MultiheadAttention(embedding_dimension = embed_dim, number_of_heads = num_heads, attention_method = StandardAttentionMethod(dropout))
    feedforward_network = FeedForwardNetwork(embed_dim, dim_feedforward, 'relu', dropout)

    pretrained_model = Geneformer(
        vocab_size=25500,
        self_attention=self_attention,
        feedforward_network=feedforward_network,
        numeric_embedding_manager=numeric_embedding_manager,
        embedding_dimension=embed_dim,
        dropout=dropout,
        num_layers=num_layers,
        padding_token=padding_token,
        learning_rate=1e-3,
        weight_decay=0.001,
        num_warmup_steps=10000,
    )

    l = torch.load('checkpoints/train-loss-epoch00-step00384800_progress_saved.ckpt', map_location=pretrained_model.device)
    pretrained_model.load_state_dict(l['state_dict'])
    pretrained_model.encoder.freeze_layers(number_of_layers=2)
    model = GeneformerForSequenceClassification(pretrained_model, number_of_classes=len(cell_tokenizer))
    trainer.fit(model, data_module)
    #torch.save(model, 'model.pth')


if __name__ == "__main__":
    print("start")
    fine_tune_model()
