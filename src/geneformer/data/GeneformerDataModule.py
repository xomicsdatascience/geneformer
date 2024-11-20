import torch
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
#from geneformer.data.MegaBatchDataLoader import MegaBatchDataLoader

class GeneformerDataModule(pl.LightningDataModule):
    def __init__(self, dataset, batch_size, num_batches_per_megabatch, padding_token, test_val_size=0.01):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_batches_per_megabatch = num_batches_per_megabatch
        self.test_val_size = test_val_size
        self.padding_token = padding_token

    def setup(self, stage=None):
        train_testvalid = self.dataset.train_test_split(test_size=self.test_val_size)
        test_valid = train_testvalid['test'].train_test_split(test_size=0.5)
        self.train_dataset = train_testvalid['train']
        self.val_dataset = test_valid['train']
        self.test_dataset = test_valid['test']

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn=self._collate_function, shuffle=True)
        #return MegaBatchDataLoader(self.train_dataset, self.batch_size, self.num_batches_per_megabatch, self.padding_token, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=self._collate_function, shuffle=True)
        #return MegaBatchDataLoader(self.val_dataset, self.batch_size, self.num_batches_per_megabatch, self.padding_token)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, collate_fn=self._collate_function, shuffle=True)
        #return MegaBatchDataLoader(self.test_dataset, self.batch_size, self.num_batches_per_megabatch, self.padding_token)

    def _collate_function(self, batch):
        input_tensors = [torch.tensor(x['input_ids']) for x in batch]
        input_tensor = nn.utils.rnn.pad_sequence(
            input_tensors,
            batch_first=True,
            padding_value=self.padding_token,
        )

        padding_mask = (input_tensor != self.padding_token)

        return input_tensor, padding_mask
