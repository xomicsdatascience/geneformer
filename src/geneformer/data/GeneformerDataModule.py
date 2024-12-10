import torch
from torch import nn
import pytorch_lightning as pl
import random
from torch.utils.data import DataLoader

class GeneformerDataModule(pl.LightningDataModule):
    def __init__(self, dataset, batch_size, num_batches_per_megabatch, padding_token, masking_token, test_val_size=0.01):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_batches_per_megabatch = num_batches_per_megabatch
        self.test_val_size = test_val_size
        self.padding_token = padding_token
        self.masking_token = masking_token

    def setup(self, stage=None):
        train_testvalid = self.dataset.train_test_split(test_size=self.test_val_size)
        test_valid = train_testvalid['test'].train_test_split(test_size=0.5)
        self.train_dataset = train_testvalid['train']
        self.val_dataset = test_valid['train']
        self.test_dataset = test_valid['test']

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn=self._collate_function, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=self._collate_function, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, collate_fn=self._collate_function, shuffle=True)

    def _collate_function(self, batch):
        input_tensors = [torch.tensor(x['input_ids']) for x in batch]
        masked_tensors, original_masked_value_tensors = mask_and_track(input_tensors, self.masking_token, self.padding_token)

        masked_tensor = nn.utils.rnn.pad_sequence(
            masked_tensors,
            batch_first=True,
            padding_value=self.padding_token,
        )
        original_masked_value_tensor = nn.utils.rnn.pad_sequence(
            original_masked_value_tensors,
            batch_first=True,
            padding_value=self.padding_token,
        )

        padding_mask = (masked_tensor != self.padding_token)

        return masked_tensor, padding_mask, original_masked_value_tensor


def mask_and_track(tensors, mask_token, padding_token, mask_prob=0.15):
    masked_tensors = []
    original_masked_value_tensors = []

    for tensor in tensors:
        mask = torch.rand(tensor.shape) < mask_prob
        original_masked = torch.clone(tensor)
        masked_tensor = torch.where(mask, torch.tensor(mask_token).to(tensor.device), tensor)

        original_masked[~mask] = padding_token
        original_masked[mask] = tensor[mask]

        masked_tensors.append(masked_tensor)
        original_masked_value_tensors.append(original_masked)

    return masked_tensors, original_masked_value_tensors
