import torch
from torch import nn
import pytorch_lightning as pl
import random
from torch.utils.data import DataLoader

data_directory = 'data'

class GeneformerDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, padding_token, masking_token):
        super().__init__()
        self.batch_size = batch_size
        self.padding_token = padding_token
        self.masking_token = masking_token

    def setup(self, stage=None):
        self.train_dataset = LineIndexDataset(f'{data_directory}/train{self.de_filepath_suffix}', f'{data_directory}/train{self.en_filepath_suffix}', self.num_training_samples)
        self.val_dataset = LineIndexDataset(f'{data_directory}/val{self.de_filepath_suffix}', f'{data_directory}/val{self.en_filepath_suffix}', self.num_training_samples)
        self.test_dataset = LineIndexDataset(f'{data_directory}/test{self.de_filepath_suffix}', f'{data_directory}/test{self.en_filepath_suffix}', self.num_training_samples)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn=self._collate_function, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=self._collate_function)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, collate_fn=self._collate_function)

    def _collate_function(self, batch):
        masked_tensors, original_masked_value_tensors = mask_and_track(batch, self.masking_token, self.padding_token)

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
