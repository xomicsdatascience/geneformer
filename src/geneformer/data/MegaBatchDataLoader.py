import torch
from torch import nn
from torch.utils.data import DataLoader


class MegaBatchDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, num_batches_per_megabatch, padding_token, shuffle=False):
        super().__init__(dataset)
        self.megabatch_size = batch_size * num_batches_per_megabatch
        self.padding_token = padding_token
        self.shuffle = shuffle


    def megabatch_collate(self, batch):
        batch.sort(key=lambda x: len(x), reverse=True)

        batches = []
        for i in range(self.megabatch_size // self.batch_size):
            if i * self.batch_size > len(dataset):
                return batches
            batch_slice = batch[i * self.batch_size:(i + 1) * self.batch_size]
            padded_batch = nn.utils.rnn.pad_sequence(
                batch_slice,
                batch_first=True,
                padding_value=self.padding_token,
            )
            padding_mask = (padded_batch != self.self.padding_token)

            batches.append((padded_batch, padding_mask))

        return batches

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            indices = torch.randperm(len(self.dataset))

        for i in range(0, len(self.dataset), self.megabatch_size):
            megabatch = [self.dataset[j] for j in indices[i * self.megabatch_size:(i+1) * self.megabatch_size]]
            for batch in self.megabatch_collate(megabatch):
                yield batch